#include "vox2tet/image/image_utils.hpp"

#include "vox2tet/core/log.hpp"
#include "vox2tet/core/paths.hpp"
#include "vox2tet/image/critical_conn.hpp"
#include "vox2tet/image/edt.hpp"
#include "vox2tet/image/ext_volume.hpp"
#include "vox2tet/image/label.hpp"
#include "vox2tet/io/image_io.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace vox2tet::image {

namespace {

// `np.bincount(labels.ravel())`: returns sizes indexed by label id.
std::vector<std::uint64_t> bincount(const std::vector<std::int32_t>& labels,
                                    std::int32_t n_components) {
    std::vector<std::uint64_t> counts(n_components, 0);
    for (auto L : labels) ++counts[L];
    return counts;
}

void write_label_text(const std::string& path,
                      const std::vector<std::uint64_t>& values) {
    paths::create_folder(path);
    std::ofstream f(path);
    if (!f) throw std::runtime_error("write_label_text: cannot open " + path);
    for (auto v : values) f << v << "\n";
}

// Reinterpret raw bytes of `vol` as integer voxels of width `isz` and run
// `func` over each linear position.
template <typename Fn>
void for_each_voxel(Volume& vol, Fn&& fn) {
    const std::size_t n = vol.voxel_count();
    switch (vol.dtype) {
        case VoxelType::U8: {
            auto* p = reinterpret_cast<std::uint8_t*>(vol.bytes.data());
            for (std::size_t i = 0; i < n; ++i) fn(i, static_cast<std::uint64_t>(p[i]),
                                                   [&](std::uint64_t v){ p[i] = static_cast<std::uint8_t>(v); });
            break;
        }
        case VoxelType::U16: {
            auto* p = reinterpret_cast<std::uint16_t*>(vol.bytes.data());
            for (std::size_t i = 0; i < n; ++i) fn(i, static_cast<std::uint64_t>(p[i]),
                                                   [&](std::uint64_t v){ p[i] = static_cast<std::uint16_t>(v); });
            break;
        }
        case VoxelType::U32: {
            auto* p = reinterpret_cast<std::uint32_t*>(vol.bytes.data());
            for (std::size_t i = 0; i < n; ++i) fn(i, static_cast<std::uint64_t>(p[i]),
                                                   [&](std::uint64_t v){ p[i] = static_cast<std::uint32_t>(v); });
            break;
        }
        case VoxelType::U64: {
            auto* p = reinterpret_cast<std::uint64_t*>(vol.bytes.data());
            for (std::size_t i = 0; i < n; ++i) fn(i, static_cast<std::uint64_t>(p[i]),
                                                   [&](std::uint64_t v){ p[i] = v; });
            break;
        }
    }
}

// Read voxel by linear index.
std::uint64_t read_voxel(const Volume& v, std::size_t lin) {
    switch (v.dtype) {
        case VoxelType::U8:  return reinterpret_cast<const std::uint8_t* >(v.bytes.data())[lin];
        case VoxelType::U16: return reinterpret_cast<const std::uint16_t*>(v.bytes.data())[lin];
        case VoxelType::U32: return reinterpret_cast<const std::uint32_t*>(v.bytes.data())[lin];
        case VoxelType::U64: return reinterpret_cast<const std::uint64_t*>(v.bytes.data())[lin];
        case VoxelType::F32:
        case VoxelType::F64: break;  // narrowed on read
    }
    return 0;
}

void write_voxel(Volume& v, std::size_t lin, std::uint64_t value) {
    switch (v.dtype) {
        case VoxelType::U8:
            reinterpret_cast<std::uint8_t* >(v.bytes.data())[lin] =
                static_cast<std::uint8_t>(value); break;
        case VoxelType::U16:
            reinterpret_cast<std::uint16_t*>(v.bytes.data())[lin] =
                static_cast<std::uint16_t>(value); break;
        case VoxelType::U32:
            reinterpret_cast<std::uint32_t*>(v.bytes.data())[lin] =
                static_cast<std::uint32_t>(value); break;
        case VoxelType::U64:
            reinterpret_cast<std::uint64_t*>(v.bytes.data())[lin] =
                value; break;
        case VoxelType::F32:
        case VoxelType::F64: break;  // narrowed on read
    }
}

// Build an int32 label volume from a LabelResult so we can hand it to the
// TIFF writer.
Volume labels_to_int32_volume(const std::vector<std::int32_t>& labels,
                              std::array<std::size_t,3> shape) {
    Volume out;
    out.shape = shape;
    out.dtype = VoxelType::U32;       // store as u32 (matches np.int32 range
                                      // for non-negative label ids; we have
                                      // no negative values after labelling)
    out.bytes.assign(out.voxel_count() * out.item_size(), 0);
    auto* p = reinterpret_cast<std::uint32_t*>(out.bytes.data());
    for (std::size_t i = 0; i < labels.size(); ++i)
        p[i] = static_cast<std::uint32_t>(labels[i]);
    return out;
}

// Replace voxels marked by `too_small_mask` with the colour of their
// Euclidean-nearest non-removed neighbour. In-place on `vox`.
void refill_via_edt(Volume& vox,
                    const std::vector<std::uint8_t>& too_small_mask) {
    auto idx = distance_transform_edt_indices(too_small_mask, vox.shape);
    const std::size_t N  = vox.voxel_count();
    const std::size_t ny = vox.shape[1];
    const std::size_t nx = vox.shape[2];

    // Snapshot voxels first; otherwise refilling in scan order can read
    // already-overwritten values.
    std::vector<std::uint8_t> snapshot = vox.bytes;
    auto read_snap = [&](std::size_t lin) -> std::uint64_t {
        switch (vox.dtype) {
            case VoxelType::U8:  return reinterpret_cast<const std::uint8_t* >(snapshot.data())[lin];
            case VoxelType::U16: return reinterpret_cast<const std::uint16_t*>(snapshot.data())[lin];
            case VoxelType::U32: return reinterpret_cast<const std::uint32_t*>(snapshot.data())[lin];
            case VoxelType::U64: return reinterpret_cast<const std::uint64_t*>(snapshot.data())[lin];
            case VoxelType::F32:
            case VoxelType::F64: break;  // narrowed on read
        }
        return 0;
    };

    for (std::size_t lin = 0; lin < N; ++lin) {
        if (!too_small_mask[lin]) continue;
        const std::size_t src =
            (static_cast<std::size_t>(idx.iz[lin]) * ny +
             static_cast<std::size_t>(idx.iy[lin])) * nx +
             static_cast<std::size_t>(idx.ix[lin]);
        write_voxel(vox, lin, read_snap(src));
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// removeSmall — port of `image_utils.removeSmall`.
// On entry `vox` and `labels` describe the same image at the same shape;
// labels[i] in [0, n_components). Returns the boolean mask of removed
// voxels and the list of removed label ids. Modifies `vox` in place.
// ---------------------------------------------------------------------------
struct RemoveSmallSubResult {
    std::vector<std::uint8_t> too_small_mask;
    std::vector<std::int32_t> removed_labels;
};

static RemoveSmallSubResult remove_small(Volume& vox,
                                         const std::vector<std::int32_t>& labels,
                                         std::int32_t n_components,
                                         int max_remove_size) {
    VOX2TET_LOG() << "Removing regions less or equal than "
                  << max_remove_size << " voxels ...";

    auto sizes = bincount(labels, n_components);
    RemoveSmallSubResult out;
    out.too_small_mask.assign(vox.voxel_count(), 0);
    std::size_t n_removed_voxels = 0;
    for (std::size_t i = 0; i < labels.size(); ++i) {
        const std::int32_t L = labels[i];
        if (sizes[L] > 0 && sizes[L] <= static_cast<std::uint64_t>(max_remove_size)) {
            out.too_small_mask[i] = 1;
            ++n_removed_voxels;
        }
    }
    for (std::int32_t L = 0; L < n_components; ++L) {
        if (sizes[L] > 0 && sizes[L] <= static_cast<std::uint64_t>(max_remove_size)) {
            out.removed_labels.push_back(L);
        }
    }

    if (n_removed_voxels > 0) {
        refill_via_edt(vox, out.too_small_mask);
    }
    VOX2TET_LOG() << "removed region = " << out.removed_labels.size()
                  << ", removed voxels = " << n_removed_voxels;
    VOX2TET_PRINT("Removing done!");
    return out;
}

// ---------------------------------------------------------------------------
RemoveSmallResult remove_small_regions(const Settings& s, const Volume& voxels_in) {
    auto labels = label_components(voxels_in, s.connectivity);

    const std::string base = s.out_path_base;
    const std::string img_path   = base + "_C" + std::to_string(s.connectivity)
                                 + "_R" + std::to_string(s.max_remove_size) + ".tif";
    const std::string lab_path   = base + "_L_C" + std::to_string(s.connectivity);
    const std::string labrm_path = base + "_L_C" + std::to_string(s.connectivity)
                                 + "_R" + std::to_string(s.max_remove_size);

    paths::create_folder(img_path);

    // 1) Drop small regions from a copy of the label image.
    std::vector<std::int32_t> labels_rem = labels.labels;
    // Build a synthetic label-volume so refill_via_edt operates on labels
    // and we can recover the new color-volume via materials_out[].
    Volume label_vol;
    label_vol.shape = labels.shape;
    label_vol.dtype = VoxelType::U32;
    label_vol.bytes.assign(label_vol.voxel_count() * 4, 0);
    {
        auto* p = reinterpret_cast<std::uint32_t*>(label_vol.bytes.data());
        for (std::size_t i = 0; i < labels_rem.size(); ++i)
            p[i] = static_cast<std::uint32_t>(labels_rem[i]);
    }
    auto small = remove_small(label_vol, labels_rem, labels.n_components,
                              s.max_remove_size);
    // Read back the refilled labels.
    {
        auto* p = reinterpret_cast<const std::uint32_t*>(label_vol.bytes.data());
        for (std::size_t i = 0; i < labels_rem.size(); ++i)
            labels_rem[i] = static_cast<std::int32_t>(p[i]);
    }

    // 2) Rebuild the colour volume: voxels[i] = materials_out[ labels_rem[i] ],
    //    after renumbering remaining labels 0..n_kept-1 (reference: renum_labels).
    const std::int32_t n_components = labels.n_components;
    std::vector<std::uint8_t> removed_mask(n_components, 0);
    for (auto L : small.removed_labels) removed_mask[L] = 1;

    std::vector<std::int32_t> renum(n_components, -1);
    std::int32_t k = 0;
    for (std::int32_t L = 0; L < n_components; ++L) {
        if (!removed_mask[L]) renum[L] = k++;
    }

    // Build the colour-replaced volume.
    Volume vox_out;
    vox_out.shape = voxels_in.shape;
    vox_out.dtype = voxels_in.dtype;
    vox_out.bytes.assign(vox_out.voxel_count() * vox_out.item_size(), 0);
    const std::size_t N = vox_out.voxel_count();
    for (std::size_t i = 0; i < N; ++i) {
        const std::int32_t L = labels_rem[i];
        write_voxel(vox_out, i, labels.materials_out[L]);
    }

    // Persist artefacts that match the reference exactly:
    //   *_C{c}_R{r}.tif         — colour-volume with removed regions
    //   *_L_C{c}.tif            — original label image
    //   *_L_C{c}_R{r}.tif       — label image after removal
    //   *_L_C{c}.txt            — materials_out for every label (full)
    //   *_L_C{c}_R{r}.txt       — materials_out for kept labels only
    io::write_tiff3(img_path, vox_out);

    Volume orig_lab = labels_to_int32_volume(labels.labels, labels.shape);
    io::write_tiff3(lab_path + ".tif", orig_lab);

    // The reference writes the *renumbered* labels image (kept labels become
    // contiguous 0..n_kept-1) — re-apply renum to labels_rem.
    std::vector<std::int32_t> labels_renum = labels_rem;
    for (auto& L : labels_renum) L = renum[L];
    Volume rem_lab = labels_to_int32_volume(labels_renum, labels.shape);
    io::write_tiff3(labrm_path + ".tif", rem_lab);

    write_label_text(lab_path + ".txt", labels.materials_out);
    std::vector<std::uint64_t> kept_materials;
    kept_materials.reserve(n_components - small.removed_labels.size());
    for (std::int32_t L = 0; L < n_components; ++L) {
        if (!removed_mask[L]) kept_materials.push_back(labels.materials_out[L]);
    }
    write_label_text(labrm_path + ".txt", kept_materials);

    // the legacy removeSmall additionally dumps the *removed label ids*
    // alongside the C{c}_R{r}.tif file (as plain integers, one per line).
    std::vector<std::uint64_t> removed_ids;
    removed_ids.reserve(small.removed_labels.size());
    for (auto L : small.removed_labels) removed_ids.push_back(static_cast<std::uint64_t>(L));
    write_label_text(paths::base_file_path(img_path) + ".txt", removed_ids);

    RemoveSmallResult R;
    R.voxels        = std::move(vox_out);
    R.img_file_path = img_path;
    return R;
}

// ---------------------------------------------------------------------------
// Critical-connectivity fix — placeholder for now: returns the extended
// volume from the input file. The full algorithm is the next thing on
// PROGRESS.md (phase 5b critical-conn).
// ---------------------------------------------------------------------------
FixCritResult fix_critical_connectivity(const std::string& in_path,
                                        const std::string& out_path_base) {
    VOX2TET_LOG() << "Fixing critical voxel connectivities in " << in_path << " ...";
    auto [vox, src_shape] = ext_volume_from_file(in_path);

    const auto ncv = fix_vertex_critical(vox);
    const auto nce = fix_edge_critical_2m(vox);

    FixCritResult R;
    R.voxels = std::move(vox);

    if (ncv.first == 0 && nce.first == 0) {
        VOX2TET_PRINT("No critical voxels found!");
        R.out_file_path = in_path;
        return R;
    }

    // Write the fixed image (interior slab only, like the reference
    // `voxels[1:-1, 1:-1, 1:-1]`).
    R.out_file_path = paths::base_file_path(out_path_base) + "_F.tif";

    Volume interior;
    interior.shape = src_shape;
    interior.dtype = R.voxels.dtype;
    const std::size_t isz = R.voxels.item_size();
    interior.bytes.assign(interior.voxel_count() * isz, 0);
    const std::size_t Nx = R.voxels.shape[2];
    const std::size_t Ny = R.voxels.shape[1];
    for (std::size_t z = 0; z < src_shape[0]; ++z)
    for (std::size_t y = 0; y < src_shape[1]; ++y) {
        const std::uint8_t* src = R.voxels.bytes.data() +
            (((z + 1) * Ny + (y + 1)) * Nx + 1) * isz;
        std::uint8_t* dst = interior.bytes.data() +
            ((z * src_shape[1]) + y) * src_shape[2] * isz;
        std::memcpy(dst, src, src_shape[2] * isz);
    }
    io::write_tiff3(R.out_file_path, interior);

    if (ncv.second == 0 && nce.second == 0)
        VOX2TET_PRINT("All critical voxel connectivities fixed!");
    else
        VOX2TET_PRINT("Not all critical voxel connectivities fixed!");
    return R;
}

}  // namespace vox2tet::image
