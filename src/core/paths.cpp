#include "vox2tet/core/paths.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <system_error>

namespace fs = std::filesystem;

namespace vox2tet::paths {

void create_folder(const std::string& path_or_base, bool do_dir_name) {
    fs::path p = fs::absolute(path_or_base);
    fs::path dir = do_dir_name ? p.parent_path() : p;
    if (dir.empty()) return;
    std::error_code ec;
    fs::create_directories(dir, ec);
}

std::string base_file_path(const std::string& path) {
    fs::path p = fs::absolute(path);
    fs::path noext = p.parent_path() / p.stem();
    return noext.string();
}

std::string file_extension(const std::string& path, bool to_upper) {
    fs::path p(path);
    std::string ext = p.extension().string();
    if (to_upper) {
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return std::toupper(c); });
    }
    return ext;
}

std::string file_name(const std::string& path) {
    return fs::path(path).filename().string();
}

std::string base_file_name(const std::string& path) {
    return fs::path(path).stem().string();
}

std::string base_folder(const std::string& path) {
    return fs::absolute(path).parent_path().string();
}

void rename_if_exist(const std::vector<std::string>& path_list) {
    for (const auto& p : path_list) {
        std::error_code ec;
        if (!fs::is_regular_file(p, ec)) continue;
        auto ftime = fs::last_write_time(p, ec);
        // Convert file_time_type -> system clock for formatting; approximate.
        auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
            ftime - decltype(ftime)::clock::now() + std::chrono::system_clock::now());
        std::time_t cftime = std::chrono::system_clock::to_time_t(sctp);
        std::tm tm_buf{};
#if defined(_WIN32)
        gmtime_s(&tm_buf, &cftime);
#else
        gmtime_r(&cftime, &tm_buf);
#endif
        char ts[32];
        std::strftime(ts, sizeof(ts), "%d.%m.%Y_%H.%M.%S", &tm_buf);

        fs::path src(p);
        fs::path dst = src.parent_path() /
                       (src.stem().string() + "_" + ts + src.extension().string());
        fs::rename(src, dst, ec);
    }
}

bool is_exist_files(const std::vector<std::string>& path_list) {
    for (const auto& p : path_list) {
        std::error_code ec;
        if (!fs::is_regular_file(p, ec)) return false;
    }
    return true;
}

std::optional<std::string> which(const std::string& program) {
#if defined(_WIN32)
    constexpr char path_sep = ';';
#else
    constexpr char path_sep = ':';
#endif

    auto is_exe = [](const fs::path& fp) {
        std::error_code ec;
        return fs::exists(fp, ec) && fs::is_regular_file(fp, ec);
    };

    fs::path fp(program);
    if (fp.has_parent_path()) {
        if (is_exe(fp)) return fp.string();
        return std::nullopt;
    }

    const char* path_env = std::getenv("PATH");
    if (!path_env) return std::nullopt;

    std::string path_str = path_env;
    std::stringstream ss(path_str);
    std::string dir;
    while (std::getline(ss, dir, path_sep)) {
        if (dir.empty()) continue;
        fs::path candidate = fs::path(dir) / program;
        if (is_exe(candidate)) return candidate.string();
#if defined(_WIN32)
        if (const char* exts = std::getenv("PATHEXT")) {
            std::stringstream es(exts);
            std::string ext;
            while (std::getline(es, ext, ';')) {
                fs::path c2 = candidate; c2 += ext;
                if (is_exe(c2)) return c2.string();
            }
        }
#endif
    }
    return std::nullopt;
}

}  // namespace vox2tet::paths
