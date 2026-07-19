# Perspectives: reseeding boundary edges to reduce element count

Status: conceptual proposal, not implemented. Captures the analysis from a
design discussion on 2026-07-03.

## Problem statement

The current pipeline never changes the length of boundary edges — edges on
triple lines (3 materials meet), quad points (4 materials meet), or the
bounding-box frame/faces. Their spacing stays proportional to the initial
voxel size regardless of `Lmax`, `n_remesh_itr`, or any other remeshing
setting, which forces an excessive element count on inputs where those
features don't need voxel-scale resolution.

## Root cause

Three places in the pipeline conspire to freeze these vertices permanently:

- **Vertex classification** ([`src/marching_cubes/contouring.cpp:737-750`](../src/marching_cubes/contouring.cpp#L737-L750)):
  only `masks[0]` vertices (voxel-face midpoints) are ever movable. Triple-line
  nodes (`masks[1]`), quad points (`masks[2]`), and bbox face/edge/corner
  nodes (`masks[3..5]`) are all classified as fixed "brep" nodes.
- **Fixed masks** ([`src/mesh/half_edge.cpp:218-280`](../src/mesh/half_edge.cpp#L218-L280)):
  `not_fixed_v = masks[0]`, so no brep vertex is ever collapsed away.
  Non-manifold half-edges (triple lines) have no opposite, so
  `not_fixed_h = 0` and they're never split either. Sharp bbox-frame edges
  are excluded explicitly. `smooth_brep_mean` moves these nodes tangentially
  but never changes how many there are — spacing stays at ≈ 1 voxel forever.
- **Sizing-field feedback** ([`src/remesh/smooth.cpp:438-456`](../src/remesh/smooth.cpp#L438-L456)):
  on brep vertices the target size is `1.7 × brep_sizing`, where
  `brep_sizing` is the distance to the nearest *other* brep vertex — i.e.
  self-referentially ≈ 1.7 voxels. Because the surface sizing field is
  distance-to-nearest-feature clipped to `[Lmin, Lmax]`, the whole surface
  *near* every feature is also forced to voxel-scale triangles. The element
  blow-up is therefore compounded: fine chains **plus** a fine halo of
  surface around them.

## Constraints any fix must satisfy

- **Geometric accuracy**: original shape and boundaries preserved (bounded,
  quantifiable deviation from the original geometry).
- **Mesh quality**: good aspect ratios, no slivers.

## Proposed methods

### Method 1 — Curvature-adaptive 1-D resampling of B-rep chains (core)

`order_bedges` already gives every feature curve as an ordered polyline with
fixed endpoints (quad points, bbox corners, chain junctions). Insert a chain
reseeding stage after pre-remesh smoothing, before `calc_sizing_field`:

1. **Parameterize** each smoothed chain by arc length (or reuse the existing
   B-spline fit — `brep_degree`, `brep_smoothness`).
2. **Compute target spacing** per point: `h(t) = min(Lmax, c/κ(t), β·lfs(t))`
   — see [lfs definition](#defining-lfs-local-feature-size) below.
   - `c/κ(t)` (chordal-error rule `h ≈ 2·sqrt(2εκ⁻¹ − ε²)`) bounds deviation
     from the curve by `ε` (≈ 0.3–0.5 voxel, consistent with the existing
     `max_D2self = 0.8` drift budget) — this is what preserves **geometric
     accuracy**.
   - `β·lfs(t)` is what protects **mesh quality** — without it, a long chain
     edge running parallel to a nearby feature forces flat, high-aspect
     triangles into the gap.
3. **Resample** along arc length at that spacing, keeping junction/corner
   endpoints exactly fixed. Straight runs (notably the bbox frame) collapse
   to a handful of segments up to `Lmax`; curved triple lines keep the nodes
   they need.
4. New nodes are placed **on the smoothed curve**, so geometry loss is
   bounded by the explicit `ε` (Hausdorff-checkable against the original
   chain).

Two ways to realize step 3, differing mainly in engineering cost:

- **1a. Feature-aware edge collapse** (incremental, reuses existing
  machinery). Let a chain vertex collapse into its chain neighbor only
  (never across a junction). Reuse existing acceptance tests — cycle-min
  dihedral guard over the blink cycle, corner-angle guard, Hoppe adjacency
  check — plus the chordal-deviation bound. `not_fixed_v` becomes a 3-state
  tag: *free* / *on-curve* / *pinned*. Least invasive; degrades gracefully
  (a collapse that would sliver or fold is simply rejected).
- **1b. Reseed-and-retriangulate** (cleaner result, more work). Resample
  chains geometrically, then locally re-triangulate the adjacent surface
  patches (each interface patch is a manifold disk/annulus bounded by
  chains — standard constrained 2-D remeshing applies per patch). Better
  fans around new coarse nodes, at the cost of new connectivity code.

### Method 2 — Sizing-field follows the coarsened chains (mandatory companion)

Reseeding chains alone barely reduces the count, and hurts quality, if the
surrounding surface stays voxel-fine (high-valence fans, needles where one
long chain edge meets many tiny surface triangles). After reseeding,
`calc_sizing_field` should derive brep-vertex sizing from the **new chain
segment lengths** (or directly from `h(t)`) instead of nearest-brep-vertex
distance, then enforce a **grading limit** when propagating into the
surface (neighbor-to-neighbor ratio ≤ g ≈ 1.2–1.5, via one Dijkstra-style
relaxation pass `L[v] ≤ L[u] + (g−1)·|uv|`). The existing
split/collapse/flip/smooth loop then equilibrates the surface on its own —
it already knows how to keep dihedrals and corner angles above threshold.
Bounded grading is the standard defense against slivers at size
transitions.

### Method 3 — Special-case the bounding-box frame (cheap, high payoff)

The 12 bbox frame edges are exactly straight and axis-aligned, so reseeding
them is error-free by construction: uniform spacing at
`min(Lmax, grading cap)`, keeping only the 8 corners and the nodes where
internal triple lines terminate on the frame (true junctions). The same
argument extends to the 6 bbox faces: exactly planar, so once the frame and
the in-plane grain-boundary traces are reseeded, face-interior triangles can
grow to `Lmax` with zero geometric error — only the trace curves need the
curvature-adaptive treatment of Method 1. A large fraction of a typical
RVE's element count sits on the outer box, so this alone gives a big
reduction for comparatively little work.

### Method 4 — Literature alternative (for a from-scratch rewrite)

Cheng–Dey–Ramos "protecting balls" (used by CGAL `Mesh_3` for 1-D features):
sample each feature curve with balls sized by local feature size, provably
preventing slivers near curves; pair with a feature-constrained
variational/CVT surface remesher for the patches. Conceptually Method 1 +
Method 2 with formal guarantees, but a much larger implementation than
extending the existing operators.

### Recommendation

Implement **1a + 2 + 3**, in that order: chain-collapse with chordal-error
and lfs caps, sizing field driven by new chain spacing with bounded grading,
then the trivial uniform reseed on the straight bbox frame. Geometric
accuracy is controlled by one explicit parameter `ε` (chordal deviation, in
voxels — natural to expose next to `max_D2self` in `Settings`); mesh quality
is enforced by the same four-layer guard system that already gates every
other remesh operation. Placing reseeding before the main remesh loop lets
that loop absorb the transition. A follow-up `tetgen -q` run then also
coarsens the interior tets, since TetGen's element size is driven by the
input surface facets.

## Defining lfs (local feature size)

The textbook definition (Amenta–Bern) is: **lfs(x) = distance from x to the
medial axis of the geometry's complement**. It automatically encodes
everything relevant — nearby features, thin layers, hairpins, acute wedges —
but computing a medial axis of a multi-material B-rep is impractical. The
practical substitute is a **minimum over explicit distance terms**, each
guarding against a specific way a long chain edge can squeeze the
surrounding mesh:

```
lfs(t) = min( d_chains(t), d_surf(t), d_self(t), σ_surf(t) )
h(t)   = clamp( min( c/κ(t), β·lfs(t) ),  h_min, Lmax )
```

### The four terms

**1. `d_chains(t)` — distance to non-incident feature curves.** All of them
in one set: other triple lines, the bbox frame edges, and grain-boundary
traces on the bbox faces.

- Chains *sharing a junction* with the current chain must be **included**
  (only the junction point itself excluded). This handles acute junctions
  for free: if two chains leave a quad point at angle θ, a point at arc
  distance s from the junction sees the sibling chain at distance ≈ s·sinθ,
  so spacing stays fine near a sharp Y-junction and grows moving away from
  it — no special junction term needed.
- Near any junction this term → 0, demanding infinitely fine spacing; that's
  what the `h_min` clamp is for. Natural choice: `h_min` ≈ current
  voxel-scale spacing, so the reseeded mesh is never finer than today's.

**2. `d_surf(t)` — distance to non-incident surface patches.** Covers a
triple line running parallel and close to a bbox face, a chain passing near
a thin grain's far wall, or two interface sheets of a thin layer. If a chain
edge of length h spans a region where a foreign sheet sits at distance
d < h, the surface strip between them is forced into flat, high-aspect
triangles — guard layers then either reject the collapse (no coarsening
gained) or slivers appear. "Non-incident" matters: the sheets *bounded by*
the chain touch it everywhere (distance ≡ 0) and must be excluded — but see
term 3 for the pitfall in how that exclusion is implemented.

**3. `d_self(t)` — distance to the chain itself outside a geodesic collar.**
The curvature term `c/κ(t)` is blind to a **hairpin**, where the chain
returns to within a voxel of itself after a long detour. Measure distance
from point t to all points of the same chain whose *arc-length* distance
exceeds a collar radius (a few × local h). Same logic applies to incident
surface patches: exclude them by **geodesic collar, not by patch id** — a
single interface patch can wrap around and approach the chain again from
the far side, and an id-based exclusion would miss it. This is the classic
failure mode of naive lfs implementations.

**4. `σ_surf(t)` — surface sizing field in a tube around the chain.** Not a
distance term. A *straight* triple line (κ ≈ 0, no nearby features) can lie
on a *highly curved* interface; terms 1–3 would allow h = Lmax, but the
surface next to the chain must stay fine to capture that curvature —
otherwise a long chain edge meets dozens of tiny surface triangles (a
high-valence fan of needles along the chain). Cap chain spacing by a small
multiple of the minimum surface target size within a one-ring/tube
neighborhood. This couples chain and surface sizing two ways: the surface
field near features follows chain spacing (Method 2), and chain spacing
bows to surface curvature where the surface is the binding constraint. An
explicit curvature estimate is needed for this (e.g. dihedral deficit per
vertex on the smoothed surface) — the existing `undo_sharp_bdangle_nodes`
trigger is a related but not identical signal.

**Not needed**: the wedge (dihedral) angle between sheets meeting at the
triple line. A thin acute wedge constrains element size *on those sheets*
(they approach each other away from the chain — that's the surface field's
own lfs problem), not spacing *along* the chain. It enters chain sizing only
indirectly through term 4.

### Properties that protect mesh quality

- **β factor**: use `h ≤ β·lfs`, β ≈ 0.5–1, not `h ≤ lfs`. Guarantees at
  least ~1/β element rows between any two features, bounding the aspect
  ratio of strip triangles between them.
- **Lipschitz grading**: each distance term is 1-Lipschitz, and a min of
  1-Lipschitz functions is 1-Lipschitz, so lfs can't jump. After sampling
  h(t), run one relaxation pass enforcing
  `h(t₂) ≤ h(t₁) + (g−1)·|t₂−t₁|` (g ≈ 1.2–1.5) along the chain, and the
  same grading when propagating into the surface field. Smooth size
  transitions are the single most effective sliver preventive — most
  slivers at reseeded features come from abrupt fine-to-coarse jumps, not
  from coarseness itself.
- **Symmetry for free**: every chain measures distance to every other
  chain, so two features approaching each other both refine — no coarse
  chain ever bridges past a fine one.

### How to compute it

Measured on the **smoothed** geometry, after `smooth_brep_mean` and before
reseeding. Two practical routes:

- **Sample + k-d tree**: sample all chains and surface vertices with entity
  tags (chain id, patch id, arc-length coordinate); for each chain sample,
  query nearest neighbors, discarding hits that are incident *and* within
  the geodesic collar. The existing `calc_brep_sizing` is already
  brute-force O(N·|B|), so even a brute-force version of this is no
  regression; nanoflann can upgrade it later.
- **Voxel EDT**: an EDT module already exists (`tests/test_edt.cpp`).
  Rasterizing features and taking distance transforms gives d_chains/d_surf
  cheaply and robustly on the grid — but incidence exclusion (term 2's
  "non-incident", term 3's collar) is awkward in image space, so EDT works
  best as a fast lower-bound field combined with tag-aware point queries
  near the chain itself.

With the β factor and Lipschitz grading on top, the lfs definition itself
becomes the main sliver defense, and the existing four-layer guard system
becomes a safety net rather than the mechanism that has to reject half the
coarsening operations.
