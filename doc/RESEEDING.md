# Boundary-edge reseeding

## The problem

The initial marching-cubes mesh samples every feature chain — internal
triple lines, the bounding-box frame, and the traces of grain
boundaries on the bbox faces — at voxel spacing, and the rest of the
pipeline never changes that: chain vertices are excluded from every
remesh operation (`not_fixed_v` / `not_fixed_h`), so their spacing
stays ≈ 1 voxel forever. Worse, the surface sizing field is driven by
the distance to the nearest chain vertex, so a voxel-fine chain forces
a voxel-fine *halo of surface triangles* around it. The element count
near features is therefore dictated by the voxel size, not by the
geometry.

## The plan (staged)

The fix is *reseeding*: coarsen the chains to a geometry-driven
spacing, then let the sizing field and the remesh loop propagate the
coarsening into the surface. It is implemented in stages, each shipping
independently and each guarded so that a failure mode is always "less
coarsening", never a broken mesh:

| Stage | Scope | Status |
|-------|-------|--------|
| **A** — bbox-frame special case | The 12 frame lines: exactly straight, so reseeding is error-free by construction. Guarded edge collapse of chain-interior nodes. | **Implemented** |
| **A2** — bbox-face traces & internal triple lines | The curved chains (edge multiplicity 3–4): non-manifold collapse, curvature adaptivity via the chordal-deviation guard, fold guard across the new chord, and a global AABB segment-intersection guard. | **Implemented** (`do_reseed_triple_lines`) |
| **B** — full lfs sizing | Composite local-feature-size `lfs = min(d_chains, d_self, d_surf)` with geodesic-collar incidence exclusion; per-vertex chain target `h = clamp(β·lfs, 1, target)`, grading-limited (`g ≈ 1.3`) along the chain graph. | **Implemented, default on** (`do_reseed_lfs`). Closes the chord-vs-chord near-miss hole; on JMA_30 it recovers the worst corner *above* the no-reseed baseline (15.6° → 18.6° vs legacy 18.0°) at the cost of ~4% of the node gain. |
| **C** — sizing-field coupling | Surface sizing seeded from the reseeded chain segment lengths (the recomputed `calc_brep_sizing` provides this automatically) plus a Dijkstra grading limit on the whole field. | **Implemented, experimental — default off** (`do_reseed_graded_sizing`). Measured on both JMA fixtures the grading limit gives *no* quality benefit: the post-reseed slivers turn out to be cap triangles pinned to the coarsened chain chords, which no sizing value can remove (see the Stage C section below). |
| **Cap fix** — near-chain collapse | The structural fix the Stage-C measurements pointed at: let the remesh loop collapse interior edges whose one-ring touches a feature chain, so cap vertices can disappear sideways. | **Implemented, default on** (`do_collapse_near_bedges`). Removes the A2 quality regression and coarsens further: JMA_30 worst corner 9.5° → 15.6°, JMA_10 21.2° → 23.6° (better than the no-reseed baseline). |

The design discussion behind this staging (guard system, robustness
limits, relation to CGAL protecting balls / TetWild) is summarised in
`chat-log.md`.

## What is implemented (Stages A + A2 + B)

`remesh::reseed_feature_chains` (in `src/remesh/reseed.cpp`) runs
inside the pre-remesh phase, after chain/surface smoothing and
*before* the sizing field is computed:

1. **Chain decomposition.** The boundary-edge graph is split only at
   *true junctions* — nodes whose graph degree differs from 2 (bbox
   corners, trace junctions on the frame, quad points, chain
   crossings). Voxel-corner (`masks[2]`) nodes of degree 2 are
   staircase artifacts on curved chains and become removable interior
   nodes; how far such a corner may be cut is decided by the
   chordal-deviation guard. (This deliberately differs from
   `order_bedges`, which splits at every `masks[2]` node and would
   fragment curved chains into unreseedable stubs.) Components with no
   junction at all are emitted as closed loops. Junction preservation
   is structural: junctions are chain endpoints and never candidates.
2. **Eligibility.** Bbox-frame chains (every vertex on `masks[4]`)
   always; curved chains and loops only when
   `do_reseed_triple_lines` is set. Closed loops keep at least 4
   surviving vertices so a ring can never pinch.
3. **Greedy anchored walk.** The next interior vertex `v` is collapsed
   into the anchor `u` (a pure topological merge — `u` never moves, so
   surviving coordinates are preserved exactly), growing one chord
   until the local spacing target would be exceeded; then the anchor
   advances. A rejected collapse also advances the anchor. The cap on
   a chord is the *minimum* `h` over every vertex it replaces (Stage
   B's per-vertex lfs target — see below; with `do_reseed_lfs` off,
   `h` is `reseed_target_len` everywhere, or `Lmax` when that is 0).
4. Every collapse must pass all guards:
   * **topology** — the chain edge (u,v) carries one triangle per
     incident sheet (multiplicity 2 on the frame, 3 on face traces and
     triple lines, 4 on quad lines; anything else rejects). Every side
     edge of every dying triangle must be manifold with clean mutual
     opposites — a side edge that is itself a chain edge (junction
     adjacency) rejects. Link condition: the common neighbours of u
     and v are exactly the dying triangles' apexes;
   * **geometry** — every removed vertex stays within `reseed_eps`
     (voxels) of the replacement chord u–w. On curved chains this *is*
     the curvature control: a tight bend refuses long chords, which is
     the chordal-error rule `h ≈ 2√(2ε/κ)` enforced against the actual
     polyline instead of a noisy discrete-curvature estimate;
   * **quality** — every surviving triangle around `v`, re-evaluated
     with `v` at `u`'s position, must keep positive area, an unflipped
     normal, and a min corner angle above the threshold
     (`min_corner_angle_boundary` for all-bbox triangles,
     `_internal` otherwise) unless it was already below. Additionally
     the fold across the new chord — the minimum pairwise dihedral
     among the sheets that will share u–w — must not drop below the
     matching `min_dangle_*` threshold unless it already was;
   * **intersection (AABB guard)** — the chord u–w must not strictly
     cross any live triangle that does not contain u, v, or w. A
     static AABB tree over all triangles is built once per reseed run;
     since collapse never moves vertices, only *grow-only* box refits
     are needed when triangles are retargeted, so the tree stays
     conservative (never misses) for the whole run. The
     Möller–Trumbore test is strict-interior and reports no hit for
     coplanar configurations — on the planar bbox faces the coplanar
     crossing case (a trace chord sweeping across another trace) is
     instead excluded by the normal-flip guard, which forbids any
     in-plane fold.
5. The mesh arrays (`tri`, `xyz`, node masks, interface ranges) are
   compacted and rebuilt; the pipeline then recomputes half-edges,
   not-fixed masks, brep sizing, normals, and the sizing field from
   the reseeded mesh.

### Stage B: composite lfs targets (`do_reseed_lfs`)

`reseed_detail::compute_chain_lfs` (in `src/remesh/reseed.cpp`) gives
every chain vertex `p` a spacing target

```
h(p) = clamp( β · lfs(p), 1 voxel, target ),
lfs(p) = min( d_chains(p), d_self(p), d_surf(p) )
```

gradient-limited along the chain graph with slope `g − 1`
(`reseed_grading`), so a tight spot cannot sit next to a long chord.
The three distance terms are exactly the plan's composite feature
size, and the crux is *incidence exclusion by geodesic collar, not by
entity id*:

* **`d_surf`** — nearest surface (non-chain) vertex that is
  *intrinsically far*: a multi-source Dijkstra from the whole chain
  set over the triangle edge graph gives every surface vertex its
  "skirt depth" (graph distance to the nearest chain vertex), and a
  candidate counts only when `skirt > collar · |p − q|` (collar = 2).
  A vertex of the chain's own skirt has `skirt ≈ |p − q|` and is
  excluded; a different sheet approaching from outside has a large
  skirt depth and caps the target. No entity ids needed.
* **`d_chains` / `d_self`** — nearest chain vertex that is either in a
  different connected component (never reached by a bounded Dijkstra
  over the chain graph) or *along-chain far* (`geo > collar · |p − q|`
  — the same chain bending back on itself). Along-chain neighbours are
  excluded by the same rule.

The walk then caps every chord at the minimum `h` over the vertices it
replaces, so coarsening stops where *other* geometry gets close — the
chord-vs-chord and chord-vs-surface near-miss configurations that the
AABB guard (which only detects actual intersections) cannot see. The
log reports how often the cap bit (`lfs capped <n> spans`).

Two things Stage B deliberately does *not* do: no curvature term
(`c/κ` from the original plan — the chordal-deviation guard already
enforces the same bound against the true polyline, without a noisy
discrete-curvature estimate), and no EDT/medial-axis lfs (voxel
staircase noise; the k-d-tree-style queries above are computed on the
smoothed geometry instead).

### Switching back to the old pipeline

```json
{ "do_reseed_bedges": false }
```

restores the previous behaviour exactly (chains stay frozen at voxel
spacing); the reseed stage is skipped entirely.

### Settings

| Field | Default | Purpose |
|-------|---------|---------|
| `do_reseed_bedges`       | `true` | Master switch for the reseeding stage. |
| `do_reseed_triple_lines` | `true` | Stage A2 scope: also reseed face traces, internal triple/quad lines, and closed loops. `false` = frame-only (Stage A). |
| `do_reseed_graded_sizing`| `false`| Stage C (experimental): grade the sizing field with `limit_sizing_gradient` after reseeding. Off by default — measured to give no quality benefit (see Stage C section). |
| `do_collapse_near_bedges`| `true` | The cap fix: allow the remesh loop to collapse interior edges whose one-ring touches a feature chain (guarded; see the cap-fix section). Only active together with `do_reseed_bedges`. |
| `do_reseed_lfs`          | `true` | Stage B: cap each chain vertex's spacing target by `β·lfs` (composite local feature size with geodesic-collar exclusion; see the Stage B section). `false` = uniform target everywhere. |
| `reseed_eps`             | 0.4    | Chordal-deviation budget in voxels (geometric-accuracy guard and curvature control). |
| `reseed_target_len`      | 0.0    | Target chain spacing; `0` means "use `Lmax`". |
| `reseed_beta`            | 0.7    | Stage B safety factor `β` in `h = clamp(β·lfs, 1, target)`. |
| `reseed_grading`         | 1.3    | Allowed size slope `g`: used along the chains by Stage B, and on the surface field by Stage C (`L[v] ≤ L[u] + (g−1)·|uv|`). Values ≤ 1 disable grading. |

### Diagnostics

* Log line per run:
  `reseed_feature_chains: chains frame/curved/loop <f>/<c>/<l>, <a>/<b> collapses accepted (rej topo/geom/qual/isect …), <r> of <V> vertices removed`.
* `<base>_RS_E.ply` — the feature chains after reseeding (compare with
  `<base>_S_E.ply`, the smoothed chains before reseeding).

### Measured effect

| Fixture | Config | Final remesh nodes | Min corner after repair | Slivers |
|---------|--------|--------------------|--------------------------|---------|
| JMA_10  | reseeding off        | 908  | 21.6° | 19 |
| JMA_10  | Stage A (frame only) | 773 (−15%) | 21.4° | — |
| JMA_10  | Stage A2, no cap fix | 715 (−21%) | 21.2° | 20 |
| JMA_10  | A2 + cap fix, no lfs | 604 (−34%) | 23.1° | 24 |
| JMA_10  | **+ Stage B lfs (default)** | **669 (−26%)** | **22.1°** | 26 |
| JMA_30  | reseeding off        | 16208 | 18.0° | 224 |
| JMA_30  | Stage A2, no cap fix | 14825 (−8.5%) | 9.5° | 268 |
| JMA_30  | A2 + cap fix, no lfs | 13530 (−17%) | 15.6° | 292 |
| JMA_30  | **+ Stage B lfs (default)** | **14068 (−13%)** | **18.6°** | **247** |

(Rows above the Stage-B ones predate the `split_edges` non-manifold
fix; re-measured after it, the no-lfs cap-fix numbers shift by ≲ 0.5%
— e.g. JMA_10 603 → 604 nodes — while the reseeding-off baselines are
bit-identical. All Stage-B-era rows are post-fix.)

Stage B's trade is deliberate: on JMA_30 the lfs cap refuses 2373
spans near approaching features, giving back ~4% of the nodes but
lifting the worst corner from 15.6° to 18.6° — *above* the 18.0°
no-reseed baseline — and cutting slivers from 292 to 247 (legacy:
224). On the sparse
JMA_10 the cap costs 65 nodes and ~1° (both configurations are above
the 20° guard); `do_reseed_lfs: false` recovers the aggressive
behaviour when raw node count matters more than worst-element
quality.

The quality guard is the dominant limiter (`rej qual` ≫ others): while
the surface next to a chain is still voxel-fine, a chord much longer
than ~2–3 voxels would create sub-threshold corner angles and is
rejected. The JMA_30 quality drop (worst corner 18.0° → 9.5°, slivers
224 → 268 after Layer-4 repair) is analysed in the Stage C section
below — it is *not* a sizing-field problem.

## Stage C: graded sizing field (experimental, default off)

`remesh::limit_sizing_gradient` (in `src/remesh/smooth.cpp`) runs a
multi-source Dijkstra relaxation over the mesh edge graph enforcing
`L[v] ≤ L[u] + (g−1)·|uv|` on every edge. The result is the
pointwise-largest field below the input that satisfies the bound, so
grading only ever refines the coarse side of a transition. It is exact
(unit-tested against an independent Bellman-Ford reference) and runs
after `calc_sizing_field` when `do_reseed_bedges` **and**
`do_reseed_graded_sizing` are both set.

**Why it is off by default — the measured story.** The Stage-C
hypothesis was that the A2 quality regression on JMA_30 came from the
ungraded fine-to-coarse sizing jump next to coarsened chains. The
measurements falsify that:

| Fixture | Grading | Final nodes | Min corner | Slivers after repair |
|---------|---------|-------------|------------|----------------------|
| JMA_10  | off     | 715  | 21.2°  | 20 |
| JMA_10  | g = 1.3 | 768  | 14.1°  | 25 |
| JMA_10  | g = 1.5 | 722  | 21.3°  | 32 |
| JMA_30  | off     | 14825 | 9.52° | 268 |
| JMA_30  | g = 1.3 | 15915 | 9.52° | 299 |
| JMA_30  | g = 1.5 | 15178 | 9.52° | 256 |

The JMA_30 worst angle is 9.52258° with grading off, at 1.3, at 1.5
and at 2.0 — the same triangle every time. Inspecting the final meshes
shows why: **every one of the ten worst JMA_30 triangles touches a
reseeded chain vertex**, and their edge lengths follow one pattern —
two short edges whose sum barely exceeds the long one (e.g. 0.70 +
0.82 against a 1.50 chord). These are *cap triangles*: a free surface
vertex lying almost on a coarsened chain chord. No sizing value can
remove them, because the remesh loop is structurally unable to: the
interior collapse pass requires both edge endpoints to be free
(`collapse_edges` rejects any edge into a fixed chain vertex), and
collapsing the cap vertex along the chord direction would be rejected
by the ring-duplicate (fold) guard anyway. Grading at g = 1.3 actually
*increases* cap incidence — it refines the surface right next to
still-frozen coarse chords (JMA_10 drops to 14.1°, below the 20°
boundary threshold) — which is why the default is off.

The structural fix that follows from this diagnosis is the near-chain
collapse below.

## The cap fix: near-chain collapse (`do_collapse_near_bedges`)

The remesh loop's interior collapse (`collapse_edges` in
`src/remesh/remesh.cpp`) carried a conservative ring guard: any
candidate whose one-ring contained a *single* fixed (chain) half-edge
was skipped outright. That freezes every vertex one ring from a chain —
which is exactly what left the cap vertices in place.

The fix relaxes that guard, on the observation that the collapse
rewiring only ever touches the four side edges of the two dying
triangles. A fixed hedge elsewhere in the ring is safe provided:

* the two dying triangles' side edges (and the collapsing pair itself)
  are free, manifold and mutually-opposite — a fixed or non-manifold
  side edge still rejects, so the rewiring never crosses a chain edge;
* every retargeted fan triangle that keeps a chain vertex passes a
  quality guard: its normal must not flip, and its min corner angle
  must stay above `min_corner_angle_boundary` unless it was already
  below and does not get worse.

A cap vertex then simply disappears *sideways* into a free neighbour
(the standard shortest-edge collapse), and the ex-cap triangle is
re-evaluated with the new, well-placed apex — the dangerous free→fixed
merge into the chain itself is never attempted. The path is gated on
`do_reseed_bedges && do_collapse_near_bedges`, so `do_reseed_bedges:
false` alone still reproduces the legacy pipeline bit-for-bit
(verified: JMA_30 16208 nodes / 17.95°, identical to baseline).

Measured effect (defaults, vs the no-cap-fix A2 state): JMA_30 worst
corner 9.52° → 15.56° with slivers 268 → 251 and 14825 → 13540 nodes;
JMA_10 21.2° → 23.6° (now *better* than the 21.6° no-reseed baseline)
with 715 → 603 nodes. The remaining sub-threshold triangles are no
longer degenerate chord caps (their two short edges sum well above the
long edge) — they are junction- and wedge-limited configurations, the
class the acceptance criteria always excluded.

**Remaining follow-ups:** with Stage B implemented the roadmap's
stages are all shipped; the graded-sizing knob (Stage C) may become
useful again now that chain spacing is lfs-driven — worth re-measuring
if chain-adjacent quality regresses on new data.
`do_reseed_triple_lines: false` remains available to restrict
reseeding to the frame.

### Known limits (honest robustness notes)

* The AABB guard tests the *chord* against distant geometry; the
  retargeted fan triangles themselves are checked only by the local
  corner/normal/dihedral guards. With `reseed_eps ≤ 0.5` voxel the fan
  stays inside a thin tube around the chain, so this is a deliberate
  cost/benefit cut — the full fan-vs-world test would be the belt to
  this suspenders.
* The chord-vs-chord near-miss problem (two chains approaching without
  intersecting, forcing high-aspect strips between them) is guarded by
  the Stage B lfs cap (`d_chains`/`d_surf` with collar exclusion) —
  but the cap is computed once, on the *pre-reseed* geometry, and
  against vertices, not chords: two chords could still pass closer
  than either endpoint pair. With `β = 0.7` and the corner-angle guard
  as backstop this has not been observed to matter.
* Chain edges of multiplicity > 4 and side edges that are themselves
  chain edges (junction-adjacent configurations) are detected and
  simply not collapsed.
* Sharp input wedges still bound the achievable element quality — no
  reseeding (or any meshing algorithm) can place well-shaped elements
  against a dihedral the geometry itself makes small.

## Tests

`tests/test_reseed.cpp` covers: exact area preservation on planar
fixtures (the strongest accuracy check available: any leaked or
flipped triangle changes the total), watertightness and Euler
characteristic after reseeding, target-spacing cap semantics, quality
guard monotonicity, the Stage A2 scope switch, multiplicity-3 collapse
on face traces and an internal triple line, closed-loop handling and a
one-sided Hausdorff bound (original → reseeded ≤ `reseed_eps`) on a
cylindrical inclusion's curved trace circles, direct unit tests of the
AABB tree (broad-phase completeness, pruning, grow-only refit) and the
strict-interior segment–triangle test, and settings round-trip.

Stage B (`ChainLfs` suite): a two-cylinder fixture where the gap
between the trace circles must cap `h` well below the global target on
the near arcs while the far arcs stay near `β·lfs` of the frame
distance (also asserts the grading bound along the chains); a
single-cylinder fixture proving the collar excludes the chain's *own*
skirt (h stays well above `β·1 voxel`); and a full-reseed run on the
two-cylinder fixture asserting every reseed-created chain edge
respects the `h` of both surviving endpoints, that the cap actually
bit (`lfs_limited > 0`), and that disabling `do_reseed_lfs` produces
strictly longer chords.

Stage C (`SizingGrading` suite): `limit_sizing_gradient` verified
against an independent Bellman-Ford reference (exact per-vertex
equality), idempotence, the g ≤ 1 no-op, and the edge-Lipschitz bound
plus `[Lmin, Lmax]` containment on the full reseed → sizing-field
pipeline path.

Cap fix (`CapFix` suite): the full remesh loop is driven over a
reseeded cylinder fixture with the near-chain collapse on and off. The
hard invariant — the non-manifold (3+ sheet) edge count, which chain
edges being frozen makes exact — must hold in both runs; the fix must
not change the Euler characteristic relative to the control run, must
keep all six bbox face areas covered to the same tolerance the legacy
loop achieves, and must produce strictly fewer vertices than both the
reseeded mesh and the control. (The legacy loop itself perturbs bbox
planes slightly through the off-plane repair layers and changes χ on
this fixture — a pre-existing property, flagged for separate
investigation.)
