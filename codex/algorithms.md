What We're Working With

  Inputs available for any fill algorithm:
  - Every splat's position, scale, quaternion, opacity, color, SH coefficients (via forEachSplat())
  - Cached world positions + colors in Float32Arrays (~12MB for 500k splats)
  - Spatial index (20x20x20 voxel grid with per-cell splat indices)
  - The deletion SDF shape(s) — exact mathematical boundary of what was removed
  - Click selection cluster — boundary splats identified by color-similarity flood fill
  - PackedSplats.pushSplat() to create new splats, SplatMesh to render them

  The fundamental constraint: Gaussian splats are 2.5D. Behind an occluded object, there may be zero captured data. We're
  not interpolating through missing samples — we're generating data that was never observed.

  ---
  Category 1: Boundary Propagation

  1A — Wavefront Diffusion
  Expand a wavefront inward from all boundary splats simultaneously. Each iteration: advance one step inward, generate a new
   splat at each front position. Properties = distance-weighted average of the frontier splats that spawned it. Fronts from
  opposite sides merge.

  - Pros: Simple, fast (O(n) per iteration, few iterations), naturally fills any shape
  - Cons: Produces smooth/blurry fill — no texture, no structure. Basically 3D blur.
  - Quality: Low. This IS interpolation, just iterative.

  1B — Directional Extrusion Along Surface Normals
  Each boundary splat has an orientation (quaternion encodes the splat's disk normal). Project inward along the surface
  continuation direction. New splats inherit properties and get gradually blended with opposing-side boundary splats.

  - Pros: Preserves surface continuity direction, handles curved surfaces
  - Cons: Normals may point in conflicting directions at concavities. No texture synthesis.
  - Quality: Medium for smooth surfaces, poor for textured ones.

  1C — Anisotropic Diffusion
  Like 1A but diffusion strength varies: strong along surfaces (parallel to splat normals), weak across surfaces
  (perpendicular). Preserves edges and surface orientation while filling.

  - Pros: Better edge preservation than isotropic diffusion
  - Cons: Still fundamentally averaging — can't create detail that doesn't exist
  - Quality: Medium-low.

  ---
  Category 2: Surface Reconstruction

  2A — Implicit Surface Continuation (Poisson-style)
  Boundary splats define a partial surface with oriented normals. Reconstruct an implicit surface (signed distance field)
  from boundary points + normals, solve for the continuation through the hole, then sample new splat positions on the
  reconstructed surface.

  - Pros: Mathematically principled, produces smooth continuous surfaces
  - Cons: Assumes a single connected surface. Holes often expose multiple surfaces (wall meets floor). Heavy computation
  (solve Poisson equation).
  - Quality: High for geometry, but properties still need a separate assignment step.
  - Feasibility: Moderate — we'd need a Poisson solver. Could use a simpler RBF-based implicit surface.

  2B — Multi-Surface Segmentation + Per-Surface Extension
  First segment boundary splats into distinct surfaces (by normal clustering — k-means on quaternion orientations). Then
  independently extend each surface through the hole using plane/quadric fitting. Handle intersections by depth-ordering
  surfaces from a viewpoint.

  - Pros: Handles the "couch seat + couch back behind deleted pillow" case correctly
  - Cons: Surface segmentation is itself hard. Junction handling (where surfaces meet) requires care.
  - Quality: High — this is the most physically accurate framing of the problem.
  - Feasibility: High. Quaternion clustering is straightforward. Plane fitting per cluster is cheap.

  2C — Alpha Shape / Convex Hull Fill
  Compute the alpha shape of boundary splats, generate points in the interior, assign properties from nearest boundary.

  - Pros: Dead simple, handles arbitrary boundary shapes
  - Cons: Fills the volume, not the surface. Would generate splats floating in mid-air inside the hole. Wasteful and wrong.
  - Quality: Low.

  ---
  Category 3: Texture Synthesis in Splat Space

  3A — Exemplar-Based 3D Texture Synthesis (Efros-Leung in 3D)
  For each new splat to generate, look at its local 3D neighborhood of existing splats. Search the rest of the scene for the
   best-matching 3D neighborhood. Copy the center splat's properties from the match. Grow outward one splat at a time.

  - Pros: Can reproduce complex textures (brick patterns, wood grain, foliage density variations)
  - Cons: O(n²) or worse without acceleration. Defining "3D neighborhood similarity" for unstructured point clouds is hard.
  Extremely slow.
  - Quality: Potentially excellent if it works.
  - Feasibility: Low for real-time. Could work as a background process.

  3B — PatchMatch in Voxel Space
  Discretize into a voxel grid. Fill the hole using PatchMatch — randomly initialize fill voxels, then iteratively improve
  by propagating good matches from neighbors and random search. Each voxel stores splat distribution statistics.

  - Pros: Well-understood algorithm, GPU-friendly, handles repetitive textures brilliantly
  - Cons: Voxelization loses splat-level detail. Converting back from voxels to individual splats is lossy.
  - Quality: Medium-high for repetitive surfaces. Poor for unique regions.
  - Feasibility: Medium. Voxelization is already built (spatial index).

  3C — Statistical Donor Region Cloning
  Find the "material signature" of the background surface from boundary splats (avg color, color variance, density,
  orientation distribution). Search the scene for donor regions with matching signatures. Clone splat distributions from
  donors, transform to fit the hole geometry.

  - Pros: Captures real texture from elsewhere in the scene. Fast (one-time search + transform).
  - Cons: May not find a good donor. Seams at junction with boundary. Assumes the occluded surface is similar to visible
  surfaces of the same material.
  - Quality: High when a good donor exists. Poor otherwise.
  - Feasibility: High. We already have per-cell statistics in the spatial index.

  ---
  Category 4: View-Dependent / Projection Methods

  4A — Depth Map Extrapolation + Backprojection
  From the current camera viewpoint: project all splats to get a depth map. The hole is a connected region in this depth
  map. Extrapolate depth into the hole (bilinear, bicubic, or PDE-based inpainting of the depth channel). Extrapolate color
  similarly from the 2D projection. Back-project new (color, depth) pixels to 3D positions. Create splats there.

  - Pros: Leverages the 2.5D nature directly. 2D inpainting is well-understood and fast. Result looks correct from the
  current viewpoint.
  - Cons: View-dependent — looks wrong from other angles. Single-surface assumption in depth. Need to handle depth
  discontinuities.
  - Quality: High from one viewpoint, degrades as camera moves.
  - Feasibility: High. Projection and back-projection are standard ops.

  4B — Multi-View Consensus Fill
  Do 4A from N virtual viewpoints. For each candidate 3D fill point, check that it's consistent across multiple viewpoints.
  Keep only points with cross-view agreement.

  - Pros: Much more robust than single-view. Approaches actual 3D correctness.
  - Cons: O(N × cost of 4A). Consensus logic is complex. Conflicting viewpoints at depth discontinuities.
  - Quality: High if N is sufficient (5-10 views).
  - Feasibility: Medium. Expensive but parallelizable.

  4C — Neural Depth Completion Proxy
  If we had a depth completion model, we could complete the depth map in 2D and back-project. We don't have one in-browser,
  but we could use the LLM (Claude with vision) to suggest what the depth structure should look like, then place splats
  accordingly.

  - Pros: Leverages semantic understanding ("wall continues flat behind pillow")
  - Cons: LLMs aren't precise enough for geometric placement. Latency.
  - Quality: Semantically correct, geometrically imprecise.
  - Feasibility: Medium — we already have the LLM pipeline.

  ---
  Category 5: Physics / Energy Minimization

  5A — Constrained Energy Minimization
  Define an energy function over fill splat configurations:
  - Smoothness: neighboring fill splats should have similar properties
  - Boundary: fill splats near boundary should match boundary splats
  - Density: local splat density should match surrounding scene density
  - Orientation: splat orientations should follow the local surface flow field
  - Flatness: prefer coplanar configurations (surfaces are locally flat)

  Optimize via gradient descent or L-BFGS.

  - Pros: Most general framework. Can encode any prior knowledge as energy terms. Other methods are special cases.
  - Cons: Slow (iterative optimization). Non-convex — may converge to bad local minima. Need to choose energy weights.
  - Quality: Depends entirely on energy function design.
  - Feasibility: Low for real-time. Background process only.

  5B — Spring-Lattice Relaxation
  Place seed splats on a grid inside the hole. Connect to boundary splats with virtual springs. Run damped simulation until
  equilibrium. Spring rest lengths and stiffnesses encode desired density and spacing.

  - Pros: Intuitive, produces uniform spacing, handles arbitrary hole shapes
  - Cons: Only positions geometry — doesn't address color/texture/orientation. Slow convergence.
  - Quality: Medium geometry, no texture intelligence.
  - Feasibility: Medium.

  5C — Gaussian Process Regression
  Model each splat property (R, G, B, opacity, scale_x, etc.) as a Gaussian Process over 3D space. Condition on observed
  boundary values. Sample or take the mean prediction at interior points.

  - Pros: Principled uncertainty quantification. Smooth interpolation with well-understood behavior. Can capture spatial
  correlations.
  - Cons: GP inference is O(n³) in boundary point count. Only practical for small boundaries (~few hundred points).
  - Quality: High for smooth surfaces. Can't generate texture beyond the correlation length.
  - Feasibility: Low-Medium. Matrix inversion scales poorly.

  ---
  Category 6: Structural / Geometric Reasoning

  6A — Symmetry Exploitation
  Many real objects and scenes have local or global symmetry. Detect symmetry axes/planes from surrounding splats. Reflect
  existing splats through symmetry transforms to fill the hole.

  - Pros: When applicable, produces perfect results (floor tile pattern, symmetric room)
  - Cons: Not always applicable. Detection is hard. Partial symmetry even harder.
  - Quality: Perfect when symmetry exists. Useless otherwise.
  - Feasibility: Medium. Symmetry detection in unstructured point clouds is a known problem.

  6B — Skeleton/Medial-Axis-Guided Fill
  Compute the medial axis of the boundary (the "spine" of the hole). Generate splats along sheets extending from the medial
  axis to the boundary, spacing them to match surrounding density.

  - Pros: Handles long thin holes and complex topologies well
  - Cons: Medial axis is unstable (sensitive to boundary noise). Overkill for simple holes.
  - Quality: Medium — good geometry, no texture.
  - Feasibility: Medium.

  6C — Plane Detection + Parameterized Fill
  Run a plane detector (RANSAC) on boundary splats. Most indoor scenes are dominated by planes (walls, floors, ceilings,
  tabletops). For each detected plane, generate a uniform splat distribution on the plane segment within the hole, with
  properties sampled from the plane's boundary splats.

  - Pros: Fast, robust, handles 80% of indoor scenes. Planes are the most common background surface.
  - Cons: Fails for curved or irregular surfaces. Multiple intersecting planes need junction handling.
  - Quality: High for planar scenes (rooms, furniture). Poor for organic scenes (outdoor, plants).
  - Feasibility: Very high. RANSAC plane fitting is ~50 lines of code. We have all the data.

  ---
  Category 7: Hybrid / Novel Approaches

  7A — "Onion Peeling" Layer Decomposition
  Observation: the scene isn't random — it has layers (foreground objects, mid-ground surfaces, background walls). Decompose
   the scene into depth layers around the hole. For each layer, extend it independently through the hole. Composite layers
  back together.

  - Pros: Handles multi-surface holes correctly. Each layer is simpler to fill independently.
  - Cons: Layer decomposition is itself hard. How many layers? Where are the boundaries?
  - Quality: High conceptually.
  - Feasibility: Medium. Could use depth discontinuity detection + flood fill.

  7B — "Scene Grammar" Structural Fill
  Use the spatial index + semantic labeling to understand the scene structure. If we know we deleted a pillow from a couch,
  we know the fill should be "couch seat texture." Query the LLM for structural reasoning, then use a simple geometric fill
  with properties cloned from the identified surface.

  - Pros: Semantic correctness — fills with the right thing, not just smooth blobs
  - Cons: Requires semantic understanding pipeline. LLM latency. Still needs a geometric backend.
  - Quality: Semantically excellent. Geometric quality depends on the fill backend.
  - Feasibility: Medium-High. We already have the LLM + spatial index + scene manifest.

  7C — The Composite Algorithm ✅ IMPLEMENTED (src/infill.ts)

  Combine the strongest practical ideas:

  1. Segment boundary into surfaces (quaternion clustering, 2B)
  2. Fit geometry per surface (RANSAC planes / quadrics, 6C)
  3. Sample fill positions on fitted surfaces at matching density
  4. Assign properties via donor cloning (find matching material elsewhere in scene, 3C) with boundary-blend falloff
  5. Handle junctions between surfaces via depth-ordering from camera

  This isn't one algorithm — it's a pipeline where each stage handles one aspect of the problem.

  ### Implementation Notes (2026-03-17)

  Implemented as `src/infill.ts`, auto-triggered by executor after delete ops. Key findings:

  **What worked:**
  - Quaternion → normal extraction (smallest scale axis) reliably identifies surface orientation
  - K-means on angular distance separates floor from wall from ceiling correctly
  - RANSAC with normal consistency check (reject planes >45° from cluster normal) converges fast
  - SDF-distance-weighted color assignment solves boundary contamination (inner boundary = object, outer = background)
  - 12% deletion oversize eats degraded grazing-angle splats, infill covers the expanded hole

  **Critical tuning discovered:**
  - Boundary width must be adaptive: 20% of smallest shape dimension, NOT a fixed softEdge multiplier. Fixed 0.25 captured 74k splats (15% of scene); adaptive 0.044 captured 19k.
  - Must cap boundary splats at ~2000 with stride subsampling. K-means on 74k splats = 3.3s. On 2000 = <100ms.
  - Fill splat scale must be computed from grid spacing (1.5x for overlap), NOT inherited from boundary. Boundary scale is fine-detail object scale, too small for flat surface coverage.
  - ≥60% jitter on grid sampling breaks visible dot patterns. 20% is insufficient.
  - Small clusters (<15% of largest) and weak planes (inlier <0.5) must be filtered — mixed normals produce 45° tilted fill disks.
  - softEdge MUST be 0 for visual deletion. Any softEdge >0 creates blurry semi-transparent residue worse than a hard cut.

  **Not yet implemented from the plan:**
  - Donor cloning (3C): currently uses boundary-only color blending, no scene-wide donor search
  - Junction handling between surfaces: currently surfaces are independent
  - Quadric fitting for curved surfaces: only planar RANSAC implemented

  **Performance:** ~400ms for typical deletions (target was <500ms). Dominated by boundary identification + KNN.

  ---
  Feasibility × Quality Matrix

  ┌──────────────────────────┬───────────────┬───────────┬────────────┬─────────────────────────────┐
  │         Approach         │    Quality    │   Speed   │ Complexity │          Best For           │
  ├──────────────────────────┼───────────────┼───────────┼────────────┼─────────────────────────────┤
  │ 1A Wavefront Diffusion   │ Low           │ Fast      │ Low        │ Proof of concept only       │
  ├──────────────────────────┼───────────────┼───────────┼────────────┼─────────────────────────────┤
  │ 1B Normal Extrusion      │ Medium        │ Fast      │ Low        │ Smooth curved surfaces      │
  ├──────────────────────────┼───────────────┼───────────┼────────────┼─────────────────────────────┤
  │ 2A Poisson Surface       │ High          │ Slow      │ High       │ Single-surface holes        │
  ├──────────────────────────┼───────────────┼───────────┼────────────┼─────────────────────────────┤
  │ 2B Multi-Surface Segment │ High          │ Medium    │ Medium     │ Multi-surface holes         │
  ├──────────────────────────┼───────────────┼───────────┼────────────┼─────────────────────────────┤
  │ 3A 3D Texture Synthesis  │ Excellent     │ Very Slow │ Very High  │ Textured surfaces (offline) │
  ├──────────────────────────┼───────────────┼───────────┼────────────┼─────────────────────────────┤
  │ 3C Donor Cloning         │ High          │ Fast      │ Medium     │ Repetitive materials        │
  ├──────────────────────────┼───────────────┼───────────┼────────────┼─────────────────────────────┤
  │ 4A Depth Extrapolation   │ High (1 view) │ Fast      │ Medium     │ Quick fill                  │
  ├──────────────────────────┼───────────────┼───────────┼────────────┼─────────────────────────────┤
  │ 4B Multi-View Consensus  │ High          │ Slow      │ High       │ Robust fill                 │
  ├──────────────────────────┼───────────────┼───────────┼────────────┼─────────────────────────────┤
  │ 5A Energy Minimization   │ Variable      │ Very Slow │ Very High  │ Research                    │
  ├──────────────────────────┼───────────────┼───────────┼────────────┼─────────────────────────────┤
  │ 5C Gaussian Process      │ High          │ Slow      │ High       │ Small holes                 │
  ├──────────────────────────┼───────────────┼───────────┼────────────┼─────────────────────────────┤
  │ 6C RANSAC Planes         │ High (indoor) │ Very Fast │ Low        │ Planar scenes               │
  ├──────────────────────────┼───────────────┼───────────┼────────────┼─────────────────────────────┤
  │ 7A Onion Peeling         │ High          │ Medium    │ Medium     │ Layered scenes              │
  ├──────────────────────────┼───────────────┼───────────┼────────────┼─────────────────────────────┤
  │ 7B Scene Grammar         │ Semantic      │ Medium    │ Medium     │ LLM-assisted                │
  ├──────────────────────────┼───────────────┼───────────┼────────────┼─────────────────────────────┤
  │ 7C Composite Pipeline    │ Highest       │ Medium    │ Medium     │ General case                │
  └──────────────────────────┴───────────────┴───────────┴────────────┴─────────────────────────────┘

  ---
  My Assessment

  The approaches worth implementing, in order of increasing sophistication:

  Tier 1 — Ship Tomorrow:
  - 6C (RANSAC Plane Fill) — covers 80% of indoor scenes. ~100 lines. Fast.

  Tier 2 — Ship This Week:
  - 2B + 3C (Multi-Surface Segmentation + Donor Cloning) — handles the general case. The boundary splats' quaternions give
  us surface normals → cluster into surfaces → fit planes/quadrics → clone matching material from elsewhere in the scene.

  Tier 3 — Full Solution:
  - 7C (Composite Pipeline) — 2B + 6C + 3C + 4A for junction handling. This is the real algorithm.

  The key insight that makes this tractable: splat quaternions encode surface orientation. We don't need to guess what
  surfaces exist behind the deleted object — the boundary splats tell us via their orientations. Clustering boundary splat
  normals directly reveals how many surfaces there are and their orientations. Everything flows from that.