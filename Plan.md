# Rasterizer Refactor Plan

## Scope and Execution Rules

This roadmap covers correctness, validation, architecture, glTF support, rendering quality, performance, cleanup, and release hygiene. Phases are ordered by dependency so later work is not built on temporary data models.

- Complete and commit bounded subphases separately.
- Keep correctness changes separate from performance changes.
- Do not retain legacy implementations, compatibility backends, or duplicate code paths after a replacement is verified.
- Run release-mode validation after every subphase.
- Add regression coverage before changing behavior that is difficult to inspect manually.
- Clean comments locally when touching a file; reserve the final global sweep for Phase 9.
- Preserve comments only when they explain invariants, coordinate systems, file-format rules, numerical choices, or non-obvious tradeoffs.

## Completed Baseline

### Phase 1: Establish a Test Baseline

Status: completed

- Added `src/lib.rs` so integration tests and tooling can import renderer modules.
- Kept `src/main.rs` focused on argument parsing and application dispatch.
- Added unit coverage for interpolation, transforms, texture sampling, mip selection, and TOML defaults.
- Added headless rendering tests for clipping, culling, depth, alpha masks, SSAA resolve, transparency, and PBR output.

### Phase 2: Correct Framebuffer and Rendering Safety

Status: completed

- Replaced shared framebuffer synchronization with exclusive horizontal-band ownership.
- Stored color and depth together so the winning fragment commits both values consistently.
- Removed framebuffer locks, atomics, `UnsafeCell`, manual `Sync`, and old per-pixel compatibility APIs.
- Split triangle preparation from band-owned rasterization while preserving transparent order within each band.
- Made the shadow pass sample masked-material alpha and exclude blended materials from depth casting.
- Associated each shadow map with its selected directional light rather than assuming light index zero.
- Rebuilt supersampling and shadow buffers during hot reload, rejected invalid sizes, and required restart for window-size changes.

Baseline after Phase 2: 35 tests, no `unsafe` in `src/` or `tests/`, and deterministic output between one and eight Rayon workers for the smoke scene.

## Phase 3: Configuration, Validation, and Error Foundations

Status: completed

This phase precedes the glTF rewrite because import, image, CLI, and GUI errors should share one stable model.

### 3A: Rust Support and CI

Status: completed

- [x] Declare `rust-version = "1.85"` as the intended edition-2024 minimum; document that it is not yet exercised by a separate MSRV job.
- [x] Add Windows CI using the runner's current stable Rust toolchain for:
  - `cargo fmt --all -- --check`;
  - `cargo clippy --release --all-targets --all-features -- -D warnings`;
  - `cargo test --release`;
  - `cargo check --release`.
- [x] Add a small headless smoke render on the Windows CI runner without GUI interaction.
- Do not install or switch to a second Rust version during routine CI; exact MSRV verification can be added later as a dedicated compatibility job if needed.

### 3B: Typed Configuration and Consistent Defaults

Status: completed

- [x] Replace projection, cull mode, and light type strings with Serde enums.
- [x] Define an empty document as the complete repository-default scene and test that behavior.
- [x] Fix partial-table defaults so omitted fields inherit their complete struct defaults rather than type-level zero values.
- [x] Rename the old `samples` field to `supersample_scale` because `2` means a 2×2 grid, not two samples; reject the old field instead of retaining a compatibility path.

### 3C: Validation and Checked Dimensions

Status: completed

- [x] Add `Config::validate()` covering dimensions, supersampling, shadow-map size, camera planes, FOV, exposure, PCF radius, transforms, and finite numeric values.
- [x] Reject zero or overflowing framebuffer dimensions before allocation.
- [x] Reject zero-length or invalid camera/light vectors before normalization.
- [x] Introduce checked renderer/framebuffer construction or a validated dimensions type so library callers cannot bypass critical size checks.

### 3D: Structured Errors and Exit Behavior

Status: completed

- [x] Introduce structured application, config, asset, glTF, image-output, and window errors.
- [x] Return errors from CLI/GUI orchestration instead of panicking or logging-and-continuing.
- [x] Stop silently replacing an unreadable or invalid requested config with the default car scene.
- [x] Return a nonzero process status when configuration, rendering, or image saving fails.
- [x] Create output parent directories where appropriate.

### 3E: Configuration-Relative Paths

Status: completed

- [x] Resolve model, texture, background, and output paths consistently relative to the loaded configuration file.
- [x] Carry the configuration base directory explicitly rather than relying on process working directory.
- [x] Test configs loaded from nested directories.

Suggested commits:

1. `declare rust support and add ci` (completed)
2. `define typed scene configuration` (completed)
3. `validate render configuration` (completed)
4. `propagate application errors` (completed)
5. `resolve assets relative to config` (completed)

Exit criteria:

- Invalid configuration fails before resource allocation.
- CLI failures return nonzero status.
- Defaults are consistent for empty and partial TOML documents.
- Paths behave identically regardless of launch working directory.
- CI runs the complete release-mode test baseline on the current stable toolchain.

## Phase 4: glTF Structural Correctness

Status: completed

This phase fixes importer topology, indexing, validation, and diagnostics without yet redesigning the full material/texture model.

### 4A: Minimal Fixtures

Status: completed

Created small fixtures under `tests/fixtures/gltf/` for:

- [x] one image referenced by multiple glTF textures;
- [x] texture indices that differ from source-image indices;
- [x] nested nodes, including legitimate names containing `plane` or `shadow`;
- [x] indexed and non-indexed triangles;
- [x] triangle strips and fans;
- [x] invalid indices and mismatched attribute counts;
- [x] unsupported primitive modes and image formats.

Keep fixtures in the kilobyte range and independent of demonstration assets.

### 4B: Contextual Import Errors

Status: completed

- [x] Make recursive node and primitive processing return structured `Result` values.
- [x] Include file path, scene, node, mesh, primitive, attribute, and texture context where available.
- [x] Remove importer `unwrap()` calls and silent magenta-image fallbacks.

### 4C: Scene Traversal and Primitive Topology

Status: completed

- [x] Remove implicit filtering of node names containing `plane` or `shadow`.
- [x] Always recurse into child nodes, even if a current mesh is intentionally excluded by future explicit configuration.
- [x] Support `Triangles` directly.
- [x] Convert `TriangleStrip` and `TriangleFan` to triangle lists with correct winding.
- [x] Reject point and line modes with an explicit unsupported-feature error.

### 4D: Image and Texture Index Mapping

Status: completed

- [x] Resolve texture references through `texture.source().index()` rather than indexing the image array with `texture.index()`.
- [x] Preserve sharing when several texture objects reference one source image.
- [x] Do not hide the mapping bug by duplicating image data.

### 4E: Attribute and Index Validation

Status: completed

- [x] Require POSITION.
- [x] Validate NORMAL, TEXCOORD, and TANGENT counts against POSITION.
- [x] Validate every index before accessing a vertex.
- [x] Reject non-finite positions and unsafe zero-length normal/tangent normalization.
- [x] Generate missing normals using a documented area-weighted method.
- [x] Return an explicit Unsupported diagnostic when a normal map requires missing tangents, pending MikkTSpace in Phase 6.

### 4F: Model Normalization Policy

Status: completed

- [x] Replace unconditional center-and-normalize behavior with explicit `preserve`, `center`, and `normalize` policies.
- [x] Keep `normalize` as the documented default to preserve existing scene behavior, and test object scale/placement across multiple assets.

Non-goals for this phase:

- animation, skinning, morph targets;
- Draco or KTX2/Basis;
- glTF cameras and lights;
- full sampler/UV-set semantics;
- MikkTSpace tangent generation.

Suggested commits:

1. `add minimal gltf fixtures`
2. `propagate contextual gltf errors`
3. `support triangle primitive modes`
4. `fix gltf texture source mapping`
5. `validate gltf mesh attributes`
6. `make model normalization explicit`

Exit criteria:

- No importer panic path remains for malformed external data.
- Supported topology is explicit and tested.
- Texture/source-image mapping follows the glTF model.
- Legitimate node names are never filtered heuristically.

## Phase 5: Rendering State and Submission Architecture

Status: completed

This phase must precede `doubleSided`, complete sampler semantics, and major performance work.

### 5A: Decouple Core Rasterization from Scene Materials

Status: completed

- [x] Remove `scene::Material` and `AlphaMode` dependencies from `core/rasterizer.rs` and `core/pipeline.rs`.
- [x] Make fragment execution return an explicit discard or RGBA result.
- [x] Keep material lookup and alpha-cutoff decisions in shaders or render-command construction.

Exit condition: `src/core/` no longer imports scene material types.

### 5B: Explicit RenderState

Status: completed

Introduced per-draw state for:

- [x] cull mode;
- [x] depth test and comparison;
- [x] depth write;
- [x] blend mode;
- [x] wireframe/debug mode.

- [x] Remove pass behavior that depends on mutating global rasterizer fields between opaque and transparent draws.

### 5C: RenderCommand and RenderQueue

Status: completed

- [x] Build explicit shadow, opaque, masked, and transparent command lists.
- [x] Prepare and bin a complete pass rather than rebuilding bins separately for each mesh.
- [x] Give commands stable insertion IDs.
- [x] Sort transparent commands with `f32::total_cmp` plus a deterministic tie-breaker.

### 5D: Per-Material Culling and Double-Sided Foundations

Status: completed

- [x] Import and store `doubleSided`.
- [x] Disable culling per command for double-sided materials.
- [x] Expose front-facing state to fragment shading so back-face normal/tangent handling can follow glTF semantics.

### 5E: Stable Scene Identity and Hot Reload

Status: completed

- [x] Add stable scene-object IDs or an explicit `SceneObjectKind`.
- [x] Stop inferring the ground object from vector length or position zero.
- [x] Classify reload changes as live-update, renderer rebuild, resource reload, or window restart.
- [x] Rebuild scene assets when object paths, object count, or mip policy changes.

### 5F: Centralized Object Transforms

Status: completed

- [x] Add one transform-construction function used by initial load and hot reload.
- [x] Document translation, Euler rotation order, handedness, and scale behavior.

Suggested commits:

1. `decouple rasterizer from scene materials`
2. `introduce explicit render state`
3. `build deterministic render queues`
4. `support per-material culling`
5. `give scene objects stable identities`
6. `centralize object transforms`

## Phase 6: glTF Material, Texture, and Geometry Semantics

Status: completed

### 6A: Texture Resource Model

Status: completed

- [x] Separate:

  - image pixel/mip data;
  - sampler state;
  - material texture binding.

- [x] Make each binding identify the image, sampler, UV set, and color-space/data usage.
- [x] Preserve image sharing when several glTF textures use different samplers.

### 6B: Sampler Semantics

Status: completed

Implement and test:

- [x] Repeat;
- [x] ClampToEdge;
- [x] MirroredRepeat;
- [x] glTF minification/magnification choices;
- [x] non-mip filters without forced trilinear sampling.

### 6C: UV Sets and Texture Bindings

Status: completed

- [x] Support `TEXCOORD_0` and `TEXCOORD_1` through import, interpolation, LOD selection, and shading.
- [x] Store a typed UV-set selection per material texture slot.
- [x] Explicitly reject unsupported higher sets and missing material-required sets.
- [x] Reject `KHR_texture_transform` explicitly as a separate extension task rather than silently ignoring it.

### 6D: Complete Material Factors

Status: completed

- [x] Multiply base-color texture RGB and alpha by their factors.
- [x] Multiply metallic and roughness texture channels by their factors.
- [x] Apply normal texture scale to tangent-space X/Y components.
- [x] Blend occlusion from fully lit to the texture R channel using occlusion strength.
- [x] Multiply emissive texture RGB by the emissive factor and explicitly reject unsupported `KHR_materials_emissive_strength`.
- [x] Replace non-finite inputs and clamp core material factors to their supported ranges.
- [x] Preserve per-material double-sided behavior from Phase 5.

The base-color RGB factor bug was fixed first, before the remaining 6D factors were completed.

### 6E: Normals, Tangents, and Mirrored Transforms

Status: completed

- [x] Preserve UV-set support through tangent generation.
- [x] Generate missing normals before tangent generation.
- [x] Generate MikkTSpace-compatible tangents using the normal map's selected UV set.
- [x] Handle negative-determinant and non-uniform node and object transforms.
- [x] Preserve tangent handedness and mirrored front-face semantics.

Covered UV seams, mirrored and degenerate UVs, negative node scales, non-uniform transforms, and runtime mirrored object transforms.

Suggested commits:

1. `separate images samplers and bindings`
2. `implement gltf sampler modes`
3. `support multiple uv sets`
4. `apply complete pbr material factors`
5. `generate normals and mikktspace tangents`
6. `handle mirrored geometry transforms`

## Phase 7: Color and Rendering Correctness

Status: completed

### 7A: Color Space

Status: completed

- [x] Replace gamma-2.2 approximations with standard piecewise sRGB transfer functions.
- [x] Decode color texels before bilinear/trilinear interpolation.
- [x] Keep metallic, roughness, normal, and AO textures as linear data.
- [x] Prevent negative or non-finite values from reaching gamma conversion.
- [x] Define TOML color triplets as linear RGB values.

### 7B: Mip Correctness

Status: completed

- [x] Generate color mips in linear space.
- [x] Downsample data textures without color conversion.
- [x] Renormalize normal-map mips.
- [x] Continue 1×N and N×1 chains until 1×1.
- [x] Add non-square, odd-sized, and single-axis texture tests.

### 7C: Rasterization Rules

Status: completed

- [x] Implement a standard top-left fill rule.
- [x] Test two triangles sharing an edge for both cracks and double coverage.
- [x] Use pixel-space edge distance for stable wireframe width.
- [x] Reject non-finite clip, screen, and depth values consistently.

### 7D: Shadow Quality

Status: completed

- [x] Separate constant and slope-scaled bias.
- [x] Treat PCF taps outside the shadow map as lit border samples instead of clamping them to edge texels.
- [x] Fit directional shadow bounds to the camera frustum and scene geometry, capping the camera-frustum reach with the configured shadow distance.
- [x] Keep point-light cubemap shadows as an explicit future feature.

### 7E: PBR Scope

Status: completed

- [x] Keep the renderer scoped to direct-light metallic-roughness PBR plus an ambient approximation.
- [x] Document that image-based lighting is intentionally not implemented.

## Phase 8: Profile-Guided Performance Work

Status: in progress

This is the reordered version of the original Phase 6. Correctness and stable submission/resource models must land first.

### 8A: Benchmark Baseline

Status: completed

- [x] Record scene loading.
- [x] Record shadow preparation and rasterization.
- [x] Record main-pass preparation.
- [x] Record combined opaque/masked and separate transparent rasterization without changing the existing whole-pass binning architecture.
- [x] Record post-processing and complete frame time.
- [x] Reject output changes across measured frames and compare hashes across worker counts.
- [x] Emit per-frame CSV, environment metadata, and a mean/p95 Markdown summary.

The committed benchmark driver covers one large triangle, 400 small triangles, the default car, the city asset, high transparency, shadows on/off, 1×/2× supersampling, and one versus all configured physical cores. The initial Windows/i5-13500H baseline is recorded under `benchmarks/baselines/`.

### 8B: Low-Risk Resource and Allocation Improvements

Status: completed

- [x] Cache background textures and reload only when path or mip policy changes.
- [x] Reuse shadow depth storage with copy-on-write protection when an older frame still holds it.
- [x] Borrow lights and shadow depth from pass resources instead of cloning them per object.
- [x] Process shared indexed vertices once per queue group; skip the cache when indices do not reuse vertices.
- [x] Cache tangent-frame transforms, winding orientation, and transparent world-space vertices on static scene objects, rebuilding them when hot reload changes the transform.
- [x] Eliminate per-triangle clipping allocations with fixed-capacity stack storage, retain horizontal-band bins across draws, reserve render queues exactly, and keep output buffers persistent in repeated-frame modes.

The generic prepared-triangle vector remains one whole-queue allocation per draw. Persisting it or the borrowed command queues across frames would require type erasure, unsafe lifetime conversion, or a larger submission-model redesign, so it is outside this low-risk subphase. Phase 8B preserves every Phase 8A output hash. On the recorded i5-13500H matrix, the clearest complete-frame changes were city at -4.1% with one worker and -1.7% with 12 workers, and high transparency at -25.4% with 12 workers; small differences elsewhere are within run-to-run noise. See `benchmarks/results/2026-08-29-phase-8b-i5-13500h.md`.

### 8C: Whole-Pass Binning

Status: completed

- [x] Confirm that rasterization already bins each complete queue group once.
- [x] Flatten all mesh triangles across the submitted queues into one ordered preparation domain.
- [x] Prepare mesh work with one whole-pass Rayon traversal instead of one traversal and temporary vector per mesh command.
- [x] Preserve command and triangle order through clipping and final collection.
- [x] Keep pure transparent queues on the lower-overhead sequential preparation path while retaining global back-to-front ordering.
- [x] Cover empty commands across queue boundaries and retain deterministic output hashes.

The city scene validates the implementation rather than an audit-only close. In a 30-frame before/after run with 12 workers, shadow preparation changed from 34.161 to 5.965 ms, main preparation from 38.524 to 7.003 ms, and complete frame time from 80.797 to 19.477 ms. The standard matrix reproduced the city improvement while preserving every Phase 8A/8B hash. Pure transparency and small-triangle preparation moved by less than 0.1 ms and varied in both directions across repeated runs, so no extra parallelism is used for pure transparent queues. See `benchmarks/results/2026-08-29-phase-8c-i5-13500h.md`.

### 8D: Horizontal Bands versus 2D Tiles

Status: completed

- [x] Benchmark 8-, 16-, and 32-row bands.
- [x] Implement and benchmark safe 16×16 and 32×16 tiles without locks, atomics, or `unsafe`.
- [x] Compare one versus 12 workers across all six benchmark scenes with 5 warmup and 30 measured frames.
- [x] Verify identical output hashes across every candidate, scene, and worker count.
- [x] Remove the experimental selector and losing backends, retaining only the winning 8-row band renderer.

The 8-row band won every 12-worker comparison against the original 16-row band, reducing measured rasterization by 10.9% to 26.9% in the shadowed car cases and by 13.1% to 23.1% in the remaining 1× scenarios. Single-worker differences were mostly within 3.2%. The tile candidates improved the two shadow-heavy car cases further, but regressed large-triangle, many-small-triangle, and city complete-frame results relative to 8-row bands, so they did not meet the consistent-win requirement. See `benchmarks/results/2026-08-29-phase-8d-i5-13500h.md`.

### 8E: Further Experiments

Status: planned

Only after measurement:

- incremental edge equations;
- hierarchical Z;
- SIMD;
- compact sample layouts;
- tuned Rayon task granularity.

## Phase 9: Code, Comment, and Documentation Cleanup

Status: planned

Perform local cleanup in every earlier phase, then run one global pass.

- Remove decorative separators such as `=====`, `-----`, and numbered step banners.
- Remove comments that merely restate the next line.
- Remove commented-out code and historical wording such as “added”, “removed”, “fix”, or “now”.
- Replace vague TODOs with tracked work or delete them.
- Preserve comments explaining coordinate systems, matrix order, glTF packing, transparent ordering, numerical stability, and genuine safety invariants.
- Split oversized test files by subsystem.
- Delete dead APIs, unused modules, temporary aliases, and superseded backends.
- Correct terminology such as MSAA versus SSAA and `shadow apnea` versus `shadow acne`.
- Ensure README, `AGENTS.md`, `scene.toml`, and implementation behavior agree.

Required scans:

```bash
rg -n '={3,}|-{3,}|^\s*//\s*[0-9]+\.' src scene.toml
rg -n 'TODO|FIXME|HACK|Added|Removed|FIX:|Now' src
```

## Phase 10: Dependencies, Repository Weight, and Release Hygiene

Status: planned

### 10A: Dependency Reduction

- Disable unused `image` default formats after supported texture formats are finalized.
- Re-run `cargo tree --duplicates`.
- Remove the dependency path that currently includes the yanked `core2 0.4.0` if dependency updates or feature reduction permit it.

### 10B: Repository and Package Size

- Exclude generated `outputs/` from Cargo packages.
- Keep only required README images in a dedicated documentation directory.
- Avoid storing packed and unpacked copies of the same model unless both are required fixtures.
- Consider Git LFS or release downloads for large demonstration assets.
- Add explicit Cargo package include/exclude rules.
- Verify licenses for all distributed third-party models and textures.

### 10C: Release Verification

Run:

```bash
cargo fmt --all -- --check
cargo clippy --release --all-targets --all-features -- -D warnings
cargo test --release
cargo check --release
cargo package --no-verify
```

Verify a clean clone or unpacked Cargo package can run a headless smoke render without repository-local assumptions.

## Standard Validation for Every Subphase

```bash
cargo fmt --all -- --check
cargo clippy --release --all-targets --all-features -- -D warnings
cargo test --release
cargo check --release
```

Rendering changes must additionally compare deterministic output between one and multiple Rayon workers and inspect or compare a small release-mode image.

## Next-Agent Handoff

Repository state after Phase 8D:

- branch: `main`;
- completed commits through Phase 8A cover the benchmark infrastructure and baseline;
- Phase 7D separates shadow bias terms, defines lit PCF borders, and fits directional shadow bounds;
- Phase 7E records direct-light PBR plus an ambient approximation as the intended scope, without image-based lighting;
- Phase 8B caches stable resources and transforms, shares pass data, deduplicates indexed vertex work, removes clipping heap allocations, and reuses band/depth storage;
- Phase 8C prepares all mesh triangles in a queue group through one ordered parallel domain and removes per-mesh prepared-triangle collections;
- Phase 8D retains 8-row horizontal bands after benchmarking three band heights and two safe 2D tile sizes across the full matrix;
- all Phase 8A matrix hashes remain unchanged between the baseline and Phase 8B and across one versus 12 workers;
- all Phase 8A matrix hashes also remain unchanged after Phases 8C and 8D;
- 131 tests are present after Phase 8D;
- no `unsafe` remains in `src/` or `tests/`.

Immediate target: Phase 8E measurement of further rasterization experiments. Retain only changes that improve the representative benchmark matrix without changing deterministic output.
