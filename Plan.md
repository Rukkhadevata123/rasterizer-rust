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

Status: in progress

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

- Add one transform-construction function used by initial load and hot reload.
- Document translation, Euler rotation order, handedness, and scale behavior.

Suggested commits:

1. `decouple rasterizer from scene materials`
2. `introduce explicit render state`
3. `build deterministic render queues`
4. `support per-material culling`
5. `give scene objects stable identities`
6. `centralize object transforms`

## Phase 6: glTF Material, Texture, and Geometry Semantics

Status: planned

### 6A: Texture Resource Model

Separate:

- image pixel/mip data;
- sampler state;
- material texture binding.

A binding should identify the image, sampler, UV set, and color-space/data usage. This permits one image to be reused by several glTF textures with different samplers.

### 6B: Sampler Semantics

Implement and test:

- Repeat;
- ClampToEdge;
- MirroredRepeat;
- glTF minification/magnification choices;
- non-mip filters without forced trilinear sampling.

### 6C: UV Sets and Texture Bindings

- Support at least `TEXCOORD_0` and `TEXCOORD_1`.
- Store a UV-set selection per material texture slot.
- Explicitly reject unsupported higher sets.
- Treat `KHR_texture_transform` as a separate extension task rather than silently ignoring it.

### 6D: Complete Material Factors

Implement and test:

- base-color texture multiplied by base-color RGB and alpha factors;
- metallic and roughness channels multiplied by their factors;
- normal texture scale;
- occlusion strength;
- emissive factor and documented extension limitations;
- clamped, finite material inputs;
- per-material double-sided behavior from Phase 5.

Base-color RGB factor is a known current rendering bug and should be the first 6D fix.

### 6E: Normals, Tangents, and Mirrored Transforms

Order the work as:

1. UV-set support;
2. missing-normal generation;
3. MikkTSpace-compatible tangent generation using the normal map's selected UV set;
4. negative-determinant and non-uniform transform handling;
5. tangent handedness verification.

Test UV seams, mirrored UVs, degenerate UVs, negative node scales, and non-uniform transforms. Do not label a simpler tangent accumulator as MikkTSpace-compatible.

Suggested commits:

1. `separate images samplers and bindings`
2. `implement gltf sampler modes`
3. `support multiple uv sets`
4. `apply complete pbr material factors`
5. `generate normals and mikktspace tangents`
6. `handle mirrored geometry transforms`

## Phase 7: Color and Rendering Correctness

Status: planned

### 7A: Color Space

- Replace gamma-2.2 approximations with standard piecewise sRGB transfer functions.
- Decode color texels before bilinear/trilinear interpolation.
- Keep metallic, roughness, normal, and AO textures as linear data.
- Prevent negative or non-finite values from reaching gamma conversion.
- Define whether TOML colors are linear or sRGB.

### 7B: Mip Correctness

- Generate color mips in linear space.
- Downsample data textures without color conversion.
- Renormalize normal-map mips.
- Continue 1×N and N×1 chains until 1×1.
- Add non-square and single-axis texture tests.

### 7C: Rasterization Rules

- Implement a standard top-left fill rule.
- Test two triangles sharing an edge for both cracks and double coverage.
- Use pixel-space edge distance for stable wireframe width.
- Reject non-finite clip, screen, and depth values consistently.

### 7D: Shadow Quality

- Separate constant and slope-scaled bias.
- Define shadow-map border behavior instead of clamping every PCF tap to an edge texel.
- Evaluate scene- or camera-fitted directional shadow bounds.
- Keep point-light cubemap shadows as an explicit future feature.

### 7E: PBR Scope

Either implement image-based lighting or update documentation to state that the renderer uses direct-light PBR plus an ambient approximation.

## Phase 8: Profile-Guided Performance Work

Status: planned

This is the reordered version of the original Phase 6. Correctness and stable submission/resource models must land first.

### 8A: Benchmark Baseline

Record timings for:

- scene loading;
- shadow preparation and rasterization;
- main-pass preparation;
- opaque, masked, and transparent rasterization;
- post-processing;
- complete frame time.

Use representative scenes: one large triangle, many small triangles, the default car, the city asset, high transparency, shadows on/off, 1×/2× supersampling, and one versus all physical cores. Preserve output hashes during optimization.

### 8B: Low-Risk Resource and Allocation Improvements

- Cache background textures and reload only when path or mip policy changes.
- Reuse shadow depth storage instead of copying it into a new `Vec<f32>` and `Arc` every frame.
- Borrow or share lights instead of cloning them per object.
- Process each indexed vertex once per pass.
- Cache static object transforms where beneficial.
- Reuse clipping buffers, prepared-triangle storage, bins, transparent queues, and output buffers.

### 8C: Whole-Pass Binning

Move from per-mesh preparation/binning to complete render-queue preparation so overlapping work and allocations are amortized across the pass.

### 8D: Horizontal Bands versus 2D Tiles

Benchmark:

- 8-, 16-, and 32-row bands;
- 16×16 and 32×16 tiles.

Compare load balance, cache locality, binning cost, large-triangle duplication, transparent scenes, and CPU core counts. Replace the band renderer only if a tile renderer wins consistently. Keep only the winning backend; do not retain parallel compatibility implementations.

### 8E: Further Experiments

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

Repository state when this roadmap was revised:

- branch: `main`;
- completed commits include `c32a12f` for band ownership and `2bc0630` for remaining Phase 2 safety;
- working tree should be clean before starting implementation;
- 54 tests are present after Phase 3E;
- no `unsafe` remains in `src/` or `tests/`.

Immediate target: Phase 5F. Do not begin the texture resource redesign, MikkTSpace work, or tile renderer in the same change set.
