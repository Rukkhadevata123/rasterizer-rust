# Rasterizer Refactor Plan

## Scope and Rules

This roadmap covers correctness fixes, test coverage, architecture cleanup, performance work, dependency reduction, documentation alignment, and comment cleanup.

- Complete and commit one phase at a time.
- Keep correctness changes separate from performance changes.
- Run release-mode validation after every phase.
- Do not begin Phase 2 until its design has been reviewed.
- Preserve useful comments that explain invariants, coordinate systems, file-format rules, or safety requirements.

## Phase 1: Establish a Test Baseline

Status: completed

- Add `src/lib.rs` so integration tests and tooling can import the renderer.
- Keep `src/main.rs` limited to argument parsing and application dispatch.
- Add unit tests for:
  - barycentric and perspective-correct interpolation;
  - view, projection, viewport, and object transforms;
  - texture wrapping, bilinear sampling, mip generation, and LOD selection;
  - TOML defaults and representative scene configuration;
  - clipping, culling, depth, alpha mask, and framebuffer resolve behavior.
- Add small headless integration renders using generated geometry and textures.
- Avoid large repository assets in automated tests.
- Establish deterministic image assertions with exact pixels where practical and tolerances for floating-point output.

## Phase 2: Correct Framebuffer and Rendering Safety

Status: blocked pending design review

- Redesign framebuffer synchronization so the winning depth and color always come from the same fragment.
- Remove concurrent creation of mutable references to the complete color vector.
- Prefer tile ownership or per-pixel storage over striped locks and `UnsafeCell<Vec<_>>`.
- If unsafe code remains, document every invariant with a focused `SAFETY` comment.
- Fix alpha-mask and blended-material behavior in the shadow pass.
- Associate the shadow map with an explicit light instead of assuming light index zero.
- Rebuild or reject incompatible renderer settings during hot reload, especially resolution, sample scale, and shadow-map size.

## Phase 3: Harden glTF Import

Status: planned

- Resolve glTF texture references through their source image indices.
- Remove implicit node-name filtering for names containing `plane` or `shadow`; replace it with explicit configuration if needed.
- Validate primitive modes and triangulate supported strip/fan modes.
- Validate attribute lengths and index bounds; replace importer `unwrap` calls with contextual errors.
- Support or explicitly report limitations for:
  - `doubleSided`;
  - normal scale and occlusion strength;
  - sampler wrap modes and alternate UV sets;
  - missing normals and tangents;
  - mirrored and non-uniform transforms.
- Either implement MikkTSpace-compatible tangent generation or remove that claim from documentation.

## Phase 4: Validate Configuration and Errors

Status: planned

- Add `Config::validate()` for dimensions, sample scale, camera planes, FOV, exposure, PCF radius, transforms, and finite vector values.
- Fix partial-struct defaults so omitted camera fields use repository defaults rather than zero arrays.
- Replace projection, cull mode, and light type strings with Serde enums.
- Return structured errors from config loading, glTF import, image saving, and GUI setup.
- Exit with a nonzero status on invalid configuration or failed output.
- Create output parent directories when appropriate.
- Separate live-update fields from settings that require renderer or resource reconstruction.
- Give scene objects stable identities instead of inferring the ground object from vector length.

## Phase 5: Simplify Rendering Architecture

Status: planned

- Decouple the core rasterizer from scene-level `Material` and `AlphaMode` types.
- Return a fragment result that can explicitly represent color, alpha, and discard.
- Introduce a `RenderState` containing culling, depth test/write, blending, and wireframe options.
- Build explicit opaque, masked, and transparent render queues.
- Use deterministic transparent sort keys with stable tie-breaking.
- Centralize object transform construction and document rotation order.
- Reconsider the `Material` enum while it contains only one variant.

## Phase 6: Profile-Guided Performance Work

Status: planned

- Add pass-level timing and representative benchmarks before optimizing.
- Cache background textures and reload only when their path or mip policy changes.
- Reuse the shadow depth storage instead of copying it into a new `Vec<f32>` and `Arc` every frame.
- Borrow or share lights instead of cloning the light list for every object.
- Process each indexed vertex once per pass instead of once per triangle.
- Replace nested triangle/scanline Rayon parallelism with tile binning and tile-level ownership.
- Reuse clipping scratch storage instead of allocating two vectors per triangle.
- Store texture mips in a uniform contiguous format instead of sampling through `DynamicImage`.
- Continue mip generation until both dimensions reach one.
- Evaluate incremental edge equations, top-left fill rules, and hierarchical depth rejection.

## Phase 7: Improve Rendering Quality

Status: planned

- Multiply glTF base-color textures by the complete base-color factor.
- Clamp and sanitize material inputs; keep roughness away from unstable zero values.
- Use the standard piecewise sRGB transfer function instead of a simple gamma approximation.
- Prevent negative or non-finite values from reaching gamma conversion.
- Generate color-texture mips in linear space while preserving data textures as linear data.
- Improve shadow border sampling and separate constant from slope-scaled bias.
- Support glTF double-sided materials independently of the global cull mode.
- Either add image-based lighting or accurately document the current direct-light plus ambient approximation.

## Phase 8: Remove Redundant Comments and Code

Status: planned

- Remove decorative separators such as `=====`, `-----`, and numbered step banners.
- Remove comments that merely repeat the following statement.
- Remove commented-out code and historical comments such as “added”, “removed”, “fix”, or “now”.
- Preserve comments that explain:
  - unsafe invariants;
  - coordinate-system and matrix conventions;
  - glTF channel packing;
  - transparent ordering;
  - non-obvious numerical or performance tradeoffs.
- Replace vague TODOs with tracked work or remove them.
- Rename misleading concepts, including `samples` if it remains a supersampling scale rather than a sample count.
- Correct stale wording such as MSAA versus SSAA and “shadow apnea” versus “shadow acne”.
- Align README, `AGENTS.md`, `scene.toml`, and implementation behavior.

## Phase 9: Reduce Dependencies and Repository Weight

Status: planned

- Disable unused default `image` formats and enable only formats required by the renderer and glTF importer.
- Recheck duplicate dependencies and remove the dependency path that currently includes the yanked `core2 0.4.0`.
- Declare `rust-version = "1.85"` in `Cargo.toml`.
- Exclude generated outputs and unnecessary large assets from Cargo packages.
- Keep only required documentation images in a dedicated directory.
- Avoid storing both packed and unpacked copies of the same model unless both are test fixtures.
- Consider Git LFS or release downloads for large demonstration models.
- Verify third-party asset licenses.
- Add CI for formatting, Clippy with warnings denied, release tests, release checks, and a small headless render.

## Validation for Every Phase

```bash
cargo fmt --all -- --check
cargo clippy --release --all-targets --all-features -- -D warnings
cargo test --release
cargo check --release
```

Rendering phases should also run a small deterministic CLI render and inspect or compare the generated image.
