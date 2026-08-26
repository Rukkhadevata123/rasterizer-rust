# Repository Guidelines

## Project Structure & Module Organization

This is a single Rust 2024 binary crate. `src/main.rs` parses CLI arguments and delegates to `src/app.rs`. Code is grouped by responsibility:

- `src/core/`: rasterization, framebuffer, geometry, math, and pipeline traits.
- `src/pipeline/`: render passes, renderer orchestration, and PBR/shadow shaders.
- `src/scene/`: cameras, lights, materials, models, meshes, and textures.
- `src/io/`: TOML configuration, glTF loading, and PNG output.
- `src/ui/`: interactive input handling.

Runtime assets live in `assets/`; `scene.toml` is the primary example configuration. Keep generated images in `outputs/` or another ignored output path.

## Build, Test, and Development Commands

Rust 1.85 or newer is required for edition 2024.

```bash
cargo run --release -- --config scene.toml
cargo run --release -- --config scene.toml --gui
cargo fmt --all -- --check
cargo clippy --all-targets --all-features
cargo test
```

Always render with `--release`; debug rasterization is impractically slow. The first command renders one PNG, while the second opens the `minifb` viewer. In GUI mode, `R` reloads configuration but not models or textures.

## Coding Style & Naming Conventions

Use standard `rustfmt` output and address Clippy warnings. Follow Rust conventions: `snake_case` for modules, functions, and variables; `PascalCase` for types and traits; `SCREAMING_SNAKE_CASE` for constants. Preserve the boundary between low-level `core` code and higher-level scene and pipeline orchestration.

## Testing Guidelines

Unit tests live in nearby `#[cfg(test)]` modules, and cross-module rendering tests live under `tests/`. Name tests after observable outcomes, such as `triangle_crossing_near_plane_is_clipped_and_rendered`. Run the full suite with `cargo test --release`. There is no coverage target yet; rendering changes should also produce a release PNG for visual regression checks.

## Concurrency & Safety

Rayon drives parallel rendering. `FrameBuffer` combines `UnsafeCell`, atomics, striped mutexes, and a manual `Sync` implementation. Changes to buffer access or depth testing must preserve documented safety invariants. Transparent triangles must remain sequential after back-to-front sorting so blending order is stable.

## Commit & Pull Request Guidelines

Recent history uses short, lowercase summaries. Prefer concise imperative subjects, for example `fix near-plane clipping`, and keep commits focused. Pull requests should explain motivation, summarize implementation, list validation commands, link relevant issues, and include before/after images for rendering or GUI changes.
