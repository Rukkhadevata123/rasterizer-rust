# AGENTS.md

## Build & Run

- **Rust edition 2024** requires Rust 1.85+. Cargo.toml: `edition = "2024"`.
- Single binary crate (not a workspace). Binary name: `rasterizer` (Cargo.toml `[[bin]]`).

```bash
# CLI mode: render one frame to PNG (default output.png)
cargo run --release -- --config scene.toml

# GUI mode: real-time interactive viewer (minifb window)
cargo run --release -- --config scene.toml --gui
```

- Always use `--release` — debug builds are unusably slow due to software rasterization.
- `cargo test` does nothing: there are zero tests in the repo.
- `cargo fmt` and `cargo clippy` use defaults (no rustfmt.toml / clippy.toml).
- Default asset path: `assets/glbs/old_rusty_car.glb`. The default config references it.

## Architecture

```
src/
  main.rs        → CLI arg parsing (clap), dispatches to app
  app.rs         → run_gui() / run_cli() orchestrators
  core/          → Engine kernel
    pipeline.rs   → Shader trait + Interpolatable trait (the programmable pipeline interface)
    rasterizer.rs → Scanline rasterization, clipping, cull modes, blend modes
    framebuffer.rs→ Thread-safe color+depth buffers (UnsafeCell + AtomicU32 + striped Mutex)
    geometry.rs   → Vertex struct
    math/         → Transform factories, barycentric interpolation
    color.rs      → ACES tone mapping, linear→sRGB
  pipeline/      → High-level render orchestration
    passes.rs     → shadow_pass + main_pass + post_process pipeline stages
    renderer.rs   → Renderer (owns Rasterizer + FrameBuffer), draw_mesh/draw_model
    shaders/      → PBR and Shadow shader implementations of Shader trait
  scene/         → Scene graph & asset management
    loader.rs     → init_scene_resources(), build_lights_from_config() — entry for scene setup
    context.rs    → RenderContext (camera + lights + objects)
    camera.rs     → Perspective/orthographic camera
    material.rs   → Material enum (Pbr), AlphaMode
    model.rs / mesh.rs → Geometry containers
    texture.rs    → Image loading, mipmapping, trilinear filtering
    light.rs      → Light enum (Directional, Point)
    scene_object.rs→ SceneObject (model + transform)
  io/            → File I/O
    config.rs     → TOML config deserialization (serde)
    gltf_loader.rs→ glTF 2.0 .glb/.gltf importer
    image.rs      → PNG output via the `image` crate
  ui/
    input.rs      → FPS camera controller (WASD + mouse)
```

## Concurrency Model

- **Rayon** used for all parallelism: vertex processing, triangle rasterization, post-processing, depth clear.
- `FrameBuffer` (`src/core/framebuffer.rs`):
  - `UnsafeCell<Vec<Vector3>>` for color, `Vec<AtomicU32>` for depth.
  - 1024 striped `Mutex` locks for color writes (locked per-pixel by index hash).
  - Manually implements `unsafe impl Sync` — be cautious modifying this.
  - Depth uses CAS loop (`test_and_update_depth`) for lock-free concurrent depth testing.
- Transparent triangles are sorted `par_sort_unstable_by` by view-space Z, then rasterized **sequentially** to preserve blending order.

## Config / Hot Reload

- Config is TOML (`scene.toml`). Struct: `io::config::Config`.
- GUI: press **R** to hot-reload config from disk (re-parses lights, transforms, render settings — does NOT reload models/textures).

## Key Conventions

- `.gitignore` blocks `TESTS.md`, `legacy/`, `python_files/`, `obj_not_upload`.
- No CI / GitHub workflows / pre-commit hooks are configured.
- No `AGENTS.md` or `CLAUDE.md` sibling files exist; this is the first.
