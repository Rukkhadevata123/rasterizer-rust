# Rust PBR Rasterizer

A high-performance, multi-threaded software rasterizer built from scratch in Rust. This project implements a modern programmable rendering pipeline featuring Physically Based Rendering (PBR), real-time shadow mapping, and transparent object handling. It supports both high-quality offline image generation and interactive real-time visualization via **minifb**.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Rukkhadevata123/rasterizer-rust)

**Real-time Interactive Mode:**

![Real-time PBR](outputs/minifb-2.png)
*Physically Based Rendering with metallic properties, normal mapping, and contact shadows.*

![Wireframe Mode](outputs/minifb-3.png)
*Debug visualization including wireframe overlays and cull mode toggling.*

## Key Features

### Physically Based Rendering (PBR)

- **Workflow:** Standard Metallic-Roughness workflow.
- **BRDF:** Cook-Torrance specular BRDF with Trowbridge-Reitz GGX distribution and Smith Geometry function.
- **Fresnel:** Fresnel-Schlick approximation for realistic light reflection at varying angles.
- **Normal Mapping:** Tangent-space normal mapping for glTF assets that provide tangent attributes; MikkTSpace-compatible generation is planned.
- **Tone Mapping:** ACES Filmic Tone Mapping for cinematic color reproduction.

### Advanced Rendering Capabilities

- **Transparency:** Correct rendering of semi-transparent objects (glass, ice) using back-to-front sorting and alpha blending logic.
- **Shadow Mapping:** Real-time soft shadows using Percentage-Closer Filtering (PCF), adaptive bias, and alpha-tested cutout casters.
- **Anti-Aliasing:** SSAA (Super-Sample Anti-Aliasing) support for smooth edges.
- **Texture Filtering:** glTF UV0/UV1 bindings with repeat, clamp, mirrored-repeat, nearest, linear, and mip-filter sampler semantics.

### High Performance Architecture

- **Massive Parallelism:** Leveraging `Rayon` for triangle preparation, exclusive horizontal-band rasterization, buffer clearing, and post-processing.
- **Thread Safety:** Lock-free safe Rust framebuffer writes through exclusive horizontal-band ownership, keeping depth and color updates consistent.
- **Clipping:** Robust Homogeneous Clip Space clipping (Sutherland–Hodgman) to handle primitives correctly near the camera plane.
- **Optimization:** Data-oriented design with pre-transform optimizations for complex scene sorting.

### Interactive Real-time GUI

- **Windowing:** Lightweight, zero-bloat window management via `minifb`.
- **Camera:** FPS-style free-roam camera with WASD movement and mouse look.
- **Hot Reloading:** Press `R` to reload lights, transforms, render settings, model/texture assets, supersampling buffers, and shadow-map buffers. Window dimensions require a restart.
- **Runtime Tools:** Toggle wireframe modes (`Middle Click`) and cull modes (`Right Click`) on the fly.

## Project Structure

The project follows a clean, modular architecture separating core engine logic from scene management and pipeline definitions.

```text
src
├── core               # The Engine Kernel
│   ├── rasterizer.rs  # Scanline rasterization & clipping logic
│   ├── framebuffer.rs # Safe color/depth sample storage and resolve
│   ├── geometry.rs    # Vertex layout & geometric primitives
│   └── math           # Transform factories & interpolation helpers
├── pipeline           # The Rendering Pipeline
│   ├── passes.rs      # High-level Render Passes (Shadow & Main)
│   ├── renderer.rs    # Render orchestrator & clear logic
│   └── shaders        # Programmable PBR & Shadow shaders
├── scene              # Scene Graph & Assets
│   ├── material.rs    # PBR Material & Alpha Mode definitions
│   ├── texture.rs     # Image resources, sampler state, bindings & mipmapping
│   ├── light.rs       # Lighting definitions
│   └── loader.rs      # Resource management & Hot-reloading
├── io                 # File I/O
│   ├── gltf_loader.rs # Robust glTF 2.0 asset importer
│   └── config.rs      # TOML-based scene configuration
└── app.rs             # Application control loops
```

## Getting Started

### Prerequisites

- Rust stable (`Cargo.toml` declares Rust 1.85 as the intended edition-2024 minimum; CI currently validates the runner's active stable toolchain)
- Cargo

### Usage

**1. Real-time GUI Mode (Recommended)**  
Launch the interactive viewer to explore the scene, test lighting, and view PBR materials in real-time.

```bash
cargo run --release -- --config scene.toml --gui
```

**2. Offline Rendering (CLI)**  
Render a single high-quality frame to an output image file (default: `output.png`).

```bash
cargo run --release -- --config scene.toml
```

## Controls (GUI Mode)

| Input                  | Action                                  |
|:-----------------------|:----------------------------------------|
| **W / A / S / D**      | Move Camera                             |
| **Mouse**              | Look Around                             |
| **Space / L-Shift**    | Move Up / Down                          |
| **Scroll Wheel**       | Adjust FOV (Zoom)                       |
| **R**                  | Reload Configuration (Hot Reload)       |
| **Right Click**         | Cycle Cull Mode (Back -> None -> Front) |
| **Middle Click**       | Toggle Wireframe Mode                   |
| **Esc**                | Exit Application                        |
