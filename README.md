# Software Rasterization Renderer in Rust

A 3D software rasterizer implementing modern graphics pipeline with PBR materials and TOML configuration.

[![Rust](https://img.shields.io/badge/rust-1.81%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project implements a complete 3D graphics pipeline in software, featuring:

- **PBR Material System** - Cook-Torrance BRDF with metallic-roughness workflow
- **Multi-threaded Rasterization** - Parallel triangle processing with intelligent load balancing  
- **Shadow Mapping** - Basic shadow casting for directional lights
- **MSAA Anti-aliasing** - Standard sampling patterns (2x/4x/8x)
- **Interactive GUI** - Real-time parameter adjustment with egui

## Quick Start

```bash
git clone https://github.com/Rukkhadevata123/Rasterizer_rust
cd Rasterizer_rust
cargo run --release

# Use example configuration
cargo run --release -- --use-example-config

# Run headless with config
cargo run --release -- --config scene.toml --headless

# Run complex example
cargo run --release -- --config test_complex_config.toml
```

## Configuration

All rendering parameters are controlled via TOML files:

```toml
[files]
obj = "models/bunny.obj"
output = "render"
texture = "textures/material.jpg"  # optional

[render]
width = 1920
height = 1080
msaa_samples = 4           # 1, 2, 4, or 8
use_zbuffer = true
backface_culling = true

[camera]
from = "2.5,1.5,4.0"      # camera position
at = "0,0.5,0"            # look-at target  
fov = 60.0                # field of view (degrees)

[material]
use_pbr = true            # PBR vs Blinn-Phong
base_color = "0.8,0.7,0.6"
metallic = 0.0            # 0.0 = dielectric, 1.0 = metallic
roughness = 0.5           # 0.0 = mirror, 1.0 = rough
alpha = 1.0               # transparency

[lighting]
ambient = 0.2
ambient_color = "0.2,0.3,0.4"

[[light]]
type = "directional"
direction = "0.3,-0.8,-0.5"
color = "1.0,0.95,0.8"
intensity = 0.8

[shadow]
enable_shadow_mapping = true
shadow_map_size = 512

[background]
enable_ground_plane = true
ground_plane_color = "0.3,0.5,0.2"
ground_plane_height = -1.0
```

## Technical Implementation

### PBR Rendering

Implements Cook-Torrance BRDF with energy conservation:

```rust
// BRDF = diffuse + specular
let f_diffuse = k_d * base_color / π;
let f_specular = (D * G * F) / (4 * (N·L) * (N·V));

// Energy conservation
let k_d = (1.0 - k_s) * (1.0 - metallic);
```

**Functions:**

- **D**: GGX/Trowbridge-Reitz normal distribution
- **G**: Smith geometry function with height correlation
- **F**: Schlick Fresnel approximation

### Rasterization Pipeline

1. **Geometry Processing** - MVP transformations with parallel vertex processing
2. **Triangle Setup** - Culling and material binding
3. **Rasterization** - Barycentric coordinate interpolation with intelligent parallelization
4. **Pixel Shading** - PBR/Phong lighting with texture sampling
5. **MSAA Resolve** - Multi-sample anti-aliasing with standard patterns

### Multi-threading Strategy

- **Large triangles**: Pixel-level parallelism
- **Small triangles**: Triangle-level parallelism
- **Mixed workloads**: Hybrid approach with Rayon work-stealing

### MSAA Implementation

Standard sampling patterns:

- **2x**: Diagonal `[(-0.25, -0.25), (0.25, 0.25)]`
- **4x**: Rotated grid for optimal coverage
- **8x**: Optimized 8-point distribution

## Project Structure(Partially)

```
src/
├── core/                     # Rendering pipeline
│   ├── renderer.rs          # Main renderer
│   ├── rasterizer/          # Rasterization subsystem
│   │   ├── msaa.rs          # Anti-aliasing
│   │   ├── pixel_processor.rs
│   │   └── shading.rs       # PBR/Phong lighting
│   └── simple_shadow_map.rs # Shadow mapping
├── material_system/          # Materials and lighting
│   ├── materials.rs         # PBR and Blinn-Phong
│   ├── light.rs             # Light sources
│   └── texture.rs           # Texture management
├── geometry/                 # Geometric processing
│   ├── transform.rs         # MVP pipeline
│   └── interpolation.rs     # Barycentric coordinates
└── io/                      # Configuration system
    └── render_settings.rs   # TOML settings
```

## Performance Guidelines

**High Performance:**

```toml
msaa_samples = 1
use_pbr = false
enable_shadow_mapping = false
backface_culling = true
```

**Balanced Quality:**

```toml
msaa_samples = 4
use_pbr = true
shadow_map_size = 256
```

**High Quality:**

```toml
msaa_samples = 8
use_pbr = true
shadow_map_size = 1024
```

## Command Line Options

```bash
cargo run --release -- [OPTIONS]

-c, --config <FILE>        # TOML configuration file
    --headless             # Run without GUI
    --use-example-config   # Generate and use example config
```

## Graphics Theory

This renderer demonstrates fundamental 3D graphics concepts:

- **Rasterization**: Converting 3D triangles to 2D pixels
- **Perspective Projection**: 3D to 2D coordinate transformation
- **Barycentric Interpolation**: Smooth attribute interpolation across triangles
- **Z-buffering**: Hidden surface removal via depth testing
- **Physically Based Rendering**: Realistic material appearance
- **Shadow Mapping**: Basic shadow casting technique
- **Multi-sampling**: Edge anti-aliasing through super-sampling

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*A software rasterizer showcasing 3D graphics pipeline implementation in Rust.*
