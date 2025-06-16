# Software Rasterization Renderer in Rust

A complete software-based 3D rasterization renderer implemented in Rust, featuring modern GUI interface and TOML-driven configuration system. Supports advanced rendering techniques including PBR materials, shadow mapping, MSAA anti-aliasing, and alpha transparency.

[![Rust Version](https://img.shields.io/badge/rust-1.81%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-2.6.0-blue.svg)](https://github.com/Rukkhadevata123/Rasterizer_rust)

## Features

### Rendering Pipeline

- **Multi-threaded rasterization** with parallel triangle processing
- **MSAA anti-aliasing** (1x/2x/4x/8x sampling levels)
- **PBR material system** with metallic-roughness workflow
- **Shadow mapping** for directional lights with ground plane shadows
- **Alpha transparency** with proper background blending
- **Multiple lighting models** - Phong and Physically Based Rendering

### Optimization

- **Background caching system** - pre-computed backgrounds and ground planes
- **Frustum culling** and small triangle elimination
- **Backface culling** for improved performance
- **Smart memory management** for large scenes

### Interface

- **Real-time GUI** with immediate parameter adjustment
- **Interactive camera controls** - pan, orbit, and dolly operations
- **TOML configuration system** for reproducible renders
- **Animation support** with video generation (requires FFmpeg)

## Installation

### Requirements

- Rust 1.81+
- FFmpeg (optional, for video generation)

```bash
git clone https://github.com/Rukkhadevata123/Rasterizer_rust
cd Rasterizer_rust
cargo build --release

# Launch GUI
cargo run --release

# Generate example configuration
cargo run --release -- --use-example-config
```

## Configuration

The renderer uses TOML configuration files for all rendering parameters:

```toml
[files]
obj = "models/bunny.obj"
output = "render_output"
texture = "textures/material.jpg"  # optional

[render]
width = 1920
height = 1080
projection = "perspective"  # or "orthographic"
msaa_samples = 4           # 1, 2, 4, or 8
use_zbuffer = true
backface_culling = true

[camera]
from = "2.5,1.5,4.0"      # camera position
at = "0,0.5,0"            # look-at target
up = "0,1,0"              # up vector
fov = 60.0                # field of view in degrees

[lighting]
use_lighting = true
ambient = 0.2
ambient_color = "0.2,0.3,0.4"

# Multiple light sources
[[light]]
type = "directional"
direction = "0.3,-0.8,-0.5"
color = "1.0,0.95,0.8"
intensity = 0.8

[[light]]
type = "point"
position = "2.0,3.0,2.0"
color = "1.0,0.8,0.6"
intensity = 2.5

[material]
use_pbr = true
base_color = "0.8,0.7,0.6"
metallic = 0.0
roughness = 0.5
alpha = 1.0               # transparency (0.0-1.0)

[shadow]
enable_shadow_mapping = true
shadow_map_size = 512
enhanced_ao = true        # ambient occlusion
soft_shadows = true
```

## Rendering Pipeline

The renderer implements a standard 3D graphics pipeline:

1. **Model Loading** - OBJ files with material (MTL) support
2. **Geometry Processing** - vertex transformations and projection
3. **Shadow Map Generation** - depth-only rendering from light's perspective
4. **Culling** - frustum, backface, and small triangle culling
5. **Rasterization** - multi-threaded triangle rasterization with MSAA
6. **Pixel Shading** - PBR or Phong lighting calculations
7. **Alpha Blending** - transparency rendering with background
8. **Post-processing** - gamma correction and final composition

### Key Components

**Vertex Processing:**

- Model-view-projection transformations
- Normal transformation for lighting
- Texture coordinate interpolation

**Rasterization:**

- Scanline-based triangle filling
- Barycentric coordinate interpolation
- Multi-sample anti-aliasing (MSAA)

**Lighting:**

- Directional and point light sources
- Blinn-Phong and PBR material models
- Shadow mapping with PCF filtering
- Ambient occlusion approximation

**Materials:**

- PBR metallic-roughness workflow
- Subsurface scattering simulation
- Emissive materials support
- Texture mapping with bilinear filtering

## Command Line Usage

```bash
# Basic usage
cargo run --release -- [OPTIONS]

# Main options
-c, --config <FILE>        # Specify TOML configuration file
    --headless             # Run without GUI
    --use-example-config   # Use built-in example configuration

# Examples
cargo run --release -- --config scene.toml
cargo run --release -- --config example.toml --headless
```

## Performance Guidelines

### High Performance

- Set `msaa_samples = 1`
- Disable `enhanced_ao` and `soft_shadows`
- Use `alpha = 1.0` (no transparency)
- Set `shadow_map_size = 128` or disable shadow mapping

### Balanced Quality

- Set `msaa_samples = 4`
- Enable basic lighting and shadows
- Use moderate shadow map sizes (256-512)

### High Quality

- Set `msaa_samples = 8`
- Enable all lighting features
- Use large shadow maps (1024+)
- Enable subsurface scattering and anisotropy

## Project Structure

```
src/
├── core/                     # Core rendering engine
│   ├── renderer.rs          # Main renderer
│   ├── frame_buffer.rs      # Framebuffer with background caching
│   ├── parallel_rasterizer.rs  # Multi-threaded rasterization
│   └── simple_shadow_map.rs # Shadow mapping implementation
├── geometry/                 # Geometric processing
│   ├── camera.rs            # Camera system
│   ├── transform.rs         # 3D transformations
│   └── culling.rs           # Visibility culling
├── material_system/          # Materials and lighting
│   ├── materials.rs         # PBR and Phong materials
│   ├── light.rs             # Light source types
│   └── texture.rs           # Texture management
├── io/                       # Input/Output systems
│   ├── config_loader.rs     # TOML configuration
│   ├── obj_loader.rs        # OBJ file parsing
│   └── render_settings.rs   # Unified settings
├── ui/                       # GUI interface
└── utils/                    # Utility functions
```

## Technical Details

**Multi-threading:**
The renderer uses Rayon for parallel processing of triangles during rasterization. Each thread processes a subset of triangles independently.

**MSAA Implementation:**
Multi-sample anti-aliasing is implemented using sub-pixel sampling with a rotated grid pattern. Sample coverage is determined using edge equations.

**Shadow Mapping:**
Implements standard shadow mapping with percentage-closer filtering (PCF) for soft shadow edges. Currently optimized for ground plane shadows.

**Alpha Transparency:**
Proper alpha blending with pre-multiplied alpha and background composition. Supports transparent materials with correct depth sorting.

## Contributing

Contributions are welcome. Please ensure code follows Rust conventions and includes appropriate tests for new features.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

**A modern software rasterizer showcasing 3D graphics**
