# Software Rasterization Renderer in Rust

A high-performance, multi-threaded 3D software rasterizer **built from scratch** in Rust. This project implements a modern, PBR-correct rendering pipeline, complete with an interactive GUI, a sophisticated caching system, and animation capabilities.

[![Rust](https://img.shields.io/badge/rust-1.78%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project is a comprehensive implementation of a 3D graphics pipeline in pure Rust. It's designed for both performance and educational value, demonstrating how modern rendering features can be built without relying on a GPU.

### Key Features

* **Dual Shading Models:** Supports both **Physically Based Rendering (PBR)** using the Cook-Torrance BRDF and the classic **Blinn-Phong** model.
* **Parallel Rasterization:** Employs efficient, triangle-level parallelism using Rayon for high performance on multi-core CPUs.
* **Dynamic Ground & Background:** Features a procedural, infinite ground plane and a skydome/background pass, which are rendered efficiently and separately from the main scene geometry.
* **Shadow Mapping:** Implements shadow casting for directional lights onto the procedural ground plane.
* **Advanced Caching System:** A smart, fine-grained caching mechanism minimizes re-computation during animations, distinguishing between camera movement and object movement to maximize performance.
* **Interactive GUI:** Built with `egui`, allowing for real-time adjustment of all rendering parameters, materials, lighting, and camera controls.
* **Animation & Video Export:** Supports camera and object animations, pre-rendering of frames for smooth playback, and video export via `ffmpeg`.
* **TOML Configuration:** All scene and render settings can be loaded from and saved to human-readable TOML files.

![image](./demo/demo.png)

Check our Video Demo:

https://github.com/user-attachments/assets/1137feff-b93c-450a-a39f-0bac5608b523

## Rendering Pipeline

The renderer is architected around a multi-pass process to handle different components of the scene efficiently.

```mermaid
graph TD
    subgraph "Setup"
        A[Load 3D Model & Textures] --> B[Parse TOML Config]
        B --> C[Initialize Scene]
    end

    subgraph "Frame Rendering Loop"
        D(Start Frame) --> E{Cache Invalidation};
        E --> F[Pass 1: Shadow Map];
        F --> G[Pass 2: Background & Ground];
        G --> H[Pass 3: Main Scene Render];
        
        subgraph "Pass 3 Details"
            H1[Geometry Processing] --> H2[Primitive Assembly];
            H2 --> H3[Rasterization & Shading];
        end
        
        H --> H1;
        H3 --> I[Final Image in FrameBuffer];
    end

    C --> D;
    
    style A fill:#e3f2fd
    style C fill:#e8eaf6
    style F fill:#fffde7,stroke:#fbc02d,stroke-width:2px
    style G fill:#e0f2f1,stroke:#00796b,stroke-width:2px
    style H fill:#fce4ec,stroke:#d81b60,stroke-width:2px
    style I fill:#e8f5e9
```

## Project Structure

The project is organized into logical modules, each responsible for a specific part of the rendering pipeline.

```mermaid
graph TD
    subgraph "Application"
        Main("main.rs")
        UI("ui/")
    end

    subgraph "Rendering Core"
        Core("core/")
        Renderer("core/renderer.rs")
        Rasterizer("core/rasterizer.rs")
        FrameBuffer("core/frame_buffer.rs")
        ShadowMap("core/shadow_map.rs")
    end
    
    subgraph "Geometry & Math"
        Geometry("geometry/")
        Transform("geometry/transform.rs")
        Camera("geometry/camera.rs")
        Interpolation("geometry/interpolation.rs")
        Culling("geometry/culling.rs")
    end

    subgraph "Material & Scene"
        MaterialSystem("material_system/")
        Materials("material_system/materials.rs")
        Light("material_system/light.rs")
        Texture("material_system/texture.rs")
        SceneLib("scene/")
    end

    subgraph "I/O & Utilities"
        IO("io/")
        Config("io/config_loader.rs")
        Model("io/model_loader.rs")
        Utils("utils/")
    end

    Main --> UI
    Main --> Core
    Main --> IO

    UI --> Core
    Renderer --> Rasterizer
    Renderer --> FrameBuffer
    Renderer --> ShadowMap
    Core --> Geometry
    Core --> MaterialSystem
    Core --> SceneLib
    
    SceneLib --> MaterialSystem
    IO --> Utils
    
    classDef module fill:#263238,color:#eceff1,stroke:#546e7a,stroke-width:2px;
    class Main,UI,Core,Geometry,MaterialSystem,SceneLib,IO,Utils module;
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Rukkhadevata123/rasterizer-rust
cd rasterizer-rust

# Run the application with release optimizations
cargo run --release

# To use a specific configuration file
cargo run --release -- -c path/to/your_config.toml

# To use a specific configuration file without GUI
cargo run --release -- -c path/to/your_config.toml --headless
```

## Configuration

All rendering parameters are controlled via a single TOML file. This allows for easy scene setup and sharing. You can check our `complex_config.toml` .

```toml
# Example configuration file (scene.toml)

# --- File paths ---
[files]
obj = "path/to/model.obj"
output = "render_output"
output_dir = "renders"
texture = "path/to/override_texture.png"  # Optional: Overrides textures from MTL
background_image_path = "path/to/background.jpg" # Optional

# --- Core Render Settings ---
[render]
width = 1280
height = 720
projection = "perspective"  # "perspective" or "orthographic"
use_zbuffer = true
use_gamma = true
backface_culling = true
wireframe = false

# --- Camera Setup ---
[camera]
from = "3.0, 2.0, 5.0"   # Position
at = "0.0, 0.5, 0.0"     # Look-at target
up = "0.0, 1.0, 0.0"     # Up vector
fov = 60.0               # Field of view (degrees)

# --- Object Transformation ---
[object]
position = "0, 0, 0"
rotation = "0, 45, 0"    # Rotation in degrees (X, Y, Z)
scale = 1.0              # Uniform scale

# --- Lighting Environment ---
[lighting]
use_lighting = true
ambient = 0.1
ambient_color = "0.1, 0.1, 0.1"

# Define one or more light sources
[[light]]
type = "directional"
enabled = true
direction = "0.5, -1.0, -0.6"
color = "1.0, 0.98, 0.95"
intensity = 1.5

[[light]]
type = "point"
enabled = true
position = "0.0, 3.0, 2.0"
color = "1.0, 0.5, 0.2"
intensity = 5.0

# --- Material Properties (Global Override) ---
[material]
use_pbr = true
use_phong = false
base_color = "0.82, 0.67, 0.16" # Used for PBR Albedo or Phong Diffuse
metallic = 0.1
roughness = 0.4
alpha = 1.0
emissive = "0,0,0"

# --- Shadow Mapping ---
[shadow]
enable_shadow_mapping = true
shadow_map_size = 512       # Higher values = better quality
shadow_bias = 0.005

# --- Background & Ground ---
[background]
enable_gradient_background = true
gradient_top_color = "0.1, 0.2, 0.4"
gradient_bottom_color = "0.7, 0.8, 1.0"
enable_ground_plane = true
ground_plane_color = "0.3, 0.3, 0.3"
ground_plane_height = 0.0

# --- Animation Settings ---
[animation]
animate = false
fps = 30
rotation_speed = 1.0 # Speed multiplier for realtime rendering
rotation_cycles = 1.0 # Number of 360-degree rotations for video export
animation_type = "CameraOrbit" # "CameraOrbit" or "ObjectLocalRotation"
rotation_axis = "Y" # "X", "Y", "Z", or "Custom"
custom_rotation_axis = "0,1,0"
```

## Core Concepts Explained

### Material & Shading System

The material system is designed for flexibility and type safety. Instead of a single, monolithic `Material` struct, the project uses a `Material` enum:

```rust
pub enum MaterialType {
    Phong,
    PBR,
}
```

This ensures that material-specific properties (like `metallic` for PBR or `shininess` for Phong) are only present where they belong, preventing invalid states at compile time. The system seamlessly calculates the correct lighting response based on the active material type for each mesh.

### Caching and Performance

To achieve high performance during interactive use and animations, the renderer employs a sophisticated, multi-level caching system for the procedural ground and background.

* **Background Cache:** The sky/background is computed once and cached. It only becomes invalid if background-related settings are changed.
* **Ground Base Cache:** The visual properties of the ground (grid lines, colors, fade-out), which depend on the camera's position and orientation, are cached separately.
* **Ground Shadow Cache:** The shadows cast on the ground are in their own cache.

This fine-grained approach enables smart optimizations:

* **Camera Movement:** Invalidates the ground base and shadow caches, but re-uses the background cache.
* **Object-Only Animation:** Invalidates only the shadow cache, re-using both the background and ground base caches for maximum efficiency.

This is managed by a clean, event-driven invalidation API (`frame_buffer.invalidate_ground_base_cache()`, `frame_buffer.invalidate_ground_shadow_cache()`) that decouples the application logic from the internal caching implementation.

### GUI and Interaction

The application provides a comprehensive GUI for real-time control.

* **Side Panel:** All settings are organized into logical, collapsible sections for easy navigation.
* **Render View:** The central view displays the rendered image.
* **Camera Controls:**
  * **Drag:** Pan the camera.
  * **Shift + Drag:** Orbit the camera around its target.
  * **Scroll Wheel:** Dolly (zoom) the camera forwards and backwards.

## Command-Line Usage

The application can be run in headless mode for batch rendering or integration into scripts.

```bash
# Basic headless render using a config file
cargo run --release -- -c scene.toml --headless

# Use the example config for a quick test
cargo run --release -- --use-example-config
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
