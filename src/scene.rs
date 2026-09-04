//! Public scene, camera, lighting, mesh, material, and texture authoring types.
//!
//! Scene data is independent of the private raster backend. Mesh vertices use the canonical
//! [`crate::render::Vertex`] layout, and renderable scenes can be submitted through the built-in
//! pass helpers exported by [`crate::render`].

pub mod camera;
pub mod context;
pub mod light;
pub mod loader;
pub mod material;
pub mod mesh;
pub mod model;
pub mod scene_object;
pub mod texture;
pub mod utils;
