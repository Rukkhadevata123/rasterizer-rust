//! Configuration, glTF import, and PNG output workflows.
//!
//! Configuration paths are resolved relative to their source file. Asset-loading failures retain
//! contextual public errors instead of falling back to placeholder geometry or textures.

pub mod config;
pub mod gltf_loader;
pub mod image;
