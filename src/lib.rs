//! Multi-threaded CPU rasterizer with programmable shaders, PBR passes, and scene loading.
//!
//! The library rendering boundary is [`render`]. It contains shader contracts,
//! immutable pipeline state, typed command recording, synchronous submission, render targets, and
//! built-in shader bindings. [`scene`] contains authoring data and [`io`] contains configuration,
//! glTF, and image workflows. The low-level `core` and `pipeline` trees are implementation details.

pub(crate) mod core;
pub mod io;
pub(crate) mod pipeline;
pub mod render;
pub mod scene;
