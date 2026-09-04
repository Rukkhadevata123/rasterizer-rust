//! Multi-threaded CPU rasterizer with programmable shaders, PBR passes, and scene loading.
//!
//! The target 5.0 library rendering boundary is [`render`]. It contains shader contracts,
//! immutable pipeline state, typed command recording, synchronous submission, render targets, and
//! built-in shader bindings. [`scene`] contains authoring data and [`io`] contains configuration,
//! glTF, and image workflows. The low-level `core` and `pipeline` trees are implementation details.
//!
//! The [`app`], [`benchmark`], and [`ui`] modules support the package's bundled executable and are
//! not alternate rendering APIs.

pub mod app;
pub mod benchmark;
pub(crate) mod core;
pub mod error;
pub mod io;
pub(crate) mod pipeline;
pub mod render;
pub mod scene;
pub mod ui;
