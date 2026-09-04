use crate::core::framebuffer::RenderTargetError;
use crate::io::ImageOutputError;
use crate::io::config::{ConfigError, ConfigValidationError};
use crate::pipeline::renderer::PresentBufferError;
use crate::render::ResolveTonemapError;
use crate::scene::AssetError;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ApplicationError {
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error(transparent)]
    Asset(#[from] AssetError),
    #[error(transparent)]
    ImageOutput(#[from] ImageOutputError),
    #[error(transparent)]
    Window(#[from] WindowError),
    #[error(transparent)]
    Benchmark(#[from] BenchmarkError),
    #[error(transparent)]
    ResolveTonemap(#[from] ResolveTonemapError),
    #[error("invalid runtime configuration: {source}")]
    InvalidConfiguration {
        #[source]
        source: ConfigValidationError,
    },
    #[error("failed to initialize {target}: {source}")]
    RenderTargetInitialization {
        target: &'static str,
        #[source]
        source: RenderTargetError,
    },
    #[error("failed to initialize present buffer: {source}")]
    PresentBufferInitialization {
        #[source]
        source: PresentBufferError,
    },
}

#[derive(Debug, Error)]
pub enum BenchmarkError {
    #[error("invalid benchmark options: {reason}")]
    InvalidOptions { reason: String },
    #[error("failed to write benchmark report '{}': {source}", path.display())]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "benchmark output changed at measured frame {frame_index}: expected {expected:016x}, got {actual:016x}"
    )]
    OutputChanged {
        frame_index: usize,
        expected: u64,
        actual: u64,
    },
}

#[derive(Debug, Error)]
pub enum WindowError {
    #[error("failed to create window: {source}")]
    Create {
        #[source]
        source: minifb::Error,
    },
    #[error("failed to present window buffer: {source}")]
    Present {
        #[source]
        source: minifb::Error,
    },
}
