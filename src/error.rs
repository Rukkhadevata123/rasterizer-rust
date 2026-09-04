use crate::core::framebuffer::RenderTargetError;
use crate::io::config::{ConfigError, ConfigValidationError};
use crate::pipeline::renderer::PresentBufferError;
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

#[derive(Debug, Error, PartialEq)]
pub enum ResolveTonemapError {
    #[error(
        "resolve-tonemap pass '{label}' source dimensions {source_width}x{source_height} do not match destination dimensions {destination_width}x{destination_height}"
    )]
    DimensionMismatch {
        label: String,
        source_width: usize,
        source_height: usize,
        destination_width: usize,
        destination_height: usize,
    },
    #[error(
        "resolve-tonemap pass '{label}' exposure must be finite and non-negative, got {exposure}"
    )]
    InvalidExposure { label: String, exposure: f32 },
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
pub enum AssetError {
    #[error("failed to load object {object_index} from '{}': {source}", path.display())]
    Model {
        object_index: usize,
        path: PathBuf,
        #[source]
        source: GltfError,
    },
    #[error("failed to load background image '{}': {source}", path.display())]
    BackgroundImage {
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },
}

#[derive(Debug, Error)]
pub enum GltfError {
    #[error("failed to import glTF '{}': {source}", path.display())]
    Import {
        path: PathBuf,
        #[source]
        source: Box<gltf::Error>,
    },
    #[error("glTF '{}' contains no scenes", path.display())]
    NoScene { path: PathBuf },
    #[error("glTF '{}' contains no meshes", path.display())]
    NoMeshes { path: PathBuf },
    #[error("unsupported feature in glTF '{}': {reason}", path.display())]
    Unsupported { path: PathBuf, reason: String },
    #[error(
        "failed to process glTF '{}' scene {}, node {} ({}), mesh {}, primitive {}: {}",
        context.path.display(),
        context.scene_index,
        context.node_index,
        context.node_name,
        context.mesh_index,
        context.primitive_index,
        context.reason
    )]
    Primitive { context: Box<PrimitiveContext> },
    #[error("failed to process image {image_index} in glTF '{}': {reason}", path.display())]
    Image {
        path: PathBuf,
        image_index: usize,
        reason: String,
    },
    #[error(
        "failed to resolve texture {texture_index} for material {material_index:?} in glTF '{}': source image {source_image_index} is unavailable",
        path.display()
    )]
    Texture {
        path: PathBuf,
        material_index: Option<usize>,
        texture_index: usize,
        source_image_index: usize,
    },
}

#[derive(Debug)]
pub struct PrimitiveContext {
    pub path: PathBuf,
    pub scene_index: usize,
    pub node_index: usize,
    pub node_name: String,
    pub mesh_index: usize,
    pub primitive_index: usize,
    pub reason: String,
}

#[derive(Debug, Error)]
pub enum ImageOutputError {
    #[error("image dimensions {width}x{height} exceed the PNG format limits")]
    InvalidDimensions { width: usize, height: usize },
    #[error("image buffer length is {actual}, expected {expected} pixels")]
    BufferLength { expected: usize, actual: usize },
    #[error("failed to create output directory '{}': {source}", path.display())]
    CreateParent {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to save image '{}': {source}", path.display())]
    Save {
        path: PathBuf,
        #[source]
        source: image::ImageError,
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
