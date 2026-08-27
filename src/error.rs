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
    #[error("invalid runtime configuration: {reason}")]
    InvalidConfiguration { reason: String },
    #[error("failed to initialize {target}: {reason}")]
    RenderInitialization {
        target: &'static str,
        reason: String,
    },
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to determine the process working directory: {source}")]
    CurrentDirectory {
        #[source]
        source: std::io::Error,
    },
    #[error("failed to read config '{}': {source}", path.display())]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse config '{}': {source}", path.display())]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
    #[error("invalid config '{}': {reason}", path.display())]
    Invalid { path: PathBuf, reason: String },
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
