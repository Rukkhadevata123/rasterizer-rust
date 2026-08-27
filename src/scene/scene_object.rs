use crate::scene::model::Model;
use nalgebra::Matrix4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneObjectKind {
    Ground,
    Model { config_index: usize },
}

/// Represents an instance of a model in the scene with its own transformation.
pub struct SceneObject {
    pub kind: SceneObjectKind,
    pub model: Model,
    pub transform: Matrix4<f32>,
}

impl SceneObject {
    pub fn new(kind: SceneObjectKind, model: Model, transform: Matrix4<f32>) -> Self {
        Self {
            kind,
            model,
            transform,
        }
    }
}
