use crate::scene::model::Model;
use nalgebra::Matrix4;

/// Represents an instance of a model in the scene with its own transformation.
pub struct SceneObject {
    pub model: Model,
    pub transform: Matrix4<f32>,
}

impl SceneObject {
    pub fn new(model: Model, transform: Matrix4<f32>) -> Self {
        Self { model, transform }
    }
}
