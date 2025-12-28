use crate::scene::camera::Camera;
use crate::scene::light::Light;
use crate::scene::scene_object::SceneObject;
use nalgebra::Point3;

/// Holds all scene resources required for rendering.
pub struct RenderContext {
    pub camera: Camera,
    pub lights: Vec<Light>,
    pub scene_objects: Vec<SceneObject>,
    pub shadow_light_pos: Point3<f32>,
}
