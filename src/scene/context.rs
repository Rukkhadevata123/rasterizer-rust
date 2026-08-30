use crate::scene::camera::Camera;
use crate::scene::light::Light;
use crate::scene::scene_object::SceneObject;
use nalgebra::Point3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShadowLight {
    pub light_index: usize,
    pub position: Point3<f32>,
}

pub struct RenderScene {
    pub camera: Camera,
    pub lights: Vec<Light>,
    pub scene_objects: Vec<SceneObject>,
    pub shadow_light: Option<ShadowLight>,
}
