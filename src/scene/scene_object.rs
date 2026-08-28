use crate::core::geometry::Vertex;
use crate::core::math::transform::TangentFrameTransform;
use crate::scene::material::{AlphaMode, Material};
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
    transform: Matrix4<f32>,
    tangent_frame_transform: TangentFrameTransform,
    front_face_inverted: bool,
    transparent_world_vertices: Vec<Option<Vec<Vertex>>>,
}

impl SceneObject {
    pub fn new(kind: SceneObjectKind, model: Model, transform: Matrix4<f32>) -> Self {
        let mut object = Self {
            kind,
            model,
            transform,
            tangent_frame_transform: TangentFrameTransform::new(
                transform.fixed_view::<3, 3>(0, 0).into_owned(),
            ),
            front_face_inverted: false,
            transparent_world_vertices: Vec::new(),
        };
        object.refresh_transform_cache();
        object
    }

    pub fn transform(&self) -> Matrix4<f32> {
        self.transform
    }

    pub fn set_transform(&mut self, transform: Matrix4<f32>) {
        self.transform = transform;
        self.refresh_transform_cache();
    }

    pub fn tangent_frame_transform(&self) -> TangentFrameTransform {
        self.tangent_frame_transform
    }

    pub fn front_face_inverted(&self) -> bool {
        self.front_face_inverted
    }

    pub fn transparent_world_vertices(&self, mesh_index: usize) -> Option<&[Vertex]> {
        self.transparent_world_vertices
            .get(mesh_index)
            .and_then(Option::as_deref)
    }

    fn refresh_transform_cache(&mut self) {
        let linear = self.transform.fixed_view::<3, 3>(0, 0).into_owned();
        self.tangent_frame_transform = TangentFrameTransform::new(linear);
        self.front_face_inverted = linear.determinant() < 0.0;
        self.transparent_world_vertices = self
            .model
            .meshes
            .iter()
            .map(|mesh| {
                let is_transparent = self
                    .model
                    .materials
                    .get(mesh.material_id)
                    .is_some_and(|material| {
                        matches!(material, Material::Pbr(material) if material.alpha_mode == AlphaMode::Blend)
                    });
                is_transparent.then(|| {
                    mesh.vertices
                        .iter()
                        .map(|vertex| {
                            let mut transformed = *vertex;
                            transformed.position =
                                self.transform.transform_point(&vertex.position);
                            (transformed.normal, transformed.tangent) = self
                                .tangent_frame_transform
                                .transform(vertex.normal, vertex.tangent);
                            transformed
                        })
                        .collect()
                })
            })
            .collect();
    }
}
