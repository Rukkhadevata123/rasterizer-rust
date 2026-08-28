use crate::core::math::transform::TransformFactory;
use crate::error::AssetError;
use crate::io::config::{Config, LightKind, ModelNormalization, ObjectConfig, ProjectionMode};
use crate::io::gltf_loader::load_gltf;
use crate::scene::camera::Camera;
use crate::scene::context::{RenderContext, ShadowLight};
use crate::scene::light::Light;
use crate::scene::material::{Material, PbrMaterial};
use crate::scene::mesh::Mesh;
use crate::scene::model::Model;
use crate::scene::scene_object::{SceneObject, SceneObjectKind};
use crate::scene::utils::{center_model, normalize_and_center_model};
use log::info;
use nalgebra::{Point3, Vector3};

/// Helper to rebuild light list from config (used in Init and Hot Reload)
pub fn build_lights_from_config(config: &Config) -> (Vec<Light>, Option<ShadowLight>) {
    let mut lights = Vec::new();
    let mut shadow_light = None;

    for light_config in &config.lights {
        let color = Vector3::from(light_config.color);
        match light_config.kind {
            LightKind::Directional => {
                if let Some(direction) = light_config.direction {
                    let direction = Vector3::from(direction).normalize();
                    let light_index = lights.len();
                    lights.push(Light::new_directional(
                        direction,
                        color,
                        light_config.intensity,
                    ));
                    shadow_light.get_or_insert(ShadowLight {
                        light_index,
                        position: Point3::origin() - direction * 10.0,
                    });
                }
            }
            LightKind::Point => {
                if let Some(position) = light_config.position {
                    let mut light =
                        Light::new_point(Point3::from(position), color, light_config.intensity);
                    if let Light::Point {
                        ref mut attenuation,
                        ..
                    } = light
                        && let Some(value) = light_config.attenuation
                    {
                        *attenuation = (value[0], value[1], value[2]);
                    }
                    lights.push(light);
                }
            }
        }
    }

    (lights, shadow_light)
}

/// Builds a right-handed object transform from config degrees.
///
/// With column vectors, `T * Rx * Ry * Rz * S` applies non-uniform scale first,
/// then Z, Y, and X Euler rotations, and translation last.
fn object_transform(config: &ObjectConfig) -> nalgebra::Matrix4<f32> {
    TransformFactory::translation(&Vector3::from(config.position))
        * TransformFactory::rotation_x(config.rotation[0].to_radians())
        * TransformFactory::rotation_y(config.rotation[1].to_radians())
        * TransformFactory::rotation_z(config.rotation[2].to_radians())
        * TransformFactory::scaling_nonuniform(&Vector3::from(config.scale))
}

/// Applies fields that do not require reloading models or rebuilding geometry.
pub fn update_scene_objects(scene_objects: &mut [SceneObject], config: &Config) {
    for scene_object in scene_objects {
        match scene_object.kind {
            SceneObjectKind::Ground => {
                if let Some(Material::Pbr(material)) = scene_object.model.materials.get_mut(0) {
                    material.albedo = config
                        .ground
                        .albedo
                        .map(Vector3::from)
                        .unwrap_or(Vector3::new(0.6, 0.6, 0.6));
                    material.metallic = config.ground.metallic.unwrap_or(0.0);
                    material.roughness = config.ground.roughness.unwrap_or(0.8);
                    material.sanitize_factors();
                }
            }
            SceneObjectKind::Model { config_index } => {
                let Some(object_config) = config.objects.get(config_index) else {
                    continue;
                };
                scene_object.transform = object_transform(object_config);
            }
        }
    }
}

/// Initial resource loading (Heavy I/O). Returns a RenderContext.
pub fn init_scene_resources(config: &Config) -> Result<RenderContext, AssetError> {
    // 1. Camera
    let cam_pos = Point3::from(config.camera.position);
    let cam_target = Point3::from(config.camera.target);
    let cam_up = Vector3::from(config.camera.up);
    let aspect_ratio = config.render.width as f32 / config.render.height as f32;

    let camera = if config.camera.projection == ProjectionMode::Orthographic {
        Camera::new_orthographic(
            cam_pos,
            cam_target,
            cam_up,
            config.camera.ortho_height,
            aspect_ratio,
            config.camera.near,
            config.camera.far,
        )
    } else {
        Camera::new_perspective(
            cam_pos,
            cam_target,
            cam_up,
            config.camera.fov.to_radians(),
            aspect_ratio,
            config.camera.near,
            config.camera.far,
        )
    };

    // 2. Lights
    let (lights, shadow_light) = build_lights_from_config(config);

    // 3. Objects
    let mut scene_objects: Vec<SceneObject> = Vec::new();

    // 3.1 Ground
    if config.ground.enabled {
        let ground_mesh = Mesh::create_plane(config.ground.size, 0);
        let mut ground_material = PbrMaterial {
            albedo: config
                .ground
                .albedo
                .map(Vector3::from)
                .unwrap_or(Vector3::new(0.6, 0.6, 0.6)),
            metallic: config.ground.metallic.unwrap_or(0.0),
            roughness: config.ground.roughness.unwrap_or(0.8),
            occlusion_strength: 1.0,
            emissive: Vector3::zeros(),
            ..Default::default()
        };
        ground_material.sanitize_factors();
        let ground_mat = Material::Pbr(ground_material);
        scene_objects.push(SceneObject::new(
            SceneObjectKind::Ground,
            Model::new(vec![ground_mesh], vec![ground_mat]),
            TransformFactory::translation(&Vector3::new(0.0, -1.0, 0.0)),
        ));
    }

    // 3.2 Loaded Objects
    for (object_index, obj_conf) in config.objects.iter().enumerate() {
        // Direct GLTF Loading
        let model_path = config.resolve_path(&obj_conf.path);
        let mut model = load_gltf(&model_path, config.render.use_mipmap).map_err(|source| {
            AssetError::Model {
                object_index,
                path: model_path,
                source,
            }
        })?;
        apply_model_normalization(&mut model, obj_conf.normalization);

        // Ensure material fallback
        if model.materials.is_empty() {
            model.materials.push(Material::default());
        }

        scene_objects.push(SceneObject::new(
            SceneObjectKind::Model {
                config_index: object_index,
            },
            model,
            object_transform(obj_conf),
        ));
    }

    info!("Scene initialized with {} objects.", scene_objects.len());

    Ok(RenderContext {
        camera,
        lights,
        scene_objects,
        shadow_light,
    })
}

fn apply_model_normalization(model: &mut Model, normalization: ModelNormalization) {
    match normalization {
        ModelNormalization::Preserve => {}
        ModelNormalization::Center => {
            center_model(model);
        }
        ModelNormalization::Normalize => {
            normalize_and_center_model(model);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::config::LightConfig;

    fn point_light() -> LightConfig {
        LightConfig {
            kind: LightKind::Point,
            position: Some([1.0, 2.0, 3.0]),
            direction: None,
            color: [1.0, 1.0, 1.0],
            intensity: 2.0,
            attenuation: None,
        }
    }

    fn directional_light(direction: [f32; 3]) -> LightConfig {
        LightConfig {
            kind: LightKind::Directional,
            position: None,
            direction: Some(direction),
            color: [1.0, 1.0, 1.0],
            intensity: 3.0,
            attenuation: None,
        }
    }

    #[test]
    fn preserve_normalization_keeps_imported_vertex_positions() {
        let mut model = Model::new(vec![Mesh::create_test_triangle(0)], vec![]);
        let positions: Vec<_> = model.meshes[0]
            .vertices
            .iter()
            .map(|vertex| vertex.position)
            .collect();

        apply_model_normalization(&mut model, ModelNormalization::Preserve);

        assert_eq!(
            model.meshes[0]
                .vertices
                .iter()
                .map(|vertex| vertex.position)
                .collect::<Vec<_>>(),
            positions
        );
    }

    #[test]
    fn shadow_light_tracks_first_valid_directional_light_index() {
        let config = Config {
            lights: vec![point_light(), directional_light([0.0, -1.0, 0.0])],
            ..Default::default()
        };

        let (lights, shadow_light) = build_lights_from_config(&config);
        let shadow_light = shadow_light.expect("directional light should cast shadows");

        assert_eq!(lights.len(), 2);
        assert_eq!(shadow_light.light_index, 1);
        assert!((shadow_light.position - Point3::new(0.0, 10.0, 0.0)).norm() < 1e-5);
    }

    #[test]
    fn point_only_scene_has_no_shadow_light() {
        let config = Config {
            lights: vec![point_light()],
            ..Default::default()
        };

        let (lights, shadow_light) = build_lights_from_config(&config);

        assert_eq!(lights.len(), 1);
        assert!(shadow_light.is_none());
    }

    #[test]
    fn object_transform_applies_scale_zyx_rotation_then_translation() {
        let config = ObjectConfig {
            path: "unused.gltf".to_string(),
            position: [1.0, 2.0, 3.0],
            rotation: [90.0, 90.0, 90.0],
            scale: [2.0, 3.0, 4.0],
            normalization: ModelNormalization::Preserve,
        };

        let transformed = object_transform(&config).transform_point(&Point3::new(1.0, 0.0, 0.0));

        assert!((transformed - Point3::new(1.0, 2.0, 5.0)).norm() < 1e-5);
    }

    #[test]
    fn scene_updates_use_stable_object_kinds_instead_of_positions() {
        let ground = SceneObject::new(
            SceneObjectKind::Ground,
            Model::new(vec![Mesh::create_plane(1.0, 0)], vec![Material::default()]),
            nalgebra::Matrix4::identity(),
        );
        let model = SceneObject::new(
            SceneObjectKind::Model { config_index: 0 },
            Model::new(
                vec![Mesh::create_test_triangle(0)],
                vec![Material::default()],
            ),
            nalgebra::Matrix4::identity(),
        );
        let mut scene_objects = vec![model, ground];
        let config = Config {
            ground: crate::io::config::GroundConfig {
                albedo: Some([0.25, 0.5, 0.75]),
                ..Default::default()
            },
            objects: vec![ObjectConfig {
                path: "unused.gltf".to_string(),
                position: [3.0, 4.0, 5.0],
                rotation: [0.0; 3],
                scale: [1.0; 3],
                normalization: ModelNormalization::Preserve,
            }],
            ..Default::default()
        };

        update_scene_objects(&mut scene_objects, &config);

        assert_eq!(
            scene_objects[0].kind,
            SceneObjectKind::Model { config_index: 0 }
        );
        assert_eq!(
            scene_objects[0]
                .transform
                .transform_point(&Point3::origin()),
            Point3::new(3.0, 4.0, 5.0)
        );
        let Material::Pbr(ground_material) = &scene_objects[1].model.materials[0];
        assert_eq!(ground_material.albedo, Vector3::new(0.25, 0.5, 0.75));

        let reset_config = Config {
            ground: crate::io::config::GroundConfig {
                albedo: None,
                metallic: None,
                roughness: None,
                ..Default::default()
            },
            ..config
        };
        update_scene_objects(&mut scene_objects, &reset_config);
        let Material::Pbr(ground_material) = &scene_objects[1].model.materials[0];
        assert_eq!(ground_material.albedo, Vector3::new(0.6, 0.6, 0.6));
        assert_eq!(ground_material.metallic, 0.0);
        assert_eq!(ground_material.roughness, 0.8);
    }

    #[test]
    fn scene_resource_rebuild_assigns_kinds_from_the_new_config() {
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/gltf/nested-named-nodes.gltf")
            .to_string_lossy()
            .into_owned();
        let object = ObjectConfig {
            path: fixture,
            position: [0.0; 3],
            rotation: [0.0; 3],
            scale: [1.0; 3],
            normalization: ModelNormalization::Preserve,
        };
        let initial_config = Config {
            objects: vec![object.clone(), object.clone()],
            ..Default::default()
        };

        let initial = init_scene_resources(&initial_config).expect("fixture scene should load");
        assert_eq!(initial.scene_objects.len(), 3);
        assert_eq!(initial.scene_objects[0].kind, SceneObjectKind::Ground);
        assert_eq!(
            initial.scene_objects[1].kind,
            SceneObjectKind::Model { config_index: 0 }
        );
        assert_eq!(
            initial.scene_objects[2].kind,
            SceneObjectKind::Model { config_index: 1 }
        );

        let rebuilt_config = Config {
            ground: crate::io::config::GroundConfig {
                enabled: false,
                ..Default::default()
            },
            objects: vec![object],
            ..initial_config
        };
        let rebuilt = init_scene_resources(&rebuilt_config).expect("fixture scene should rebuild");

        assert_eq!(rebuilt.scene_objects.len(), 1);
        assert_eq!(
            rebuilt.scene_objects[0].kind,
            SceneObjectKind::Model { config_index: 0 }
        );
    }
}
