use crate::core::math::transform::TransformFactory;
use crate::io::config::Config;
use crate::io::gltf_loader::load_gltf;
use crate::scene::camera::Camera;
use crate::scene::context::{RenderContext, ShadowLight};
use crate::scene::light::Light;
use crate::scene::material::{Material, PbrMaterial};
use crate::scene::mesh::Mesh;
use crate::scene::model::Model;
use crate::scene::scene_object::SceneObject;
use crate::scene::utils::normalize_and_center_model;
use log::{error, info};
use nalgebra::{Point3, Vector3};

/// Helper to rebuild light list from config (used in Init and Hot Reload)
pub fn build_lights_from_config(config: &Config) -> (Vec<Light>, Option<ShadowLight>) {
    let mut lights = Vec::new();
    let mut shadow_light = None;

    for light_config in &config.lights {
        let color = Vector3::from(light_config.color);
        match light_config.r#type.as_str() {
            "directional" => {
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
            "point" => {
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
            _ => {}
        }
    }

    (lights, shadow_light)
}

/// Helper to update existing SceneObjects with new parameters from config.
/// Only updates Transform now.
pub fn update_scene_objects(scene_objects: &mut [SceneObject], config: &Config) {
    let num_loaded_objects = config.objects.len();
    let total_scene_objects = scene_objects.len();

    // Check if ground exists in memory (index 0)
    let has_ground_in_memory = total_scene_objects > num_loaded_objects;
    let obj_start_index = if has_ground_in_memory { 1 } else { 0 };

    // 1. Update Ground
    if has_ground_in_memory && let Some(ground_obj) = scene_objects.get_mut(0) {
        if config.ground.enabled {
            ground_obj.transform = TransformFactory::translation(&Vector3::new(0.0, -1.0, 0.0));
            if let Some(Material::Pbr(mat)) = ground_obj.model.materials.get_mut(0) {
                if let Some(c) = config.ground.albedo {
                    mat.albedo = Vector3::from(c);
                }
                if let Some(m) = config.ground.metallic {
                    mat.metallic = m;
                }
                if let Some(r) = config.ground.roughness {
                    mat.roughness = r;
                }
            }
        } else {
            ground_obj.transform = TransformFactory::scaling_nonuniform(&Vector3::zeros());
        }
    }

    // 2. Update Loaded Objects (Transforms Only)
    for (i, obj_conf) in config.objects.iter().enumerate() {
        let scene_idx = obj_start_index + i;
        if let Some(scene_obj) = scene_objects.get_mut(scene_idx) {
            let translation = TransformFactory::translation(&Vector3::from(obj_conf.position));
            let rotation = TransformFactory::rotation_x(obj_conf.rotation[0].to_radians())
                * TransformFactory::rotation_y(obj_conf.rotation[1].to_radians())
                * TransformFactory::rotation_z(obj_conf.rotation[2].to_radians());
            let scale = TransformFactory::scaling_nonuniform(&Vector3::from(obj_conf.scale));

            scene_obj.transform = translation * rotation * scale;
        }
    }
}

/// Initial resource loading (Heavy I/O). Returns a RenderContext.
pub fn init_scene_resources(config: &Config) -> RenderContext {
    // 1. Camera
    let cam_pos = Point3::from(config.camera.position);
    let cam_target = Point3::from(config.camera.target);
    let cam_up = Vector3::from(config.camera.up);
    let aspect_ratio = config.render.width as f32 / config.render.height as f32;

    let camera = if config.camera.projection == "orthographic" {
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
        let ground_mat = Material::Pbr(PbrMaterial {
            albedo: config
                .ground
                .albedo
                .map(Vector3::from)
                .unwrap_or(Vector3::new(0.6, 0.6, 0.6)),
            metallic: config.ground.metallic.unwrap_or(0.0),
            roughness: config.ground.roughness.unwrap_or(0.8),
            ao: 1.0,
            emissive: Vector3::zeros(),
            ..Default::default()
        });
        scene_objects.push(SceneObject::new(
            Model::new(vec![ground_mesh], vec![ground_mat]),
            TransformFactory::translation(&Vector3::new(0.0, -1.0, 0.0)),
        ));
    }

    // 3.2 Loaded Objects
    for obj_conf in &config.objects {
        // Direct GLTF Loading
        let mut model = match load_gltf(&obj_conf.path, config.render.use_mipmap) {
            Ok(mut m) => {
                normalize_and_center_model(&mut m);
                m
            }
            Err(e) => {
                error!(
                    "Error loading GLTF '{}': {}. Using fallback mesh.",
                    obj_conf.path, e
                );
                let mesh = Mesh::create_test_triangle(0);
                let mat = PbrMaterial {
                    albedo: Vector3::new(1.0, 0.0, 1.0),
                    ..Default::default()
                };
                Model::new(vec![mesh], vec![Material::Pbr(mat)])
            }
        };

        // Ensure material fallback
        if model.materials.is_empty() {
            model.materials.push(Material::default());
        }

        // Apply Transform
        let translation = TransformFactory::translation(&Vector3::from(obj_conf.position));
        let rotation = TransformFactory::rotation_x(obj_conf.rotation[0].to_radians())
            * TransformFactory::rotation_y(obj_conf.rotation[1].to_radians())
            * TransformFactory::rotation_z(obj_conf.rotation[2].to_radians());
        let scale = TransformFactory::scaling_nonuniform(&Vector3::from(obj_conf.scale));

        scene_objects.push(SceneObject::new(model, translation * rotation * scale));
    }

    info!("Scene initialized with {} objects.", scene_objects.len());

    RenderContext {
        camera,
        lights,
        scene_objects,
        shadow_light,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::config::LightConfig;

    fn point_light() -> LightConfig {
        LightConfig {
            r#type: "point".to_string(),
            position: Some([1.0, 2.0, 3.0]),
            direction: None,
            color: [1.0, 1.0, 1.0],
            intensity: 2.0,
            attenuation: None,
        }
    }

    fn directional_light(direction: [f32; 3]) -> LightConfig {
        LightConfig {
            r#type: "directional".to_string(),
            position: None,
            direction: Some(direction),
            color: [1.0, 1.0, 1.0],
            intensity: 3.0,
            attenuation: None,
        }
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
}
