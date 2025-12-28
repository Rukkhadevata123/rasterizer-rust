use crate::core::math::transform::TransformFactory;
use crate::io::config::Config;
use crate::io::obj_loader::load_obj;
use crate::scene::camera::Camera;
use crate::scene::context::RenderContext;
use crate::scene::light::Light;
use crate::scene::material::{Material, PbrMaterial};
use crate::scene::mesh::Mesh;
use crate::scene::model::Model;
use crate::scene::scene_object::SceneObject;
use crate::scene::texture::Texture;
use crate::scene::utils::normalize_and_center_model;
use log::{error, info, warn};
use nalgebra::{Point3, Vector3};
use std::sync::Arc;

/// Helper to rebuild light list from config (used in Init and Hot Reload)
pub fn build_lights_from_config(config: &Config) -> (Vec<Light>, Point3<f32>) {
    let mut lights = Vec::new();
    let mut shadow_light_pos = Point3::new(0.0, 10.0, 0.0);
    let mut has_shadow_light = false;

    for l in &config.lights {
        let color = Vector3::from(l.color);
        match l.r#type.as_str() {
            "directional" => {
                if let Some(dir) = l.direction {
                    let dir_vec = Vector3::from(dir).normalize();
                    lights.push(Light::new_directional(dir_vec, color, l.intensity));
                    if !has_shadow_light {
                        shadow_light_pos = Point3::origin() - dir_vec * 10.0;
                        has_shadow_light = true;
                    }
                }
            }
            "point" => {
                if let Some(pos) = l.position {
                    let mut light = Light::new_point(Point3::from(pos), color, l.intensity);
                    if let Light::Point {
                        ref mut attenuation,
                        ..
                    } = light
                        && let Some(a) = l.attenuation
                    {
                        *attenuation = (a[0], a[1], a[2]);
                    }
                    lights.push(light);
                }
            }
            _ => {}
        }
    }
    (lights, shadow_light_pos)
}

/// Helper to update existing SceneObjects with new parameters from config.
pub fn update_scene_objects(scene_objects: &mut [SceneObject], config: &Config) {
    let num_loaded_objects = config.objects.len();
    let total_scene_objects = scene_objects.len();

    // Check if ground exists in memory (index 0)
    let has_ground_in_memory = total_scene_objects > num_loaded_objects;
    let obj_start_index = if has_ground_in_memory { 1 } else { 0 };

    // 1. Update Ground
    if has_ground_in_memory {
        if let Some(ground_obj) = scene_objects.get_mut(0) {
            if config.ground.enabled {
                ground_obj.transform = TransformFactory::translation(&Vector3::new(0.0, -1.0, 0.0));
                let Material::Pbr(mat) = &mut ground_obj.model.materials[0];
                if let Some(c) = config.ground.albedo {
                    mat.albedo = Vector3::from(c);
                }
                if let Some(m) = config.ground.metallic {
                    mat.metallic = m;
                }
                if let Some(r) = config.ground.roughness {
                    mat.roughness = r;
                }
            } else {
                // Hide ground
                ground_obj.transform = TransformFactory::scaling_nonuniform(&Vector3::zeros());
            }
        }
    } else if config.ground.enabled {
        warn!("Cannot enable ground dynamically because it wasn't loaded at startup.");
    }

    // 2. Update Loaded Objects
    for (i, obj_conf) in config.objects.iter().enumerate() {
        let scene_idx = obj_start_index + i;
        if let Some(scene_obj) = scene_objects.get_mut(scene_idx) {
            // Update Transform
            let translation = TransformFactory::translation(&Vector3::from(obj_conf.position));
            let rotation = TransformFactory::rotation_x(obj_conf.rotation[0].to_radians())
                * TransformFactory::rotation_y(obj_conf.rotation[1].to_radians())
                * TransformFactory::rotation_z(obj_conf.rotation[2].to_radians());
            let scale = TransformFactory::scaling_nonuniform(&Vector3::from(obj_conf.scale));
            scene_obj.transform = translation * rotation * scale;

            // Update Material
            let Material::Pbr(mat) = &mut scene_obj.model.materials[0];
            if let Some(c) = obj_conf.albedo {
                mat.albedo = Vector3::from(c);
            }
            if let Some(m) = obj_conf.metallic {
                mat.metallic = m;
            }
            if let Some(r) = obj_conf.roughness {
                mat.roughness = r;
            }
            if let Some(ao) = obj_conf.ao {
                mat.ao = ao;
            }
            if let Some(e) = obj_conf.emissive {
                mat.emissive = Vector3::from(e) * obj_conf.emissive_intensity;
            }
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
    let (lights, shadow_light_pos) = build_lights_from_config(config);

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
                .unwrap_or(Vector3::new(0.9, 0.9, 0.9)),
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
        let mut model = match load_obj(&obj_conf.path, config.render.use_mipmap) {
            Ok(mut m) => {
                normalize_and_center_model(&mut m);
                m
            }
            Err(e) => {
                error!(
                    "Error loading model '{}': {}. Using fallback mesh.",
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

        if model.materials.is_empty() {
            model.materials.push(Material::default());
        }

        // Apply config overrides
        let Material::Pbr(ref mut mat) = model.materials[0];
        if let Some(c) = obj_conf.albedo {
            mat.albedo = Vector3::from(c);
        }
        if let Some(m) = obj_conf.metallic {
            mat.metallic = m;
        }
        if let Some(r) = obj_conf.roughness {
            mat.roughness = r;
        }
        if let Some(ao) = obj_conf.ao {
            mat.ao = ao;
        }
        if let Some(e) = obj_conf.emissive {
            mat.emissive = Vector3::from(e) * obj_conf.emissive_intensity;
        }

        // Load textures
        if let Some(path) = &obj_conf.albedo_texture {
            if let Ok(tex) = Texture::load(path, config.render.use_mipmap) {
                mat.albedo_texture = Some(Arc::new(tex));
            } else {
                warn!("Failed to load Albedo texture '{}'", path);
            }
        }
        if let Some(path) = &obj_conf.metallic_roughness_texture {
            if let Ok(tex) = Texture::load(path, config.render.use_mipmap) {
                mat.metallic_roughness_texture = Some(Arc::new(tex));
            } else {
                warn!("Failed to load Metallic/Roughness texture '{}'", path);
            }
        }
        if let Some(path) = &obj_conf.normal_texture {
            if let Ok(tex) = Texture::load(path, config.render.use_mipmap) {
                mat.normal_texture = Some(Arc::new(tex));
            } else {
                warn!("Failed to load Normal texture '{}'", path);
            }
        }

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
        shadow_light_pos,
    }
}
