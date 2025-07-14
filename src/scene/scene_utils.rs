use crate::geometry::camera::Camera;
use crate::io::render_settings::{RenderSettings, parse_point3, parse_vec3};
use crate::material_system::light::Light;
use crate::material_system::materials::ModelData;
use crate::material_system::materials::material_applicator::{
    apply_pbr_parameters, apply_phong_parameters,
};
use crate::scene::scene_object::SceneObject;
use nalgebra::Vector3;

/// 表示一个 3D 场景，包含对象、光源和相机
#[derive(Debug, Clone)]
pub struct Scene {
    /// 场景中的主要对象（简化为单个对象）
    pub object: SceneObject,

    /// 场景中的光源
    pub lights: Vec<Light>,

    /// 当前活动相机
    pub active_camera: Camera,

    /// 环境光强度
    pub ambient_intensity: f32,

    /// 环境光颜色
    pub ambient_color: Vector3<f32>,
}

impl Scene {
    /// 链式创建场景，自动应用所有设置
    pub fn new(model_data: ModelData, settings: &RenderSettings) -> Result<Self, String> {
        let mut model_data = model_data.clone();
        // 应用材质参数
        if settings.use_pbr {
            apply_pbr_parameters(&mut model_data, settings);
        }
        if settings.use_phong {
            apply_phong_parameters(&mut model_data, settings);
        }

        // 创建对象
        let mut object = SceneObject::from_model_data(model_data);

        // 应用对象变换
        let (position, rotation_rad, scale) = settings.get_object_transform_components();
        let final_scale = if settings.object_scale != 1.0 {
            scale * settings.object_scale
        } else {
            scale
        };
        object.set_transform_from_components(position, rotation_rad, final_scale);

        // 相机
        let aspect_ratio = settings.width as f32 / settings.height as f32;
        let camera_from =
            parse_point3(&settings.camera_from).map_err(|e| format!("无效的相机位置格式: {e}"))?;
        let camera_at =
            parse_point3(&settings.camera_at).map_err(|e| format!("无效的相机目标格式: {e}"))?;
        let camera_up =
            parse_vec3(&settings.camera_up).map_err(|e| format!("无效的相机上方向格式: {e}"))?;
        let camera = match settings.projection.as_str() {
            "perspective" => Camera::perspective(
                camera_from,
                camera_at,
                camera_up,
                settings.camera_fov,
                aspect_ratio,
                0.1,
                100.0,
            ),
            "orthographic" => {
                let height = 4.0;
                let width = height * aspect_ratio;
                Camera::orthographic(camera_from, camera_at, camera_up, width, height, 0.1, 100.0)
            }
            _ => return Err(format!("不支持的投影类型: {}", settings.projection)),
        };

        // 光源
        let lights = settings.lights.clone();

        // 环境光
        let ambient_intensity = settings.ambient;
        let ambient_color = settings.get_ambient_color_vec();

        Ok(Scene {
            object,
            lights,
            active_camera: camera,
            ambient_intensity,
            ambient_color,
        })
    }

    /// 链式设置对象变换
    pub fn set_object_transform(
        &mut self,
        position: Vector3<f32>,
        rotation_rad: Vector3<f32>,
        scale: Vector3<f32>,
    ) -> &mut Self {
        self.object
            .set_transform_from_components(position, rotation_rad, scale);
        self
    }

    /// 链式设置光源
    pub fn set_lights(&mut self, lights: Vec<Light>) -> &mut Self {
        self.lights = lights;
        self
    }

    /// 链式设置相机
    pub fn set_camera(&mut self, camera: Camera) -> &mut Self {
        self.active_camera = camera;
        self
    }

    /// 链式设置环境光
    pub fn set_ambient(&mut self, intensity: f32, color: Vector3<f32>) -> &mut Self {
        self.ambient_intensity = intensity;
        self.ambient_color = color;
        self
    }

    /// 获取场景统计信息
    pub fn get_scene_stats(&self) -> SceneStats {
        let mut vertex_count = 0;
        let mut triangle_count = 0;
        let material_count = self.object.model_data.materials.len();
        let mesh_count = self.object.model_data.meshes.len();

        for mesh in &self.object.model_data.meshes {
            vertex_count += mesh.vertices.len();
            triangle_count += mesh.indices.len() / 3;
        }

        SceneStats {
            vertex_count,
            triangle_count,
            material_count,
            mesh_count,
            light_count: self.lights.len(),
        }
    }
}

/// 场景统计信息
#[derive(Debug, Clone)]
pub struct SceneStats {
    pub vertex_count: usize,
    pub triangle_count: usize,
    pub material_count: usize,
    pub mesh_count: usize,
    pub light_count: usize,
}
