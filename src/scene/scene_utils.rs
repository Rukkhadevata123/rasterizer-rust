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
    /// 从模型数据创建场景
    pub fn from_model_data(model_data: ModelData, default_camera: Camera) -> Self {
        Scene {
            object: SceneObject::from_model_data(model_data),
            lights: Vec::new(),
            active_camera: default_camera,
            ambient_intensity: 0.2,
            ambient_color: Vector3::new(1.0, 1.0, 1.0),
        }
    }

    /// 从模型数据和渲染设置创建完整场景
    pub fn create_from_model_and_settings(
        model_data: ModelData,
        settings: &RenderSettings,
    ) -> Result<Self, String> {
        let camera = Self::setup_camera_from_settings(settings)?;
        let mut modified_model_data = model_data.clone();

        Self::apply_material_parameters(&mut modified_model_data, settings);

        let mut scene = Self::from_model_data(modified_model_data, camera);

        // 应用对象变换
        scene.update_object_transform(settings);

        // 直接使用设置中的光源，无需重复创建
        scene.lights = settings.lights.clone();

        // 使用按需计算方法获取环境光颜色
        scene.set_ambient_light(settings.ambient, settings.get_ambient_color_vec());

        Ok(scene)
    }

    /// 更新对象变换
    pub fn update_object_transform(&mut self, settings: &RenderSettings) {
        let (position, rotation_rad, scale) = settings.get_object_transform_components();

        // 应用全局缩放
        let final_scale = if settings.object_scale != 1.0 {
            scale * settings.object_scale
        } else {
            scale
        };

        self.object
            .set_transform_from_components(position, rotation_rad, final_scale);
    }

    /// 根据渲染设置创建相机
    pub fn setup_camera_from_settings(settings: &RenderSettings) -> Result<Camera, String> {
        let aspect_ratio = settings.width as f32 / settings.height as f32;
        let camera_from = parse_point3(&settings.camera_from)
            .map_err(|e| format!("无效的相机位置格式: {}", e))?;
        let camera_at =
            parse_point3(&settings.camera_at).map_err(|e| format!("无效的相机目标格式: {}", e))?;
        let camera_up =
            parse_vec3(&settings.camera_up).map_err(|e| format!("无效的相机上方向格式: {}", e))?;

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

        Ok(camera)
    }

    /// 设置活动相机
    pub fn set_camera(&mut self, camera: Camera) {
        self.active_camera = camera;
    }

    /// 设置环境光
    pub fn set_ambient_light(&mut self, intensity: f32, color: Vector3<f32>) {
        self.ambient_intensity = intensity;
        self.ambient_color = color;
    }

    /// 获取场景统计信息（直接计算，无中间转换）
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

    // 私有辅助方法
    fn apply_material_parameters(model_data: &mut ModelData, settings: &RenderSettings) {
        if settings.use_pbr {
            apply_pbr_parameters(model_data, settings);
        }
        if settings.use_phong {
            apply_phong_parameters(model_data, settings);
        }
    }
}

/// 统一的场景统计信息（删除ObjectStats，只用一个结构体）
#[derive(Debug, Clone)]
pub struct SceneStats {
    pub vertex_count: usize,
    pub triangle_count: usize,
    pub material_count: usize,
    pub mesh_count: usize,
    pub light_count: usize,
}
