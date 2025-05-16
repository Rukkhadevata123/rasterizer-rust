use crate::core::scene_object::SceneObject;
use crate::geometry::camera::Camera;
use crate::materials::material_system::Light;
use crate::utils::model_types::ModelData;
use nalgebra::{Point3, Vector3};
use std::collections::HashMap;

/// 表示一个完整的 3D 场景，包含模型、对象实例、光源和相机
#[derive(Debug, Clone)]
pub struct Scene {
    /// 所有加载的模型数据，由标识符索引
    pub models: Vec<ModelData>,

    /// 场景中的所有对象实例
    pub objects: Vec<SceneObject>,

    /// 场景中的光源 (使用特征对象)
    pub lights: Vec<Light>,

    /// 当前活动相机
    pub active_camera: Camera,

    /// 命名对象的映射，允许通过名称查找对象
    object_names: HashMap<String, usize>,
}

impl Scene {
    /// 创建一个新的空场景
    pub fn new(default_camera: Camera) -> Self {
        Scene {
            models: Vec::new(),
            objects: Vec::new(),
            lights: Vec::new(),
            active_camera: default_camera,
            object_names: HashMap::new(),
        }
    }

    /// 向场景添加模型数据，返回其分配的ID
    pub fn add_model(&mut self, model: ModelData) -> usize {
        let model_id = self.models.len();
        self.models.push(model);
        model_id
    }

    /// 向场景添加一个对象实例，可选地为其命名
    pub fn add_object(&mut self, object: SceneObject, name: Option<&str>) -> usize {
        let object_id = self.objects.len();

        // 如果提供了名称，添加到映射
        if let Some(name_str) = name {
            self.object_names.insert(name_str.to_string(), object_id);
        }

        self.objects.push(object);
        object_id
    }

    /// 向场景添加光源 (接受特征对象)
    pub fn add_light(&mut self, light: Light) -> usize {
        let light_id = self.lights.len();
        self.lights.push(light);
        light_id
    }

    /// 设置场景的活动相机
    pub fn set_camera(&mut self, camera: Camera) {
        self.active_camera = camera;
    }

    /// 在场景中创建一个定向光
    pub fn create_directional_light(
        &mut self,
        direction: Vector3<f32>,
        intensity: Vector3<f32>,
    ) -> usize {
        self.add_light(Light::directional(direction, intensity))
    }

    /// 在场景中创建一个点光源
    pub fn create_point_light(
        &mut self,
        position: Point3<f32>,
        intensity: Vector3<f32>,
        attenuation: (f32, f32, f32),
    ) -> usize {
        self.add_light(Light::point(position, intensity, Some(attenuation)))
    }

    /// 在场景中创建一个环境光
    pub fn create_ambient_light(&mut self, intensity: Vector3<f32>) -> usize {
        self.add_light(Light::ambient(intensity))
    }

    /// 在场景中以圆形阵列创建多个对象实例
    pub fn create_object_ring(
        &mut self,
        model_id: usize,
        count: usize,
        radius: f32,
        base_name: Option<&str>,
    ) -> Vec<usize> {
        let mut object_ids = Vec::with_capacity(count);

        for i in 0..count {
            let angle = (i as f32) * (std::f32::consts::PI * 2.0 / (count as f32));

            // 在 XZ 平面上围绕圆形摆放
            let x = radius * angle.cos();
            let z = radius * angle.sin();

            // 围绕 Y 轴旋转，面向圆心
            let rotation = Vector3::new(0.0, angle + std::f32::consts::PI, 0.0);

            let position = Point3::new(x, 0.0, z);
            let scale = Vector3::new(1.0, 1.0, 1.0); // 默认缩放

            let object = SceneObject::new(model_id, position, rotation, scale);

            // 如果提供了基础名称，为每个对象创建唯一名称
            let name = base_name.map(|base| format!("{}_{}", base, i));

            let object_id = self.add_object(object, name.as_deref());
            object_ids.push(object_id);
        }

        object_ids
    }

    /// 获取场景中的对象数量
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// 获取场景中的光源数量
    pub fn light_count(&self) -> usize {
        self.lights.len()
    }
}
