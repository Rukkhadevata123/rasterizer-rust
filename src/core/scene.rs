use crate::core::scene_object::SceneObject;
use crate::geometry::camera::Camera;
use crate::materials::material_system::Light;
use crate::materials::model_types::ModelData;
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

    /// 环境光强度 - 控制场景整体亮度 [0.0, 1.0]
    pub ambient_intensity: f32,

    /// 环境光颜色 - 控制场景基础色调 (RGB)
    pub ambient_color: Vector3<f32>,
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
            ambient_intensity: 0.2,                     // 默认环境光强度
            ambient_color: Vector3::new(1.0, 1.0, 1.0), // 默认白色环境光
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

    /// 设置场景的环境光参数
    ///
    /// 根据参数设置场景环境光的强度和颜色
    pub fn setup_ambient_light(&mut self, ambient: f32, ambient_color: &str) -> Result<(), String> {
        use crate::io::args::parse_vec3;

        // 设置环境光强度
        self.ambient_intensity = ambient;

        // 设置环境光颜色
        if !ambient_color.is_empty() {
            if let Ok(parsed_color) = parse_vec3(ambient_color) {
                self.ambient_color = parsed_color;
            } else {
                // 如果颜色格式无效，使用统一的RGB值
                self.ambient_color = Vector3::new(1.0, 1.0, 1.0);
            }
        }

        Ok(())
    }

    /// 设置场景的光照系统
    ///
    /// 根据参数配置定向光或点光源，并设置环境光
    #[allow(clippy::too_many_arguments)]
    pub fn setup_lighting(
        &mut self,
        use_lighting: bool,
        light_type: &str,
        light_dir: &str,
        light_pos: &str,
        light_atten: &str,
        diffuse: f32,
        ambient: f32,
        ambient_color: &str,
    ) -> Result<(), String> {
        // 首先设置环境光 - 无论是否使用主光源
        self.setup_ambient_light(ambient, ambient_color)?;

        // 如果不使用光照，就到此为止
        if !use_lighting {
            return Ok(());
        }

        use crate::io::args::parse_point3;
        use crate::io::args::parse_vec3;

        // 设置主光源 (定向光或点光源)
        let light_intensity = Vector3::new(1.0, 1.0, 1.0) * diffuse;

        match light_type.to_lowercase().as_str() {
            "point" => {
                let light_pos =
                    parse_point3(light_pos).map_err(|e| format!("无效的光源位置格式: {}", e))?;

                let atten_parts: Vec<Result<f32, _>> = light_atten
                    .split(',')
                    .map(|s| s.trim().parse::<f32>())
                    .collect();

                if atten_parts.len() != 3 || atten_parts.iter().any(|r| r.is_err()) {
                    return Err(format!("无效的光衰减格式: '{}'. 应为 'c,l,q'", light_atten));
                }

                let attenuation = (
                    atten_parts[0].as_ref().map_or(0.0, |v| *v).max(0.0),
                    atten_parts[1].as_ref().map_or(0.0, |v| *v).max(0.0),
                    atten_parts[2].as_ref().map_or(0.0, |v| *v).max(0.0),
                );

                self.create_point_light(light_pos, light_intensity, attenuation);
            }
            _ => {
                // 默认为定向光
                let mut light_dir =
                    parse_vec3(light_dir).map_err(|e| format!("无效的光源方向格式: {}", e))?;
                light_dir = -light_dir.normalize(); // 朝向光源的方向

                self.create_directional_light(light_dir, light_intensity);
            }
        }

        Ok(())
    }

    /// 从模型数据和参数创建并设置完整场景
    pub fn setup_from_model_data(
        &mut self,
        model_data: ModelData,
        object_count: Option<usize>,
    ) -> usize {
        // 添加模型和主对象
        let model_id = self.add_model(model_data);
        let main_object = SceneObject::new_default(model_id);
        self.add_object(main_object, Some("main"));

        // 添加多个对象实例(如果需要)
        if let Some(count) = object_count {
            if count > 1 {
                // 创建环形对象阵列
                let radius = 2.0;
                self.create_object_ring(model_id, count - 1, radius, Some("satellite"));
            }
        }

        model_id
    }
}
