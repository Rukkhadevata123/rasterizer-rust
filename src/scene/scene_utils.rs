// scene_utils.rs
use crate::geometry::camera::Camera;
use crate::io::render_settings::{RenderSettings, parse_point3, parse_vec3};
use crate::material_system::light::Light;
use crate::material_system::materials::ModelData;
use crate::material_system::materials::material_applicator::{
    apply_pbr_parameters, apply_phong_parameters,
};
use crate::scene::scene_object::SceneObject;
use crate::scene::scene_object::Transformable;
use nalgebra::{Point3, Vector3};
use std::collections::HashMap;

/// 表示一个完整的 3D 场景，包含模型、对象实例、光源和相机
#[derive(Debug, Clone)]
pub struct Scene {
    /// 所有加载的模型数据，由标识符索引
    pub models: Vec<ModelData>,

    /// 场景中的所有对象实例
    pub objects: Vec<SceneObject>,

    /// 场景中的光源
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
            ambient_intensity: 0.2,
            ambient_color: Vector3::new(1.0, 1.0, 1.0),
        }
    }

    /// 根据渲染设置创建相机
    ///
    /// # 参数
    /// * `settings` - 渲染设置
    ///
    /// # 返回值
    /// 创建的相机对象
    pub fn setup_camera_from_settings(settings: &RenderSettings) -> Result<Camera, String> {
        let aspect_ratio = settings.width as f32 / settings.height as f32;
        let camera_from = parse_point3(&settings.camera_from)
            .map_err(|e| format!("无效的相机位置格式: {}", e))?;
        let camera_at =
            parse_point3(&settings.camera_at).map_err(|e| format!("无效的相机目标格式: {}", e))?;
        let camera_up =
            parse_vec3(&settings.camera_up).map_err(|e| format!("无效的相机上方向格式: {}", e))?;

        Ok(Camera::new(
            camera_from,
            camera_at,
            camera_up,
            settings.camera_fov,
            aspect_ratio,
            0.1,   // 近平面距离
            100.0, // 远平面距离
        ))
    }

    /// 从模型数据和渲染设置创建完整场景
    ///
    /// 这个函数是场景创建的统一接口，处理相机设置、材质应用、场景对象和光照
    ///
    /// # 参数
    /// * `model_data` - 预加载的模型数据
    /// * `settings` - 渲染设置
    ///
    /// # 返回值
    /// 完整配置的场景对象
    pub fn create_from_model_and_settings(
        model_data: ModelData,
        settings: &RenderSettings,
    ) -> Result<Self, String> {
        // 创建相机和基础场景
        let camera = Self::setup_camera_from_settings(settings)?;
        let mut scene = Self::new(camera);

        // 创建并配置模型和场景对象
        let mut modified_model_data = model_data.clone();

        // 应用材质参数
        scene.apply_material_parameters(&mut modified_model_data, settings);

        // 添加模型和对象
        let model_id = scene.add_model(modified_model_data);

        // 创建主对象和额外对象
        let object_count = settings
            .object_count
            .as_ref()
            .and_then(|count_str| count_str.parse::<usize>().ok());

        // 添加主对象
        let main_object = SceneObject::new(model_id).with_name("main");
        scene.add_object(main_object);

        // 添加额外对象（如果需要）
        if let Some(count) = object_count {
            if count > 1 {
                let radius = 2.0;
                scene.create_object_ring(model_id, count - 1, radius, Some("satellite"));
                println!("创建了环形排列的 {} 个附加对象", count - 1);
            }
        }

        // 设置场景光照
        scene.setup_lighting_from_settings(settings)?;

        Ok(scene)
    }

    /// 应用材质参数到模型数据
    fn apply_material_parameters(&self, model_data: &mut ModelData, settings: &RenderSettings) {
        // 应用PBR材质参数(如果需要)
        if settings.use_pbr {
            println!(
                "应用PBR材质参数 - 金属度: {}, 粗糙度: {}",
                settings.metallic, settings.roughness
            );
            apply_pbr_parameters(model_data, settings);
        }

        // 应用Phong材质参数(如果需要)
        if settings.use_phong {
            println!(
                "应用Phong材质参数 - 高光系数: {}, 光泽度: {}",
                settings.specular, settings.shininess
            );
            apply_phong_parameters(model_data, settings);
        }
    }

    /// 向场景添加模型数据，返回其分配的ID
    pub fn add_model(&mut self, model: ModelData) -> usize {
        let model_id = self.models.len();
        self.models.push(model);
        model_id
    }

    /// 向场景添加一个对象实例
    pub fn add_object(&mut self, object: SceneObject) -> usize {
        let object_id = self.objects.len();

        // 如果对象有名称，添加到映射
        if let Some(name) = &object.name {
            self.object_names.insert(name.clone(), object_id);
        }

        self.objects.push(object);
        object_id
    }

    /// 根据名称查找对象
    pub fn find_object(&self, name: &str) -> Option<&SceneObject> {
        self.object_names.get(name).map(|id| &self.objects[*id])
    }

    /// 根据名称查找可变对象
    pub fn find_object_mut(&mut self, name: &str) -> Option<&mut SceneObject> {
        if let Some(id) = self.object_names.get(name).cloned() {
            return Some(&mut self.objects[id]);
        }
        None
    }

    /// 向场景添加光源
    pub fn add_light(&mut self, light: Light) -> usize {
        let light_id = self.lights.len();
        self.lights.push(light);
        light_id
    }

    /// 清除所有光源
    pub fn clear_lights(&mut self) {
        self.lights.clear();
    }

    /// 设置场景的活动相机
    pub fn set_camera(&mut self, camera: Camera) {
        self.active_camera = camera;
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
            let position = Point3::new(x, 0.0, z);

            // 创建对象，面向圆心
            let object = SceneObject::new(model_id).with_position(position);

            // 添加到场景
            let mut object = object;
            object.rotate_y(angle + std::f32::consts::PI);

            // 如果提供了基础名称，为每个对象创建唯一名称
            if let Some(base) = base_name {
                object.name = Some(format!("{}_{}", base, i));
            }

            let object_id = self.add_object(object);
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
    pub fn set_ambient_light(&mut self, intensity: f32, color: Vector3<f32>) {
        self.ambient_intensity = intensity;
        self.ambient_color = color;
    }

    /// 从渲染设置设置灯光
    fn setup_lighting_from_settings(&mut self, settings: &RenderSettings) -> Result<(), String> {
        // 设置环境光
        let color = if !settings.ambient_color.is_empty() {
            parse_vec3(&settings.ambient_color).unwrap_or_else(|_| Vector3::new(1.0, 1.0, 1.0))
        } else {
            Vector3::new(1.0, 1.0, 1.0)
        };

        self.set_ambient_light(settings.ambient, color);

        // 如果不使用光照，直接返回
        if !settings.use_lighting {
            return Ok(());
        }

        // 清除现有灯光
        self.clear_lights();

        // 从预设创建光源
        if settings.directional_lights.is_empty() && settings.point_lights.is_empty() {
            // 如果没有明确配置，从预设创建
            let preset_lights = crate::material_system::light::create_lights_from_preset(
                settings.lighting_preset.clone(),
                settings.main_light_intensity,
            );

            for light in preset_lights {
                self.lights.push(light);
            }
        } else {
            // 否则使用配置的光源
            // 添加方向光源
            for (i, light) in settings.directional_lights.iter().enumerate() {
                if light.enabled {
                    match light.to_light() {
                        Ok(l) => self.lights.push(l),
                        Err(e) => eprintln!("方向光 #{} 配置错误: {}", i + 1, e),
                    }
                }
            }

            // 添加点光源
            for (i, light) in settings.point_lights.iter().enumerate() {
                if light.enabled {
                    match light.to_light() {
                        Ok(l) => self.lights.push(l),
                        Err(e) => eprintln!("点光源 #{} 配置错误: {}", i + 1, e),
                    }
                }
            }
        }

        // 如果没有光源，添加一个默认的方向光源
        if self.lights.is_empty() {
            let default_direction = Vector3::new(0.0, -1.0, -1.0).normalize();
            let default_color = Vector3::new(1.0, 1.0, 1.0);
            self.add_light(Light::directional(default_direction, default_color, 0.8));
        }

        println!("已设置 {} 个光源", self.lights.len());
        Ok(())
    }
}
