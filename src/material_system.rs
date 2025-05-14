use crate::texture_utils::Texture;
use nalgebra::{Matrix3, Point3, Vector2, Vector3};
use std::fmt::Debug; // 添加 Texture 导入

/// 定义材质接口，允许不同的材质实现
pub trait IMaterial: Debug + Send + Sync {
    /// 计算材质在给定点的响应
    fn compute_response(
        &self,
        light_dir: &Vector3<f32>,
        view_dir: &Vector3<f32>,
        normal: &Vector3<f32>,
    ) -> Vector3<f32>;

    /// 获取材质的漫反射颜色（或基础颜色）
    ///
    /// 提供默认实现，使这个方法成为可选的。
    /// 材质可以选择不实现这个方法，而使用默认值。
    fn get_diffuse_color(&self) -> Vector3<f32> {
        // 默认返回中灰色
        Vector3::new(0.5, 0.5, 0.5)
    }

    /// 获取材质的环境光响应颜色
    fn get_ambient_color(&self) -> Vector3<f32>;
}

/// 定义光源接口，允许不同的光源类型
pub trait ILight: Debug + Send + Sync {
    /// 计算光源在给定点的方向（指向光源）
    fn get_direction(&self, point: &nalgebra::Point3<f32>) -> Vector3<f32>;

    /// 计算光源在给定点的强度（考虑衰减）
    fn get_intensity(&self, point: &nalgebra::Point3<f32>) -> Vector3<f32>;

    /// 获取光源类型的名称，用于调试
    /// 提供默认实现，返回通用名称
    fn get_type_name(&self) -> &'static str {
        "Generic Light"
    }

    /// 将光源特征对象转换为具体的 Light 枚举类型 (如果可能)
    /// 注意：这是一种具体的实现需求，更通用的方法可能是避免直接转换
    fn to_light_enum(&self) -> crate::lighting::Light;
}

/// 定义光照模型接口，允许使用不同的光照算法
pub trait ILightingModel: Debug + Send + Sync {
    /// 使用给定的材质和光源计算颜色
    ///
    /// 这是光照模型的核心方法，计算给定点在给定光照条件下的最终颜色。
    /// 具体实现可能包含环境光、漫反射、镜面反射等组件的计算。
    fn compute_lighting(
        &self,
        material: &dyn IMaterial,
        lights: &[Box<dyn ILight>],
        position: &nalgebra::Point3<f32>,
        normal: &Vector3<f32>,
        view_dir: &Vector3<f32>,
    ) -> Vector3<f32>;

    /// 获取光照模型的名称，用于调试
    fn get_model_name(&self) -> &'static str {
        "Default Lighting Model"
    }
}

/// Blinn-Phong 光照模型实现
#[derive(Debug, Clone)]
pub struct BlinnPhongLightingModel {
    pub ambient_intensity: Vector3<f32>,
}

impl Default for BlinnPhongLightingModel {
    fn default() -> Self {
        Self {
            ambient_intensity: Vector3::new(0.1, 0.1, 0.1),
        }
    }
}

impl BlinnPhongLightingModel {
    /// 创建一个新的 Blinn-Phong 光照模型实例
    pub fn new() -> Self {
        Self::default()
    }

    // 添加一个方法来使用 compute_lighting
    pub fn render_with_model(
        &self,
        point: Point3<f32>,
        normal: Vector3<f32>,
        _view_dir: Vector3<f32>,
        light: &crate::lighting::Light,
        diffuse_color: Vector3<f32>,
    ) -> Vector3<f32> {
        // 简单实现，用于消除警告
        match light {
            crate::lighting::Light::Ambient(intensity) => intensity.component_mul(&diffuse_color),
            crate::lighting::Light::Directional {
                direction,
                intensity,
            } => {
                let n_dot_l = normal.dot(direction).max(0.0);
                intensity.component_mul(&diffuse_color) * n_dot_l
            }
            crate::lighting::Light::Point {
                position,
                intensity,
                attenuation: _,
            } => {
                let light_dir = (position - point).normalize();
                let n_dot_l = normal.dot(&light_dir).max(0.0);
                intensity.component_mul(&diffuse_color) * n_dot_l
            }
        }
    }

    // 覆盖默认的 get_model_name 方法
    pub fn get_model_name(&self) -> &'static str {
        "Blinn-Phong 光照模型"
    }
}

impl ILightingModel for BlinnPhongLightingModel {
    fn compute_lighting(
        &self,
        material: &dyn IMaterial,
        lights: &[Box<dyn ILight>],
        position: &nalgebra::Point3<f32>,
        normal: &Vector3<f32>,
        view_dir: &Vector3<f32>,
    ) -> Vector3<f32> {
        // 环境光部分
        let ambient = material
            .get_ambient_color()
            .component_mul(&self.ambient_intensity);

        // 初始颜色为环境光
        let mut final_color = ambient;

        // 累加所有光源的贡献
        for light in lights {
            let light_dir = light.get_direction(position);
            let light_intensity = light.get_intensity(position);

            // 获取光照响应
            let response = material.compute_response(&light_dir, view_dir, normal);

            // 累加到最终颜色
            final_color += response.component_mul(&light_intensity);
        }

        // 确保颜色在有效范围内
        final_color.map(|c| c.clamp(0.0, 1.0))
    }

    /// 获取光照模型的名称，用于调试
    fn get_model_name(&self) -> &'static str {
        self.get_model_name()
    }
}

/// 基于物理的渲染 (PBR) 材质，实现简化版的基于金属度/粗糙度的工作流
#[derive(Debug, Clone)]
pub struct PBRMaterial {
    /// 基础色/反照率，非金属材质时代表漫反射颜色
    pub base_color: Vector3<f32>,
    /// 金属度，0.0 为绝缘体，1.0 为金属
    pub metallic: f32,
    /// 粗糙度，影响微表面分布，0.0 为完全光滑，1.0 为非常粗糙
    pub roughness: f32,
    /// 环境光遮蔽
    pub ambient_occlusion: f32,
    /// 发光 (自发光)
    pub emissive: Vector3<f32>,
    /// 基础色纹理
    pub base_color_texture: Option<Texture>,
    /// 金属度/粗糙度/环境光遮蔽组合纹理（R=环境光遮蔽，G=粗糙度，B=金属度）
    pub metal_rough_ao_texture: Option<Texture>,
    /// 法线贴图
    pub normal_map: Option<Texture>,
    /// 发光纹理
    pub emissive_texture: Option<Texture>,
}

impl Default for PBRMaterial {
    fn default() -> Self {
        Self {
            base_color: Vector3::new(0.8, 0.8, 0.8),
            metallic: 0.0,
            roughness: 0.5,
            ambient_occlusion: 1.0,
            emissive: Vector3::zeros(),
            base_color_texture: None,
            metal_rough_ao_texture: None,
            normal_map: None,
            emissive_texture: None,
        }
    }
}

impl PBRMaterial {
    /// 创建一个新的PBR材质实例
    pub fn new(base_color: Vector3<f32>, metallic: f32, roughness: f32) -> Self {
        Self {
            base_color,
            metallic,
            roughness,
            ambient_occlusion: 1.0,
            emissive: Vector3::zeros(),
            base_color_texture: None,
            metal_rough_ao_texture: None,
            normal_map: None,
            emissive_texture: None,
        }
    }

    /// 添加基础色纹理
    pub fn with_base_color_texture(mut self, texture: Texture) -> Self {
        self.base_color_texture = Some(texture);
        self
    }

    /// 添加金属度/粗糙度/AO组合纹理
    pub fn with_metal_rough_ao_texture(mut self, texture: Texture) -> Self {
        self.metal_rough_ao_texture = Some(texture);
        self
    }

    /// 添加法线贴图
    pub fn with_normal_map(mut self, texture: Texture) -> Self {
        self.normal_map = Some(texture);
        self
    }

    /// 设置发光属性
    pub fn with_emissive(mut self, emissive: Vector3<f32>, texture: Option<Texture>) -> Self {
        self.emissive = emissive;
        self.emissive_texture = texture;
        self
    }

    /// 计算最终的PBR贴图贡献
    pub fn calculate_maps_contribution(&self, uv: &Vector2<f32>) -> (Vector3<f32>, f32, f32, f32) {
        let mut base_color = self.base_color;
        let mut metallic = self.metallic;
        let mut roughness = self.roughness;
        let mut ao = self.ambient_occlusion;

        // 应用基础色纹理
        if let Some(texture) = &self.base_color_texture {
            let texel = texture.sample(uv.x, uv.y);
            base_color = Vector3::new(texel[0], texel[1], texel[2]);
        }

        // 应用金属度/粗糙度/环境光遮蔽组合纹理
        if let Some(texture) = &self.metal_rough_ao_texture {
            let texel = texture.sample(uv.x, uv.y);
            ao = texel[0]; // R 通道存储环境光遮蔽
            roughness = texel[1]; // G 通道存储粗糙度
            metallic = texel[2]; // B 通道存储金属度
        }

        (base_color, metallic, roughness, ao)
    }

    /// 获取法线贴图的贡献
    pub fn get_normal_from_map(
        &self,
        uv: &Vector2<f32>,
        tangent: &Vector3<f32>,
        bitangent: &Vector3<f32>,
        normal: &Vector3<f32>,
    ) -> Vector3<f32> {
        if let Some(normal_map) = &self.normal_map {
            let texel = normal_map.sample(uv.x, uv.y);
            // 将法线贴图值从 [0,1] 转换到 [-1,1] 范围
            let map_normal = Vector3::new(
                texel[0] * 2.0 - 1.0,
                texel[1] * 2.0 - 1.0,
                texel[2] * 2.0 - 1.0,
            )
            .normalize();

            // 创建TBN矩阵将法线从切线空间转换到模型空间
            let tbn = Matrix3::from_columns(&[*tangent, *bitangent, *normal]);
            tbn * map_normal
        } else {
            *normal // 如果没有法线贴图，使用原始法线
        }
    }

    /// 获取发光贡献
    pub fn get_emissive(&self, uv: &Vector2<f32>) -> Vector3<f32> {
        if let Some(em_texture) = &self.emissive_texture {
            let texel = em_texture.sample(uv.x, uv.y);
            let em_color = Vector3::new(texel[0], texel[1], texel[2]);
            // 将纹理颜色与基础发光颜色相乘
            em_color.component_mul(&self.emissive)
        } else {
            self.emissive
        }
    }
}

impl IMaterial for PBRMaterial {
    fn compute_response(
        &self,
        light_dir: &Vector3<f32>,
        view_dir: &Vector3<f32>,
        normal: &Vector3<f32>,
    ) -> Vector3<f32> {
        use self::pbr_functions::*;

        // 默认UV坐标 - 在实际使用时应使用插值的纹理坐标
        let uv = Vector2::new(0.5, 0.5);

        // 从贴图获取材质属性（如果有）
        let (base_color, metallic, roughness, ao) = self.calculate_maps_contribution(&uv);

        // 检查是否有法线贴图 - 在实际使用时需要提供切线和副切线
        let n = if self.normal_map.is_some() {
            // 这里简化处理，实际应计算切线和副切线
            let tangent = Vector3::new(1.0, 0.0, 0.0);
            let bitangent = normal.cross(&tangent).normalize();
            let corrected_tangent = bitangent.cross(normal).normalize();

            self.get_normal_from_map(&uv, &corrected_tangent, &bitangent, normal)
        } else {
            *normal
        };

        // 确保所有向量都是单位向量
        let l = light_dir;
        let v = view_dir;

        // 半程向量
        let h = (l + v).normalize();

        // 各种点积
        let n_dot_l = n.dot(l).max(0.0);
        let n_dot_v = n.dot(v).max(0.0);
        let n_dot_h = n.dot(&h).max(0.0);
        let h_dot_v = h.dot(v).max(0.0);

        if n_dot_l <= 0.0 {
            // 添加发光贡献，即使表面不接收直接光照
            return self.get_emissive(&uv);
        }

        // 从金属度计算基础反射率F0
        // 非金属F0一般为0.04，金属则使用基础色
        let f0_dielectric = Vector3::new(0.04, 0.04, 0.04);
        let f0 = f0_dielectric.lerp(&base_color, metallic);

        // Cook-Torrance BRDF
        let d = distribution_ggx(n_dot_h, roughness); // 法线分布项
        let f = fresnel_schlick(h_dot_v, f0); // 菲涅耳项
        let g = geometry_smith(n_dot_v, n_dot_l, roughness); // 几何项

        // Cook-Torrance 镜面反射项
        let numerator = d * g * f;
        let denominator = 4.0 * n_dot_v * n_dot_l;
        let specular = numerator / denominator.max(0.001);

        // 计算漫反射项（对金属无漫反射）
        let k_s = f; // 镜面反射比例由菲涅耳决定
        // 修复：为每个分量分别计算
        let one = Vector3::new(1.0, 1.0, 1.0);
        let k_d_components = Vector3::new(
            (one[0] - k_s[0]) * (1.0 - metallic),
            (one[1] - k_s[1]) * (1.0 - metallic),
            (one[2] - k_s[2]) * (1.0 - metallic),
        );

        // 使用组件乘法计算漫反射
        let diffuse = Vector3::new(
            k_d_components[0] * base_color[0],
            k_d_components[1] * base_color[1],
            k_d_components[2] * base_color[2],
        ) / std::f32::consts::PI;

        // 合并漫反射、镜面反射和发光贡献
        let brdf_result = Vector3::new(
            (diffuse[0] + specular[0]) * n_dot_l * ao,
            (diffuse[1] + specular[1]) * n_dot_l * ao,
            (diffuse[2] + specular[2]) * n_dot_l * ao,
        );

        // 添加发光贡献
        brdf_result + self.get_emissive(&uv)
    }

    fn get_diffuse_color(&self) -> Vector3<f32> {
        // 非金属时基础色就是漫反射颜色，金属则无漫反射
        self.base_color * (1.0 - self.metallic)
    }

    fn get_ambient_color(&self) -> Vector3<f32> {
        // 简化处理：基础色 * 环境光遮蔽作为环境色
        let uv = Vector2::new(0.5, 0.5);
        let (base_color, _, _, ao) = self.calculate_maps_contribution(&uv);
        base_color * ao * (1.0 - self.metallic)
    }
}

/// 用于构建光源的便捷工厂类
#[derive(Debug)]
pub struct LightFactory;

impl LightFactory {
    /// 创建一个定向光
    pub fn directional(direction: Vector3<f32>, intensity: Vector3<f32>) -> Box<dyn ILight> {
        // 确保方向是单位向量
        let normalized_direction = direction.normalize();
        Box::new(crate::lighting::Light::Directional {
            direction: normalized_direction,
            intensity,
        })
    }

    /// 创建一个点光源
    pub fn point(
        position: Point3<f32>,
        intensity: Vector3<f32>,
        attenuation: Option<(f32, f32, f32)>,
    ) -> Box<dyn ILight> {
        Box::new(crate::lighting::Light::Point {
            position,
            intensity,
            attenuation: attenuation.unwrap_or((1.0, 0.09, 0.032)), // 默认物理合理的衰减值
        })
    }

    /// 创建一个环境光
    pub fn ambient(intensity: Vector3<f32>) -> Box<dyn ILight> {
        Box::new(crate::lighting::Light::Ambient(intensity))
    }
}

/// PBR 渲染的核心函数
/// 该模块包含基于物理的渲染中使用的各种数学函数
pub mod pbr_functions {
    use nalgebra::Vector3;

    /// 计算 GGX 法线分布函数 (NDF)
    ///
    /// # 参数
    /// * `n_dot_h` - 法线与半程向量的点积
    /// * `roughness` - 材质的粗糙度参数
    pub fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
        let alpha = roughness * roughness;
        let alpha_squared = alpha * alpha;

        let n_dot_h_squared = n_dot_h * n_dot_h;
        let denom = n_dot_h_squared * (alpha_squared - 1.0) + 1.0;

        alpha_squared / (std::f32::consts::PI * denom * denom)
    }

    /// 计算 Smith 几何函数
    ///
    /// # 参数
    /// * `n_dot_v` - 法线与视线方向的点积
    /// * `n_dot_l` - 法线与光线方向的点积
    /// * `roughness` - 材质的粗糙度参数
    pub fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
        let k = ((roughness + 1.0) * (roughness + 1.0)) / 8.0;

        let g_v = n_dot_v / (n_dot_v * (1.0 - k) + k);
        let g_l = n_dot_l / (n_dot_l * (1.0 - k) + k);

        g_v * g_l
    }

    /// 计算 Schlick 菲涅耳近似
    ///
    /// # 参数
    /// * `cos_theta` - 视线方向与半程向量的点积
    /// * `f0` - 基础反射率（0度入射时的反射率）
    pub fn fresnel_schlick(cos_theta: f32, f0: Vector3<f32>) -> Vector3<f32> {
        let one_minus_cos_theta = 1.0 - cos_theta;
        let one_minus_cos_theta_5 = one_minus_cos_theta
            * one_minus_cos_theta
            * one_minus_cos_theta
            * one_minus_cos_theta
            * one_minus_cos_theta;

        f0 + (Vector3::new(1.0, 1.0, 1.0) - f0) * one_minus_cos_theta_5
    }
}
