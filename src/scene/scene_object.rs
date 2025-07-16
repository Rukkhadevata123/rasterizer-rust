use crate::geometry::transform::TransformFactory;
use crate::material_system::materials::Model;
use nalgebra::{Matrix4, Vector3};

/// 表示场景中的单个对象实例
///
/// 包含几何数据（模型）和变换信息，是渲染器的基本单位
#[derive(Debug, Clone)]
pub struct SceneObject {
    /// 对象的几何数据（网格、材质等）
    pub model: Model,

    /// 对象在世界空间中的变换矩阵
    pub transform: Matrix4<f32>,
}

impl SceneObject {
    /// 从模型数据创建新的场景对象
    pub fn from_model_data(model: Model) -> Self {
        Self {
            model,
            transform: Matrix4::identity(),
        }
    }

    /// 创建空的场景对象（用于测试或占位）
    pub fn empty(name: &str) -> Self {
        Self {
            model: Model {
                meshes: Vec::new(),
                materials: Vec::new(),
                name: name.to_string(),
            },
            transform: Matrix4::identity(),
        }
    }

    /// 设置完整变换（从组件构建变换矩阵）
    pub fn set_transform_from_components(
        &mut self,
        position: Vector3<f32>,
        rotation_rad: Vector3<f32>,
        scale: Vector3<f32>,
    ) {
        // 按正确顺序组合变换：缩放 -> 旋转 -> 平移
        let scale_matrix = TransformFactory::scaling_nonuniform(&scale);
        let rotation_x_matrix = TransformFactory::rotation_x(rotation_rad.x);
        let rotation_y_matrix = TransformFactory::rotation_y(rotation_rad.y);
        let rotation_z_matrix = TransformFactory::rotation_z(rotation_rad.z);
        let translation_matrix = TransformFactory::translation(&position);

        // 组合变换矩阵：T * Rz * Ry * Rx * S
        self.transform = translation_matrix
            * rotation_z_matrix
            * rotation_y_matrix
            * rotation_x_matrix
            * scale_matrix;
    }

    /// 应用增量旋转（用于动画）
    pub fn rotate(&mut self, axis: &Vector3<f32>, angle_rad: f32) {
        let rotation_matrix = TransformFactory::rotation(axis, angle_rad);
        self.transform = rotation_matrix * self.transform;
    }
}

impl Default for SceneObject {
    fn default() -> Self {
        Self::empty("Default")
    }
}
