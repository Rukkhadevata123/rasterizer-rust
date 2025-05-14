use nalgebra::{Matrix4, Point3, Rotation3, Unit, UnitQuaternion, Vector3};

/// 表示场景中的一个对象，包含模型ID和变换矩阵
#[derive(Debug, Clone)]
pub struct SceneObject {
    /// 模型数据的标识符（例如，Vec<ModelData> 中的索引）
    pub model_id: usize,
    /// 该对象在世界空间中的变换矩阵
    pub transform: Matrix4<f32>,
    /// 可选的材质ID（覆盖模型默认材质）
    pub material_id: Option<usize>,
}

impl SceneObject {
    /// 创建一个新的场景对象，使用组合变换矩阵
    ///
    /// # 参数
    /// * `model_id` - 模型数据的标识符
    /// * `position` - 对象的位置
    /// * `euler_angles_rad` - 以欧拉角（roll, pitch, yaw）表示的旋转，单位为弧度
    /// * `scale` - x、y、z轴的缩放因子
    pub fn new(
        model_id: usize,
        position: Point3<f32>,
        euler_angles_rad: Vector3<f32>,
        scale: Vector3<f32>,
    ) -> Self {
        let rotation = UnitQuaternion::from_euler_angles(
            euler_angles_rad.x,
            euler_angles_rad.y,
            euler_angles_rad.z,
        );
        // 创建变换: 缩放 -> 旋转 -> 平移
        let transform = Matrix4::new_translation(&position.coords)
            * Matrix4::from(rotation)
            * Matrix4::new_nonuniform_scaling(&scale);
        SceneObject {
            model_id,
            transform,
            material_id: None, // 默认无材质ID
        }
    }

    /// 创建一个带有单位变换的新场景对象
    pub fn new_default(model_id: usize) -> Self {
        SceneObject {
            model_id,
            transform: Matrix4::identity(),
            material_id: None, // 默认无材质ID
        }
    }

    /// 设置对象的位置，更新变换矩阵
    /// 直接设置矩阵的平移部分，保留现有的旋转和缩放
    pub fn set_position(&mut self, position: Point3<f32>) {
        self.transform.m14 = position.x;
        self.transform.m24 = position.y;
        self.transform.m34 = position.z;
        // 假设 m44 为 1，这是仿射变换矩阵的标准
    }

    /// 按给定向量平移对象
    /// 相对于当前方向和位置应用平移
    pub fn translate(&mut self, translation_vector: &Vector3<f32>) {
        let translation_matrix = Matrix4::new_translation(translation_vector);
        self.transform = translation_matrix * self.transform;
    }

    /// 围绕给定轴旋转对象（角度以弧度为单位）
    /// 在对象的局部坐标系中应用旋转
    pub fn rotate_local(&mut self, axis: &Vector3<f32>, angle_rad: f32) {
        // 使用轴角表示法创建旋转矩阵
        let axis_unit = Unit::new_normalize(*axis);
        let rotation = Rotation3::from_axis_angle(&axis_unit, angle_rad);
        let rotation_matrix = Matrix4::from(rotation);

        self.transform *= rotation_matrix; // 后乘以实现局部旋转
    }

    /// 围绕给定轴旋转对象（角度以弧度为单位）
    /// 在世界坐标系中应用旋转
    pub fn rotate_global(&mut self, axis: &Vector3<f32>, angle_rad: f32) {
        // 获取当前变换矩阵
        let rotation_quat =
            UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(*axis), angle_rad);
        let rotation_matrix = Matrix4::from(rotation_quat);
        self.transform = rotation_matrix * self.transform; // 前乘以实现全局旋转
    }

    /// 在局部坐标系中缩放对象
    pub fn scale_local(&mut self, scale_vector: &Vector3<f32>) {
        let scaling_matrix = Matrix4::new_nonuniform_scaling(scale_vector);
        self.transform *= scaling_matrix; // 后乘以实现局部缩放
    }

    /// 创建一个具有指定模型变换的新场景对象
    pub fn with_transform(
        model_id: usize,
        transform: Matrix4<f32>,
        material_id: Option<usize>,
    ) -> Self {
        SceneObject {
            model_id,
            transform,
            material_id,
        }
    }
}

/// 为 SceneObject 实现标准的 Default trait
impl Default for SceneObject {
    /// 创建一个位于世界坐标原点的默认场景对象
    fn default() -> Self {
        Self {
            model_id: 0,
            transform: Matrix4::identity(),
            material_id: None,
        }
    }
}
