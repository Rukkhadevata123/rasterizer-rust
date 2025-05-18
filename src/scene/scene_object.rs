use crate::geometry::transform::TransformFactory;
use nalgebra::{Matrix4, Point3, UnitQuaternion, Vector3};

/// 可变换对象特性，定义了对象变换的标准接口
pub trait Transformable {
    /// 获取对象的变换矩阵
    fn get_transform(&self) -> &Matrix4<f32>;

    /// 设置对象的变换矩阵
    fn set_transform(&mut self, transform: Matrix4<f32>);
    /// 在局部坐标系中应用变换矩阵
    fn apply_local(&mut self, transform: Matrix4<f32>);
    /// 在全局坐标系中应用变换矩阵
    fn apply_global(&mut self, transform: Matrix4<f32>);
}

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
    pub fn set_position(&mut self, position: Point3<f32>) {
        // 保留当前矩阵的旋转和缩放部分，只更新平移部分
        self.transform.m14 = position.x;
        self.transform.m24 = position.y;
        self.transform.m34 = position.z;
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

impl Transformable for SceneObject {
    fn get_transform(&self) -> &Matrix4<f32> {
        &self.transform
    }

    fn set_transform(&mut self, transform: Matrix4<f32>) {
        self.transform = transform;
    }

    fn apply_local(&mut self, transform: Matrix4<f32>) {
        // 局部变换：后乘
        self.transform *= transform;
    }

    fn apply_global(&mut self, transform: Matrix4<f32>) {
        // 全局变换：前乘
        self.transform = transform * self.transform;
    }
}

/// 提供标准变换操作的辅助扩展特性
pub trait TransformOperations: Transformable {
    /// 在局部坐标系中平移对象
    fn translate_local(&mut self, translation: &Vector3<f32>) {
        let translation_matrix = TransformFactory::translation(translation);
        self.apply_local(translation_matrix);
    }

    /// 在全局坐标系中平移对象
    fn translate_global(&mut self, translation: &Vector3<f32>) {
        let translation_matrix = TransformFactory::translation(translation);
        self.apply_global(translation_matrix);
    }

    /// 在全局坐标系中平移对象（translate的简化版本）
    fn translate(&mut self, translation: &Vector3<f32>) {
        self.translate_global(translation);
    }

    /// 在局部坐标系中旋转对象
    fn rotate_local(&mut self, axis: &Vector3<f32>, angle_rad: f32) {
        let rotation_matrix = TransformFactory::rotation(axis, angle_rad);
        self.apply_local(rotation_matrix);
    }

    /// 在全局坐标系中旋转对象
    fn rotate_global(&mut self, axis: &Vector3<f32>, angle_rad: f32) {
        let rotation_matrix = TransformFactory::rotation(axis, angle_rad);
        self.apply_global(rotation_matrix);
    }

    /// 局部X轴旋转
    fn rotate_local_x(&mut self, angle_rad: f32) {
        let rotation_matrix = TransformFactory::rotation_x(angle_rad);
        self.apply_local(rotation_matrix);
    }

    /// 局部Y轴旋转
    fn rotate_local_y(&mut self, angle_rad: f32) {
        let rotation_matrix = TransformFactory::rotation_y(angle_rad);
        self.apply_local(rotation_matrix);
    }

    /// 局部Z轴旋转
    fn rotate_local_z(&mut self, angle_rad: f32) {
        let rotation_matrix = TransformFactory::rotation_z(angle_rad);
        self.apply_local(rotation_matrix);
    }

    /// 全局X轴旋转
    fn rotate_global_x(&mut self, angle_rad: f32) {
        let rotation_matrix = TransformFactory::rotation_x(angle_rad);
        self.apply_global(rotation_matrix);
    }

    /// 全局Y轴旋转
    fn rotate_global_y(&mut self, angle_rad: f32) {
        let rotation_matrix = TransformFactory::rotation_y(angle_rad);
        self.apply_global(rotation_matrix);
    }

    /// 全局Z轴旋转
    fn rotate_global_z(&mut self, angle_rad: f32) {
        let rotation_matrix = TransformFactory::rotation_z(angle_rad);
        self.apply_global(rotation_matrix);
    }

    /// 在局部坐标系中缩放对象（非均匀缩放）
    fn scale_local(&mut self, scale: &Vector3<f32>) {
        let scaling_matrix = TransformFactory::scaling_nonuniform(scale);
        self.apply_local(scaling_matrix);
    }

    /// 在全局坐标系中缩放对象（非均匀缩放）
    fn scale_global(&mut self, scale: &Vector3<f32>) {
        let scaling_matrix = TransformFactory::scaling_nonuniform(scale);
        self.apply_global(scaling_matrix);
    }

    /// 在局部坐标系中均匀缩放对象
    fn scale_local_uniform(&mut self, scale: f32) {
        let scaling_matrix = TransformFactory::scaling(scale);
        self.apply_local(scaling_matrix);
    }

    /// 在全局坐标系中均匀缩放对象
    fn scale_global_uniform(&mut self, scale: f32) {
        let scaling_matrix = TransformFactory::scaling(scale);
        self.apply_global(scaling_matrix);
    }
}

// 为SceneObject实现TransformOperations特性
impl TransformOperations for SceneObject {}

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
