use crate::geometry::transform::TransformFactory;
// use crate::geometry::transform::direct_transform; // 移除 direct_transform 的使用
use nalgebra::{Matrix4, Point3, Vector3};

/// 可变换对象特性，定义了对象变换的标准接口和便捷方法
pub trait Transformable {
    /// 获取对象的变换矩阵
    fn get_transform(&self) -> &Matrix4<f32>;

    /// 设置对象的变换矩阵
    fn set_transform(&mut self, transform: Matrix4<f32>);

    /// 在局部坐标系中应用变换矩阵
    fn apply_local(&mut self, transform: Matrix4<f32>);

    /// 在全局坐标系中应用变换矩阵
    fn apply_global(&mut self, transform: Matrix4<f32>);

    /// 在全局坐标系中平移对象
    fn translate(&mut self, translation: &Vector3<f32>) {
        let mut current_transform = *self.get_transform();
        // 直接修改平移分量实现全局平移
        current_transform.m14 += translation.x;
        current_transform.m24 += translation.y;
        current_transform.m34 += translation.z;
        self.set_transform(current_transform);
    }

    /// 在局部坐标系中平移对象
    fn translate_local(&mut self, translation: &Vector3<f32>) {
        let translation_matrix = TransformFactory::translation(translation);
        self.apply_local(translation_matrix);
    }

    /// 在局部坐标系中旋转对象
    fn rotate(&mut self, axis: &Vector3<f32>, angle_rad: f32) {
        let rotation_matrix = TransformFactory::rotation(axis, angle_rad);
        self.apply_local(rotation_matrix);
    }

    /// 绕全局X轴旋转
    fn rotate_x(&mut self, angle_rad: f32) {
        self.apply_global(TransformFactory::rotation_x(angle_rad));
    }

    /// 绕全局Y轴旋转
    fn rotate_y(&mut self, angle_rad: f32) {
        self.apply_global(TransformFactory::rotation_y(angle_rad));
    }

    /// 绕全局Z轴旋转
    fn rotate_z(&mut self, angle_rad: f32) {
        self.apply_global(TransformFactory::rotation_z(angle_rad));
    }

    /// 沿全局轴均匀缩放对象的几何形状（不包括平移）
    fn scale(&mut self, scale_factor: f32) {
        // M_new = S * M
        // 这会缩放旋转/缩放部分和平移部分。
        // 如果只想缩放几何体本身，通常是局部缩放或更复杂的处理。
        // direct_transform::apply_scale 的行为是 M_3x3_new = S_3x3 * M_3x3_old
        // 这相当于对整个矩阵左乘一个缩放矩阵 S = diag(s,s,s,1)
        self.apply_global(TransformFactory::scaling(scale_factor));
    }

    /// 沿局部轴非均匀缩放对象
    fn scale_xyz(&mut self, scale_vector: &Vector3<f32>) {
        // M_new = M * S_local
        // direct_transform::apply_scale_xyz 的行为是 M_new_cols = M_cols * diag(sx,sy,sz)
        // 这相当于 M_new = M_old * S_local
        self.apply_local(TransformFactory::scaling_nonuniform(scale_vector));
    }

    /// 设置对象位置
    fn set_position(&mut self, position: Point3<f32>) {
        let mut transform = *self.get_transform();
        transform.m14 = position.x;
        transform.m24 = position.y;
        transform.m34 = position.z;
        self.set_transform(transform);
    }

    /// 获取对象当前位置
    fn get_position(&self) -> Point3<f32> {
        // direct_transform::extract_position(self.get_transform())
        Point3::new(
            self.get_transform().m14,
            self.get_transform().m24,
            self.get_transform().m34,
        )
    }
}

/// 表示场景中的一个对象，包含模型ID和变换矩阵
#[derive(Debug, Clone)]
pub struct SceneObject {
    /// 模型数据的标识符
    pub model_id: usize,
    /// 对象在世界空间中的变换矩阵
    pub transform: Matrix4<f32>,
    /// 可选的材质ID（覆盖模型默认材质）
    pub material_id: Option<usize>,
    /// 对象名称
    pub name: Option<String>,
}

impl SceneObject {
    /// 创建一个新的场景对象
    pub fn new(model_id: usize) -> Self {
        Self {
            model_id,
            transform: Matrix4::identity(),
            material_id: None,
            name: None,
        }
    }

    /// 使用指定变换创建对象
    pub fn with_transform(mut self, transform: Matrix4<f32>) -> Self {
        self.transform = transform;
        self
    }

    /// 使用指定位置创建对象
    pub fn with_position(mut self, position: Point3<f32>) -> Self {
        self.transform.m14 = position.x;
        self.transform.m24 = position.y;
        self.transform.m34 = position.z;
        self
    }

    /// 使用位置、旋转和缩放创建对象
    pub fn with_transform_components(
        mut self,
        position: Point3<f32>,
        euler_angles_rad: Vector3<f32>, // 假设为绕X, Y, Z轴的局部旋转顺序
        scale: Vector3<f32>,
    ) -> Self {
        let s_matrix = TransformFactory::scaling_nonuniform(&scale);
        let rx_matrix = TransformFactory::rotation_x(euler_angles_rad.x);
        let ry_matrix = TransformFactory::rotation_y(euler_angles_rad.y);
        let rz_matrix = TransformFactory::rotation_z(euler_angles_rad.z);
        let t_matrix = TransformFactory::translation(&position.coords);

        // 标准的变换顺序：Scale (local) -> Rotate (local Z, then Y, then X) -> Translate (world)
        // M = T * R_x * R_y * R_z * S
        // 或者 T * R_z * R_y * R_x * S (如果欧拉角是按 ZYX 顺序应用的 intrinsic 旋转)
        // direct_transform 的顺序是 S -> Rx -> Ry -> Rz -> T (effectively T_world * R_local_z * R_local_y * R_local_x * S_local)
        self.transform = t_matrix * rz_matrix * ry_matrix * rx_matrix * s_matrix;
        self
    }

    /// 添加材质重写
    pub fn with_material(mut self, material_id: usize) -> Self {
        self.material_id = Some(material_id);
        self
    }

    /// 添加名称
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

impl Transformable for SceneObject {
    fn get_transform(&self) -> &Matrix4<f32> {
        &self.transform
    }

    fn set_transform(&mut self, transform: Matrix4<f32>) {
        self.transform = transform;
    }

    fn apply_local(&mut self, transform_matrix: Matrix4<f32>) {
        // 局部变换：后乘 M_new = M_old * T_local
        self.transform *= transform_matrix;
    }

    fn apply_global(&mut self, transform_matrix: Matrix4<f32>) {
        // 全局变换：前乘 M_new = T_global * M_old
        self.transform = transform_matrix * self.transform;
    }
}

impl Default for SceneObject {
    fn default() -> Self {
        Self::new(0)
    }
}
