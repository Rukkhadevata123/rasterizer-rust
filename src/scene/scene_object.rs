use crate::geometry::transform::TransformFactory;
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

    /// 在全局坐标系中平移对象 - 使用 TransformFactory
    fn translate(&mut self, translation: &Vector3<f32>) {
        // 分步操作避免借用冲突
        let current_translation = TransformFactory::extract_translation(self.get_transform());
        let new_translation = current_translation + translation;
        let mut new_transform = *self.get_transform();
        TransformFactory::set_translation(&mut new_transform, &new_translation);
        self.set_transform(new_transform);
    }

    /// 在局部坐标系中平移对象 - 使用 TransformFactory
    fn translate_local(&mut self, translation: &Vector3<f32>) {
        let translation_matrix = TransformFactory::translation(translation);
        self.apply_local(translation_matrix);
    }

    /// 在局部坐标系中旋转对象 - 使用 TransformFactory
    fn rotate(&mut self, axis: &Vector3<f32>, angle_rad: f32) {
        let rotation_matrix = TransformFactory::rotation(axis, angle_rad);
        self.apply_local(rotation_matrix);
    }

    /// 绕全局X轴旋转 - 使用 TransformFactory
    fn rotate_x(&mut self, angle_rad: f32) {
        self.apply_global(TransformFactory::rotation_x(angle_rad));
    }

    /// 绕全局Y轴旋转 - 使用 TransformFactory
    fn rotate_y(&mut self, angle_rad: f32) {
        self.apply_global(TransformFactory::rotation_y(angle_rad));
    }

    /// 绕全局Z轴旋转 - 使用 TransformFactory
    fn rotate_z(&mut self, angle_rad: f32) {
        self.apply_global(TransformFactory::rotation_z(angle_rad));
    }

    /// 沿全局轴均匀缩放对象 - 使用 TransformFactory
    fn scale(&mut self, scale_factor: f32) {
        self.apply_global(TransformFactory::scaling(scale_factor));
    }

    /// 沿局部轴非均匀缩放对象 - 使用 TransformFactory
    fn scale_xyz(&mut self, scale_vector: &Vector3<f32>) {
        self.apply_local(TransformFactory::scaling_nonuniform(scale_vector));
    }

    /// 设置对象位置 - 使用 TransformFactory
    fn set_position(&mut self, position: Point3<f32>) {
        let mut transform = *self.get_transform();
        TransformFactory::set_translation(&mut transform, &position.coords);
        self.set_transform(transform);
    }

    /// 获取对象当前位置 - 使用 TransformFactory
    fn get_position(&self) -> Point3<f32> {
        TransformFactory::extract_position(self.get_transform())
    }

    /// 高级变换方法：组合平移和旋转
    fn transform_by(
        &mut self,
        translation: &Vector3<f32>,
        rotation_axis: &Vector3<f32>,
        rotation_rad: f32,
    ) {
        let translation_matrix = TransformFactory::translation(translation);
        let rotation_matrix = TransformFactory::rotation(rotation_axis, rotation_rad);
        // 先平移，再旋转
        let combined_transform = rotation_matrix * translation_matrix;
        self.apply_global(combined_transform);
    }

    /// 围绕指定点旋转
    fn rotate_around_point(&mut self, pivot: &Point3<f32>, axis: &Vector3<f32>, angle_rad: f32) {
        // 1. 移动到原点（相对于pivot）
        let to_origin = TransformFactory::translation(&(-pivot.coords));
        // 2. 执行旋转
        let rotation = TransformFactory::rotation(axis, angle_rad);
        // 3. 移回原位置
        let from_origin = TransformFactory::translation(&pivot.coords);

        // 组合变换：T * R * T^-1
        let combined_transform = from_origin * rotation * to_origin;
        self.apply_global(combined_transform);
    }

    /// 朝向指定目标（只旋转Y轴，保持对象"站立"）
    fn look_at(&mut self, target: &Point3<f32>) {
        let current_pos = self.get_position();
        let direction = (target - current_pos).normalize();

        // 计算Y轴旋转角度
        let angle_y = direction.z.atan2(direction.x);

        // 提取当前变换的平移、缩放部分，重新构建带新旋转的矩阵
        let current_translation = TransformFactory::extract_translation(self.get_transform());
        let translation_matrix = TransformFactory::translation(&current_translation);
        let rotation_matrix = TransformFactory::rotation_y(angle_y);

        // 注意：这里假设没有缩放，如果需要保持缩放需要额外提取缩放信息
        let new_transform = translation_matrix * rotation_matrix;
        self.set_transform(new_transform);
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

    /// 使用指定位置创建对象 - 使用 TransformFactory
    pub fn with_position(mut self, position: Point3<f32>) -> Self {
        self.transform = TransformFactory::position_matrix(position);
        self
    }

    /// 使用位置、旋转和缩放创建对象 - 使用 TransformFactory
    pub fn with_transform_components(
        mut self,
        position: Point3<f32>,
        euler_angles_rad: Vector3<f32>,
        scale: Vector3<f32>,
    ) -> Self {
        // 使用 TransformFactory 构建复合变换
        let s_matrix = TransformFactory::scaling_nonuniform(&scale);
        let rx_matrix = TransformFactory::rotation_x(euler_angles_rad.x);
        let ry_matrix = TransformFactory::rotation_y(euler_angles_rad.y);
        let rz_matrix = TransformFactory::rotation_z(euler_angles_rad.z);
        let t_matrix = TransformFactory::translation(&position.coords);

        // 标准的变换顺序：T * Rz * Ry * Rx * S
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

    /// 重置为默认变换（单位矩阵）
    pub fn reset_transform(&mut self) {
        self.transform = Matrix4::identity();
    }

    /// 获取对象的边界球心（基于变换矩阵的位置）
    pub fn get_bounding_sphere_center(&self) -> Point3<f32> {
        self.get_position()
    }

    /// 克隆对象并应用新的变换
    pub fn clone_with_transform(&self, new_transform: Matrix4<f32>) -> Self {
        let mut cloned = self.clone();
        cloned.transform = new_transform;
        cloned
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
