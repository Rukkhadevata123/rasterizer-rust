use crate::geometry::transform::TransformFactory;
use nalgebra::{Matrix4, Point3, Vector3};
use std::cell::Cell;

/// 投影类型枚举，提供类型安全的投影方式选择
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectionType {
    Perspective {
        fov_y_degrees: f32,
        aspect_ratio: f32,
    },
    Orthographic {
        width: f32,
        height: f32,
    },
}

impl ProjectionType {
    /// 获取宽高比
    pub fn aspect_ratio(&self) -> f32 {
        match self {
            ProjectionType::Perspective { aspect_ratio, .. } => *aspect_ratio,
            ProjectionType::Orthographic { width, height } => width / height,
        }
    }

    /// 判断是否为透视投影
    pub fn is_perspective(&self) -> bool {
        matches!(self, ProjectionType::Perspective { .. })
    }
}

/// 相机参数结构体，包含所有相机配置信息
#[derive(Debug, Clone)]
pub struct CameraParams {
    pub position: Point3<f32>,
    pub target: Point3<f32>,
    pub up: Vector3<f32>,
    pub projection: ProjectionType,
    pub near: f32,
    pub far: f32,
}

impl Default for CameraParams {
    fn default() -> Self {
        Self {
            position: Point3::new(0.0, 0.0, 3.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            projection: ProjectionType::Perspective {
                fov_y_degrees: 45.0,
                aspect_ratio: 1.0,
            },
            near: 0.1,
            far: 100.0,
        }
    }
}

/// 相机类，负责管理视角和投影变换
/// 使用内部可变性解决矩阵懒更新问题
#[derive(Debug)]
pub struct Camera {
    // 基本参数
    params: CameraParams,

    // 计算得出的基向量
    right: Cell<Vector3<f32>>,
    forward: Cell<Vector3<f32>>,

    // 缓存的变换矩阵
    view_matrix: Cell<Matrix4<f32>>,
    projection_matrix: Cell<Matrix4<f32>>,
    view_projection_matrix: Cell<Matrix4<f32>>,

    // 矩阵是否需要更新的标志
    matrices_dirty: Cell<bool>,
}

impl Camera {
    /// 使用参数结构体创建相机
    pub fn new(params: CameraParams) -> Self {
        let camera = Camera {
            params,
            right: Cell::new(Vector3::x()),
            forward: Cell::new(Vector3::z()),
            view_matrix: Cell::new(Matrix4::identity()),
            projection_matrix: Cell::new(Matrix4::identity()),
            view_projection_matrix: Cell::new(Matrix4::identity()),
            matrices_dirty: Cell::new(true),
        };
        camera.update_matrices_internal();
        camera
    }

    /// 创建透视投影相机的便捷方法
    pub fn perspective(
        position: Point3<f32>,
        target: Point3<f32>,
        up: Vector3<f32>,
        fov_y_degrees: f32,
        aspect_ratio: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let params = CameraParams {
            position,
            target,
            up: up.normalize(),
            projection: ProjectionType::Perspective {
                fov_y_degrees,
                aspect_ratio,
            },
            near,
            far,
        };
        Self::new(params)
    }

    /// 创建正交投影相机的便捷方法
    pub fn orthographic(
        position: Point3<f32>,
        target: Point3<f32>,
        up: Vector3<f32>,
        width: f32,
        height: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let params = CameraParams {
            position,
            target,
            up: up.normalize(),
            projection: ProjectionType::Orthographic { width, height },
            near,
            far,
        };
        Self::new(params)
    }

    // ============ 参数访问和修改方法 ============

    /// 获取相机参数的只读引用
    pub fn params(&self) -> &CameraParams {
        &self.params
    }

    /// 获取相机位置
    pub fn position(&self) -> Point3<f32> {
        self.params.position
    }

    /// 获取相机目标
    pub fn target(&self) -> Point3<f32> {
        self.params.target
    }

    /// 获取上方向
    pub fn up(&self) -> Vector3<f32> {
        self.params.up
    }

    /// 获取右方向
    pub fn right(&self) -> Vector3<f32> {
        self.ensure_matrices_updated();
        self.right.get()
    }

    /// 获取前方向
    pub fn forward(&self) -> Vector3<f32> {
        self.ensure_matrices_updated();
        self.forward.get()
    }

    /// 获取宽高比
    pub fn aspect_ratio(&self) -> f32 {
        self.params.projection.aspect_ratio()
    }

    /// 获取近裁剪面
    pub fn near(&self) -> f32 {
        self.params.near
    }

    /// 获取远裁剪面
    pub fn far(&self) -> f32 {
        self.params.far
    }

    /// 设置相机位置
    pub fn set_position(&mut self, position: Point3<f32>) {
        self.params.position = position;
        self.mark_dirty();
    }

    /// 设置相机目标
    pub fn set_target(&mut self, target: Point3<f32>) {
        self.params.target = target;
        self.mark_dirty();
    }

    /// 设置上方向
    pub fn set_up(&mut self, up: Vector3<f32>) {
        self.params.up = up.normalize();
        self.mark_dirty();
    }

    /// 设置投影类型
    pub fn set_projection(&mut self, projection: ProjectionType) {
        self.params.projection = projection;
        self.mark_dirty();
    }

    /// 设置近远裁剪平面
    pub fn set_clipping_planes(&mut self, near: f32, far: f32) {
        self.params.near = near;
        self.params.far = far;
        self.mark_dirty();
    }

    /// 批量更新相机参数（减少重复计算）
    pub fn update_params<F>(&mut self, updater: F)
    where
        F: FnOnce(&mut CameraParams),
    {
        updater(&mut self.params);
        self.mark_dirty();
    }

    // ============ 矩阵访问方法 ============

    /// 获取视图矩阵
    pub fn view_matrix(&self) -> Matrix4<f32> {
        self.ensure_matrices_updated();
        self.view_matrix.get()
    }

    /// 获取投影矩阵
    pub fn projection_matrix(&self) -> Matrix4<f32> {
        self.ensure_matrices_updated();
        self.projection_matrix.get()
    }

    /// 获取视图-投影矩阵
    pub fn view_projection_matrix(&self) -> Matrix4<f32> {
        self.ensure_matrices_updated();
        self.view_projection_matrix.get()
    }

    // ============ 相机运动方法 ============

    /// 围绕目标点进行任意轴旋转
    pub fn orbit(&mut self, axis: &Vector3<f32>, angle_rad: f32) {
        let camera_to_target = self.params.position - self.params.target;
        let rotation_matrix = TransformFactory::rotation(axis, angle_rad);
        let rotated_vector = rotation_matrix.transform_vector(&camera_to_target);
        self.params.position = self.params.target + rotated_vector;
        self.mark_dirty();
    }

    /// 在XZ平面上平移相机
    pub fn pan(&mut self, right_amount: f32, forward_amount: f32) {
        self.ensure_matrices_updated();

        let forward_xz = Vector3::new(self.forward.get().x, 0.0, self.forward.get().z).normalize();
        let right_xz = Vector3::new(self.right.get().x, 0.0, self.right.get().z).normalize();
        let translation = right_xz * right_amount + forward_xz * forward_amount;

        self.params.position += translation;
        self.params.target += translation;
        self.mark_dirty();
    }

    /// 相机沿视线方向移动
    pub fn dolly(&mut self, amount: f32) {
        let direction = (self.params.target - self.params.position).normalize();
        let translation = direction * amount;
        self.params.position += translation;
        self.mark_dirty();
    }

    /// 移动相机（同时移动位置和目标点）
    pub fn translate(&mut self, translation: &Vector3<f32>) {
        self.params.position += translation;
        self.params.target += translation;
        self.mark_dirty();
    }

    // ============ 内部实现方法 ============

    /// 标记矩阵需要更新
    fn mark_dirty(&self) {
        self.matrices_dirty.set(true);
    }

    /// 确保矩阵是最新的（支持内部可变性）
    fn ensure_matrices_updated(&self) {
        if self.matrices_dirty.get() {
            self.update_matrices_internal();
        }
    }

    /// 手动触发矩阵更新（公共接口）
    pub fn update_matrices(&mut self) {
        self.update_matrices_internal();
    }

    /// 内部矩阵更新实现
    fn update_matrices_internal(&self) {
        if self.matrices_dirty.get() {
            self.update_basis_vectors();
            self.update_view_matrix();
            self.update_projection_matrix();
            self.update_view_projection_matrix();
            self.matrices_dirty.set(false);
        }
    }

    /// 更新相机基向量
    fn update_basis_vectors(&self) {
        let forward = (self.params.target - self.params.position).normalize();
        let right = forward.cross(&self.params.up).normalize();

        self.forward.set(forward);
        self.right.set(right);
    }

    /// 更新视图矩阵 - 使用 TransformFactory
    fn update_view_matrix(&self) {
        let view_matrix =
            TransformFactory::view(&self.params.position, &self.params.target, &self.params.up);
        self.view_matrix.set(view_matrix);
    }

    /// 更新投影矩阵 - 使用 TransformFactory
    fn update_projection_matrix(&self) {
        let projection_matrix = match &self.params.projection {
            ProjectionType::Perspective {
                fov_y_degrees,
                aspect_ratio,
            } => TransformFactory::perspective(
                *aspect_ratio,
                fov_y_degrees.to_radians(),
                self.params.near,
                self.params.far,
            ),
            ProjectionType::Orthographic { width, height } => TransformFactory::orthographic(
                -width / 2.0,
                width / 2.0,
                -height / 2.0,
                height / 2.0,
                self.params.near,
                self.params.far,
            ),
        };
        self.projection_matrix.set(projection_matrix);
    }

    /// 更新视图-投影矩阵 - 使用 TransformFactory
    fn update_view_projection_matrix(&self) {
        let view_projection_matrix = TransformFactory::model_view_projection(
            &Matrix4::identity(), // 单位矩阵作为模型矩阵
            &self.view_matrix.get(),
            &self.projection_matrix.get(),
        );
        self.view_projection_matrix.set(view_projection_matrix);
    }
}

impl Clone for Camera {
    fn clone(&self) -> Self {
        Camera {
            params: self.params.clone(),
            right: Cell::new(self.right.get()),
            forward: Cell::new(self.forward.get()),
            view_matrix: Cell::new(self.view_matrix.get()),
            projection_matrix: Cell::new(self.projection_matrix.get()),
            view_projection_matrix: Cell::new(self.view_projection_matrix.get()),
            matrices_dirty: Cell::new(self.matrices_dirty.get()),
        }
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new(CameraParams::default())
    }
}
