use crate::geometry::transform::TransformFactory;
use nalgebra::{Matrix4, Point3, Vector3};

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

/// 简化的相机类
#[derive(Debug, Clone)]
pub struct Camera {
    pub params: CameraParams,

    // 预计算的矩阵 - 每次创建时计算一次
    view_matrix: Matrix4<f32>,
    projection_matrix: Matrix4<f32>,
}

impl Camera {
    /// 使用参数结构体创建相机
    pub fn new(params: CameraParams) -> Self {
        let mut camera = Camera {
            params,
            view_matrix: Matrix4::identity(),
            projection_matrix: Matrix4::identity(),
        };
        camera.update_matrices();
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

    // ============ 基本访问方法 ============

    /// 获取相机位置
    pub fn position(&self) -> Point3<f32> {
        self.params.position
    }

    /// 获取宽高比
    pub fn aspect_ratio(&self) -> f32 {
        self.params.projection.aspect_ratio()
    }

    /// 获取远裁剪面
    pub fn far(&self) -> f32 {
        self.params.far
    }

    /// 获取近裁剪面  
    pub fn near(&self) -> f32 {
        self.params.near
    }

    // ============ 矩阵访问方法 ============

    /// 获取视图矩阵
    pub fn view_matrix(&self) -> Matrix4<f32> {
        self.view_matrix
    }

    /// 获取投影矩阵
    pub fn projection_matrix(&self) -> Matrix4<f32> {
        self.projection_matrix
    }

    // ============ 相机运动方法（用于动画）============

    /// 围绕目标点进行任意轴旋转（用于轨道动画）
    pub fn orbit(&mut self, axis: &Vector3<f32>, angle_rad: f32) {
        let camera_to_target = self.params.position - self.params.target;
        let rotation_matrix = TransformFactory::rotation(axis, angle_rad);
        let rotated_vector = rotation_matrix.transform_vector(&camera_to_target);
        self.params.position = self.params.target + rotated_vector;
        self.update_matrices();
    }

    /// 相机沿视线方向移动（保留用于动画）
    pub fn dolly(&mut self, amount: f32) {
        let direction = (self.params.target - self.params.position).normalize();
        let translation = direction * amount;
        self.params.position += translation;
        self.update_matrices();
    }

    // ============ GUI 交互方法 ============

    /// 屏幕拖拽转换为世界坐标平移（GUI专用）
    /// 返回值：是否需要清除地面缓存
    pub fn pan_from_screen_delta(
        &mut self,
        screen_delta: egui::Vec2,
        screen_size: egui::Vec2,
        sensitivity: f32,
    ) -> bool {
        // 计算世界坐标增量
        let distance_to_target = (self.params.position - self.params.target).magnitude();

        let world_scale = match &self.params.projection {
            ProjectionType::Perspective { fov_y_degrees, .. } => {
                let fov_rad = fov_y_degrees.to_radians();
                distance_to_target * (fov_rad / 2.0).tan() * 2.0 / screen_size.y
            }
            ProjectionType::Orthographic { height, .. } => height / screen_size.y,
        };

        // 应用敏感度
        let adjusted_scale = world_scale * sensitivity;
        let world_delta_x = -screen_delta.x * adjusted_scale;
        let world_delta_y = screen_delta.y * adjusted_scale;

        // 计算相机的右向量和上向量
        let forward = (self.params.target - self.params.position).normalize();
        let right = forward.cross(&self.params.up).normalize();
        let up = right.cross(&forward).normalize();

        // 计算世界空间中的平移向量
        let translation = right * world_delta_x + up * world_delta_y;

        // 同时移动相机位置和目标点
        self.params.position += translation;
        self.params.target += translation;

        self.update_matrices();

        // 相机位置变化，需要清除地面缓存
        true
    }

    /// 滚轮缩放转换为相机推拉（GUI专用）
    /// 返回值：是否需要清除地面缓存
    pub fn dolly_from_scroll(&mut self, scroll_delta: f32, sensitivity: f32) -> bool {
        let distance_to_target = (self.params.position - self.params.target).magnitude();

        // 基础敏感度：距离的 10%
        let base_sensitivity = distance_to_target * 0.1;

        // 应用用户敏感度设置
        let dolly_amount = scroll_delta * base_sensitivity * sensitivity;

        // 确保不会推得太近（最小距离 0.1）
        let min_distance = 0.1;
        if distance_to_target - dolly_amount > min_distance {
            self.dolly(dolly_amount);
        } else {
            // 如果会太近，就移动到最小距离位置
            let direction = (self.params.position - self.params.target).normalize();
            self.params.position = self.params.target + direction * min_distance;
            self.update_matrices();
        }

        // 相机位置变化，需要清除地面缓存
        true
    }

    /// 屏幕拖拽转换为轨道旋转（GUI专用）
    /// 返回值：是否需要清除地面缓存
    pub fn orbit_from_screen_delta(&mut self, screen_delta: egui::Vec2, sensitivity: f32) -> bool {
        // 基础旋转敏感度
        let base_rotation_sensitivity = 0.01;
        let adjusted_sensitivity = base_rotation_sensitivity * sensitivity;

        let angle_x = -screen_delta.y * adjusted_sensitivity;
        let angle_y = -screen_delta.x * adjusted_sensitivity;

        // Y轴旋转（水平拖拽）
        if angle_y.abs() > 1e-6 {
            self.orbit(&Vector3::y(), angle_y);
        }

        // X轴旋转（垂直拖拽） - 围绕相机的右向量
        if angle_x.abs() > 1e-6 {
            let forward = (self.params.target - self.params.position).normalize();
            let right = forward.cross(&self.params.up).normalize();

            // 限制垂直旋转角度，避免翻转
            let camera_to_target = self.params.position - self.params.target;
            let current_elevation = camera_to_target
                .y
                .atan2((camera_to_target.x.powi(2) + camera_to_target.z.powi(2)).sqrt());

            // 限制在 -85° 到 85° 之间
            let max_elevation = 85.0_f32.to_radians();
            let new_elevation = current_elevation + angle_x;

            if new_elevation.abs() < max_elevation {
                self.orbit(&right, angle_x);
            }
        }

        // 相机位置变化，需要清除地面缓存
        true
    }

    /// 重置相机到默认视角（GUI专用）
    /// 返回值：是否需要清除地面缓存
    pub fn reset_to_default_view(&mut self) -> bool {
        self.params.position = Point3::new(0.0, 0.0, 3.0);
        self.params.target = Point3::new(0.0, 0.0, 0.0);
        self.params.up = Vector3::new(0.0, 1.0, 0.0);
        self.update_matrices();

        // 相机位置变化，需要清除地面缓存
        true
    }

    /// 聚焦到物体（自动调整距离）（GUI专用）
    /// 返回值：是否需要清除地面缓存
    pub fn focus_on_object(&mut self, object_center: Point3<f32>, object_radius: f32) -> bool {
        // 计算合适的距离（确保物体完全可见）
        let optimal_distance = match &self.params.projection {
            ProjectionType::Perspective { fov_y_degrees, .. } => {
                let fov_rad = fov_y_degrees.to_radians();
                object_radius / (fov_rad / 2.0).tan() * 1.5 // 1.5倍确保有边距
            }
            ProjectionType::Orthographic { .. } => {
                object_radius * 3.0 // 正交投影下的合适距离
            }
        };

        // 保持当前的观察方向，但调整距离
        let current_direction = (self.params.position - self.params.target).normalize();

        self.params.target = object_center;
        self.params.position = object_center + current_direction * optimal_distance;

        self.update_matrices();

        // 相机位置变化，需要清除地面缓存
        true
    }

    // ============ 内部实现方法 ============

    /// 更新所有矩阵（创建时和修改后调用）
    pub fn update_matrices(&mut self) {
        self.update_view_matrix();
        self.update_projection_matrix();
    }

    /// 更新视图矩阵
    fn update_view_matrix(&mut self) {
        self.view_matrix =
            TransformFactory::view(&self.params.position, &self.params.target, &self.params.up);
    }

    /// 更新投影矩阵
    fn update_projection_matrix(&mut self) {
        self.projection_matrix = match &self.params.projection {
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
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new(CameraParams::default())
    }
}
