use crate::geometry::transform::TransformFactory;
use crate::scene::scene_object::{TransformOperations, Transformable};
use nalgebra::{Matrix4, Point3, Vector3};

/// 相机类，负责管理视角和投影变换
#[derive(Debug, Clone)]
pub struct Camera {
    /// 相机位置（眼睛位置）
    pub position: Point3<f32>,
    /// 相机观察点（目标位置）
    pub target: Point3<f32>,
    /// 相机上方向（定义相机的"上"是哪个方向）
    pub up: Vector3<f32>,
    /// 相机在x轴方向的单位向量
    pub right: Vector3<f32>,
    /// 视场角（垂直方向，以弧度为单位）
    pub fov_y: f32,
    /// 宽高比（视口宽度/高度）
    pub aspect_ratio: f32,
    /// 近裁剪平面距离
    pub near: f32,
    /// 远裁剪平面距离
    pub far: f32,
    /// 视图矩阵（世界坐标 -> 相机坐标）
    pub view_matrix: Matrix4<f32>,
    /// 投影矩阵（相机坐标 -> 裁剪坐标）
    pub projection_matrix: Matrix4<f32>,
    /// 视图-投影矩阵（组合矩阵：世界坐标 -> 裁剪坐标）
    pub view_projection_matrix: Matrix4<f32>,
}

impl Camera {
    /// 创建一个新的透视投影相机
    pub fn new_perspective(
        position: Point3<f32>,
        target: Point3<f32>,
        up: Vector3<f32>,
        fov_y_degrees: f32,
        aspect_ratio: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let mut camera = Camera {
            position,
            target,
            up: up.normalize(),
            right: Vector3::x(), // 初始值，会在update_camera_basis中更新
            fov_y: fov_y_degrees.to_radians(),
            aspect_ratio,
            near,
            far,
            view_matrix: Matrix4::identity(),
            projection_matrix: Matrix4::identity(),
            view_projection_matrix: Matrix4::identity(),
        };
        camera.update_matrices();
        camera
    }

    /// 创建一个新的正交投影相机
    pub fn new_orthographic(
        position: Point3<f32>,
        target: Point3<f32>,
        up: Vector3<f32>,
        width: f32,
        height: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let aspect_ratio = width / height;
        let mut camera = Camera {
            position,
            target,
            up: up.normalize(),
            right: Vector3::x(), // 初始值，会在update_camera_basis中更新
            fov_y: 0.0,          // 正交投影不使用FOV
            aspect_ratio,
            near,
            far,
            view_matrix: Matrix4::identity(),
            projection_matrix: Matrix4::identity(),
            view_projection_matrix: Matrix4::identity(),
        };

        // 设置正交投影矩阵
        let half_width = width / 2.0;
        let half_height = height / 2.0;
        camera.projection_matrix = TransformFactory::orthographic(
            -half_width,
            half_width,
            -half_height,
            half_height,
            near,
            far,
        );

        camera.update_matrices();
        camera
    }

    /// 更新相机的基向量（right）
    fn update_camera_basis(&mut self) {
        // 计算相机的前方向（从相机指向目标）
        let forward = (self.target - self.position).normalize();

        // 计算右方向：前方向与上方向的叉积
        self.right = forward.cross(&self.up).normalize();

        // 重新计算正交化的上方向
        self.up = self.right.cross(&forward).normalize();
    }

    /// 更新所有相机矩阵
    pub fn update_matrices(&mut self) {
        // 更新相机基向量
        self.update_camera_basis();

        // 更新视图矩阵
        self.view_matrix = TransformFactory::view(&self.position, &self.target, &self.up);

        // 如果不是正交投影，则更新透视投影矩阵
        if self.fov_y > 0.0 {
            self.projection_matrix =
                TransformFactory::perspective(self.aspect_ratio, self.fov_y, self.near, self.far);
        }

        // 计算组合的视图-投影矩阵
        self.view_projection_matrix = self.projection_matrix * self.view_matrix;
    }

    /// 围绕目标点进行任意轴旋转
    pub fn orbit(&mut self, axis: &Vector3<f32>, angle_rad: f32) {
        // 计算从目标点到相机的向量
        let camera_to_target = self.position - self.target;

        // 使用TransformFactory创建旋转矩阵
        let rotation_matrix = TransformFactory::rotation(axis, angle_rad);

        // 应用旋转到相机位置向量
        let rotated_vector = rotation_matrix.transform_vector(&camera_to_target);

        // 更新相机位置
        self.position = self.target + rotated_vector;

        // 更新矩阵
        self.update_matrices();
    }

    /// 围绕Y轴旋转相机（简化的orbit方法）
    pub fn orbit_y(&mut self, angle_degrees: f32) {
        // 将角度转换为弧度
        let angle_rad = angle_degrees.to_radians();
        // 调用orbit方法，围绕Y轴旋转
        self.orbit(&Vector3::y_axis(), angle_rad);
    }

    /// 在XZ平面上平移相机（保持相机高度不变）
    pub fn pan(&mut self, right_amount: f32, forward_amount: f32) {
        // 获取相机的前方向（忽略Y分量以保持在XZ平面上）
        let forward = (self.target - self.position).normalize();
        let forward_xz = Vector3::new(forward.x, 0.0, forward.z).normalize();

        // 获取右方向（忽略Y分量以保持在XZ平面上）
        let right_xz = Vector3::new(self.right.x, 0.0, self.right.z).normalize();

        // 计算平移向量
        let translation = right_xz * right_amount + forward_xz * forward_amount;

        // 平移相机位置和目标点
        self.position += translation;
        self.target += translation;

        // 更新矩阵
        self.update_matrices();
    }

    /// 相机沿视线方向移动（正值接近目标，负值远离目标）
    pub fn dolly(&mut self, amount: f32) {
        // 计算从相机到目标的方向向量
        let direction = (self.target - self.position).normalize();

        // 计算平移向量
        let translation = direction * amount;

        // 平移相机位置
        self.position += translation;

        // 更新矩阵
        self.update_matrices();
    }

    /// 改变相机的视场角（垂直FOV）
    pub fn set_fov(&mut self, fov_y_degrees: f32) {
        self.fov_y = fov_y_degrees.to_radians();

        // 更新投影矩阵和视图-投影矩阵
        self.projection_matrix =
            TransformFactory::perspective(self.aspect_ratio, self.fov_y, self.near, self.far);
        self.view_projection_matrix = self.projection_matrix * self.view_matrix;
    }

    /// 改变相机的宽高比
    pub fn set_aspect_ratio(&mut self, aspect_ratio: f32) {
        self.aspect_ratio = aspect_ratio;

        // 更新投影矩阵和视图-投影矩阵
        if self.fov_y > 0.0 {
            self.projection_matrix =
                TransformFactory::perspective(self.aspect_ratio, self.fov_y, self.near, self.far);
        } else {
            // 如果是正交投影，根据宽高比调整投影矩阵
            let height = 2.0;
            let width = height * aspect_ratio;
            self.projection_matrix = TransformFactory::orthographic(
                -width / 2.0,
                width / 2.0,
                -height / 2.0,
                height / 2.0,
                self.near,
                self.far,
            );
        }

        self.view_projection_matrix = self.projection_matrix * self.view_matrix;
    }

    /// 获取视图矩阵
    pub fn get_view_matrix(&self) -> Matrix4<f32> {
        self.view_matrix
    }

    /// 获取投影矩阵
    pub fn get_projection_matrix(&self, projection_type: &str) -> Matrix4<f32> {
        // 如果请求特定投影类型，则可能需要重新计算投影矩阵
        if projection_type == "orthographic" && self.fov_y > 0.0 {
            // 需要创建正交投影矩阵
            let height = 2.0;
            let width = height * self.aspect_ratio;
            TransformFactory::orthographic(
                -width / 2.0,
                width / 2.0,
                -height / 2.0,
                height / 2.0,
                self.near,
                self.far,
            )
        } else if projection_type == "perspective" && self.fov_y <= 0.0 {
            // 需要创建透视投影矩阵
            TransformFactory::perspective(
                self.aspect_ratio,
                60.0f32.to_radians(),
                self.near,
                self.far,
            )
        } else {
            // 使用当前投影矩阵
            self.projection_matrix
        }
    }

    /// 创建通用相机（支持透视和正交）
    pub fn new(
        position: Point3<f32>,
        target: Point3<f32>,
        up: Vector3<f32>,
        fov_y_degrees: f32,
        aspect_ratio: f32,
        near: f32,
        far: f32,
    ) -> Self {
        // 根据FOV决定使用透视还是正交投影
        if fov_y_degrees > 0.0 {
            Self::new_perspective(position, target, up, fov_y_degrees, aspect_ratio, near, far)
        } else {
            // 对于正交投影，使用与视图的宽高比一致的视口大小
            let height = 2.0;
            let width = height * aspect_ratio;
            Self::new_orthographic(position, target, up, width, height, near, far)
        }
    }

    /// 从视图矩阵更新相机位置、目标和方向
    fn update_camera_from_view_matrix(&mut self) {
        // 视图矩阵的逆变换包含相机的世界变换
        if let Some(view_inverse) = self.view_matrix.try_inverse() {
            // 从视图矩阵的逆提取相机位置
            self.position = Point3::new(
                view_inverse[(0, 3)],
                view_inverse[(1, 3)],
                view_inverse[(2, 3)],
            );

            // 提取相机的方向向量
            let forward = Vector3::new(
                -view_inverse[(0, 2)],
                -view_inverse[(1, 2)],
                -view_inverse[(2, 2)],
            );

            // 计算目标点
            self.target = self.position + forward;

            // 提取上方向
            self.up = Vector3::new(
                view_inverse[(0, 1)],
                view_inverse[(1, 1)],
                view_inverse[(2, 1)],
            );

            // 提取右方向
            self.right = Vector3::new(
                view_inverse[(0, 0)],
                view_inverse[(1, 0)],
                view_inverse[(2, 0)],
            );
        }
    }
}

impl Transformable for Camera {
    fn get_transform(&self) -> &Matrix4<f32> {
        &self.view_matrix
    }

    fn set_transform(&mut self, transform: Matrix4<f32>) {
        // 设置视图矩阵并更新其他相关矩阵
        self.view_matrix = transform;
        self.view_projection_matrix = self.projection_matrix * self.view_matrix;

        // 注意：这不会更新position、target和up，如果需要完全一致性，应该从视图矩阵提取这些值
    }

    fn apply_local(&mut self, transform: Matrix4<f32>) {
        // 相机的局部变换是视图矩阵的逆变换的局部变换
        // 对于相机，"局部"是指相机坐标系
        let view_inverse = self
            .view_matrix
            .try_inverse()
            .unwrap_or_else(Matrix4::identity);
        let new_view_inverse = view_inverse * transform;
        self.view_matrix = new_view_inverse
            .try_inverse()
            .unwrap_or_else(Matrix4::identity);
        self.view_projection_matrix = self.projection_matrix * self.view_matrix;

        // 从视图矩阵更新相机属性
        self.update_camera_from_view_matrix();
    }

    fn apply_global(&mut self, transform: Matrix4<f32>) {
        // 相机的全局变换是视图矩阵的变换
        // 对于相机，"全局"是指世界坐标系
        let view_inverse = self
            .view_matrix
            .try_inverse()
            .unwrap_or_else(Matrix4::identity);
        let new_view_inverse = transform * view_inverse;
        self.view_matrix = new_view_inverse
            .try_inverse()
            .unwrap_or_else(Matrix4::identity);
        self.view_projection_matrix = self.projection_matrix * self.view_matrix;

        // 从视图矩阵更新相机属性
        self.update_camera_from_view_matrix();
    }
}

// 为Camera实现TransformOperations特性
impl TransformOperations for Camera {}
