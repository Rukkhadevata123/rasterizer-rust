use crate::geometry::transform::TransformFactory;
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
        // 如果是正交投影，其投影矩阵在 new_orthographic 中已设置
        if self.fov_y > 0.0 {
            self.projection_matrix =
                TransformFactory::perspective(self.aspect_ratio, self.fov_y, self.near, self.far);
        }
        // 注意: 如果是正交投影 (self.fov_y <= 0.0)，
        // 它的 projection_matrix 是在 new_orthographic 中初始化的。
        // 如果宽高比在相机创建后需要改变（例如通过 args），
        // 并且不重新创建相机，那么需要一种机制来更新正交投影矩阵。
        // 由于我们删除了 set_aspect_ratio，所以现在的假设是
        // 当宽高比变化时，会重新创建相机。

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

    // set_fov 方法已删除

    // set_aspect_ratio 方法已删除

    /// 获取视图矩阵
    pub fn get_view_matrix(&self) -> Matrix4<f32> {
        self.view_matrix
    }

    /// 获取投影矩阵
    /// 注意：此方法现在依赖于相机实例化时设置的投影类型。
    /// 如果需要动态切换投影类型而不重新创建相机，则需要更复杂的逻辑。
    pub fn get_projection_matrix(&self, projection_type: &str) -> Matrix4<f32> {
        // 简单的实现：如果请求的类型与当前fov_y暗示的类型不匹配，
        // 并且我们想要严格按请求类型返回，可能需要按需计算。
        // 但更稳健的做法是确保相机在创建时就配置了正确的投影类型。
        // 为简化，这里假设相机已正确配置。

        let is_currently_perspective = self.fov_y > 0.0;

        if projection_type == "orthographic" && is_currently_perspective {
            // 当前是透视，但请求正交：按需计算一个通用的正交矩阵
            // 这里的宽高比和尺寸可能需要根据具体场景调整
            // 或者，更好的做法是确保相机在创建时就是正确的投影类型
            // 如果应用总是重新创建相机，这种情况可能不会发生
            // 或者，应该返回错误或当前矩阵
            // 为简单起见，我们创建一个基于当前宽高比的通用正交投影
            let height = 2.0; // 假设正交视图高度
            let width = height * self.aspect_ratio;
            TransformFactory::orthographic(
                -width / 2.0,
                width / 2.0,
                -height / 2.0,
                height / 2.0,
                self.near,
                self.far,
            )
        } else if projection_type == "perspective" && !is_currently_perspective {
            // 当前是正交，但请求透视：按需计算一个通用的透视矩阵
            // 使用一个默认的FOV，例如60度
            TransformFactory::perspective(
                self.aspect_ratio,
                60.0f32.to_radians(), // 默认FOV
                self.near,
                self.far,
            )
        } else {
            // 请求的类型与当前类型匹配，或者不关心特定类型，返回当前存储的投影矩阵
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
            // 这里的 height = 2.0 是一个惯用值，定义了正交视图在相机空间中的垂直大小。
            let height = 2.0;
            let width = height * aspect_ratio;
            Self::new_orthographic(position, target, up, width, height, near, far)
        }
    }

    /// 移动相机（同时移动位置和目标点）
    pub fn move_camera(&mut self, translation: &Vector3<f32>) {
        self.position += translation;
        self.target += translation;
        self.update_matrices();
    }
}

impl Default for Camera {
    fn default() -> Self {
        Camera::new_perspective(
            Point3::new(0.0, 0.0, 3.0),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            45.0,
            1.0,
            0.1,
            100.0,
        )
    }
}
