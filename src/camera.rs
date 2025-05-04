// 依赖nalgebra库进行3D数学计算
// nalgebra是Rust中常用的线性代数库，提供向量、矩阵、点、旋转等数学结构和操作
// 不同于Python版本的手动矩阵计算，Rust版本利用nalgebra的高性能实现
use nalgebra::{Matrix4, Point3, Rotation3, Unit, Vector3};

pub struct Camera {
    pub look_from: Point3<f32>,
    pub look_at: Point3<f32>,
    world_up: Unit<Vector3<f32>>, // 用Unit类型保证向量被规范化，不同于Python版本的手动规范化
    fov: f32,                     // Vertical FOV in degrees
    aspect_ratio: f32,
    near: f32,
    far: f32,

    // Calculated basis vectors (camera local coordinates)
    forward: Unit<Vector3<f32>>, // Points from look_at to look_from (opposite of view direction)
    right: Unit<Vector3<f32>>,   // Camera's positive X axis
    up: Unit<Vector3<f32>>,      // Camera's positive Y axis

    // 缓存计算后的矩阵，避免重复计算
    view_matrix: Matrix4<f32>,
    perspective_matrix: Matrix4<f32>,
    orthographic_matrix: Matrix4<f32>,
}

impl Camera {
    pub fn new(
        look_from: Point3<f32>,
        look_at: Point3<f32>,
        world_up: Vector3<f32>,
        fov_degrees: f32,
        aspect_ratio: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let mut cam = Camera {
            look_from,
            look_at,
            world_up: Unit::new_normalize(world_up),
            fov: fov_degrees,
            aspect_ratio,
            near,
            far,
            // Initialize dummy values, will be calculated next
            forward: Unit::new_unchecked(Vector3::z()),
            right: Unit::new_unchecked(Vector3::x()),
            up: Unit::new_unchecked(Vector3::y()),
            view_matrix: Matrix4::identity(),
            perspective_matrix: Matrix4::identity(),
            orthographic_matrix: Matrix4::identity(),
        };
        cam.update_camera_basis();
        cam.update_matrices();
        cam
    }

    fn update_camera_basis(&mut self) {
        // 计算前向量 - 与Python版本相同的计算，但使用nalgebra的Unit类型确保规范化
        // Python: forward = (look_at - look_from) / ||look_at - look_from||
        // Rust: 使用Unit<T>自动处理规范化
        let forward_vec = self.look_from - self.look_at;
        self.forward =
            Unit::try_new(forward_vec, 1e-8).unwrap_or_else(|| Unit::new_unchecked(Vector3::z())); // Default if look_from == look_at

        // 计算相机右向量 (相机的+X轴)
        // Python: right = cross(forward, world_up) / ||cross(forward, world_up)||
        // 这是构建相机坐标系的关键步骤
        let right_vec = self.world_up.cross(&self.forward);
        self.right = Unit::try_new(right_vec, 1e-8).unwrap_or_else(|| {
            // 处理forward与world_up平行的情况
            // 选择一个不同的临时上向量
            let temp_up = if self.forward.dot(&Vector3::x()).abs() < 0.9 {
                Vector3::x()
            } else {
                Vector3::y()
            };
            Unit::new_normalize(temp_up.cross(&self.forward))
        });

        // 计算实际的上向量 (相机的+Y轴)
        // Python: up = cross(right, forward) / ||cross(right, forward)||
        // 确保相机坐标系是正交的
        self.up = Unit::new_normalize(self.forward.cross(&self.right));
    }

    fn update_matrices(&mut self) {
        self.view_matrix = self._compute_view_matrix();
        self.perspective_matrix = self._compute_perspective_matrix();
        self.orthographic_matrix = self._compute_orthographic_matrix();
    }

    // 计算视图矩阵 (World -> Camera space)
    // 与Python版本不同，这里直接使用nalgebra的look_at_rh函数计算视图矩阵
    // 而不是手动构建旋转和平移矩阵再组合
    fn _compute_view_matrix(&self) -> Matrix4<f32> {
        // nalgebra的look_at_rh函数创建一个右手坐标系的视图矩阵
        // 在Python版本中，视图矩阵的计算公式为:
        // R = [right.x, right.y, right.z, 0]
        //     [up.x,    up.y,    up.z,    0]
        //     [-fw.x,   -fw.y,   -fw.z,   0]
        //     [0,       0,       0,       1]
        // T = [1, 0, 0, -eye.x]
        //     [0, 1, 0, -eye.y]
        //     [0, 0, 1, -eye.z]
        //     [0, 0, 0, 1     ]
        // View = R * T
        //
        // nalgebra内部实现相同的计算，但优化了性能和精度
        Matrix4::look_at_rh(&self.look_from, &self.look_at, &self.up)
    }

    // 计算透视投影矩阵
    // 使用nalgebra的new_perspective函数，不同于Python版本的手动矩阵构建
    fn _compute_perspective_matrix(&self) -> Matrix4<f32> {
        // Python版本中，透视投影矩阵的计算公式为:
        // f = 1 / tan(fov/2)
        // [f/aspect, 0,       0,                   0                  ]
        // [0,        f,       0,                   0                  ]
        // [0,        0,       (far+near)/(n-f),    2*far*near/(n-f)   ]
        // [0,        0,       -1,                  0                  ]
        //
        // nalgebra内部使用相同的数学公式构建透视矩阵
        Matrix4::new_perspective(
            self.aspect_ratio,
            self.fov.to_radians(),
            self.near,
            self.far,
        )
    }

    // 计算正交投影矩阵
    fn _compute_orthographic_matrix(&self) -> Matrix4<f32> {
        // 计算正交投影边界，这部分逻辑与Python版本类似
        // 使用FOV计算视窗高度
        let fovy_rad = self.fov.to_radians();
        let top = (fovy_rad / 2.0).tan(); // Height/2 at distance 1
        let bottom = -top;
        let right = top * self.aspect_ratio;
        let left = -right;

        // Python版本中，正交投影矩阵的计算公式为:
        // [1/right, 0,       0,                      0                       ]
        // [0,       1/top,   0,                      0                       ]
        // [0,       0,       -2/(far-near),          -(far+near)/(far-near)  ]
        // [0,       0,       0,                      1                       ]
        //
        // nalgebra内部使用相同的数学公式，将[left,right]x[bottom,top]x[near,far]映射到[-1,1]^3
        Matrix4::new_orthographic(left, right, bottom, top, self.near, self.far)
    }

    pub fn get_view_matrix(&self) -> &Matrix4<f32> {
        &self.view_matrix
    }

    pub fn get_projection_matrix(&self, projection_type: &str) -> &Matrix4<f32> {
        match projection_type.to_lowercase().as_str() {
            "perspective" => &self.perspective_matrix,
            "orthographic" | _ => &self.orthographic_matrix, // Default to orthographic if unknown
        }
    }

    /// 使相机围绕look_at点在Y轴上旋转（水平轨道运动）
    /// angle_degrees: 旋转角度。正值在俯视Y轴时表示逆时针旋转。
    /// 使用nalgebra的Rotation3类型创建并应用旋转
    pub fn orbit_y(&mut self, angle_degrees: f32) {
        // 1. 获取从目标点(look_at)到当前相机位置(look_from)的向量
        let mut current_vector = self.look_from - self.look_at;

        // 2. 围绕世界Y轴创建旋转
        let angle_rad = angle_degrees.to_radians();

        // 罗德里格斯旋转公式的详细说明:
        // 给定旋转轴k(单位向量)、旋转角度θ和待旋转向量v，旋转后的向量v_rot计算如下:
        // v_rot = v·cos(θ) + (k×v)·sin(θ) + k·(k·v)·(1-cos(θ))
        //
        // 对于绕Y轴的旋转，k = [0,1,0]，则公式简化为:
        // 假设v = [x,y,z]
        // k×v = [0,1,0]×[x,y,z] = [z,0,-x]
        // k·v = y
        // v_rot = [x·cos(θ) + z·sin(θ), y, z·cos(θ) - x·sin(θ)]
        //
        // 这与Y轴旋转矩阵的效果相同:
        // R_y(θ) = [ cos(θ), 0, sin(θ)]
        //          [      0, 1,      0]
        //          [-sin(θ), 0, cos(θ)]
        //
        // Rust代码使用nalgebra的Rotation3::from_axis_angle自动处理这些计算
        let rotation = Rotation3::from_axis_angle(&self.world_up, angle_rad);

        // 3. 应用旋转到向量 - 等效于手动应用罗德里格斯公式
        current_vector = rotation * current_vector;

        // 以下是不使用nalgebra库的罗德里格斯公式实现 (仅作参考):
        // let cos_angle = angle_rad.cos();
        // let sin_angle = angle_rad.sin();
        // let k = self.world_up.into_inner(); // 获取旋转轴向量
        // let v = current_vector;
        //
        // // 计算k×v (叉积)
        // let cross = Vector3::new(
        //     k.y * v.z - k.z * v.y,
        //     k.z * v.x - k.x * v.z,
        //     k.x * v.y - k.y * v.x
        // );
        //
        // // 计算k·v (点积)
        // let dot = k.dot(&v);
        //
        // // 应用罗德里格斯公式
        // current_vector = v * cos_angle + cross * sin_angle + k * dot * (1.0 - cos_angle);

        // 4. 计算新的相机位置
        self.look_from = self.look_at + current_vector;

        // 5. 更新相机的内部矩阵（视图矩阵依赖于look_from）
        self.update_camera_basis();
        self.update_matrices();
    }

    /// 使相机围绕look_at点在任意轴上旋转
    /// 实现通用罗德里格斯旋转
    /// axis: 旋转轴(单位向量)
    /// angle_degrees: 旋转角度(度)
    pub fn orbit_around_axis(&mut self, axis: &Vector3<f32>, angle_degrees: f32) {
        // 1. 获取从目标点到相机的向量
        let mut current_vector = self.look_from - self.look_at;

        // 2. 确保旋转轴是单位向量
        let rotation_axis = Unit::new_normalize(*axis);

        // 3. 将角度转换为弧度
        let angle_rad = angle_degrees.to_radians();

        // 4. 创建旋转对象
        // 这里使用轴角表示法创建旋转，内部实现了罗德里格斯公式:
        // v_rot = v·cos(θ) + (axis×v)·sin(θ) + axis·(axis·v)·(1-cos(θ))
        let rotation = Rotation3::from_axis_angle(&rotation_axis, angle_rad);

        // 5. 应用旋转到相机位置向量
        current_vector = rotation * current_vector;

        // 6. 更新相机位置
        self.look_from = self.look_at + current_vector;

        // 7. 更新相机矩阵
        self.update_camera_basis();
        self.update_matrices();
    }

    // 与Python版本相比，Rust版本移除了transform_vertices函数
    // 在Rust实现中，顶点变换通常在renderer.rs中处理
    // Python版本中的transform_vertices函数执行以下步骤:
    // 1. 将顶点转换为齐次坐标
    // 2. 应用视图变换: world_vertices → view_space (通过view_matrix)
    // 3. 应用投影变换: view_space → clip_space (通过projection_matrix)
    // 4. 执行透视除法: clip_space → ndc_space (通过w分量)
}
