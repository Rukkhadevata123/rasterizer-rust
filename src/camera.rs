use nalgebra::{Matrix4, Point3, Unit, Vector3};

pub struct Camera {
    pub look_from: Point3<f32>,
    pub look_at: Point3<f32>,
    world_up: Unit<Vector3<f32>>, // Keep track of the original world up
    fov: f32,                     // Vertical FOV in degrees
    aspect_ratio: f32,
    near: f32,
    far: f32,

    // Calculated basis vectors (camera local coordinates)
    forward: Unit<Vector3<f32>>, // Points from look_at to look_from (opposite of view direction)
    right: Unit<Vector3<f32>>,   // Camera's positive X axis
    up: Unit<Vector3<f32>>,      // Camera's positive Y axis

    // Cached matrices
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
        // Calculate forward direction (points from target to camera, -Z axis of camera)
        let forward_vec = self.look_from - self.look_at;
        self.forward =
            Unit::try_new(forward_vec, 1e-8).unwrap_or_else(|| Unit::new_unchecked(Vector3::z())); // Default if look_from == look_at

        // Calculate right direction (camera's +X axis)
        let right_vec = self.world_up.cross(&self.forward);
        self.right = Unit::try_new(right_vec, 1e-8).unwrap_or_else(|| {
            // Handle case where forward is parallel to world_up
            // Choose a different temporary up vector
            let temp_up = if self.forward.dot(&Vector3::x()).abs() < 0.9 {
                Vector3::x()
            } else {
                Vector3::y()
            };
            Unit::new_normalize(temp_up.cross(&self.forward))
        });

        // Calculate actual up direction (camera's +Y axis)
        self.up = Unit::new_normalize(self.forward.cross(&self.right));
    }

    fn update_matrices(&mut self) {
        self.view_matrix = self._compute_view_matrix();
        self.perspective_matrix = self._compute_perspective_matrix();
        self.orthographic_matrix = self._compute_orthographic_matrix();
    }

    // Computes the view matrix (World -> Camera space)
    // Uses nalgebra's look_at_rh function for simplicity and correctness
    fn _compute_view_matrix(&self) -> Matrix4<f32> {
        // nalgebra's look_at_rh creates a view matrix that transforms world coordinates
        // into the camera's coordinate system (right-handed).
        // It takes the camera's position (eye), the target point, and the up direction.
        // Note: The camera's local -Z axis points towards look_at.
        Matrix4::look_at_rh(&self.look_from, &self.look_at, &self.up)
    }

    // Computes the perspective projection matrix
    fn _compute_perspective_matrix(&self) -> Matrix4<f32> {
        // nalgebra's new_perspective creates a right-handed perspective projection matrix.
        Matrix4::new_perspective(
            self.aspect_ratio,
            self.fov.to_radians(),
            self.near,
            self.far,
        )
    }

    // Computes the orthographic projection matrix
    fn _compute_orthographic_matrix(&self) -> Matrix4<f32> {
        // Calculate orthographic bounds based on FOV and aspect ratio
        // We determine the height at the near plane (or conceptually at distance 1)
        // and derive width from aspect ratio.
        let fovy_rad = self.fov.to_radians();
        let top = (fovy_rad / 2.0).tan(); // Height/2 at distance 1
        let bottom = -top;
        let right = top * self.aspect_ratio;
        let left = -right;

        // nalgebra's new_orthographic creates a right-handed orthographic projection matrix.
        // Note: The Python version's Z mapping might differ slightly from nalgebra's default.
        // nalgebra maps [near, far] to [-1, 1]. The Python one maps to [-1, 1] as well.
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
    
    pub fn near(&self) -> f32 {
        self.near
    }

    pub fn far(&self) -> f32 {
        self.far
    }

    /// Transforms vertices from world space to NDC space and also returns view space coordinates.
    /// Returns (ndc_coords: Vec<Point3<f32>>, view_coords: Vec<Point3<f32>>)
    pub fn transform_vertices(
        &self,
        vertices: &[Point3<f32>],
        projection_type: &str,
    ) -> (Vec<Point3<f32>>, Vec<Point3<f32>>) {
        let view_matrix = self.get_view_matrix();
        let projection_matrix = self.get_projection_matrix(projection_type);
        let view_projection_matrix = projection_matrix * view_matrix;

        let mut ndc_coords = Vec::with_capacity(vertices.len());
        let mut view_coords = Vec::with_capacity(vertices.len());

        for vertex in vertices {
            // World -> View Space
            let view_h = view_matrix * vertex.to_homogeneous(); // Point3 -> Point4
            view_coords.push(Point3::from_homogeneous(view_h).unwrap_or_else(|| Point3::origin())); // Store non-homogeneous view coords

            // World -> Clip Space
            let clip_h = view_projection_matrix * vertex.to_homogeneous();

            // Clip Space -> NDC Space (Perspective Divide)
            let w = clip_h.w;
            if w.abs() > 1e-8 {
                ndc_coords.push(Point3::new(clip_h.x / w, clip_h.y / w, clip_h.z / w));
            } else {
                // Avoid division by zero, handle appropriately (e.g., push origin or a specific value)
                // Pushing the non-divided coordinates might be useful for clipping later.
                // For now, just push origin as a placeholder.
                ndc_coords.push(Point3::origin());
                // Or maybe: ndc_coords.push(Point3::new(clip_h.x, clip_h.y, clip_h.z));
            }
        }

        (ndc_coords, view_coords)
    }
}
