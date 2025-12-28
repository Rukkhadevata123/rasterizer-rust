use crate::scene::camera::{Camera, ProjectionType};
use minifb::{Key, MouseButton, MouseMode, Window};
use nalgebra::Vector3;
use std::f32::consts::PI;

pub struct CameraController {
    // Movement parameters
    pub speed: f32,
    pub sensitivity: f32,

    // Rotation state
    pub yaw: f32,
    pub pitch: f32,

    // FOV control state
    pub fov: f32, // radians
    pub min_fov: f32,
    pub max_fov: f32,

    // Zoom parameters
    pub zoom_speed: f32,

    last_mouse_pos: Option<(f32, f32)>,
}

impl CameraController {
    /// Creates a new camera controller.
    pub fn new(
        speed: f32,
        sensitivity: f32,
        initial_fov_degrees: f32,
        zoom_speed: f32,
        camera: &Camera,
    ) -> Self {
        // Calculate direction vector from position to target
        let direction = (camera.target - camera.position).normalize();

        // Calculate initial Pitch (asin of y component)
        let pitch = direction.y.asin();

        // Calculate initial Yaw (atan2 of z and x)
        // We use direction.z and direction.x.
        let yaw = direction.z.atan2(direction.x);

        Self {
            speed,
            sensitivity,
            yaw,   // Synced with config
            pitch, // Synced with config
            fov: initial_fov_degrees.to_radians(),
            min_fov: 10.0f32.to_radians(),
            max_fov: 120.0f32.to_radians(),
            zoom_speed,
            last_mouse_pos: None,
        }
    }

    pub fn update(&mut self, window: &Window, camera: &mut Camera, dt: f32) {
        // --- 1. Scroll to Zoom (FOV) ---
        if let Some((_, scroll_y)) = window.get_scroll_wheel()
            && scroll_y != 0.0
        {
            self.fov += scroll_y * self.zoom_speed;
            self.fov = self.fov.clamp(self.min_fov, self.max_fov);

            if let ProjectionType::Perspective {
                ref mut fov_y_rad, ..
            } = camera.projection_type
            {
                *fov_y_rad = self.fov;
            }
        }

        // --- 2. Keyboard Movement (WASD + Space + Shift) ---
        let mut move_dir = Vector3::zeros();

        // Since we sync rotation at the end, this is safe provided we update target too.
        let forward = (camera.target - camera.position).normalize();
        let right = forward.cross(&camera.up).normalize();
        let up = Vector3::y(); // World Up

        if window.is_key_down(Key::W) {
            move_dir += forward;
        }
        if window.is_key_down(Key::S) {
            move_dir -= forward;
        }
        if window.is_key_down(Key::A) {
            move_dir -= right;
        }
        if window.is_key_down(Key::D) {
            move_dir += right;
        }

        if window.is_key_down(Key::Space) {
            move_dir += up;
        }
        if window.is_key_down(Key::LeftShift) {
            move_dir -= up;
        }

        if window.is_key_down(Key::Z) {
            // Freeze
            move_dir = Vector3::zeros();
        }

        // Apply movement
        if move_dir.norm_squared() > 1e-6 {
            let offset = move_dir.normalize() * self.speed * dt;

            // Update BOTH position and target to maintain the view direction.
            // If we only update position, the camera will rotate to look at the old target.
            camera.position += offset;
            camera.target += offset;
        }

        // --- 3. Mouse Rotation (Left Click) ---
        if window.get_mouse_down(MouseButton::Left) {
            if let Some((x, y)) = window.get_mouse_pos(MouseMode::Pass) {
                if let Some((last_x, last_y)) = self.last_mouse_pos {
                    let dx = x - last_x;
                    let dy = y - last_y;

                    self.yaw += dx * self.sensitivity;
                    self.pitch -= dy * self.sensitivity;

                    // Clamp pitch to avoid gimbal lock
                    self.pitch = self.pitch.clamp(-PI / 2.0 + 0.01, PI / 2.0 - 0.01);

                    // Update Camera Target based on Yaw/Pitch
                    // This overwrites any implicit rotation caused by movement, ensuring consistency.
                    let front = Vector3::new(
                        self.yaw.cos() * self.pitch.cos(),
                        self.pitch.sin(),
                        self.yaw.sin() * self.pitch.cos(),
                    )
                    .normalize();

                    camera.target = camera.position + front;
                }
                self.last_mouse_pos = Some((x, y));
            }
        } else {
            self.last_mouse_pos = None;
        }

        // Important: Recalculate matrices after modifying position/target/fov
        camera.update_matrices();
    }
}
