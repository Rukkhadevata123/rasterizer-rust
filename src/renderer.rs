use crate::camera::Camera;
use crate::color_utils::get_face_color;
use crate::lighting::{Light, SimpleMaterial, calculate_blinn_phong}; // Import lighting components
use crate::loaders::{Material, Mesh, ModelData, Vertex};
use crate::rasterizer::{TriangleData, rasterize_triangle};
use crate::transform::ndc_to_pixel;
use atomic_float::AtomicF32;
use nalgebra::{Matrix3, Point2, Point3, Vector2, Vector3}; // Import Matrix3
use rayon::prelude::*;
use std::sync::Mutex;
use std::sync::atomic::Ordering;
use std::time::Instant;

pub struct FrameBuffer {
    pub width: usize,
    pub height: usize,
    /// Stores positive depth values, smaller is closer. Atomic for parallel writes.
    pub depth_buffer: Vec<AtomicF32>,
    /// Stores RGB color values [0, 255] as u8. Mutex for parallel writes.
    pub color_buffer: Mutex<Vec<u8>>,
}

impl FrameBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        let num_pixels = width * height;
        let depth_buffer = (0..num_pixels)
            .map(|_| AtomicF32::new(f32::INFINITY))
            .collect();
        let color_buffer = Mutex::new(vec![0u8; num_pixels * 3]); // Initialize black
        FrameBuffer {
            width,
            height,
            depth_buffer,
            color_buffer,
        }
    }

    pub fn clear(&self) {
        // Reset depth buffer using parallel iteration for potential speedup
        self.depth_buffer.par_iter().for_each(|atomic_depth| {
            atomic_depth.store(f32::INFINITY, Ordering::Relaxed);
        });

        // Reset color buffer
        let mut color_guard = self.color_buffer.lock().unwrap();
        // Consider parallelizing this if locking becomes a bottleneck for clearing large buffers
        color_guard.fill(0);
    }

    pub fn get_color_buffer_bytes(&self) -> Vec<u8> {
        self.color_buffer.lock().unwrap().clone()
    }

    pub fn get_depth_buffer_f32(&self) -> Vec<f32> {
        self.depth_buffer
            .iter()
            .map(|atomic_depth| atomic_depth.load(Ordering::Relaxed))
            .collect()
    }
}

pub struct RenderSettings {
    pub projection_type: String,
    pub use_zbuffer: bool,
    pub use_face_colors: bool,
    pub use_texture: bool,
    pub light: Light, // Store the configured light source
                      // Add more settings like backface culling, wireframe etc.
}

pub struct Renderer {
    pub width: usize,
    pub height: usize,
    pub frame_buffer: FrameBuffer,
}

impl Renderer {
    pub fn new(width: usize, height: usize) -> Self {
        Renderer {
            width,
            height,
            frame_buffer: FrameBuffer::new(width, height),
        }
    }

    /// Normalizes and centers the model's vertices in place.
    /// Returns the original center and scaling factor.
    fn normalize_and_center_model(model_data: &mut ModelData) -> (Vector3<f32>, f32) {
        if model_data.meshes.is_empty() {
            return (Vector3::zeros(), 1.0);
        }

        // Calculate bounding box or centroid of all vertices
        let mut min_coord = Point3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max_coord = Point3::new(f32::MIN, f32::MIN, f32::MIN);
        let mut vertex_sum = Vector3::zeros();
        let mut vertex_count = 0;

        for mesh in &model_data.meshes {
            for vertex in &mesh.vertices {
                min_coord = min_coord.inf(&vertex.position);
                max_coord = max_coord.sup(&vertex.position);
                vertex_sum += vertex.position.coords;
                vertex_count += 1;
            }
        }

        if vertex_count == 0 {
            return (Vector3::zeros(), 1.0);
        }

        let center = vertex_sum / (vertex_count as f32);
        let extent = max_coord - min_coord;
        let max_extent = extent.x.max(extent.y).max(extent.z);

        let scale_factor = if max_extent > 1e-6 {
            1.6 / max_extent // Scale to fit roughly in [-0.8, 0.8] cube (like Python's 0.8 factor)
        } else {
            1.0
        };

        // Apply transformation to all vertices
        for mesh in &mut model_data.meshes {
            for vertex in &mut mesh.vertices {
                vertex.position = Point3::from((vertex.position.coords - center) * scale_factor);
            }
        }

        (center, scale_factor)
    }

    pub fn render(&self, model_data: &mut ModelData, camera: &Camera, settings: &RenderSettings) {
        let start_time = Instant::now();

        println!("Clearing buffers...");
        self.frame_buffer.clear();
        let clear_duration = start_time.elapsed();

        println!("Normalizing model...");
        let norm_start_time = Instant::now();
        let (original_center, scale_factor) = Self::normalize_and_center_model(model_data);
        let norm_duration = norm_start_time.elapsed();
        println!(
            "Model normalized. Original Center: {:.3?}, Scale Factor: {:.3}",
            original_center, scale_factor
        );

        println!("Transforming vertices...");
        let transform_start_time = Instant::now();

        // Collect all vertices from all meshes for batch transformation
        let mut all_vertices_world: Vec<Point3<f32>> = Vec::new();
        let mut mesh_vertex_offsets: Vec<usize> = vec![0];
        for mesh in &model_data.meshes {
            all_vertices_world.extend(mesh.vertices.iter().map(|v| v.position));
            mesh_vertex_offsets.push(all_vertices_world.len());
        }

        // Perform transformations: World -> View -> NDC
        // Also transform normals to View space for lighting
        let view_matrix = camera.get_view_matrix();
        let projection_matrix = camera.get_projection_matrix(&settings.projection_type);
        let view_projection_matrix = projection_matrix * view_matrix;
        // Normal matrix: transpose(inverse(view_matrix)) - for transforming normals correctly
        let normal_matrix = view_matrix.try_inverse().map_or_else(
            || {
                println!("Warning: View matrix not invertible, using identity for normals.");
                Matrix3::identity()
            },
            |inv_view| inv_view.transpose().fixed_slice::<3, 3>(0, 0).into_owned(),
        );

        let mut all_ndc_coords = Vec::with_capacity(all_vertices_world.len());
        let mut all_view_coords = Vec::with_capacity(all_vertices_world.len());
        let mut all_view_normals = Vec::with_capacity(all_vertices_world.len()); // Store view-space normals

        // Transform vertices and normals
        for (mesh_idx, mesh) in model_data.meshes.iter().enumerate() {
            let vertex_offset = mesh_vertex_offsets[mesh_idx];
            for i in 0..mesh.vertices.len() {
                let world_vertex = all_vertices_world[vertex_offset + i];
                let world_normal = mesh.vertices[i].normal; // Get normal from original mesh data

                // World -> View Space (Position)
                let view_h = view_matrix * world_vertex.to_homogeneous();
                let view_pos = Point3::from_homogeneous(view_h).unwrap_or_else(|| Point3::origin());
                all_view_coords.push(view_pos);

                // World -> View Space (Normal) - Use normal matrix
                let view_normal = (normal_matrix * world_normal).normalize();
                all_view_normals.push(view_normal);

                // World -> Clip Space
                let clip_h = view_projection_matrix * world_vertex.to_homogeneous();

                // Clip Space -> NDC Space (Perspective Divide)
                let w = clip_h.w;
                if w.abs() > 1e-8 {
                    all_ndc_coords.push(Point3::new(clip_h.x / w, clip_h.y / w, clip_h.z / w));
                } else {
                    all_ndc_coords.push(Point3::origin()); // Avoid division by zero
                }
            }
        }

        // Perform viewport transformation: NDC -> Pixel
        let all_pixel_coords = ndc_to_pixel(&all_ndc_coords, self.width as f32, self.height as f32);
        let transform_duration = transform_start_time.elapsed();
        println!(
            "View space Z range: [{:.3}, {:.3}]",
            all_view_coords
                .iter()
                .map(|p| p.z)
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0),
            all_view_coords
                .iter()
                .map(|p| p.z)
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0)
        );

        println!("Rasterizing meshes...");
        let raster_start_time = Instant::now();

        // --- Parallel Rasterization using Rayon ---
        let all_pixel_coords_ref = &all_pixel_coords;
        let all_view_coords_ref = &all_view_coords;
        let all_view_normals_ref = &all_view_normals; // Reference to view-space normals
        let mesh_vertex_offsets_ref = &mesh_vertex_offsets;
        let model_materials_ref = &model_data.materials;
        // Define view direction (from origin towards +Z in view space)
        // More accurately, it's the direction from the point being shaded to the camera origin (0,0,0 in view space)
        // let view_dir_view = Vector3::z(); // This is direction camera is looking, not towards camera

        let triangles_to_render: Vec<_> = model_data
            .meshes
            .iter()
            .enumerate()
            .flat_map(|(mesh_idx, mesh)| {
                let vertex_offset = mesh_vertex_offsets_ref[mesh_idx];
                let material_opt: Option<&Material> =
                    mesh.material_id.and_then(|id| model_materials_ref.get(id));

                let texture = if settings.use_texture {
                    material_opt.and_then(|m| m.diffuse_texture.as_ref())
                } else {
                    None
                };

                // Get material properties for lighting, or use default
                let simple_material =
                    material_opt.map_or_else(SimpleMaterial::default, |m| SimpleMaterial {
                        ambient: m.ambient,
                        diffuse: m.diffuse,
                        specular: m.specular,
                        shininess: m.shininess,
                    });
                let default_color = simple_material.diffuse; // Use diffuse as fallback color

                mesh.indices
                    .chunks_exact(3)
                    .enumerate()
                    .map(move |(face_idx_in_mesh, indices)| {
                        let i0 = indices[0] as usize;
                        let i1 = indices[1] as usize;
                        let i2 = indices[2] as usize;

                        let v0 = &mesh.vertices[i0];
                        let v1 = &mesh.vertices[i1];
                        let v2 = &mesh.vertices[i2];

                        let global_i0 = vertex_offset + i0;
                        let global_i1 = vertex_offset + i1;
                        let global_i2 = vertex_offset + i2;

                        let pix0 = all_pixel_coords_ref[global_i0];
                        let pix1 = all_pixel_coords_ref[global_i1];
                        let pix2 = all_pixel_coords_ref[global_i2];

                        let view_pos0 = all_view_coords_ref[global_i0];
                        let view_pos1 = all_view_coords_ref[global_i1];
                        let view_pos2 = all_view_coords_ref[global_i2];

                        // --- Basic Flat Shading (using face normal) ---
                        // Calculate geometric face normal in view space for flat shading / backface culling
                        let edge1 = view_pos1 - view_pos0;
                        let edge2 = view_pos2 - view_pos0;
                        let face_normal_view = edge1.cross(&edge2).normalize();

                        // Backface Culling (Optional but recommended)
                        // If dot product of face normal and view direction is positive, face is pointing away
                        let view_dir_to_face = (view_pos0 - Point3::origin()).normalize(); // Vector from origin to face
                        if face_normal_view.dot(&view_dir_to_face) > -1e-6 { // Use negative epsilon for tolerance
                            // Skip this triangle (return None or a special marker)
                            return None; // Uncomment for backface culling
                        }

                        // Use average vertex normal for simple smooth shading approximation
                        let avg_normal_view = (all_view_normals_ref[global_i0]
                            + all_view_normals_ref[global_i1]
                            + all_view_normals_ref[global_i2])
                            .normalize();

                        // Calculate lighting at face center (approximation)
                        let face_center_view = Point3::from(
                            (view_pos0.coords + view_pos1.coords + view_pos2.coords) / 3.0,
                        );
                        // Direction from the surface point towards the camera (origin in view space)
                        let view_dir_from_face = (-face_center_view.coords).normalize();

                        let lit_color = calculate_blinn_phong(
                            face_center_view,
                            avg_normal_view, // Use averaged vertex normal
                            view_dir_from_face,
                            &settings.light,
                            &simple_material,
                        );

                        // Determine final base color before lighting/texturing
                        let base_color = if settings.use_face_colors {
                            get_face_color(mesh_idx * 1000 + face_idx_in_mesh, true)
                        } else {
                            default_color // Use material diffuse or default grey
                        };

                        // If texturing, lighting modifies the *sampled texture color*.
                        // If not texturing, lighting modifies the *base color*.
                        let final_color_or_signal = if texture.is_some() {
                            // Signal to use texture. Lighting will be applied *after* sampling in rasterizer.
                            // For now, just pass None and let rasterizer use texture color directly (no lighting yet for textured surfaces)
                            None // Signal texture usage
                        } else {
                            // Apply lighting to the base color
                            // Multiply base_color by light intensity components
                            Some(base_color.component_mul(&lit_color)) // Modulate base color by calculated light color
                        };

                        // Some(...) is used here because flat_map requires Option
                        Some(TriangleData {
                            v1_pix: Point2::new(pix0.x, pix0.y),
                            v2_pix: Point2::new(pix1.x, pix1.y),
                            v3_pix: Point2::new(pix2.x, pix2.y),
                            z1_view: view_pos0.z, // Use view space Z
                            z2_view: view_pos1.z,
                            z3_view: view_pos2.z,
                            color: final_color_or_signal, // Pass lit color or None (for texture)
                            tc1: texture.map(|_| v0.texcoord),
                            tc2: texture.map(|_| v1.texcoord),
                            tc3: texture.map(|_| v2.texcoord),
                            texture: texture,
                            is_perspective: settings.projection_type == "perspective",
                            use_zbuffer: settings.use_zbuffer,
                        })
                    })
            })
            .filter_map(|x| x) // Filter out None values if backface culling is enabled
            .collect();

        // Parallel loop over triangles
        triangles_to_render.par_iter().for_each(|triangle_data| {
            rasterize_triangle(
                triangle_data,
                self.width,
                self.height,
                &self.frame_buffer.depth_buffer,
                &self.frame_buffer.color_buffer,
            );
        });

        let raster_duration = raster_start_time.elapsed();
        let total_duration = start_time.elapsed();

        println!(
            "Rendering finished. Clear: {:?}, Norm: {:?}, Transform: {:?}, Raster: {:?}, Total: {:?}",
            clear_duration, norm_duration, transform_duration, raster_duration, total_duration
        );
        println!("Rendered {} triangles.", triangles_to_render.len());
    }
}
