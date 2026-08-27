use crate::core::geometry::Vertex;
use crate::error::GltfError;
use crate::scene::material::{AlphaMode, Material, PbrMaterial};
use crate::scene::mesh::Mesh;
use crate::scene::model::Model;
use crate::scene::texture::Texture;
use image::{DynamicImage, RgbImage, RgbaImage};
use log::{info, warn};
use nalgebra::{Matrix4, Point3, Quaternion, UnitQuaternion, Vector2, Vector3, Vector4};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Loads a GLTF/GLB file, baking node transforms into mesh vertices.
/// Returns a single Model where all meshes share the same root coordinate system.
pub fn load_gltf<P: AsRef<Path>>(path: P, use_mipmap: bool) -> Result<Model, GltfError> {
    let path = path.as_ref();
    info!("Loading GLTF/GLB: {:?}", path);

    // 1. Import (This auto-detects .gltf or .glb)
    let (document, buffers, images) = gltf::import(path).map_err(|source| GltfError::Import {
        path: path.to_path_buf(),
        source: Box::new(source),
    })?;

    // 2. Load Textures into memory
    // gltf::import gives us raw image structs. We convert them to our engine's Texture type.
    let mut loaded_textures = Vec::new();
    for image_data in images {
        let tex = process_gltf_image(image_data, use_mipmap);
        loaded_textures.push(Arc::new(tex));
    }

    // 3. Prepare Accumulators
    let mut final_meshes: Vec<Mesh> = Vec::new();
    let mut final_materials: Vec<Material> = Vec::new();

    // Cache: Map (GLTF Material Index) -> (Local Material Index)
    // To avoid duplicating materials if multiple primitives use the same one.
    let mut material_cache: HashMap<usize, usize> = HashMap::new();

    // 4. Traverse Scene Graph
    // If no scene is selected, use the default or the first one.
    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next())
        .ok_or_else(|| GltfError::NoScene {
            path: path.to_path_buf(),
        })?;

    for node in scene.nodes() {
        process_node(
            &node,
            &Matrix4::identity(), // Root transform is Identity
            &buffers,
            &loaded_textures,
            &mut final_meshes,
            &mut final_materials,
            &mut material_cache,
        );
    }

    if final_meshes.is_empty() {
        return Err(GltfError::NoMeshes {
            path: path.to_path_buf(),
        });
    }

    info!(
        "GLTF loaded. Meshes: {}, Materials: {}",
        final_meshes.len(),
        final_materials.len()
    );

    Ok(Model::new(final_meshes, final_materials))
}

/// Recursive function to bake transforms
fn process_node(
    node: &gltf::Node,
    parent_transform: &Matrix4<f32>,
    buffers: &[gltf::buffer::Data],
    textures: &[Arc<Texture>],
    meshes: &mut Vec<Mesh>,
    materials: &mut Vec<Material>,
    material_cache: &mut HashMap<usize, usize>,
) {
    // 1. Calculate Global Transform for this node
    let (t, r, s) = node.transform().decomposed();
    let translation = Matrix4::new_translation(&Vector3::from(t));
    let rotation =
        UnitQuaternion::from_quaternion(Quaternion::new(r[3], r[0], r[1], r[2])).to_homogeneous();
    let scale = Matrix4::new_nonuniform_scaling(&Vector3::from(s));

    // Parent * Local = Global
    let global_transform = parent_transform * translation * rotation * scale;

    // Normal Matrix: Inverse Transpose of the upper-left 3x3.
    // Necessary to transform normals correctly if there is non-uniform scaling.
    let global_rotation_scale = global_transform.fixed_view::<3, 3>(0, 0).into_owned();
    let normal_matrix = global_rotation_scale
        .try_inverse()
        .map(|m| m.transpose())
        .unwrap_or(global_rotation_scale); // Fallback if singular

    // 2. Process Mesh
    if let Some(mesh) = node.mesh() {
        // --- Skip Shadow/Cheat Planes ---
        // If the node name indicates it's a shadow plane, we simply skip processing its mesh.
        if let Some(name) = node.name() {
            let name_lower = name.to_lowercase();
            if name_lower.contains("plane") || name_lower.contains("shadow") {
                info!("Skipping shadow plane node: {}", name);
                return; // Do not process this mesh
            }
        }

        for primitive in mesh.primitives() {
            let _mat = primitive.material();
            // Removed skipping of AlphaMode::Blend

            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            // --- Indices ---
            let indices: Vec<u32> = if let Some(iter) = reader.read_indices() {
                iter.into_u32().collect()
            } else {
                let count = reader.read_positions().map(|i| i.count()).unwrap_or(0);
                (0..count as u32).collect()
            };

            // --- Attributes ---
            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .map(|iter| iter.collect())
                .unwrap_or_default();
            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|iter| iter.collect())
                .unwrap_or_default();
            let uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|iter| iter.into_f32().collect())
                .unwrap_or_default();
            let tangents: Vec<[f32; 4]> = reader
                .read_tangents()
                .map(|iter| iter.collect())
                .unwrap_or_default();

            // --- Bake Vertices ---
            let mut vertices = Vec::with_capacity(positions.len());
            for i in 0..positions.len() {
                // Position: Apply full Global Transform
                let pos_local = Point3::from(positions[i]);
                let pos_world = global_transform.transform_point(&pos_local);

                let normal_local = if !normals.is_empty() {
                    Vector3::from(normals[i])
                } else {
                    Vector3::y()
                };
                let normal_world = (normal_matrix * normal_local).normalize();

                // Tangent Handling with Sign
                let tangent_world = if !tangents.is_empty() {
                    let t_vec_local = Vector3::new(tangents[i][0], tangents[i][1], tangents[i][2]);
                    let t_sign = tangents[i][3]; // Extract Sign (W)

                    // Rotate the vector part
                    let t_vec_world = (normal_matrix * t_vec_local).normalize();

                    // Store as Vector4
                    Vector4::new(t_vec_world.x, t_vec_world.y, t_vec_world.z, t_sign)
                } else {
                    Vector4::new(0.0, 0.0, 0.0, 1.0)
                };

                let uv = if !uvs.is_empty() {
                    Vector2::from(uvs[i])
                } else {
                    Vector2::zeros()
                };

                vertices.push(Vertex {
                    position: pos_world,
                    normal: normal_world,
                    texcoord: uv,
                    tangent: tangent_world, // Now Vector4
                });
            }

            // --- Material Handling ---
            let prim_mat = primitive.material();
            let mat_idx = if let Some(gltf_idx) = prim_mat.index() {
                // Check cache
                if let Some(&local_idx) = material_cache.get(&gltf_idx) {
                    local_idx
                } else {
                    // Create new material
                    let new_mat = convert_material(&prim_mat, textures);
                    let local_idx = materials.len();
                    materials.push(new_mat);
                    material_cache.insert(gltf_idx, local_idx);
                    local_idx
                }
            } else {
                // Default material handling
                // We don't cache "None" materials essentially, or make a default "Geometry" material
                let new_mat = convert_material(&prim_mat, textures);
                let local_idx = materials.len();
                materials.push(new_mat);
                local_idx
            };

            // Push the processed sub-mesh
            meshes.push(Mesh::new(vertices, indices, mat_idx));
        }
    }

    // 3. Recursion
    for child in node.children() {
        process_node(
            &child,
            &global_transform,
            buffers,
            textures,
            meshes,
            materials,
            material_cache,
        );
    }
}

/// Converts glTF material to Engine PbrMaterial
fn convert_material(mat: &gltf::Material, textures: &[Arc<Texture>]) -> Material {
    let pbr = mat.pbr_metallic_roughness();

    // Factors
    let albedo_factor = pbr.base_color_factor();
    let albedo = Vector3::new(albedo_factor[0], albedo_factor[1], albedo_factor[2]);
    let alpha = albedo_factor[3];
    let metallic = pbr.metallic_factor();
    let roughness = pbr.roughness_factor();
    let emissive_factor = mat.emissive_factor();
    let emissive = Vector3::from(emissive_factor);

    // Texture Helper
    let get_tex = |info: Option<gltf::texture::Info>| -> Option<Arc<Texture>> {
        info.map(|i| textures[i.texture().index()].clone())
    };

    // Valid for normal texture specific struct
    let get_normal_tex = |info: Option<gltf::material::NormalTexture>| -> Option<Arc<Texture>> {
        info.map(|i| textures[i.texture().index()].clone())
    };

    // Valid for occlusion
    let get_occlusion_tex =
        |info: Option<gltf::material::OcclusionTexture>| -> Option<Arc<Texture>> {
            info.map(|i| textures[i.texture().index()].clone())
        };

    let albedo_texture = get_tex(pbr.base_color_texture());

    // GLTF packs Metallic (B) and Roughness (G). Our shader supports this.
    let metallic_roughness_texture = get_tex(pbr.metallic_roughness_texture());

    let normal_texture = get_normal_tex(mat.normal_texture());
    let ao_texture = get_occlusion_tex(mat.occlusion_texture());
    let emissive_texture = get_tex(mat.emissive_texture());

    let alpha_mode = match mat.alpha_mode() {
        gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
        gltf::material::AlphaMode::Mask => AlphaMode::Mask(mat.alpha_cutoff().unwrap_or(0.5)),
        gltf::material::AlphaMode::Blend => AlphaMode::Blend,
    };

    Material::Pbr(PbrMaterial {
        albedo,
        alpha,
        metallic,
        roughness,
        ao: 1.0, // Base factor, usually 1.0 in GLTF if texture exists
        emissive,
        alpha_mode,
        albedo_texture,
        metallic_roughness_texture,
        normal_texture,
        ao_texture,
        emissive_texture,
    })
}

/// Convert glTF raw image data to Engine Texture
fn process_gltf_image(data: gltf::image::Data, use_mipmap: bool) -> Texture {
    let width = data.width;
    let height = data.height;

    let dyn_img = match data.format {
        // RGB
        gltf::image::Format::R8G8B8 => {
            DynamicImage::ImageRgb8(RgbImage::from_raw(width, height, data.pixels).unwrap())
        }
        // RGBA
        gltf::image::Format::R8G8B8A8 => {
            DynamicImage::ImageRgba8(RgbaImage::from_raw(width, height, data.pixels).unwrap())
        }
        // Grayscale -> RGB
        gltf::image::Format::R8 => {
            let pixels: Vec<u8> = data.pixels.iter().flat_map(|&b| [b, b, b]).collect();
            DynamicImage::ImageRgb8(RgbImage::from_raw(width, height, pixels).unwrap())
        }
        // RG -> RGBA (Roughness/Metallic sometimes?)
        gltf::image::Format::R8G8 => {
            let pixels: Vec<u8> = data
                .pixels
                .chunks(2)
                .flat_map(|c| [c[0], c[1], 0, 255])
                .collect();
            DynamicImage::ImageRgba8(RgbaImage::from_raw(width, height, pixels).unwrap())
        }
        _ => {
            warn!(
                "Unsupported texture format: {:?}. Using 1x1 magenta fallback.",
                data.format
            );
            let buf = vec![255, 0, 255, 255];
            DynamicImage::ImageRgba8(RgbaImage::from_raw(1, 1, buf).unwrap())
        }
    };

    Texture::from_image(dyn_img, use_mipmap)
}
