use crate::core::geometry::Vertex;
use crate::error::{GltfError, PrimitiveContext};
use crate::scene::material::{AlphaMode, Material, PbrMaterial};
use crate::scene::mesh::Mesh;
use crate::scene::model::Model;
use crate::scene::texture::Texture;
use image::DynamicImage;
use log::info;
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
    let gltf = gltf::Gltf::open(path).map_err(|source| GltfError::Import {
        path: path.to_path_buf(),
        source: Box::new(source),
    })?;
    let base = path.parent().unwrap_or_else(|| Path::new("./"));
    let buffers =
        gltf::import_buffers(&gltf.document, Some(base), gltf.blob).map_err(|source| {
            GltfError::Import {
                path: path.to_path_buf(),
                source: Box::new(source),
            }
        })?;
    let mut images = Vec::new();
    for image in gltf.document.images() {
        let image_index = image.index();
        let data = gltf::image::Data::from_source(image.source(), Some(base), &buffers).map_err(
            |source| GltfError::Image {
                path: path.to_path_buf(),
                image_index,
                reason: source.to_string(),
            },
        )?;
        images.push(data);
    }
    let document = gltf.document;

    // 2. Load Textures into memory
    // gltf::import gives us raw image structs. We convert them to our engine's Texture type.
    let mut loaded_textures = Vec::new();
    for (image_index, image_data) in images.into_iter().enumerate() {
        let tex = process_gltf_image(path, image_index, image_data, use_mipmap)?;
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

    let mut importer = SceneImporter {
        path,
        scene_index: scene.index(),
        buffers: &buffers,
        textures: &loaded_textures,
        meshes: &mut final_meshes,
        materials: &mut final_materials,
        material_cache: &mut material_cache,
    };
    for node in scene.nodes() {
        importer.process_node(
            &node,
            &Matrix4::identity(), // Root transform is Identity
        )?;
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
struct SceneImporter<'a> {
    path: &'a Path,
    scene_index: usize,
    buffers: &'a [gltf::buffer::Data],
    textures: &'a [Arc<Texture>],
    meshes: &'a mut Vec<Mesh>,
    materials: &'a mut Vec<Material>,
    material_cache: &'a mut HashMap<usize, usize>,
}

impl SceneImporter<'_> {
    fn process_node(
        &mut self,
        node: &gltf::Node,
        parent_transform: &Matrix4<f32>,
    ) -> Result<(), GltfError> {
        // 1. Calculate Global Transform for this node
        let (t, r, s) = node.transform().decomposed();
        let translation = Matrix4::new_translation(&Vector3::from(t));
        let rotation = UnitQuaternion::from_quaternion(Quaternion::new(r[3], r[0], r[1], r[2]))
            .to_homogeneous();
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
            for primitive in mesh.primitives() {
                let primitive_index = primitive.index();
                let _mat = primitive.material();
                // Removed skipping of AlphaMode::Blend

                let reader = primitive.reader(|buffer| Some(&self.buffers[buffer.index()]));
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

                if positions.is_empty() {
                    return Err(primitive_error(
                        self.path,
                        self.scene_index,
                        node,
                        mesh.index(),
                        primitive_index,
                        "missing required POSITION attribute",
                    ));
                }

                for (attribute, count) in [
                    ("NORMAL", normals.len()),
                    ("TEXCOORD_0", uvs.len()),
                    ("TANGENT", tangents.len()),
                ] {
                    if count != 0 && count != positions.len() {
                        return Err(primitive_error(
                            self.path,
                            self.scene_index,
                            node,
                            mesh.index(),
                            primitive_index,
                            format!(
                                "attribute {attribute} has {count} values, expected {} to match POSITION",
                                positions.len()
                            ),
                        ));
                    }
                }

                let indices =
                    triangle_list_indices(primitive.mode(), &indices).map_err(|reason| {
                        primitive_error(
                            self.path,
                            self.scene_index,
                            node,
                            mesh.index(),
                            primitive_index,
                            reason,
                        )
                    })?;

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
                        let t_vec_local =
                            Vector3::new(tangents[i][0], tangents[i][1], tangents[i][2]);
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
                    if let Some(&local_idx) = self.material_cache.get(&gltf_idx) {
                        local_idx
                    } else {
                        // Create new material
                        let new_mat = convert_material(self.path, &prim_mat, self.textures)?;
                        let local_idx = self.materials.len();
                        self.materials.push(new_mat);
                        self.material_cache.insert(gltf_idx, local_idx);
                        local_idx
                    }
                } else {
                    // Default material handling
                    // We don't cache "None" materials essentially, or make a default "Geometry" material
                    let new_mat = convert_material(self.path, &prim_mat, self.textures)?;
                    let local_idx = self.materials.len();
                    self.materials.push(new_mat);
                    local_idx
                };

                // Push the processed sub-mesh
                self.meshes.push(Mesh::new(vertices, indices, mat_idx));
            }
        }

        // 3. Recursion
        for child in node.children() {
            self.process_node(&child, &global_transform)?;
        }

        Ok(())
    }
}

fn primitive_error(
    path: &Path,
    scene_index: usize,
    node: &gltf::Node,
    mesh_index: usize,
    primitive_index: usize,
    reason: impl Into<String>,
) -> GltfError {
    GltfError::Primitive {
        context: Box::new(PrimitiveContext {
            path: path.to_path_buf(),
            scene_index,
            node_index: node.index(),
            node_name: node.name().unwrap_or("<unnamed>").to_owned(),
            mesh_index,
            primitive_index,
            reason: reason.into(),
        }),
    }
}

fn triangle_list_indices(mode: gltf::mesh::Mode, indices: &[u32]) -> Result<Vec<u32>, String> {
    match mode {
        gltf::mesh::Mode::Triangles => {
            if indices.len() % 3 != 0 {
                return Err(format!(
                    "Triangles primitive has {} indices, expected a multiple of 3",
                    indices.len()
                ));
            }
            Ok(indices.to_vec())
        }
        gltf::mesh::Mode::TriangleStrip => {
            let mut triangles = Vec::with_capacity(indices.len().saturating_sub(2) * 3);
            for (triangle_index, window) in indices.windows(3).enumerate() {
                if triangle_index % 2 == 0 {
                    triangles.extend_from_slice(&[window[0], window[1], window[2]]);
                } else {
                    triangles.extend_from_slice(&[window[1], window[0], window[2]]);
                }
            }
            Ok(triangles)
        }
        gltf::mesh::Mode::TriangleFan => {
            let mut triangles = Vec::with_capacity(indices.len().saturating_sub(2) * 3);
            if let Some(&center) = indices.first() {
                for edge in indices[1..].windows(2) {
                    triangles.extend_from_slice(&[center, edge[0], edge[1]]);
                }
            }
            Ok(triangles)
        }
        unsupported => Err(format!("unsupported primitive mode {unsupported:?}")),
    }
}

/// Converts glTF material to Engine PbrMaterial
fn convert_material(
    path: &Path,
    mat: &gltf::Material,
    textures: &[Arc<Texture>],
) -> Result<Material, GltfError> {
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
    let get_tex = |info: Option<gltf::texture::Info>| -> Result<Option<Arc<Texture>>, GltfError> {
        info.map(|info| resolve_texture(path, mat.index(), info.texture(), textures))
            .transpose()
    };

    // Valid for normal texture specific struct
    let get_normal_tex =
        |info: Option<gltf::material::NormalTexture>| -> Result<Option<Arc<Texture>>, GltfError> {
            info.map(|info| resolve_texture(path, mat.index(), info.texture(), textures))
                .transpose()
        };

    // Valid for occlusion
    let get_occlusion_tex =
        |info: Option<gltf::material::OcclusionTexture>| -> Result<Option<Arc<Texture>>, GltfError> {
            info.map(|info| resolve_texture(path, mat.index(), info.texture(), textures))
                .transpose()
        };

    let albedo_texture = get_tex(pbr.base_color_texture())?;

    // GLTF packs Metallic (B) and Roughness (G). Our shader supports this.
    let metallic_roughness_texture = get_tex(pbr.metallic_roughness_texture())?;

    let normal_texture = get_normal_tex(mat.normal_texture())?;
    let ao_texture = get_occlusion_tex(mat.occlusion_texture())?;
    let emissive_texture = get_tex(mat.emissive_texture())?;

    let alpha_mode = match mat.alpha_mode() {
        gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
        gltf::material::AlphaMode::Mask => AlphaMode::Mask(mat.alpha_cutoff().unwrap_or(0.5)),
        gltf::material::AlphaMode::Blend => AlphaMode::Blend,
    };

    Ok(Material::Pbr(PbrMaterial {
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
    }))
}

fn resolve_texture(
    path: &Path,
    material_index: Option<usize>,
    texture: gltf::Texture,
    textures: &[Arc<Texture>],
) -> Result<Arc<Texture>, GltfError> {
    let texture_index = texture.index();
    let source_image_index = texture.source().index();
    textures
        .get(source_image_index)
        .cloned()
        .ok_or_else(|| GltfError::Texture {
            path: path.to_path_buf(),
            material_index,
            texture_index,
            source_image_index,
        })
}

/// Convert glTF raw image data to Engine Texture
fn process_gltf_image(
    path: &Path,
    image_index: usize,
    data: gltf::image::Data,
    use_mipmap: bool,
) -> Result<Texture, GltfError> {
    let width = data.width;
    let height = data.height;
    let format = data.format;

    let dyn_img = match format {
        // RGB
        gltf::image::Format::R8G8B8 => DynamicImage::ImageRgb8(image_buffer(
            path,
            image_index,
            format,
            width,
            height,
            data.pixels,
        )?),
        // RGBA
        gltf::image::Format::R8G8B8A8 => DynamicImage::ImageRgba8(image_buffer(
            path,
            image_index,
            format,
            width,
            height,
            data.pixels,
        )?),
        // Grayscale -> RGB
        gltf::image::Format::R8 => {
            let pixels: Vec<u8> = data.pixels.iter().flat_map(|&b| [b, b, b]).collect();
            DynamicImage::ImageRgb8(image_buffer(
                path,
                image_index,
                format,
                width,
                height,
                pixels,
            )?)
        }
        // RG -> RGBA (Roughness/Metallic sometimes?)
        gltf::image::Format::R8G8 => {
            let pixels: Vec<u8> = data
                .pixels
                .chunks(2)
                .flat_map(|c| [c[0], c[1], 0, 255])
                .collect();
            DynamicImage::ImageRgba8(image_buffer(
                path,
                image_index,
                format,
                width,
                height,
                pixels,
            )?)
        }
        _ => {
            return Err(GltfError::Image {
                path: path.to_path_buf(),
                image_index,
                reason: format!("unsupported decoded pixel format {format:?}"),
            });
        }
    };

    Ok(Texture::from_image(dyn_img, use_mipmap))
}

fn image_buffer<P: image::Pixel + 'static>(
    path: &Path,
    image_index: usize,
    format: gltf::image::Format,
    width: u32,
    height: u32,
    pixels: Vec<P::Subpixel>,
) -> Result<image::ImageBuffer<P, Vec<P::Subpixel>>, GltfError>
where
    P::Subpixel: image::Primitive,
{
    image::ImageBuffer::from_raw(width, height, pixels).ok_or_else(|| GltfError::Image {
        path: path.to_path_buf(),
        image_index,
        reason: format!("pixel data length does not match {width}x{height} {format:?} image"),
    })
}
