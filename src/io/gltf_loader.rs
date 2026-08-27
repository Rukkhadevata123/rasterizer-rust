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
    let mut image_textures = Vec::new();
    for (image_index, image_data) in images.into_iter().enumerate() {
        let tex = process_gltf_image(path, image_index, image_data, use_mipmap)?;
        image_textures.push(Arc::new(tex));
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
        image_textures: &image_textures,
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
    image_textures: &'a [Arc<Texture>],
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

                if let Some((position_index, _)) = positions
                    .iter()
                    .enumerate()
                    .find(|(_, position)| position.iter().any(|value| !value.is_finite()))
                {
                    return Err(primitive_error(
                        self.path,
                        self.scene_index,
                        node,
                        mesh.index(),
                        primitive_index,
                        format!("POSITION[{position_index}] contains a non-finite value"),
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

                if let Some(&index) = indices
                    .iter()
                    .find(|&&index| index as usize >= positions.len())
                {
                    return Err(primitive_error(
                        self.path,
                        self.scene_index,
                        node,
                        mesh.index(),
                        primitive_index,
                        format!(
                            "index {index} is out of bounds for {} POSITION values",
                            positions.len()
                        ),
                    ));
                }

                let normals = if normals.is_empty() {
                    generate_area_weighted_normals(&positions, &indices).map_err(|reason| {
                        primitive_error(
                            self.path,
                            self.scene_index,
                            node,
                            mesh.index(),
                            primitive_index,
                            reason,
                        )
                    })?
                } else {
                    normals
                        .into_iter()
                        .enumerate()
                        .map(|(normal_index, normal)| {
                            normalized_vector3(normal, "NORMAL", normal_index)
                        })
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|reason| {
                            primitive_error(
                                self.path,
                                self.scene_index,
                                node,
                                mesh.index(),
                                primitive_index,
                                reason,
                            )
                        })?
                };

                let tangents = tangents
                    .into_iter()
                    .enumerate()
                    .map(|(tangent_index, tangent)| {
                        let direction = normalized_vector3(
                            [tangent[0], tangent[1], tangent[2]],
                            "TANGENT",
                            tangent_index,
                        )?;
                        if !tangent[3].is_finite() {
                            return Err(format!(
                                "TANGENT[{tangent_index}] handedness is non-finite"
                            ));
                        }
                        Ok([direction.x, direction.y, direction.z, tangent[3]])
                    })
                    .collect::<Result<Vec<_>, String>>()
                    .map_err(|reason| {
                        primitive_error(
                            self.path,
                            self.scene_index,
                            node,
                            mesh.index(),
                            primitive_index,
                            reason,
                        )
                    })?;

                let prim_mat = primitive.material();
                if prim_mat.normal_texture().is_some() && tangents.is_empty() {
                    return Err(primitive_error(
                        self.path,
                        self.scene_index,
                        node,
                        mesh.index(),
                        primitive_index,
                        "unsupported normal map without TANGENT attribute; tangent generation is planned for Phase 6",
                    ));
                }

                // --- Bake Vertices ---
                let mut vertices = Vec::with_capacity(positions.len());
                for i in 0..positions.len() {
                    // Position: Apply full Global Transform
                    let pos_local = Point3::from(positions[i]);
                    let pos_world = global_transform.transform_point(&pos_local);

                    let normal_world =
                        normalize_transformed_vector(normal_matrix * normals[i], "NORMAL", i)
                            .map_err(|reason| {
                                primitive_error(
                                    self.path,
                                    self.scene_index,
                                    node,
                                    mesh.index(),
                                    primitive_index,
                                    reason,
                                )
                            })?;

                    // Tangent Handling with Sign
                    let tangent_world = if !tangents.is_empty() {
                        let t_vec_local =
                            Vector3::new(tangents[i][0], tangents[i][1], tangents[i][2]);
                        let t_sign = tangents[i][3]; // Extract Sign (W)

                        // Rotate the vector part
                        let t_vec_world =
                            normalize_transformed_vector(normal_matrix * t_vec_local, "TANGENT", i)
                                .map_err(|reason| {
                                    primitive_error(
                                        self.path,
                                        self.scene_index,
                                        node,
                                        mesh.index(),
                                        primitive_index,
                                        reason,
                                    )
                                })?;

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
                let mat_idx = if let Some(gltf_idx) = prim_mat.index() {
                    // Check cache
                    if let Some(&local_idx) = self.material_cache.get(&gltf_idx) {
                        local_idx
                    } else {
                        // Create new material
                        let new_mat = convert_material(self.path, &prim_mat, self.image_textures)?;
                        let local_idx = self.materials.len();
                        self.materials.push(new_mat);
                        self.material_cache.insert(gltf_idx, local_idx);
                        local_idx
                    }
                } else {
                    // Default material handling
                    // We don't cache "None" materials essentially, or make a default "Geometry" material
                    let new_mat = convert_material(self.path, &prim_mat, self.image_textures)?;
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

/// Accumulates unnormalized face normals, weighting each contribution by triangle area.
fn generate_area_weighted_normals(
    positions: &[[f32; 3]],
    indices: &[u32],
) -> Result<Vec<Vector3<f32>>, String> {
    let mut normals = vec![Vector3::zeros(); positions.len()];
    for triangle in indices.chunks_exact(3) {
        let first = Vector3::from(positions[triangle[0] as usize]);
        let second = Vector3::from(positions[triangle[1] as usize]);
        let third = Vector3::from(positions[triangle[2] as usize]);
        let face_normal = (second - first).cross(&(third - first));
        for &index in triangle {
            normals[index as usize] += face_normal;
        }
    }

    normals
        .into_iter()
        .enumerate()
        .map(|(normal_index, normal)| {
            normalize_transformed_vector(normal, "generated NORMAL", normal_index)
        })
        .collect()
}

fn normalized_vector3(
    value: [f32; 3],
    attribute: &str,
    index: usize,
) -> Result<Vector3<f32>, String> {
    normalize_transformed_vector(Vector3::from(value), attribute, index)
}

fn normalize_transformed_vector(
    value: Vector3<f32>,
    attribute: &str,
    index: usize,
) -> Result<Vector3<f32>, String> {
    let length_squared = value.norm_squared();
    if !value.iter().all(|component| component.is_finite()) || !length_squared.is_finite() {
        return Err(format!("{attribute}[{index}] contains a non-finite value"));
    }
    if length_squared <= f32::EPSILON {
        return Err(format!("{attribute}[{index}] has zero length"));
    }
    Ok(value / length_squared.sqrt())
}

/// Converts glTF material to Engine PbrMaterial
fn convert_material(
    path: &Path,
    mat: &gltf::Material,
    image_textures: &[Arc<Texture>],
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
        info.map(|info| resolve_texture(path, mat.index(), info.texture(), image_textures))
            .transpose()
    };

    // Valid for normal texture specific struct
    let get_normal_tex =
        |info: Option<gltf::material::NormalTexture>| -> Result<Option<Arc<Texture>>, GltfError> {
            info.map(|info| resolve_texture(path, mat.index(), info.texture(), image_textures))
                .transpose()
        };

    // Valid for occlusion
    let get_occlusion_tex =
        |info: Option<gltf::material::OcclusionTexture>| -> Result<Option<Arc<Texture>>, GltfError> {
            info.map(|info| resolve_texture(path, mat.index(), info.texture(), image_textures))
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
    image_textures: &[Arc<Texture>],
) -> Result<Arc<Texture>, GltfError> {
    let texture_index = texture.index();
    let source_image_index = texture.source().index();
    image_textures
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
