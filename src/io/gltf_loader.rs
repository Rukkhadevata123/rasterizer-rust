use crate::core::geometry::{SUPPORTED_TEXCOORD_SETS, Vertex};
use crate::core::math::transform::TangentFrameTransform;
use crate::scene::material::{AlphaMode, Material, PbrMaterial};
use crate::scene::mesh::Mesh;
use crate::scene::model::Model;
use crate::scene::texture::{
    MagFilter, MinFilter, SamplerState, TexCoordSet, TextureBinding, TextureImage, TextureUsage,
    WrapMode,
};
use image::DynamicImage;
use log::info;
use mikktspace::Geometry;
use nalgebra::{Matrix4, Point3, Quaternion, UnitQuaternion, Vector2, Vector3, Vector4};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GltfError {
    #[error("failed to import glTF '{}': {source}", path.display())]
    Import {
        path: PathBuf,
        #[source]
        source: Box<gltf::Error>,
    },
    #[error("glTF '{}' contains no scenes", path.display())]
    NoScene { path: PathBuf },
    #[error("glTF '{}' contains no meshes", path.display())]
    NoMeshes { path: PathBuf },
    #[error("unsupported feature in glTF '{}': {reason}", path.display())]
    Unsupported { path: PathBuf, reason: String },
    #[error(
        "failed to process glTF '{}' scene {}, node {} ({}), mesh {}, primitive {}: {}",
        context.path.display(),
        context.scene_index,
        context.node_index,
        context.node_name,
        context.mesh_index,
        context.primitive_index,
        context.reason
    )]
    Primitive { context: Box<PrimitiveContext> },
    #[error("failed to process image {image_index} in glTF '{}': {reason}", path.display())]
    Image {
        path: PathBuf,
        image_index: usize,
        reason: String,
    },
    #[error(
        "failed to resolve texture {texture_index} for material {material_index:?} in glTF '{}': source image {source_image_index} is unavailable",
        path.display()
    )]
    Texture {
        path: PathBuf,
        material_index: Option<usize>,
        texture_index: usize,
        source_image_index: usize,
    },
}

#[derive(Debug)]
pub struct PrimitiveContext {
    pub path: PathBuf,
    pub scene_index: usize,
    pub node_index: usize,
    pub node_name: String,
    pub mesh_index: usize,
    pub primitive_index: usize,
    pub reason: String,
}

/// Loads a GLTF/GLB file, baking node transforms into mesh vertices.
/// Returns a single Model where all meshes share the same root coordinate system.
pub fn load_gltf<P: AsRef<Path>>(path: P, use_mipmap: bool) -> Result<Model, GltfError> {
    let path = path.as_ref();
    info!("Loading GLTF/GLB: {:?}", path);

    let gltf = gltf::Gltf::open(path).map_err(|source| GltfError::Import {
        path: path.to_path_buf(),
        source: Box::new(source),
    })?;
    if let Some(extension) = gltf.document.extensions_used().find(|extension| {
        matches!(
            *extension,
            "KHR_texture_transform" | "KHR_materials_emissive_strength"
        )
    }) {
        return Err(GltfError::Unsupported {
            path: path.to_path_buf(),
            reason: format!("{extension} is not supported"),
        });
    }
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

    // Keep decoded images independent from glTF texture objects so several bindings can share
    // pixels while retaining their own sampler and material-slot metadata.
    let mut image_resources = Vec::new();
    for (image_index, image_data) in images.into_iter().enumerate() {
        let image = process_gltf_image(path, image_index, image_data, use_mipmap)?;
        image_resources.push(Arc::new(image));
    }

    let mut final_meshes: Vec<Mesh> = Vec::new();
    let mut final_materials: Vec<Material> = Vec::new();

    // Reuse materials shared by multiple primitives.
    let mut material_cache: HashMap<usize, usize> = HashMap::new();

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
        image_resources: &image_resources,
        meshes: &mut final_meshes,
        materials: &mut final_materials,
        material_cache: &mut material_cache,
    };
    for node in scene.nodes() {
        importer.process_node(&node, &Matrix4::identity())?;
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

/// Recursively bakes node transforms into mesh vertices.
struct SceneImporter<'a> {
    path: &'a Path,
    scene_index: usize,
    buffers: &'a [gltf::buffer::Data],
    image_resources: &'a [Arc<TextureImage>],
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
        let (t, r, s) = node.transform().decomposed();
        let translation = Matrix4::new_translation(&Vector3::from(t));
        let rotation = UnitQuaternion::from_quaternion(Quaternion::new(r[3], r[0], r[1], r[2]))
            .to_homogeneous();
        let scale = Matrix4::new_nonuniform_scaling(&Vector3::from(s));

        let global_transform = parent_transform * translation * rotation * scale;

        let global_rotation_scale = global_transform.fixed_view::<3, 3>(0, 0).into_owned();
        let tangent_frame_transform = TangentFrameTransform::new(global_rotation_scale);

        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                let primitive_index = primitive.index();
                if let Some(set) = primitive
                    .attributes()
                    .find_map(|(semantic, _)| match semantic {
                        gltf::Semantic::TexCoords(set)
                            if set as usize >= SUPPORTED_TEXCOORD_SETS =>
                        {
                            Some(set)
                        }
                        _ => None,
                    })
                {
                    return Err(GltfError::Unsupported {
                        path: self.path.to_path_buf(),
                        reason: format!(
                            "scene {}, node {}, mesh {}, primitive {} uses unsupported TEXCOORD_{set}",
                            self.scene_index,
                            node.index(),
                            mesh.index(),
                            primitive_index
                        ),
                    });
                }
                let reader = primitive.reader(|buffer| Some(&self.buffers[buffer.index()]));
                let indices: Vec<u32> = if let Some(iter) = reader.read_indices() {
                    iter.into_u32().collect()
                } else {
                    let count = reader.read_positions().map(|i| i.count()).unwrap_or(0);
                    (0..count as u32).collect()
                };

                let positions: Vec<[f32; 3]> = reader
                    .read_positions()
                    .map(|iter| iter.collect())
                    .unwrap_or_default();
                let normals: Vec<[f32; 3]> = reader
                    .read_normals()
                    .map(|iter| iter.collect())
                    .unwrap_or_default();
                let uv_sets: [Vec<[f32; 2]>; SUPPORTED_TEXCOORD_SETS] =
                    std::array::from_fn(|set| {
                        reader
                            .read_tex_coords(set as u32)
                            .map(|iter| iter.into_f32().collect())
                            .unwrap_or_default()
                    });
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
                    ("TEXCOORD_0", uv_sets[0].len()),
                    ("TEXCOORD_1", uv_sets[1].len()),
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
                for (slot, tex_coord) in material_tex_coords(&prim_mat).into_iter().flatten() {
                    let set = TexCoordSet::try_from(tex_coord).map_err(|unsupported| {
                        GltfError::Unsupported {
                            path: self.path.to_path_buf(),
                            reason: format!(
                                "material {:?} {slot} uses unsupported TEXCOORD_{unsupported}",
                                prim_mat.index()
                            ),
                        }
                    })?;
                    if uv_sets[set.index()].is_empty() {
                        return Err(primitive_error(
                            self.path,
                            self.scene_index,
                            node,
                            mesh.index(),
                            primitive_index,
                            format!(
                                "material {slot} requires TEXCOORD_{} but the primitive does not provide it",
                                set.index()
                            ),
                        ));
                    }
                }

                let generated_tangents = if tangents.is_empty() {
                    prim_mat
                        .normal_texture()
                        .map(|normal_texture| {
                            let tex_coord = TexCoordSet::try_from(normal_texture.tex_coord())
                                .expect("material UV sets were validated above");
                            generate_mikktspace_tangents(
                                &positions,
                                &normals,
                                &uv_sets[tex_coord.index()],
                                &indices,
                            )
                            .map_err(|reason| {
                                primitive_error(
                                    self.path,
                                    self.scene_index,
                                    node,
                                    mesh.index(),
                                    primitive_index,
                                    reason,
                                )
                            })
                        })
                        .transpose()?
                } else {
                    None
                };

                let (vertex_indices, mut mesh_indices) = if generated_tangents.is_some() {
                    (
                        indices
                            .iter()
                            .map(|&index| index as usize)
                            .collect::<Vec<_>>(),
                        (0..indices.len() as u32).collect(),
                    )
                } else {
                    ((0..positions.len()).collect(), indices)
                };
                if global_rotation_scale.determinant() < 0.0 {
                    for triangle in mesh_indices.chunks_exact_mut(3) {
                        triangle.swap(1, 2);
                    }
                }

                let mut vertices = Vec::with_capacity(vertex_indices.len());
                for (vertex_index, &i) in vertex_indices.iter().enumerate() {
                    let pos_local = Point3::from(positions[i]);
                    let pos_world = global_transform.transform_point(&pos_local);

                    let tangent = generated_tangents
                        .as_ref()
                        .map(|generated| generated[vertex_index])
                        .or_else(|| tangents.get(i).copied());
                    let tangent_local = tangent
                        .map(Vector4::from)
                        .unwrap_or(Vector4::new(0.0, 0.0, 0.0, 1.0));
                    let (normal_world, tangent_world) =
                        tangent_frame_transform.transform(normals[i], tangent_local);
                    let normal_world = normalize_transformed_vector(normal_world, "NORMAL", i)
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

                    let texcoords = std::array::from_fn(|set| {
                        uv_sets[set]
                            .get(i)
                            .copied()
                            .map(Vector2::from)
                            .unwrap_or_else(Vector2::zeros)
                    });

                    vertices.push(Vertex {
                        position: pos_world,
                        normal: normal_world,
                        texcoords,
                        tangent: tangent_world,
                    });
                }

                let mat_idx = if let Some(gltf_idx) = prim_mat.index() {
                    if let Some(&local_idx) = self.material_cache.get(&gltf_idx) {
                        local_idx
                    } else {
                        let new_mat = convert_material(self.path, &prim_mat, self.image_resources)?;
                        let local_idx = self.materials.len();
                        self.materials.push(new_mat);
                        self.material_cache.insert(gltf_idx, local_idx);
                        local_idx
                    }
                } else {
                    let new_mat = convert_material(self.path, &prim_mat, self.image_resources)?;
                    let local_idx = self.materials.len();
                    self.materials.push(new_mat);
                    local_idx
                };

                self.meshes.push(Mesh::new(vertices, mesh_indices, mat_idx));
            }
        }

        for child in node.children() {
            self.process_node(&child, &global_transform)?;
        }

        Ok(())
    }
}

fn material_tex_coords(material: &gltf::Material) -> [Option<(&'static str, u32)>; 5] {
    let pbr = material.pbr_metallic_roughness();
    [
        pbr.base_color_texture()
            .map(|info| ("base-color texture", info.tex_coord())),
        pbr.metallic_roughness_texture()
            .map(|info| ("metallic-roughness texture", info.tex_coord())),
        material
            .normal_texture()
            .map(|info| ("normal texture", info.tex_coord())),
        material
            .occlusion_texture()
            .map(|info| ("occlusion texture", info.tex_coord())),
        material
            .emissive_texture()
            .map(|info| ("emissive texture", info.tex_coord())),
    ]
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

struct MikktspaceGeometry<'a> {
    positions: &'a [[f32; 3]],
    normals: &'a [Vector3<f32>],
    texcoords: &'a [[f32; 2]],
    indices: &'a [u32],
    tangents: Vec<[f32; 4]>,
}

impl MikktspaceGeometry<'_> {
    fn source_index(&self, face: usize, vertex: usize) -> usize {
        self.indices[face * 3 + vertex] as usize
    }
}

impl Geometry for MikktspaceGeometry<'_> {
    fn num_faces(&self) -> usize {
        self.indices.len() / 3
    }

    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vertex: usize) -> [f32; 3] {
        self.positions[self.source_index(face, vertex)]
    }

    fn normal(&self, face: usize, vertex: usize) -> [f32; 3] {
        self.normals[self.source_index(face, vertex)].into()
    }

    fn tex_coord(&self, face: usize, vertex: usize) -> [f32; 2] {
        self.texcoords[self.source_index(face, vertex)]
    }

    fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vertex: usize) {
        self.tangents[face * 3 + vertex] = tangent;
    }
}

fn generate_mikktspace_tangents(
    positions: &[[f32; 3]],
    normals: &[Vector3<f32>],
    texcoords: &[[f32; 2]],
    indices: &[u32],
) -> Result<Vec<[f32; 4]>, String> {
    let mut geometry = MikktspaceGeometry {
        positions,
        normals,
        texcoords,
        indices,
        tangents: vec![[0.0; 4]; indices.len()],
    };
    if !mikktspace::generate_tangents(&mut geometry) {
        return Err("MikkTSpace tangent generation failed for the normal texture UV set".into());
    }
    if geometry
        .tangents
        .iter()
        .flatten()
        .any(|component| !component.is_finite())
    {
        return Err("MikkTSpace tangent generation produced a non-finite value".into());
    }
    if let Some((corner, _)) = geometry.tangents.iter().enumerate().find(|(_, tangent)| {
        Vector3::new(tangent[0], tangent[1], tangent[2]).norm_squared() <= f32::EPSILON
    }) {
        return Err(format!(
            "MikkTSpace tangent generation produced a zero-length tangent at corner {corner}"
        ));
    }
    Ok(geometry.tangents)
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

/// Converts a glTF material into the renderer's PBR material.
fn convert_material(
    path: &Path,
    mat: &gltf::Material,
    image_resources: &[Arc<TextureImage>],
) -> Result<Material, GltfError> {
    let pbr = mat.pbr_metallic_roughness();

    let albedo_factor = pbr.base_color_factor();
    let albedo = Vector3::new(albedo_factor[0], albedo_factor[1], albedo_factor[2]);
    let alpha = albedo_factor[3];
    let metallic = pbr.metallic_factor();
    let roughness = pbr.roughness_factor();
    let emissive_factor = mat.emissive_factor();
    let emissive = Vector3::from(emissive_factor);
    let normal_scale = mat
        .normal_texture()
        .map(|texture| texture.scale())
        .unwrap_or(1.0);
    let occlusion_strength = mat
        .occlusion_texture()
        .map(|texture| texture.strength())
        .unwrap_or(1.0);

    let get_tex = |info: Option<gltf::texture::Info>, usage| {
        info.map(|info| {
            resolve_texture_binding(
                path,
                mat.index(),
                info.texture(),
                info.tex_coord(),
                usage,
                image_resources,
            )
        })
        .transpose()
    };

    let get_normal_tex = |info: Option<gltf::material::NormalTexture>| {
        info.map(|info| {
            resolve_texture_binding(
                path,
                mat.index(),
                info.texture(),
                info.tex_coord(),
                TextureUsage::Normal,
                image_resources,
            )
        })
        .transpose()
    };

    let get_occlusion_tex = |info: Option<gltf::material::OcclusionTexture>| {
        info.map(|info| {
            resolve_texture_binding(
                path,
                mat.index(),
                info.texture(),
                info.tex_coord(),
                TextureUsage::Data,
                image_resources,
            )
        })
        .transpose()
    };

    let albedo_texture = get_tex(pbr.base_color_texture(), TextureUsage::Color)?;

    // glTF packs roughness in G and metallic in B.
    let metallic_roughness_texture = get_tex(pbr.metallic_roughness_texture(), TextureUsage::Data)?;

    let normal_texture = get_normal_tex(mat.normal_texture())?;
    let ao_texture = get_occlusion_tex(mat.occlusion_texture())?;
    let emissive_texture = get_tex(mat.emissive_texture(), TextureUsage::Color)?;

    let alpha_mode = match mat.alpha_mode() {
        gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
        gltf::material::AlphaMode::Mask => AlphaMode::Mask(mat.alpha_cutoff().unwrap_or(0.5)),
        gltf::material::AlphaMode::Blend => AlphaMode::Blend,
    };

    let mut material = PbrMaterial {
        albedo,
        alpha,
        metallic,
        roughness,
        normal_scale,
        occlusion_strength,
        emissive,
        alpha_mode,
        double_sided: mat.double_sided(),
        albedo_texture,
        metallic_roughness_texture,
        normal_texture,
        ao_texture,
        emissive_texture,
    };
    material.sanitize_factors();

    Ok(Material::Pbr(material))
}

fn resolve_texture_binding(
    path: &Path,
    material_index: Option<usize>,
    texture: gltf::Texture,
    tex_coord: u32,
    usage: TextureUsage,
    image_resources: &[Arc<TextureImage>],
) -> Result<TextureBinding, GltfError> {
    let texture_index = texture.index();
    let source_image_index = texture.source().index();
    let sampler = sampler_state(texture.sampler());
    let tex_coord = TexCoordSet::try_from(tex_coord).map_err(|unsupported| {
        GltfError::Unsupported {
            path: path.to_path_buf(),
            reason: format!(
                "texture {texture_index} for material {material_index:?} uses unsupported TEXCOORD_{unsupported}"
            ),
        }
    })?;
    let image = image_resources
        .get(source_image_index)
        .cloned()
        .ok_or_else(|| GltfError::Texture {
            path: path.to_path_buf(),
            material_index,
            texture_index,
            source_image_index,
        })?;

    Ok(TextureBinding::new(image, sampler, tex_coord, usage))
}

fn sampler_state(sampler: gltf::texture::Sampler) -> SamplerState {
    let wrap_mode = |mode| match mode {
        gltf::texture::WrappingMode::ClampToEdge => WrapMode::ClampToEdge,
        gltf::texture::WrappingMode::MirroredRepeat => WrapMode::MirroredRepeat,
        gltf::texture::WrappingMode::Repeat => WrapMode::Repeat,
    };
    let mag_filter = match sampler.mag_filter() {
        Some(gltf::texture::MagFilter::Nearest) => MagFilter::Nearest,
        Some(gltf::texture::MagFilter::Linear) | None => MagFilter::Linear,
    };
    let min_filter = match sampler.min_filter() {
        Some(gltf::texture::MinFilter::Nearest) => MinFilter::Nearest,
        Some(gltf::texture::MinFilter::Linear) | None => MinFilter::Linear,
        Some(gltf::texture::MinFilter::NearestMipmapNearest) => MinFilter::NearestMipmapNearest,
        Some(gltf::texture::MinFilter::LinearMipmapNearest) => MinFilter::LinearMipmapNearest,
        Some(gltf::texture::MinFilter::NearestMipmapLinear) => MinFilter::NearestMipmapLinear,
        Some(gltf::texture::MinFilter::LinearMipmapLinear) => MinFilter::LinearMipmapLinear,
    };

    SamplerState {
        wrap_u: wrap_mode(sampler.wrap_s()),
        wrap_v: wrap_mode(sampler.wrap_t()),
        mag_filter,
        min_filter,
    }
}

/// Converts decoded glTF image data into a shareable image resource.
fn process_gltf_image(
    path: &Path,
    image_index: usize,
    data: gltf::image::Data,
    use_mipmap: bool,
) -> Result<TextureImage, GltfError> {
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

    Ok(TextureImage::from_image(dyn_img, use_mipmap))
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

#[cfg(test)]
mod tangent_tests {
    use super::*;

    #[test]
    fn mikktspace_preserves_mirrored_uv_handedness_per_corner() {
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let normals = vec![Vector3::z(); positions.len()];
        let texcoords = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
        ];
        let indices = [0, 1, 2, 3, 4, 5];

        let tangents = generate_mikktspace_tangents(&positions, &normals, &texcoords, &indices)
            .expect("valid mirrored UV islands should generate tangents");

        for tangent in &tangents[..3] {
            let direction = Vector3::new(tangent[0], tangent[1], tangent[2]);
            assert!((direction - Vector3::x()).norm() < 1e-5);
            assert_eq!(tangent[3], 1.0);
        }
        for tangent in &tangents[3..] {
            let direction = Vector3::new(tangent[0], tangent[1], tangent[2]);
            assert!((direction + Vector3::x()).norm() < 1e-5);
            assert_eq!(tangent[3], -1.0);
        }
    }

    #[test]
    fn mikktspace_handles_degenerate_uvs_with_finite_tangents() {
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let normals = vec![Vector3::z(); positions.len()];
        let texcoords = [[0.0, 0.0]; 3];

        let tangents = generate_mikktspace_tangents(&positions, &normals, &texcoords, &[0, 1, 2])
            .expect("MikkTSpace should provide its deterministic degenerate-UV fallback");

        assert_eq!(tangents.len(), 3);
        for tangent in tangents {
            let direction = Vector3::new(tangent[0], tangent[1], tangent[2]);
            assert!(direction.iter().all(|component| component.is_finite()));
            assert!((direction.norm() - 1.0).abs() < 1e-5);
            assert!(tangent[3].abs() == 1.0);
        }
    }
}
