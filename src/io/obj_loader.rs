use crate::core::geometry::Vertex;
use crate::scene::mesh::Mesh;
use log::{info, warn};
use nalgebra::{Point3, Vector2, Vector3};
use std::path::Path;

/// Loads an OBJ file and returns a unified Mesh.
///
/// # Arguments
/// * `path` - The file path to the .obj file.
///
/// # Returns
/// * `Result<Mesh, String>` - The loaded mesh or an error message.
pub fn load_obj(path: &str) -> Result<Mesh, String> {
    let path_obj = Path::new(path);
    if !path_obj.exists() {
        return Err(format!("File not found: {}", path));
    }

    info!("Loading OBJ file: {}", path);

    // TODO: Allow configuring load options via arguments?
    let load_options = tobj::LoadOptions {
        triangulate: true,
        single_index: true, // Important: Unifies indices for Position/Normal/UV
        ..Default::default()
    };

    // Load the OBJ file
    let (models, _materials) = tobj::load_obj(path_obj, &load_options)
        .map_err(|e| format!("Failed to load OBJ: {}", e))?;

    // TODO: Handle materials (MTL files).
    // Currently, we ignore `_materials` and use a default shader/material for everything.

    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut index_offset = 0;

    // Iterate over all models found in the OBJ file
    for model in models {
        let mesh = &model.mesh;
        let num_vertices = mesh.positions.len() / 3;

        // TODO: Instead of merging all sub-meshes into one big Mesh,
        // we should probably return a `Model` struct containing a list of `Mesh`es.
        // This would allow different parts of the model to have different materials.

        // Check for data availability
        let has_normals = !mesh.normals.is_empty();
        let has_texcoords = !mesh.texcoords.is_empty();

        if !has_normals {
            warn!(
                "Mesh '{}' is missing normals. Using default (0, 1, 0).",
                model.name
            );
            // TODO: Implement algorithm to calculate smooth normals based on face connectivity.
        }

        // Process vertices
        for i in 0..num_vertices {
            // 1. Position (x, y, z)
            let px = mesh.positions[i * 3];
            let py = mesh.positions[i * 3 + 1];
            let pz = mesh.positions[i * 3 + 2];

            // 2. Normal (nx, ny, nz)
            let (nx, ny, nz) = if has_normals {
                (
                    mesh.normals[i * 3],
                    mesh.normals[i * 3 + 1],
                    mesh.normals[i * 3 + 2],
                )
            } else {
                (0.0, 1.0, 0.0) // Default up vector
            };

            // 3. Texture Coordinates (u, v)
            let (u, v) = if has_texcoords {
                (mesh.texcoords[i * 2], mesh.texcoords[i * 2 + 1])
            } else {
                (0.0, 0.0)
            };

            vertices.push(Vertex {
                position: Point3::new(px, py, pz),
                normal: Vector3::new(nx, ny, nz),
                texcoord: Vector2::new(u, v),
            });
        }

        // Process indices
        // Since we are merging meshes, we need to offset the indices by the number of vertices
        // already added from previous meshes.
        for index in &mesh.indices {
            indices.push(index + index_offset);
        }

        index_offset += num_vertices as u32;
    }

    info!(
        "OBJ loaded successfully. Total vertices: {}, Total indices: {}",
        vertices.len(),
        indices.len()
    );

    Ok(Mesh::new(vertices, indices))
}

// TODO: smooth normals calculation function here?
