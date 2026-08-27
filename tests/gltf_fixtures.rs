use std::path::{Path, PathBuf};

use rasterizer_rust::error::GltfError;
use rasterizer_rust::io::gltf_loader::load_gltf;

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/gltf")
        .join(name)
}

#[test]
fn shared_image_fixture_keeps_texture_and_source_indices_distinct() {
    let gltf = gltf::Gltf::open(fixture_path("shared-image-textures.gltf"))
        .expect("shared image fixture should be valid glTF");
    let textures: Vec<_> = gltf.textures().collect();
    let base_color_texture = gltf
        .materials()
        .next()
        .expect("fixture should contain a material")
        .pbr_metallic_roughness()
        .base_color_texture()
        .expect("fixture material should have a base color texture")
        .texture();

    assert_eq!(textures.len(), 2);
    assert_eq!(textures[0].source().index(), 0);
    assert_eq!(textures[1].source().index(), 0);
    assert_eq!(base_color_texture.index(), 1);
    assert_eq!(base_color_texture.source().index(), 0);
}

#[test]
fn shared_image_fixture_loads_texture_by_source_image_index() {
    let model = match load_gltf(fixture_path("shared-image-textures.gltf"), false) {
        Ok(model) => model,
        Err(error) => panic!("shared image fixture should load: {error}"),
    };
    let material = model
        .materials
        .first()
        .expect("fixture should produce a material");
    let rasterizer_rust::scene::material::Material::Pbr(material) = material;

    assert!(material.albedo_texture.is_some());
}

#[test]
fn nested_named_nodes_fixture_preserves_hierarchy_and_names() {
    let gltf = gltf::Gltf::open(fixture_path("nested-named-nodes.gltf"))
        .expect("nested node fixture should be valid glTF");
    let root = gltf
        .default_scene()
        .expect("fixture should have a default scene")
        .nodes()
        .next()
        .expect("fixture should have a root node");
    let child = root.children().next().expect("root should have a child");

    assert_eq!(root.name(), Some("display_plane"));
    assert_eq!(child.name(), Some("shadow_archive"));
    assert!(child.mesh().is_some());
}

#[test]
fn phase_4c_nested_nodes_with_plane_and_shadow_names_are_loaded() {
    let model = match load_gltf(fixture_path("nested-named-nodes.gltf"), false) {
        Ok(model) => model,
        Err(error) => panic!("legitimately named nested nodes should load: {error}"),
    };

    assert_eq!(model.meshes.len(), 1);
    assert_eq!(model.meshes[0].vertices[0].position.x, 1.5);
}

#[test]
fn triangle_modes_fixture_covers_indexed_and_non_indexed_topologies() {
    let gltf = gltf::Gltf::open(fixture_path("triangle-modes.gltf"))
        .expect("triangle modes fixture should be valid glTF");
    let primitives: Vec<_> = gltf
        .meshes()
        .next()
        .expect("fixture should have a mesh")
        .primitives()
        .collect();

    assert_eq!(primitives.len(), 4);
    assert_eq!(primitives[0].mode(), gltf::mesh::Mode::Triangles);
    assert!(primitives[0].indices().is_none());
    assert_eq!(primitives[1].mode(), gltf::mesh::Mode::Triangles);
    assert!(primitives[1].indices().is_some());
    assert_eq!(primitives[2].mode(), gltf::mesh::Mode::TriangleStrip);
    assert_eq!(primitives[3].mode(), gltf::mesh::Mode::TriangleFan);
}

#[test]
fn phase_4c_supported_triangle_modes_are_converted_to_triangle_lists() {
    let model = match load_gltf(fixture_path("triangle-modes.gltf"), false) {
        Ok(model) => model,
        Err(error) => panic!("supported triangle modes should load: {error}"),
    };

    assert_eq!(model.meshes.len(), 4);
    assert_eq!(model.meshes[0].indices, [0, 1, 2]);
    assert_eq!(model.meshes[1].indices, [0, 1, 2]);
    assert_eq!(model.meshes[2].indices, [0, 1, 2, 2, 1, 3]);
    assert_eq!(model.meshes[3].indices, [0, 1, 2, 0, 2, 3]);
}

#[test]
fn malformed_mesh_fixtures_are_structurally_parseable() {
    for name in ["invalid-index.gltf", "mismatched-attributes.gltf"] {
        gltf::Gltf::open(fixture_path(name))
            .unwrap_or_else(|error| panic!("{name} should reach importer validation: {error}"));
    }
}

#[test]
fn mismatched_attributes_return_primitive_context() {
    let path = fixture_path("mismatched-attributes.gltf");
    let error = match load_gltf(&path, false) {
        Ok(_) => panic!("mismatched attributes should be rejected"),
        Err(error) => error,
    };

    match error {
        GltfError::Primitive { context } => {
            assert_eq!(context.path, path);
            assert_eq!(context.scene_index, 0);
            assert_eq!(context.node_index, 0);
            assert_eq!(context.mesh_index, 0);
            assert_eq!(context.primitive_index, 0);
            assert!(context.reason.contains("NORMAL"));
            assert!(context.reason.contains("2 values"));
            assert!(context.reason.contains("3"));
        }
        error => panic!("expected contextual primitive error, got {error}"),
    }
}

#[test]
fn unsupported_feature_fixtures_reach_importer_validation() {
    let lines = gltf::Gltf::open(fixture_path("unsupported-lines.gltf"))
        .expect("line mode fixture should be valid glTF");
    let primitive = lines
        .meshes()
        .next()
        .expect("fixture should have a mesh")
        .primitives()
        .next()
        .expect("fixture should have a primitive");
    assert_eq!(primitive.mode(), gltf::mesh::Mode::Lines);

    let image_error = gltf::import(fixture_path("unsupported-image-format.gltf"))
        .expect_err("unsupported image fixture should fail during image import");
    assert!(image_error.to_string().contains("image"));
}

#[test]
fn phase_4c_unsupported_primitive_mode_returns_primitive_context() {
    for (name, mode) in [
        ("unsupported-points.gltf", "Points"),
        ("unsupported-lines.gltf", "Lines"),
    ] {
        let path = fixture_path(name);
        let error = match load_gltf(&path, false) {
            Ok(_) => panic!("{mode} primitives should be rejected"),
            Err(error) => error,
        };

        match error {
            GltfError::Primitive { context } => {
                assert_eq!(context.path, path);
                assert_eq!(context.scene_index, 0);
                assert_eq!(context.node_index, 0);
                assert_eq!(context.mesh_index, 0);
                assert_eq!(context.primitive_index, 0);
                assert!(context.reason.contains(mode));
                assert!(context.reason.contains("unsupported"));
            }
            error => panic!("expected contextual primitive error, got {error}"),
        }
    }
}

#[test]
fn unsupported_image_encoding_returns_image_context() {
    let path = fixture_path("unsupported-image-format.gltf");
    let error = match load_gltf(&path, false) {
        Ok(_) => panic!("unsupported image encoding should be rejected"),
        Err(error) => error,
    };

    match error {
        GltfError::Image {
            path: error_path,
            image_index,
            reason,
        } => {
            assert_eq!(error_path, path);
            assert_eq!(image_index, 0);
            assert!(reason.contains("image encoding"));
        }
        error => panic!("expected contextual image error, got {error}"),
    }
}
