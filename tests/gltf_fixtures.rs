use std::path::{Path, PathBuf};

use nalgebra::{Vector2, Vector3};
use rasterizer_rust::error::GltfError;
use rasterizer_rust::io::gltf_loader::load_gltf;
use rasterizer_rust::scene::material::AlphaMode;
use rasterizer_rust::scene::texture::{MagFilter, MinFilter, TexCoordSet, TextureUsage, WrapMode};

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/gltf")
        .join(name)
}

#[test]
fn phase_6a_texture_bindings_share_images_and_preserve_metadata() {
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

    let model = match load_gltf(fixture_path("shared-image-textures.gltf"), false) {
        Ok(model) => model,
        Err(error) => panic!("shared image fixture should load: {error}"),
    };
    let material = model
        .materials
        .first()
        .expect("fixture should produce a material");
    let rasterizer_rust::scene::material::Material::Pbr(material) = material;

    let albedo = material
        .albedo_texture
        .as_ref()
        .expect("texture 1 should resolve its source image");
    let emissive = material
        .emissive_texture
        .as_ref()
        .expect("texture 0 should resolve its source image");
    let metallic_roughness = material
        .metallic_roughness_texture
        .as_ref()
        .expect("texture 0 should bind as linear data");

    assert!(std::sync::Arc::ptr_eq(&albedo.image, &emissive.image));
    assert!(std::sync::Arc::ptr_eq(
        &emissive.image,
        &metallic_roughness.image
    ));
    assert_eq!(albedo.tex_coord, TexCoordSet::TexCoord1);
    assert_eq!(albedo.usage, TextureUsage::Color);
    assert_eq!(albedo.sampler.mag_filter, MagFilter::Linear);
    assert_eq!(albedo.sampler.min_filter, MinFilter::LinearMipmapLinear);
    assert_eq!(albedo.sampler.wrap_u, WrapMode::Repeat);
    assert_eq!(albedo.sampler.wrap_v, WrapMode::Repeat);

    assert_eq!(emissive.tex_coord, TexCoordSet::TexCoord0);
    assert_eq!(emissive.usage, TextureUsage::Color);
    assert_eq!(emissive.sampler.mag_filter, MagFilter::Nearest);
    assert_eq!(emissive.sampler.min_filter, MinFilter::NearestMipmapNearest);
    assert_eq!(emissive.sampler.wrap_u, WrapMode::ClampToEdge);
    assert_eq!(emissive.sampler.wrap_v, WrapMode::MirroredRepeat);
    assert_eq!(metallic_roughness.tex_coord, TexCoordSet::TexCoord0);
    assert_eq!(metallic_roughness.usage, TextureUsage::Data);
    assert_eq!(metallic_roughness.sampler, emissive.sampler);
}

#[test]
fn phase_6c_imports_two_uv_sets_for_material_bindings() {
    let model = load_gltf(fixture_path("shared-image-textures.gltf"), false)
        .expect("two-UV-set fixture should load");
    let vertices = &model.meshes[0].vertices;

    assert_eq!(vertices[0].texcoords[0], Vector2::new(0.0, 0.0));
    assert_eq!(vertices[1].texcoords[0], Vector2::new(1.0, 0.0));
    assert_eq!(vertices[2].texcoords[0], Vector2::new(0.0, 1.0));
    assert_eq!(vertices[0].texcoords[1], Vector2::new(1.0, 1.0));
    assert_eq!(vertices[1].texcoords[1], Vector2::new(0.0, 1.0));
    assert_eq!(vertices[2].texcoords[1], Vector2::new(1.0, 0.0));

    let rasterizer_rust::scene::material::Material::Pbr(material) = &model.materials[0];
    assert_eq!(
        material.albedo_texture.as_ref().unwrap().tex_coord,
        TexCoordSet::TexCoord1
    );
    assert_eq!(
        material.emissive_texture.as_ref().unwrap().tex_coord,
        TexCoordSet::TexCoord0
    );
}

#[test]
fn phase_6c_rejects_higher_uv_sets() {
    let path = fixture_path("unsupported-texcoord-set.gltf");
    let error = match load_gltf(&path, false) {
        Ok(_) => panic!("TEXCOORD_2 should be unsupported"),
        Err(error) => error,
    };

    match error {
        GltfError::Unsupported {
            path: error_path,
            reason,
        } => {
            assert_eq!(error_path, path);
            assert!(reason.contains("TEXCOORD_2"));
        }
        error => panic!("expected unsupported-feature error, got {error}"),
    }
}

#[test]
fn phase_6c_rejects_missing_uv_set_required_by_material() {
    let path = fixture_path("missing-required-texcoord.gltf");
    let error = match load_gltf(&path, false) {
        Ok(_) => panic!("a required TEXCOORD_1 attribute should not be replaced with zeros"),
        Err(error) => error,
    };

    match error {
        GltfError::Primitive { context } => {
            assert_eq!(context.path, path);
            assert!(context.reason.contains("base-color texture"));
            assert!(context.reason.contains("requires TEXCOORD_1"));
        }
        error => panic!("expected contextual primitive error, got {error}"),
    }
}

#[test]
fn phase_6c_rejects_texture_transform_extension() {
    let path = fixture_path("texture-transform.gltf");
    let error = match load_gltf(&path, false) {
        Ok(_) => panic!("texture transforms should be explicit"),
        Err(error) => error,
    };

    match error {
        GltfError::Unsupported {
            path: error_path,
            reason,
        } => {
            assert_eq!(error_path, path);
            assert!(reason.contains("KHR_texture_transform"));
        }
        error => panic!("expected unsupported-feature error, got {error}"),
    }
}

#[test]
fn phase_6d_imports_core_material_factors() {
    let model = load_gltf(fixture_path("shared-image-textures.gltf"), false)
        .expect("material-factor fixture should load");
    let rasterizer_rust::scene::material::Material::Pbr(material) = &model.materials[0];

    assert_eq!(material.albedo, Vector3::new(0.5, 0.25, 0.75));
    assert_eq!(material.alpha, 0.4);
    assert_eq!(material.metallic, 0.8);
    assert_eq!(material.roughness, 0.6);
    assert_eq!(material.normal_scale, 0.5);
    assert_eq!(material.occlusion_strength, 0.25);
    assert_eq!(material.emissive, Vector3::new(0.2, 0.4, 0.6));
    assert_eq!(material.alpha_mode, AlphaMode::Mask(0.35));
    assert!(material.double_sided);
    assert!(material.normal_texture.is_some());
    assert!(material.ao_texture.is_some());
}

#[test]
fn phase_6d_rejects_unsupported_emissive_strength_extension() {
    let path = fixture_path("emissive-strength.gltf");
    let error = match load_gltf(&path, false) {
        Ok(_) => panic!("emissive-strength extension should not be silently ignored"),
        Err(error) => error,
    };

    match error {
        GltfError::Unsupported {
            path: error_path,
            reason,
        } => {
            assert_eq!(error_path, path);
            assert!(reason.contains("KHR_materials_emissive_strength"));
        }
        error => panic!("expected unsupported-feature error, got {error}"),
    }
}

#[test]
fn phase_4c_nested_nodes_with_plane_and_shadow_names_are_loaded() {
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

    let model = match load_gltf(fixture_path("nested-named-nodes.gltf"), false) {
        Ok(model) => model,
        Err(error) => panic!("legitimately named nested nodes should load: {error}"),
    };

    assert_eq!(model.meshes.len(), 1);
    assert_eq!(model.meshes[0].vertices[0].position.x, 1.5);
    let rasterizer_rust::scene::material::Material::Pbr(material) = &model.materials[0];
    assert!(material.double_sided);
}

#[test]
fn phase_4c_supported_triangle_modes_are_converted_to_triangle_lists() {
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

    let model = match load_gltf(fixture_path("triangle-modes.gltf"), false) {
        Ok(model) => model,
        Err(error) => panic!("supported triangle modes should load: {error}"),
    };

    assert_eq!(model.meshes.len(), 4);
    assert_eq!(model.meshes[0].indices, [0, 1, 2]);
    assert_eq!(model.meshes[1].indices, [0, 1, 2]);
    assert_eq!(model.meshes[2].indices, [0, 1, 2, 2, 1, 3]);
    assert_eq!(model.meshes[3].indices, [0, 1, 3, 0, 3, 2]);
}

#[test]
fn mismatched_attributes_return_primitive_context() {
    let path = fixture_path("mismatched-attributes.gltf");
    gltf::Gltf::open(&path).unwrap_or_else(|error| {
        panic!("mismatched-attributes.gltf should reach importer validation: {error}")
    });
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
fn phase_4e_out_of_bounds_indices_return_primitive_context() {
    let path = fixture_path("invalid-index.gltf");
    gltf::Gltf::open(&path).unwrap_or_else(|error| {
        panic!("invalid-index.gltf should reach importer validation: {error}")
    });
    let error = match load_gltf(&path, false) {
        Ok(_) => panic!("out-of-bounds indices should be rejected"),
        Err(error) => error,
    };

    match error {
        GltfError::Primitive { context } => {
            assert_eq!(context.path, path);
            assert!(context.reason.contains("index 3"));
            assert!(context.reason.contains("3 POSITION"));
        }
        error => panic!("expected contextual primitive error, got {error}"),
    }
}

#[test]
fn phase_4e_missing_normals_are_generated_from_triangle_area() {
    let model = match load_gltf(fixture_path("triangle-modes.gltf"), false) {
        Ok(model) => model,
        Err(error) => panic!("triangle fixture should load: {error}"),
    };

    for mesh in &model.meshes {
        for vertex in &mesh.vertices {
            assert!((vertex.normal.x - 0.0).abs() < 1.0e-6);
            assert!((vertex.normal.y - 0.0).abs() < 1.0e-6);
            assert!((vertex.normal.z - 1.0).abs() < 1.0e-6);
        }
    }
}

#[test]
fn phase_4e_invalid_attributes_return_contextual_errors() {
    let missing_position_path = fixture_path("missing-position.gltf");
    let missing_position_error = match load_gltf(&missing_position_path, false) {
        Ok(_) => panic!("missing-position.gltf should be rejected"),
        Err(error) => error,
    };
    match missing_position_error {
        GltfError::Import { path, source } => {
            assert_eq!(path, missing_position_path);
            assert!(source.to_string().contains("POSITION"));
        }
        error => panic!("expected glTF validation error, got {error}"),
    }

    for (name, diagnostic) in [
        ("non-finite-position.gltf", "POSITION[0]"),
        ("zero-normal.gltf", "NORMAL[0]"),
        ("zero-tangent.gltf", "TANGENT[0]"),
    ] {
        let error = match load_gltf(fixture_path(name), false) {
            Ok(_) => panic!("{name} should be rejected"),
            Err(error) => error,
        };
        match error {
            GltfError::Primitive { context } => assert!(
                context.reason.contains(diagnostic),
                "{name} diagnostic was: {}",
                context.reason
            ),
            error => panic!("expected contextual primitive error, got {error}"),
        }
    }
}

#[test]
fn phase_6e_generates_mikktspace_tangents_for_normal_maps() {
    let model = load_gltf(fixture_path("normal-map-without-tangents.gltf"), false)
        .expect("normal-mapped geometry should generate missing tangents");
    let mesh = &model.meshes[0];

    assert_eq!(mesh.vertices.len(), 3);
    assert_eq!(mesh.indices, [0, 2, 1]);
    for vertex in &mesh.vertices {
        assert!((vertex.normal - Vector3::z()).norm() < 1e-5);
        assert!((vertex.tangent.xyz() + Vector3::x()).norm() < 1e-5);
        assert_eq!(vertex.tangent.w, -1.0);
    }
    assert_eq!(
        mesh.vertices[0].position.coords,
        Vector3::new(1.0, -1.5, 0.0)
    );
    assert_eq!(
        mesh.vertices[1].position.coords,
        Vector3::new(-1.0, -1.5, 0.0)
    );
    assert_eq!(
        mesh.vertices[2].position.coords,
        Vector3::new(0.0, 1.5, 0.0)
    );
}

#[test]
fn phase_4c_unsupported_primitive_mode_returns_primitive_context() {
    for (name, expected_mode, diagnostic) in [
        (
            "unsupported-points.gltf",
            gltf::mesh::Mode::Points,
            "Points",
        ),
        ("unsupported-lines.gltf", gltf::mesh::Mode::Lines, "Lines"),
    ] {
        let path = fixture_path(name);
        let gltf = gltf::Gltf::open(&path)
            .unwrap_or_else(|error| panic!("{name} should reach importer validation: {error}"));
        let primitive = gltf
            .meshes()
            .next()
            .unwrap_or_else(|| panic!("{name} should contain a mesh"))
            .primitives()
            .next()
            .unwrap_or_else(|| panic!("{name} should contain a primitive"));
        assert_eq!(primitive.mode(), expected_mode, "unexpected mode in {name}");

        let error = match load_gltf(&path, false) {
            Ok(_) => panic!("{diagnostic} primitives should be rejected"),
            Err(error) => error,
        };

        match error {
            GltfError::Primitive { context } => {
                assert_eq!(context.path, path);
                assert_eq!(context.scene_index, 0);
                assert_eq!(context.node_index, 0);
                assert_eq!(context.mesh_index, 0);
                assert_eq!(context.primitive_index, 0);
                assert!(context.reason.contains(diagnostic));
                assert!(context.reason.contains("unsupported"));
            }
            error => panic!("expected contextual primitive error, got {error}"),
        }
    }
}

#[test]
fn unsupported_image_encoding_returns_image_context() {
    let path = fixture_path("unsupported-image-format.gltf");
    let image_error =
        gltf::import(&path).expect_err("unsupported image fixture should fail during image import");
    assert!(image_error.to_string().contains("image"));

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
