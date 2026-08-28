use std::process::Command;

fn write_temp_config(name: &str, content: &str) -> std::path::PathBuf {
    let directory =
        std::env::temp_dir().join(format!("rasterizer-cli-tests-{}", std::process::id()));
    std::fs::create_dir_all(&directory).expect("temporary test directory should be created");
    let path = directory.join(name);
    std::fs::write(&path, content).expect("temporary config should be written");
    path
}

fn run_with_config(path: &std::path::Path) -> std::process::Output {
    run_with_config_from(path, std::path::Path::new(env!("CARGO_MANIFEST_DIR")))
}

fn run_with_config_from(
    path: &std::path::Path,
    working_directory: &std::path::Path,
) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_rasterizer"))
        .args(["--config", &path.to_string_lossy()])
        .current_dir(working_directory)
        .output()
        .expect("rasterizer process should start")
}

fn write_minimal_gltf(directory: &std::path::Path) {
    let mut buffer = Vec::new();
    for value in [-0.5_f32, -0.5, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, 0.0] {
        buffer.extend_from_slice(&value.to_le_bytes());
    }
    for value in [0.0_f32, 0.0, 1.0, 0.0, 0.0, 1.0] {
        buffer.extend_from_slice(&value.to_le_bytes());
    }
    for index in [0_u16, 1, 2] {
        buffer.extend_from_slice(&index.to_le_bytes());
    }
    std::fs::write(directory.join("triangle.bin"), buffer).expect("glTF buffer should be written");
    image::RgbaImage::from_pixel(1, 1, image::Rgba([255, 255, 255, 255]))
        .save(directory.join("model-texture.png"))
        .expect("glTF texture should be saved");
    std::fs::write(
        directory.join("triangle.gltf"),
        r#"{
            "asset": { "version": "2.0" },
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "meshes": [{ "primitives": [{
                "attributes": { "POSITION": 0, "TEXCOORD_0": 1 },
                "indices": 2,
                "material": 0
            }] }],
            "images": [{ "uri": "model-texture.png" }],
            "textures": [{ "source": 0 }],
            "materials": [{
                "pbrMetallicRoughness": {
                    "baseColorTexture": { "index": 0 }
                }
            }],
            "buffers": [{ "uri": "triangle.bin", "byteLength": 66 }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962 },
                { "buffer": 0, "byteOffset": 36, "byteLength": 24, "target": 34962 },
                { "buffer": 0, "byteOffset": 60, "byteLength": 6, "target": 34963 }
            ],
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,
                    "count": 3,
                    "type": "VEC3",
                    "min": [-0.5, -0.5, 0.0],
                    "max": [0.5, 0.5, 0.0]
                },
                {
                    "bufferView": 1,
                    "componentType": 5126,
                    "count": 3,
                    "type": "VEC2"
                },
                {
                    "bufferView": 2,
                    "componentType": 5123,
                    "count": 3,
                    "type": "SCALAR"
                }
            ]
        }"#,
    )
    .expect("glTF document should be written");
}

fn output_config(path: &std::path::Path) -> String {
    format!(
        r#"
            lights = []
            objects = []

            [render]
            width = 1
            height = 1
            use_shadows = false
            output = '{}'

            [ground]
            enabled = false
        "#,
        path.display()
    )
}

#[test]
fn missing_requested_config_returns_failure() {
    let missing_config = std::env::temp_dir().join(format!(
        "rasterizer-missing-config-{}.toml",
        std::process::id()
    ));
    let output = run_with_config(&missing_config);

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("failed to read config"));
    assert!(stderr.contains(missing_config.to_string_lossy().as_ref()));
    assert!(!stderr.contains("Using defaults"));
}

#[test]
fn invalid_requested_config_returns_failure() {
    let config = write_temp_config("invalid-config.toml", "[render]\nwidth = 0");
    let output = run_with_config(&config);

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("invalid config"));
    assert!(stderr.contains("dimensions must be greater than zero"));
    assert!(!stderr.contains("Using defaults"));
}

#[test]
fn missing_model_returns_failure_without_fallback() {
    let config = write_temp_config(
        "missing-model.toml",
        r#"
            lights = []

            [render]
            width = 1
            height = 1
            use_shadows = false

            [ground]
            enabled = false

            [[objects]]
            path = "definitely-missing.glb"
        "#,
    );
    let output = run_with_config(&config);

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("failed to load object 0"));
    assert!(stderr.contains("definitely-missing.glb"));
    assert!(stderr.contains(config.parent().unwrap().to_string_lossy().as_ref()));
    assert!(!stderr.contains("fallback mesh"));
}

#[test]
fn missing_background_image_returns_failure() {
    let config = write_temp_config(
        "missing-background.toml",
        r#"
            lights = []
            objects = []

            [render]
            width = 1
            height = 1
            use_shadows = false
            background_image = "definitely-missing.png"

            [ground]
            enabled = false
        "#,
    );
    let output = run_with_config(&config);

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("failed to load background image"));
    assert!(stderr.contains("definitely-missing.png"));
}

#[test]
fn output_parent_directories_are_created() {
    let output_path = std::env::temp_dir()
        .join(format!("rasterizer-output-test-{}", std::process::id()))
        .join("nested")
        .join("image.png");
    if let Some(root) = output_path.parent().and_then(std::path::Path::parent) {
        let _ = std::fs::remove_dir_all(root);
    }
    let config = write_temp_config("nested-output.toml", &output_config(&output_path));
    let output = run_with_config(&config);

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output_path.is_file());
}

#[test]
fn output_directory_creation_failure_returns_failure() {
    let root = std::env::temp_dir().join(format!(
        "rasterizer-output-parent-file-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&root);
    let _ = std::fs::remove_dir_all(&root);
    std::fs::write(&root, b"not a directory").expect("parent fixture should be written");
    let output_path = root.join("image.png");
    let config = write_temp_config("invalid-output-parent.toml", &output_config(&output_path));
    let output = run_with_config(&config);

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("failed to create output directory"));
    assert!(stderr.contains(root.to_string_lossy().as_ref()));
}

#[test]
fn image_save_failure_returns_failure() {
    let output_path = std::env::temp_dir().join(format!(
        "rasterizer-unsupported-output-{}.unsupported",
        std::process::id()
    ));
    let config = write_temp_config("unsupported-output.toml", &output_config(&output_path));
    let output = run_with_config(&config);

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("failed to save image"));
    assert!(stderr.contains(output_path.to_string_lossy().as_ref()));
}

#[test]
fn nested_config_resolves_all_paths_independently_of_working_directory() {
    let root =
        std::env::temp_dir().join(format!("rasterizer-relative-paths-{}", std::process::id()));
    let config_directory = root.join("configs").join("nested");
    let working_directory = root.join("unrelated-working-directory");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&config_directory).expect("config directory should be created");
    std::fs::create_dir_all(&working_directory).expect("working directory should be created");
    write_minimal_gltf(&config_directory);
    image::RgbImage::from_pixel(1, 1, image::Rgb([32, 64, 128]))
        .save(config_directory.join("background.png"))
        .expect("background fixture should be saved");

    let config_path = config_directory.join("scene.toml");
    std::fs::write(
        &config_path,
        r#"
            lights = []

            [render]
            width = 2
            height = 2
            use_shadows = false
            background_image = "background.png"
            output = "outputs/render.png"

            [ground]
            enabled = false

            [[objects]]
            path = "triangle.gltf"
        "#,
    )
    .expect("nested config should be written");

    let relative_config_path = std::path::Path::new("../configs/nested/scene.toml");
    let output = run_with_config_from(relative_config_path, &working_directory);

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(config_directory.join("outputs/render.png").is_file());
    assert!(!working_directory.join("outputs/render.png").exists());
}
