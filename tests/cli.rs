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
    Command::new(env!("CARGO_BIN_EXE_rasterizer"))
        .args(["--config", &path.to_string_lossy()])
        .output()
        .expect("rasterizer process should start")
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
