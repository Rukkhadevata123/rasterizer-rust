use crate::io::config_loader::TomlConfigLoader;
use crate::io::render_settings::RenderSettings;
use clap::Parser;

/// 🔥 **极简CLI** - 专注配置文件和GUI控制
#[derive(Parser, Debug)]
#[command(name = "rasterizer")]
#[command(about = "🎨 TOML驱动的光栅化渲染器")]
pub struct SimpleCli {
    /// 📁 配置文件路径（TOML格式）
    #[arg(short, long, value_name = "FILE")]
    pub config: Option<String>,

    /// 🚀 无头模式（不启动GUI）
    #[arg(long)]
    pub headless: bool,

    /// 📋 使用示例配置（临时创建并加载）
    #[arg(long)]
    pub use_example_config: bool,
}

impl SimpleCli {
    /// 🔥 **处理CLI参数并返回RenderSettings和是否启动GUI**
    pub fn process() -> Result<(RenderSettings, bool), String> {
        let cli = Self::parse();

        // 处理示例配置
        if cli.use_example_config {
            let temp_config_path = "temp_example_config.toml";

            // 创建临时示例配置文件
            TomlConfigLoader::create_example_config(temp_config_path)
                .map_err(|e| format!("创建示例配置失败: {}", e))?;

            println!("✅ 已创建临时示例配置: {}", temp_config_path);

            // 加载示例配置
            let settings = TomlConfigLoader::load_from_file(temp_config_path)
                .map_err(|e| format!("加载示例配置失败: {}", e))?;

            // 不需要删除临时文件，用户可以当模板
            // let _ = std::fs::remove_file(temp_config_path);

            let should_start_gui = !cli.headless;
            return Ok((settings, should_start_gui));
        }

        // 加载配置文件或使用默认设置
        let settings = if let Some(config_path) = &cli.config {
            println!("📁 加载配置文件: {}", config_path);
            TomlConfigLoader::load_from_file(config_path)
                .map_err(|e| format!("配置文件加载失败: {}", e))?
        } else {
            println!("💡 使用默认设置");
            RenderSettings::default()
        };

        let should_start_gui = !cli.headless;
        Ok((settings, should_start_gui))
    }
}
