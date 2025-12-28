mod app;
mod core;
mod io;
mod pipeline;
mod scene;
mod ui;

use clap::Parser;
use io::config::Config;
use log::{info, warn};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "scene.toml")]
    config: String,

    /// Start in GUI mode with real-time rendering
    #[arg(long)]
    gui: bool,
}

fn main() {
    env_logger::init();
    let args = Args::parse();

    info!("Loading configuration from: {}", args.config);
    let config = match Config::load(&args.config) {
        Ok(c) => c,
        Err(e) => {
            warn!(
                "Failed to load config '{}': {}. Using defaults.",
                args.config, e
            );
            Config::default()
        }
    };

    if args.gui {
        app::run_gui(config, &args.config);
    } else {
        app::run_cli(config);
    }
}
