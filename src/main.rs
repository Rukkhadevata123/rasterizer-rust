use clap::Parser;
use log::info;
use rasterizer_rust::app;
use rasterizer_rust::error::ApplicationError;
use rasterizer_rust::io::config::Config;
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "scene.toml")]
    config: String,

    /// Start in GUI mode with real-time rendering
    #[arg(long)]
    gui: bool,
}

fn run() -> Result<(), ApplicationError> {
    let args = Args::parse();

    info!("Loading configuration from: {}", args.config);
    let config = Config::load(&args.config)?;

    if args.gui {
        app::run_gui(config, &args.config)?;
    } else {
        app::run_cli(config)?;
    }

    Ok(())
}

fn main() -> ExitCode {
    env_logger::init();
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
