use clap::Parser;
use log::info;
use rasterizer_rust::app;
use rasterizer_rust::benchmark::{BenchmarkOptions, run_benchmark};
use rasterizer_rust::error::ApplicationError;
use rasterizer_rust::io::config::Config;
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "car-scene.toml")]
    config: String,

    /// Start in GUI mode with real-time rendering
    #[arg(long)]
    gui: bool,

    /// Measure repeatable headless frame timings instead of saving one image
    #[arg(long, conflicts_with = "gui")]
    benchmark: bool,

    /// Human-readable scenario label stored in benchmark output
    #[arg(long, default_value = "configured-scene")]
    benchmark_scenario: String,

    /// Frames rendered before benchmark measurement begins
    #[arg(long, default_value_t = 3)]
    benchmark_warmup: usize,

    /// Frames included in benchmark statistics
    #[arg(long, default_value_t = 20)]
    benchmark_frames: usize,

    /// CSV path, resolved relative to the configuration file
    #[arg(long, default_value = "outputs/benchmark.csv")]
    benchmark_output: String,
}

fn run() -> Result<(), ApplicationError> {
    let args = Args::parse();

    info!("Loading configuration from: {}", args.config);
    let config = Config::load(&args.config)?;

    if args.benchmark {
        let options = BenchmarkOptions {
            scenario: args.benchmark_scenario,
            warmup_frames: args.benchmark_warmup,
            measured_frames: args.benchmark_frames,
            output: config.resolve_path(args.benchmark_output),
        };
        let report = run_benchmark(config, &options)?;
        report.write_csv(&options.output)?;
        println!("{}", report.summary());
        println!("benchmark CSV: {}", options.output.display());
    } else if args.gui {
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
