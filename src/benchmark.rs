//! Benchmark tooling used by the bundled CLI and repository scenario runner.
//!
//! New reports use CSV schema v2. Timing classes separate pass setup, command recording,
//! attachment processing, backend preparation, rasterization, inclusive synchronous submission,
//! post-processing, and complete-frame duration. This tooling module is not an alternate render
//! submission API; measured frames use [`rasterizer_rust::render`] internally.

use crate::error::{ApplicationError, BenchmarkError};
use rasterizer_rust::io::config::{Config, CullModeConfig};
use rasterizer_rust::render::{
    CullMode, FrameResources, GraphicsPipelineState, GraphicsQueue, MainHdrTarget, PolygonMode,
    PresentBuffer, PrimitiveState, RenderDevice, RenderTarget, ResolveTonemapPassDescriptor,
    TonemapOperator, execute_resolve_tonemap_pass, render_main_pass_profiled,
    render_shadow_pass_profiled,
};
use rasterizer_rust::scene::loader::init_scene_resources;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct BenchmarkOptions {
    pub scenario: String,
    pub warmup_frames: usize,
    pub measured_frames: usize,
    pub output: PathBuf,
}

impl BenchmarkOptions {
    pub fn validate(&self) -> Result<(), BenchmarkError> {
        if self.scenario.trim().is_empty() {
            return Err(BenchmarkError::InvalidOptions {
                reason: "scenario name must not be empty".to_string(),
            });
        }
        if self.measured_frames == 0 {
            return Err(BenchmarkError::InvalidOptions {
                reason: "measured frame count must be greater than zero".to_string(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameTimings {
    pub shadow_pass_setup: Duration,
    pub shadow_recording: Duration,
    pub shadow_attachment_processing: Duration,
    pub shadow_backend_preparation: Duration,
    pub shadow_rasterization: Duration,
    pub shadow_submission_total: Duration,
    pub main_pass_setup: Duration,
    pub main_recording: Duration,
    pub main_attachment_processing: Duration,
    pub main_backend_preparation: Duration,
    pub opaque_masked_rasterization: Duration,
    pub transparent_rasterization: Duration,
    pub main_submission_total: Duration,
    pub post_processing: Duration,
    pub complete_frame: Duration,
}

#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    pub scenario: String,
    pub width: usize,
    pub height: usize,
    pub supersample_scale: usize,
    pub shadows: bool,
    pub rayon_threads: usize,
    pub warmup_frames: usize,
    pub scene_loading: Duration,
    pub output_hash: u64,
    pub frames: Vec<FrameTimings>,
}

impl BenchmarkReport {
    pub fn csv(&self) -> String {
        let mut csv = String::from(
            "schema_version,scenario,frame,width,height,supersample_scale,shadows,rayon_threads,warmup_frames,scene_loading_ms,shadow_pass_setup_ms,shadow_recording_ms,shadow_attachment_processing_ms,shadow_backend_preparation_ms,shadow_rasterization_ms,shadow_submission_total_ms,main_pass_setup_ms,main_recording_ms,main_attachment_processing_ms,main_backend_preparation_ms,main_rasterization_ms,main_opaque_masked_rasterization_ms,main_transparent_rasterization_ms,main_submission_total_ms,post_processing_ms,complete_frame_ms,output_hash\n",
        );
        let scenario = csv_field(&self.scenario);
        for (frame_index, frame) in self.frames.iter().enumerate() {
            let main_rasterization =
                frame.opaque_masked_rasterization + frame.transparent_rasterization;
            let fields = [
                "2".to_string(),
                scenario.clone(),
                frame_index.to_string(),
                self.width.to_string(),
                self.height.to_string(),
                self.supersample_scale.to_string(),
                self.shadows.to_string(),
                self.rayon_threads.to_string(),
                self.warmup_frames.to_string(),
                format!("{:.6}", milliseconds(self.scene_loading)),
                format!("{:.6}", milliseconds(frame.shadow_pass_setup)),
                format!("{:.6}", milliseconds(frame.shadow_recording)),
                format!("{:.6}", milliseconds(frame.shadow_attachment_processing)),
                format!("{:.6}", milliseconds(frame.shadow_backend_preparation)),
                format!("{:.6}", milliseconds(frame.shadow_rasterization)),
                format!("{:.6}", milliseconds(frame.shadow_submission_total)),
                format!("{:.6}", milliseconds(frame.main_pass_setup)),
                format!("{:.6}", milliseconds(frame.main_recording)),
                format!("{:.6}", milliseconds(frame.main_attachment_processing)),
                format!("{:.6}", milliseconds(frame.main_backend_preparation)),
                format!("{:.6}", milliseconds(main_rasterization)),
                format!("{:.6}", milliseconds(frame.opaque_masked_rasterization)),
                format!("{:.6}", milliseconds(frame.transparent_rasterization)),
                format!("{:.6}", milliseconds(frame.main_submission_total)),
                format!("{:.6}", milliseconds(frame.post_processing)),
                format!("{:.6}", milliseconds(frame.complete_frame)),
                format!("{:016x}", self.output_hash),
            ];
            writeln!(csv, "{}", fields.join(",")).expect("writing to a String cannot fail");
        }
        csv
    }

    pub fn summary(&self) -> String {
        let complete_frames: Vec<_> = self
            .frames
            .iter()
            .map(|frame| frame.complete_frame)
            .collect();
        let mean = mean_duration(&complete_frames);
        let p95 = percentile_duration(&complete_frames, 0.95);
        let shadow_submission = mean_duration(
            &self
                .frames
                .iter()
                .map(|frame| frame.shadow_submission_total)
                .collect::<Vec<_>>(),
        );
        let main_submission = mean_duration(
            &self
                .frames
                .iter()
                .map(|frame| frame.main_submission_total)
                .collect::<Vec<_>>(),
        );
        format!(
            "benchmark '{}' | schema v2 | {}x{} | {}x SSAA | shadows={} | Rayon threads={} | scene load={:.3} ms | shadow submit mean={:.3} ms | main submit mean={:.3} ms | frame mean={:.3} ms | frame p95={:.3} ms | hash={:016x}",
            self.scenario,
            self.width,
            self.height,
            self.supersample_scale,
            self.shadows,
            self.rayon_threads,
            milliseconds(self.scene_loading),
            milliseconds(shadow_submission),
            milliseconds(main_submission),
            milliseconds(mean),
            milliseconds(p95),
            self.output_hash,
        )
    }

    pub fn write_csv(&self, path: &Path) -> Result<(), BenchmarkError> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(|source| BenchmarkError::Write {
                path: path.to_path_buf(),
                source,
            })?;
        }
        std::fs::write(path, self.csv()).map_err(|source| BenchmarkError::Write {
            path: path.to_path_buf(),
            source,
        })
    }
}

pub fn run_benchmark(
    config: Config,
    options: &BenchmarkOptions,
) -> Result<BenchmarkReport, ApplicationError> {
    options.validate()?;
    config
        .validate()
        .map_err(|source| ApplicationError::InvalidConfiguration { source })?;

    let scene_loading_started = Instant::now();
    let context = init_scene_resources(&config)?;
    let scene_loading = scene_loading_started.elapsed();

    let mut queue = RenderDevice::new().create_queue();
    let mut target = MainHdrTarget::new(
        config.render.width,
        config.render.height,
        config.render.supersample_scale,
    )
    .map_err(|source| ApplicationError::RenderTargetInitialization {
        target: "main framebuffer",
        source,
    })?;
    let mut shadow_target = RenderTarget::new(
        config.render.shadow_map_size,
        config.render.shadow_map_size,
        1,
    )
    .map_err(|source| ApplicationError::RenderTargetInitialization {
        target: "shadow framebuffer",
        source,
    })?;
    let mut frame_resources = FrameResources::new();
    let mut present = PresentBuffer::new(config.render.width, config.render.height)
        .map_err(|source| ApplicationError::PresentBufferInitialization { source })?;
    let pipeline_state = GraphicsPipelineState {
        primitive: PrimitiveState {
            cull_mode: match config.render.cull_mode {
                CullModeConfig::None => CullMode::None,
                CullModeConfig::Front => CullMode::Front,
                CullModeConfig::Back => CullMode::Back,
            },
            polygon_mode: if config.render.wireframe {
                PolygonMode::Line
            } else {
                PolygonMode::Fill
            },
            ..Default::default()
        },
        ..Default::default()
    };

    for _ in 0..options.warmup_frames {
        let _ = render_profiled_frame(
            &config,
            &context,
            &mut queue,
            (&mut shadow_target, &mut target),
            &mut frame_resources,
            &mut present,
            pipeline_state,
        )?;
    }

    let mut frames = Vec::with_capacity(options.measured_frames);
    let mut output_hash = None;
    for frame_index in 0..options.measured_frames {
        frames.push(render_profiled_frame(
            &config,
            &context,
            &mut queue,
            (&mut shadow_target, &mut target),
            &mut frame_resources,
            &mut present,
            pipeline_state,
        )?);
        let hash = fnv1a_hash(present.pixels());
        if let Some(expected) = output_hash
            && hash != expected
        {
            return Err(BenchmarkError::OutputChanged {
                frame_index,
                expected,
                actual: hash,
            }
            .into());
        }
        output_hash = Some(hash);
    }

    Ok(BenchmarkReport {
        scenario: options.scenario.clone(),
        width: config.render.width,
        height: config.render.height,
        supersample_scale: config.render.supersample_scale,
        shadows: config.render.use_shadows,
        rayon_threads: rayon::current_num_threads(),
        warmup_frames: options.warmup_frames,
        scene_loading,
        output_hash: output_hash.expect("measured frame count was validated as nonzero"),
        frames,
    })
}

fn render_profiled_frame(
    config: &Config,
    context: &rasterizer_rust::scene::context::RenderScene,
    queue: &mut GraphicsQueue,
    targets: (&mut RenderTarget, &mut MainHdrTarget),
    resources: &mut FrameResources,
    present: &mut PresentBuffer,
    pipeline_state: GraphicsPipelineState,
) -> Result<FrameTimings, ApplicationError> {
    let frame_started = Instant::now();
    let (shadow_target, target) = targets;
    let (shadow, shadow_timings) =
        render_shadow_pass_profiled(config, context, queue, shadow_target, resources);
    let main_timings = render_main_pass_profiled(
        config,
        context,
        queue,
        target,
        resources,
        &shadow,
        pipeline_state,
    )?;
    let post_started = Instant::now();
    execute_resolve_tonemap_pass(ResolveTonemapPassDescriptor {
        label: Some("benchmark-present"),
        source: target,
        destination: present,
        exposure: config.render.exposure,
        tonemap: if config.render.use_aces {
            TonemapOperator::Aces
        } else {
            TonemapOperator::None
        },
    })?;
    let post_processing = post_started.elapsed();

    Ok(FrameTimings {
        shadow_pass_setup: shadow_timings.pass_setup,
        shadow_recording: shadow_timings.recording,
        shadow_attachment_processing: shadow_timings.attachment_processing,
        shadow_backend_preparation: shadow_timings.backend_preparation,
        shadow_rasterization: shadow_timings.rasterization,
        shadow_submission_total: shadow_timings.submission_total,
        main_pass_setup: main_timings.pass_setup,
        main_recording: main_timings.recording,
        main_attachment_processing: main_timings.attachment_processing,
        main_backend_preparation: main_timings.backend_preparation,
        opaque_masked_rasterization: main_timings.opaque_masked_rasterization,
        transparent_rasterization: main_timings.transparent_rasterization,
        main_submission_total: main_timings.submission_total,
        post_processing,
        complete_frame: frame_started.elapsed(),
    })
}

fn fnv1a_hash(buffer: &[u32]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for pixel in buffer {
        for byte in pixel.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn mean_duration(samples: &[Duration]) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    Duration::from_secs_f64(
        samples.iter().map(Duration::as_secs_f64).sum::<f64>() / samples.len() as f64,
    )
}

fn percentile_duration(samples: &[Duration], percentile: f64) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let index = ((sorted.len() - 1) as f64 * percentile).ceil() as usize;
    sorted[index]
}

fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_options_require_a_scenario_and_measured_frame() {
        let mut options = BenchmarkOptions {
            scenario: " ".to_string(),
            warmup_frames: 0,
            measured_frames: 1,
            output: PathBuf::from("benchmark.csv"),
        };
        assert!(
            options
                .validate()
                .unwrap_err()
                .to_string()
                .contains("scenario")
        );

        options.scenario = "fixture".to_string();
        options.measured_frames = 0;
        assert!(
            options
                .validate()
                .unwrap_err()
                .to_string()
                .contains("greater than zero")
        );
    }

    #[test]
    fn benchmark_csv_contains_each_frame_and_stable_metadata() {
        let report = BenchmarkReport {
            scenario: "small,fixture".to_string(),
            width: 64,
            height: 32,
            supersample_scale: 2,
            shadows: false,
            rayon_threads: 1,
            warmup_frames: 3,
            scene_loading: Duration::from_millis(5),
            output_hash: 0x1234,
            frames: vec![FrameTimings {
                complete_frame: Duration::from_millis(10),
                ..Default::default()
            }],
        };

        let csv = report.csv();
        assert!(csv.starts_with("schema_version,scenario,frame,width"));
        assert!(csv.contains("2,\"small,fixture\",0,64,32,2,false,1,3"));
        assert!(csv.contains("0000000000001234"));
    }

    #[test]
    fn benchmark_csv_v2_reports_inclusive_and_nested_main_timings() {
        let report = BenchmarkReport {
            scenario: "timing-fixture".to_string(),
            width: 1,
            height: 1,
            supersample_scale: 1,
            shadows: false,
            rayon_threads: 1,
            warmup_frames: 0,
            scene_loading: Duration::ZERO,
            output_hash: 0,
            frames: vec![FrameTimings {
                main_backend_preparation: Duration::from_millis(3),
                opaque_masked_rasterization: Duration::from_millis(5),
                transparent_rasterization: Duration::from_millis(7),
                main_submission_total: Duration::from_millis(20),
                ..Default::default()
            }],
        };

        let csv = report.csv();
        let mut lines = csv.lines();
        let columns = lines.next().unwrap().split(',');
        let values = lines.next().unwrap().split(',');
        let row: std::collections::HashMap<_, _> = columns.zip(values).collect();

        assert_eq!(row["schema_version"], "2");
        assert_eq!(row["main_backend_preparation_ms"], "3.000000");
        assert_eq!(row["main_rasterization_ms"], "12.000000");
        assert_eq!(row["main_submission_total_ms"], "20.000000");
    }

    #[test]
    fn output_hash_changes_with_pixel_order_and_value() {
        assert_ne!(fnv1a_hash(&[1, 2]), fnv1a_hash(&[2, 1]));
        assert_ne!(fnv1a_hash(&[1, 2]), fnv1a_hash(&[1, 3]));
        assert_eq!(fnv1a_hash(&[1, 2]), fnv1a_hash(&[1, 2]));
    }

    #[test]
    fn percentile_uses_nearest_rank_at_or_above_requested_percentile() {
        let samples = [
            Duration::from_millis(1),
            Duration::from_millis(2),
            Duration::from_millis(3),
            Duration::from_millis(4),
        ];
        assert_eq!(
            percentile_duration(&samples, 0.95),
            Duration::from_millis(4)
        );
    }
}
