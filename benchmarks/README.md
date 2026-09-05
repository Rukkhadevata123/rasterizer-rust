# Performance Benchmarks

The benchmark suite measures the current renderer and enforces its release performance contract.
Always run it in release mode on an otherwise idle machine.

## Single-scene measurement

Run one configured scene:

```bash
cargo run --release -- --config car-scene.toml --benchmark --benchmark-scenario default-car
```

The default run renders 3 warmup frames and records 20 measured frames in
`outputs/benchmark.csv`. The CLI emits schema v2 and writes an explicit `schema_version` in every
row. The matrix runner and performance comparator require schema v2.

## Timing schema

Schema v2 reports the following timing classes independently for the shadow and main passes:

- `pass_setup`: camera, binding, render-state, sort-key input, and background-resource derivation outside command recording.
- `recording`: queue/command construction and phase sorting.
- `attachment_processing`: depth/color clear and background initialization. The main-pass value includes solid, gradient, or image background work.
- `backend_preparation`: vertex-cache construction, vertex shading, clipping, prepared primitives, and band bins inside submission.
- `rasterization`: ordered execution of prepared band bins. The main pass also retains opaque/masked and transparent rasterization subcolumns.
- `submission_total`: inclusive synchronous submission duration. It contains backend preparation and rasterization, so these nested values must not be added to it.

Scene loading, the fused CPU resolve-tonemap pass (`post_processing_ms`), complete-frame duration,
and the stable FNV-1a output hash are also reported. Timings are wall-clock durations: categories
identify ownership and need not sum exactly to `complete_frame_ms` because framework overhead and
clock sampling are not redistributed.

Opaque and masked draws share `main_opaque_masked_rasterization_ms` because they occupy the same
ordered phase. Transparent draws retain their own `main_transparent_rasterization_ms` value.
`main_rasterization_ms` is the sum of those two rasterization values.

## Scenario matrix

Run the representative scenario matrix:

```bash
node scripts/run-benchmarks.mjs
```

The matrix covers a large triangle, 400 small triangles, the default car, the city asset, a
high-transparency scene, and a deterministic image background, including shadows on/off and 1x/2x
supersampling where applicable. Every workload runs for each distinct thread count among one
worker and the configured all-worker count. The image-background case uses the tracked 8x8 color
fixture under `benchmarks/fixtures/` and places its geometry outside the view so
attachment/background traversal cost stays visible.

The runner writes one CSV per scenario and the fixed aggregate files `baseline.csv`, `baseline.md`,
and `metadata.json` below the ignored `outputs/benchmarks/` directory. The filenames describe the
format; an output root may hold either side of a comparison.

Environment variables control suite sampling and the all-worker comparison:

```bash
set BENCHMARK_WARMUP=3
set BENCHMARK_FRAMES=10
set BENCHMARK_THREADS=8
node scripts/run-benchmarks.mjs
```

`BENCHMARK_OUTPUT_ROOT` selects an independent output directory. Capture both sides from adjacent
revisions with the same machine, toolchain, configuration, worker counts, warmups, and measured
frame count. The commands below assume the working tree is at the intended revision before each
matrix run:

```bash
set BENCHMARK_OUTPUT_ROOT=outputs/benchmarks/baseline
node scripts/run-benchmarks.mjs
set BENCHMARK_OUTPUT_ROOT=outputs/benchmarks/candidate
node scripts/run-benchmarks.mjs
node scripts/compare-benchmarks.mjs outputs/benchmarks/baseline outputs/benchmarks/candidate benchmarks/performance-exceptions.json
```

## Comparison contract

The comparison command is the performance gate. It requires:

- schema v2 on both sides;
- identical platform, architecture, CPU, available parallelism, benchmark thread count, warmup and
  measured frame counts, and full `rustc -Vv` metadata;
- identical scenario sets and matching dimensions, supersampling, shadow state, worker count, and
  warmup count within every scenario;
- at least five measured frames per scenario;
- one stable output hash within each scenario and the same hash across both revisions.

Hash validation is a correctness gate independent of timing. A hash mismatch must be investigated
as an output change, not classified as performance noise.

The gate fails when any candidate full-frame mean is more than 5% slower than its adjacent
baseline. p95 changes are reported for diagnosis but do not independently fail the gate. Absolute
times and tracked results captured on another machine are evidence about their own environment, not
a denominator for a new comparison.

CPU power policy, thermals, system load, and run order are not fully captured by metadata. If a
short comparison is noisy, repeat baseline and candidate symmetrically, reverse their order when
useful, and use a longer targeted measurement for any remaining suspect scenario. Do not select a
favorable run; report the result as inconclusive if it does not stabilize.

## Reviewed exceptions

An intentional, reviewed tradeoff may be recorded in the tracked `performance-exceptions.json`;
noise alone is not an exception. Each exception must name an existing scenario, raise its allowed
threshold above 5%, and provide a non-empty rationale:

```json
{
  "schemaVersion": 1,
  "exceptions": [
    {
      "scenario": "example-threads-12",
      "thresholdPercent": 7.5,
      "rationale": "Reviewed tradeoff and affected workload."
    }
  ]
}
```

Set `BENCHMARK_THREADS` to the machine's physical-core count when simultaneous multithreading makes
the runtime's available parallelism larger than the worker topology being evaluated.
