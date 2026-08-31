# Performance Benchmarks

The benchmark suite measures the renderer before and during performance-architecture changes. Always build and run it in release mode on an otherwise idle machine.

Run one configured scene:

```bash
cargo run --release -- --config car-scene.toml --benchmark --benchmark-scenario default-car
```

The default run renders 3 warmup frames and records 20 measured frames in `outputs/benchmark.csv`. New files use schema v2 and carry an explicit `schema_version` in every row. The v1 format remains a read-only historical contract; the matrix runner requires v2 for new results.

Schema v2 reports the following timing classes independently for the shadow and main passes:

- `pass_setup`: camera, binding, render-state, sort-key input, and background-resource derivation outside command recording.
- `recording`: queue/command construction and phase sorting.
- `attachment_processing`: depth/color clear and background initialization. The main-pass value includes solid, gradient, or image background work.
- `backend_preparation`: vertex-cache construction, vertex shading, clipping, prepared primitives, and band bins inside submission.
- `rasterization`: ordered execution of prepared band bins. The main pass also retains opaque/masked and transparent rasterization subcolumns.
- `submission_total`: inclusive synchronous submission duration. It contains backend preparation and rasterization, so these nested values must not be added to it.

Scene loading, post-processing, complete-frame duration, and the stable FNV-1a output hash remain directly reported. Timings are wall-clock durations: categories identify ownership and need not sum exactly to `complete_frame` because framework overhead and clock sampling are not redistributed.

Run the representative scenario matrix:

```bash
node scripts/run-benchmarks.mjs
```

The matrix covers a large triangle, 400 small triangles, the default car, the city asset, a high-transparency scene, a deterministic image background, shadows on/off, 1x/2x supersampling, and one versus all configured workers. The image-background case uses the tracked 8x8 color fixture under `benchmarks/fixtures/` and no geometry so attachment/background traversal cost stays visible. The runner writes per-run CSV files plus `baseline.csv`, `baseline.md`, and `metadata.json` under ignored `outputs/benchmarks/`.

Environment variables control suite sampling and the all-worker comparison:

```bash
set BENCHMARK_WARMUP=3
set BENCHMARK_FRAMES=10
set BENCHMARK_THREADS=8
node scripts/run-benchmarks.mjs
```

`BENCHMARK_OUTPUT_ROOT` selects an independent output directory, which makes before/after runs directly comparable:

```bash
set BENCHMARK_OUTPUT_ROOT=outputs/benchmarks/baseline
node scripts/run-benchmarks.mjs
set BENCHMARK_OUTPUT_ROOT=outputs/benchmarks/candidate
node scripts/run-benchmarks.mjs
node scripts/compare-benchmarks.mjs outputs/benchmarks/baseline outputs/benchmarks/candidate benchmarks/performance-exceptions.json
```

The comparison command is the performance gate. It requires schema v2, identical environments and scenario sets, stable matching output hashes, matching workload metadata, and at least five measured frames per scenario. It fails when any candidate full-frame mean is more than 5% slower than its baseline. p95 changes are reported for diagnosis but do not independently fail the gate.

Accepted exceptions belong in the tracked `performance-exceptions.json`, never only in an uncommitted report. Each exception must name the affected scenario, raise its allowed threshold above 5%, and provide a non-empty rationale:

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

Set `BENCHMARK_THREADS` to the machine's physical-core count when simultaneous multithreading makes the runtime's available parallelism larger. Compare only runs with matching schema version, metadata, and output hashes. The opaque and masked queues intentionally share one rasterization measurement because the current renderer prepares and bins them together as one submission.

Committed machine baselines live under `baselines/`; benchmark-backed comparisons for completed optimization phases live under `results/`. The Phase 11 baseline directory retains its raw schema-v2 CSV and metadata so it can be passed directly to `compare-benchmarks.mjs` rather than reconstructed from rounded Markdown values.
