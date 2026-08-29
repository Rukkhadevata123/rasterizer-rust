# Performance Benchmarks

Phase 8 benchmarks measure the existing renderer before changing its performance architecture. Always build and run them in release mode on an otherwise idle machine.

Run one configured scene:

```bash
cargo run --release -- --config car-scene.toml --benchmark --benchmark-scenario default-car
```

The default run renders 3 warmup frames and records 20 measured frames in `outputs/benchmark.csv`. The CSV contains scene-loading time, shadow preparation and rasterization, main-pass preparation, combined opaque/masked rasterization, transparent rasterization, post-processing, complete frame time, and a stable FNV-1a output hash.

Run the representative Phase 8A matrix:

```bash
node scripts/run-benchmarks.mjs
```

The matrix covers a large triangle, 400 small triangles, the default car, the city asset, a high-transparency scene, shadows on/off, 1x/2x supersampling, and one versus all configured workers. It writes per-run CSV files plus `baseline.csv`, `baseline.md`, and `metadata.json` under ignored `outputs/benchmarks/`.

Environment variables control suite sampling and the all-worker comparison:

```bash
set BENCHMARK_WARMUP=3
set BENCHMARK_FRAMES=10
set BENCHMARK_THREADS=8
node scripts/run-benchmarks.mjs
```

Set `BENCHMARK_THREADS` to the machine's physical-core count when simultaneous multithreading makes the runtime's available parallelism larger. Compare only runs with matching metadata and output hashes. The opaque and masked queues intentionally share one rasterization measurement because the current renderer prepares and bins them together as one pass.

Committed machine baselines live under `baselines/`; benchmark-backed comparisons for completed optimization phases live under `results/`.
