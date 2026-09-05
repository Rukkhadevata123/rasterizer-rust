# Release Benchmark Validation

This directory records the final same-machine comparison between the rendering implementation
before the modernization work and the release candidate.

## Inputs

- Baseline source: `f306e62da39c235adff5461dbe7cde9467d1464d`.
- Candidate source: `951ff2858a258f5188249ae1db4f8fdacf30bff7`.
- Scenario harness: the candidate versions of `scripts/run-benchmarks.mjs`,
  `scripts/benchmark-csv.mjs`, and `benchmarks/fixtures/background-checker.png` on both sides. The
  harness was overlaid in a detached baseline worktree without changing its compiled Rust source.
- Sampling: 3 warmup frames and 10 measured frames per scenario.
- Worker counts: 1 and 12; the machine exposed 16 available hardware threads.
- CPU: 13th Gen Intel Core i5-13500H.
- Platform: Windows x64.
- Toolchain: rustc 1.98.0 (`88d9e12ae`, LLVM 22.1.8, MSVC host).
- Gate: every candidate full-frame mean must be no more than 5% slower than its baseline; hashes
  must be stable within each scenario and identical across sources and worker counts.

The comparison was run twice with reversed order. No other workload was intentionally run between
the adjacent captures.

## Results

| Scenario | Baseline first | Candidate first | Result |
|:--|--:|--:|:--|
| large-triangle-threads-1 | -4.43% | -4.91% | pass |
| large-triangle-threads-12 | -5.05% | -4.60% | pass |
| many-small-triangles-threads-1 | -4.94% | -4.51% | pass |
| many-small-triangles-threads-12 | -7.29% | -11.15% | pass |
| default-car-threads-1 | -2.30% | -1.70% | pass |
| default-car-threads-12 | +0.26% | -7.22% | pass |
| default-car-2x-ssaa-threads-1 | -2.07% | +0.05% | pass |
| default-car-2x-ssaa-threads-12 | +3.37% | +2.98% | pass |
| city-threads-1 | +1.49% | +2.17% | pass |
| city-threads-12 | +4.03% | +1.59% | pass |
| high-transparency-threads-1 | -8.93% | -6.54% | pass |
| high-transparency-threads-12 | -13.34% | -11.35% | pass |
| image-background-threads-1 | +1.87% | +3.26% | pass |
| image-background-threads-12 | -0.63% | +1.98% | pass |

Every scenario passed in both orders without an exception. The largest mean regression was +4.03%
in the baseline-first run and +3.26% in the candidate-first run. All output hashes matched.

The aggregate schema-v2 CSV files retain every per-frame sample. Each adjacent `metadata.json`
records the complete environment contract checked by the comparator. Reproduce the two reports
from the repository root with:

```bash
node scripts/compare-benchmarks.mjs benchmarks/release-validation/2026-09-05-i5-13500h/baseline-first/baseline benchmarks/release-validation/2026-09-05-i5-13500h/baseline-first/candidate benchmarks/performance-exceptions.json
node scripts/compare-benchmarks.mjs benchmarks/release-validation/2026-09-05-i5-13500h/candidate-first/baseline benchmarks/release-validation/2026-09-05-i5-13500h/candidate-first/candidate benchmarks/performance-exceptions.json
```

These measurements apply only to the recorded environment. Future changes require a fresh
same-machine adjacent baseline rather than reusing these values as a performance denominator.
