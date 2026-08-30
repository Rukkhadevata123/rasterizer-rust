# Phase 11.0 Pre-11.1 Benchmark Baseline

Measured from `f306e62` (`enforce benchmark regression budget`) immediately before the Phase 11.0 decision gate. The committed `baseline.csv` and `metadata.json` are the direct inputs for subsequent same-machine performance-gate comparisons.

Generated: 2026-08-30T06:31:09.221Z

CPU: 13th Gen Intel(R) Core(TM) i5-13500H

Platform: win32 x64; available parallelism: 16; benchmark all-thread count: 12

Sampling: 3 warmup frames, 10 measured frames

CSV schema: v2

All durations are milliseconds. Submission columns include their backend and rasterization columns.

| Scenario | Load | Shadow setup | Shadow record | Shadow attach | Shadow backend | Shadow raster | Shadow submit | Main setup | Main record | Main attach/bg | Main backend | Main raster | Main submit | Post | Frame mean | Frame p95 | Hash |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| large-triangle-threads-1 | 0.321 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.006 | 0.001 | 0.381 | 0.009 | 9.554 | 9.564 | 12.054 | 22.010 | 22.941 | `df2082fe18b3d495` |
| large-triangle-threads-12 | 0.226 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 | 0.001 | 0.185 | 0.007 | 1.453 | 1.460 | 1.546 | 3.199 | 3.657 | `df2082fe18b3d495` |
| many-small-triangles-threads-1 | 14.821 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.087 | 0.032 | 0.379 | 0.360 | 10.493 | 10.854 | 12.014 | 23.454 | 24.232 | `42e77b74b7b57ff5` |
| many-small-triangles-threads-12 | 15.892 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.100 | 0.024 | 0.128 | 0.306 | 1.646 | 1.953 | 1.569 | 3.846 | 4.567 | `42e77b74b7b57ff5` |
| default-car-threads-1 | 131.949 | 0.017 | 0.002 | 0.364 | 1.625 | 1.557 | 3.183 | 0.005 | 0.020 | 0.298 | 1.757 | 26.906 | 28.664 | 12.141 | 45.060 | 46.307 | `f3b2e2e39737e8e9` |
| default-car-threads-12 | 119.635 | 0.017 | 0.001 | 0.140 | 0.816 | 0.524 | 1.341 | 0.007 | 0.020 | 0.123 | 1.024 | 4.886 | 5.911 | 1.514 | 9.447 | 10.890 | `f3b2e2e39737e8e9` |
| default-car-2x-ssaa-threads-1 | 125.476 | 0.024 | 0.001 | 0.551 | 1.767 | 1.590 | 3.357 | 0.006 | 0.025 | 1.278 | 1.877 | 104.261 | 106.139 | 13.144 | 124.884 | 127.440 | `2880675024a09507` |
| default-car-2x-ssaa-threads-12 | 120.867 | 0.019 | 0.003 | 0.198 | 1.024 | 0.621 | 1.645 | 0.009 | 0.034 | 0.303 | 1.224 | 20.341 | 21.565 | 2.109 | 26.407 | 31.422 | `2880675024a09507` |
| city-threads-1 | 93.886 | 0.330 | 0.050 | 0.490 | 15.256 | 4.149 | 19.406 | 0.011 | 0.025 | 0.505 | 16.716 | 16.002 | 32.718 | 12.528 | 67.544 | 70.159 | `890bb424f63e53e6` |
| city-threads-12 | 79.977 | 0.291 | 0.095 | 0.225 | 5.252 | 1.056 | 6.308 | 0.014 | 0.072 | 0.192 | 6.975 | 2.805 | 9.780 | 1.784 | 20.913 | 22.767 | `890bb424f63e53e6` |
| high-transparency-threads-1 | 24.241 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.092 | 0.093 | 0.421 | 0.198 | 14.944 | 15.143 | 12.365 | 28.173 | 29.107 | `545259e2fe4dda9f` |
| high-transparency-threads-12 | 15.997 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.133 | 0.062 | 0.184 | 0.153 | 2.304 | 2.457 | 1.672 | 4.567 | 4.958 | `545259e2fe4dda9f` |

Raw per-frame samples are in `baseline.csv`.
