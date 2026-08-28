import { spawnSync } from "node:child_process";
import { availableParallelism, cpus } from "node:os";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repository = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const outputRoot = join(repository, "outputs", "benchmarks");
const configRoot = join(outputRoot, "configs");
mkdirSync(configRoot, { recursive: true });

const measuredFrames = positiveInteger("BENCHMARK_FRAMES", 10);
const warmupFrames = nonNegativeInteger("BENCHMARK_WARMUP", 3);
const allThreads = positiveInteger("BENCHMARK_THREADS", availableParallelism());
const binary = join(
  repository,
  "target",
  "release",
  process.platform === "win32" ? "rasterizer.exe" : "rasterizer",
);

run("cargo", ["build", "--release"]);

const opaqueTriangle = join(repository, "tests", "fixtures", "gltf", "nested-named-nodes.gltf");
const transparentTriangle = join(repository, "benchmarks", "fixtures", "transparent-triangle.gltf");
const car = join(repository, "assets", "glbs", "old_rusty_car.glb");
const city = join(repository, "assets", "glbs", "ccity_building_set_1.glb");

const gridObjects = (path, count, transparent = false) => {
  const side = Math.ceil(Math.sqrt(count));
  return Array.from({ length: count }, (_, index) => {
    const x = (index % side) - (side - 1) / 2;
    const y = Math.floor(index / side) - (side - 1) / 2;
    const z = transparent ? -(index % 8) * 0.015 : 0;
    return object(path, [x * 0.48, y * 0.32, z], [0.42, 0.42, 0.42]);
  }).join("\n");
};

const cases = [
  {
    name: "large-triangle",
    ssaa: 1,
    shadows: false,
    camera: orthographicCamera(),
    objects: object(opaqueTriangle, [0, 0, 0], [10, 7, 1]),
  },
  {
    name: "many-small-triangles",
    ssaa: 1,
    shadows: false,
    camera: orthographicCamera(),
    objects: gridObjects(opaqueTriangle, 400),
  },
  {
    name: "default-car",
    ssaa: 1,
    shadows: true,
    camera: perspectiveCamera(),
    ground: true,
    objects: object(car, [0, -0.4, 0], [2, 2, 2], [0, -45, 0]),
  },
  {
    name: "default-car-2x-ssaa",
    ssaa: 2,
    shadows: true,
    camera: perspectiveCamera(),
    ground: true,
    objects: object(car, [0, -0.4, 0], [2, 2, 2], [0, -45, 0]),
  },
  {
    name: "city",
    ssaa: 1,
    shadows: true,
    camera: perspectiveCamera([3, 2.5, 5], [0, 0.5, 0]),
    ground: true,
    objects: object(city, [0, 0, 0], [2, 2, 2], [0, -25, 0]),
  },
  {
    name: "high-transparency",
    ssaa: 1,
    shadows: false,
    camera: orthographicCamera(),
    objects: gridObjects(transparentTriangle, 400, true),
  },
];

const mergedRows = [];
const hashes = new Map();
for (const benchmarkCase of cases) {
  const configPath = join(configRoot, `${benchmarkCase.name}.toml`);
  writeFileSync(configPath, config(benchmarkCase), "utf8");
  for (const threads of [...new Set([1, allThreads])]) {
    const scenario = `${benchmarkCase.name}-threads-${threads}`;
    const csvPath = join(outputRoot, `${scenario}.csv`);
    const result = run(
      binary,
      [
        "--config",
        configPath,
        "--benchmark",
        "--benchmark-scenario",
        scenario,
        "--benchmark-warmup",
        String(warmupFrames),
        "--benchmark-frames",
        String(measuredFrames),
        "--benchmark-output",
        csvPath,
      ],
      { RAYON_NUM_THREADS: String(threads) },
    );
    process.stdout.write(result.stdout);
    const [, ...rows] = readFileSync(csvPath, "utf8").trimEnd().split(/\r?\n/);
    const hash = rows[0].split(",").at(-1);
    const expectedHash = hashes.get(benchmarkCase.name);
    if (expectedHash !== undefined && hash !== expectedHash) {
      throw new Error(
        `${benchmarkCase.name} output changed across thread counts: ${expectedHash} != ${hash}`,
      );
    }
    hashes.set(benchmarkCase.name, hash);
    mergedRows.push(...rows);
  }
}

const header = "scenario,frame,width,height,supersample_scale,shadows,rayon_threads,warmup_frames,scene_loading_ms,shadow_preparation_ms,shadow_rasterization_ms,main_preparation_ms,opaque_masked_rasterization_ms,transparent_rasterization_ms,post_processing_ms,complete_frame_ms,output_hash";
const mergedPath = join(outputRoot, "baseline.csv");
writeFileSync(mergedPath, `${header}\n${mergedRows.join("\n")}\n`, "utf8");
process.stdout.write(`merged benchmark CSV: ${mergedPath}\n`);

const metadata = {
  generatedAt: new Date().toISOString(),
  platform: process.platform,
  architecture: process.arch,
  cpu: cpus()[0]?.model ?? "unknown",
  availableParallelism: availableParallelism(),
  benchmarkThreads: allThreads,
  measuredFrames,
  warmupFrames,
  rustc: run("rustc", ["-Vv"]).stdout.trim(),
};
writeFileSync(
  join(outputRoot, "metadata.json"),
  `${JSON.stringify(metadata, null, 2)}\n`,
  "utf8",
);
writeFileSync(
  join(outputRoot, "baseline.md"),
  markdownSummary(metadata, mergedRows),
  "utf8",
);

function config(benchmarkCase) {
  return `[render]
width = 640
height = 360
output = "unused.png"
supersample_scale = ${benchmarkCase.ssaa}
ambient_light = [0.15, 0.15, 0.15]
background_color = [0.02, 0.02, 0.03]
use_shadows = ${benchmarkCase.shadows}
shadow_map_size = 512
shadow_ortho_size = 12.0
use_pcf = true
pcf_kernel_size = 1
use_aces = true
cull_mode = "back"
use_mipmap = true

${benchmarkCase.camera}

[ground]
enabled = ${benchmarkCase.ground ?? false}
size = 10.0

[[lights]]
type = "directional"
direction = [-0.5, -1.0, -0.5]
color = [1.0, 0.9, 0.8]
intensity = 4.0

${benchmarkCase.objects}
`;
}

function object(path, position, scale, rotation = [0, 0, 0]) {
  return `[[objects]]
path = ${JSON.stringify(path.replaceAll("\\", "/"))}
normalization = "normalize"
position = [${position.join(", ")}]
rotation = [${rotation.join(", ")}]
scale = [${scale.join(", ")}]
`;
}

function perspectiveCamera(position = [2, 1.5, 3], target = [0, 0.5, 0]) {
  return `[camera]
projection = "perspective"
position = [${position.join(", ")}]
target = [${target.join(", ")}]
up = [0.0, 1.0, 0.0]
fov = 45.0
near = 0.1
far = 100.0`;
}

function orthographicCamera() {
  return `[camera]
projection = "orthographic"
position = [0.0, 0.0, 10.0]
target = [0.0, 0.0, 0.0]
up = [0.0, 1.0, 0.0]
ortho_height = 8.0
near = 0.1
far = 100.0`;
}

function run(command, args, extraEnvironment = {}) {
  const result = spawnSync(command, args, {
    cwd: repository,
    encoding: "utf8",
    env: { ...process.env, ...extraEnvironment },
    maxBuffer: 64 * 1024 * 1024,
  });
  if (result.status !== 0) {
    process.stderr.write(result.stdout ?? "");
    process.stderr.write(result.stderr ?? "");
    throw new Error(`${command} exited with status ${result.status}`);
  }
  return result;
}

function positiveInteger(name, fallback) {
  const value = nonNegativeInteger(name, fallback);
  if (value === 0) throw new Error(`${name} must be greater than zero`);
  return value;
}

function nonNegativeInteger(name, fallback) {
  const source = process.env[name];
  if (source === undefined) return fallback;
  const value = Number(source);
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`${name} must be a non-negative integer`);
  }
  return value;
}

function markdownSummary(metadata, rows) {
  const groups = new Map();
  for (const row of rows) {
    const columns = row.split(",");
    const scenario = columns[0];
    const samples = groups.get(scenario) ?? [];
    samples.push(columns);
    groups.set(scenario, samples);
  }
  const lines = [
    "# Phase 8A Benchmark Baseline",
    "",
    `Generated: ${metadata.generatedAt}`,
    "",
    `CPU: ${metadata.cpu}`,
    "",
    `Platform: ${metadata.platform} ${metadata.architecture}; available parallelism: ${metadata.availableParallelism}; benchmark all-thread count: ${metadata.benchmarkThreads}`,
    "",
    `Sampling: ${metadata.warmupFrames} warmup frames, ${metadata.measuredFrames} measured frames`,
    "",
    "| Scenario | Load ms | Shadow prep ms | Shadow raster ms | Main prep ms | Opaque/masked ms | Transparent ms | Post ms | Frame mean ms | Frame p95 ms | Hash |",
    "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|",
  ];
  for (const [scenario, samples] of groups) {
    const average = (index) => mean(samples.map((sample) => Number(sample[index])));
    const complete = samples.map((sample) => Number(sample[15]));
    lines.push(
      `| ${scenario} | ${average(8).toFixed(3)} | ${average(9).toFixed(3)} | ${average(10).toFixed(3)} | ${average(11).toFixed(3)} | ${average(12).toFixed(3)} | ${average(13).toFixed(3)} | ${average(14).toFixed(3)} | ${mean(complete).toFixed(3)} | ${percentile(complete, 0.95).toFixed(3)} | \`${samples[0][16]}\` |`,
    );
  }
  lines.push("", "Raw per-frame samples are in `baseline.csv`.", "");
  return lines.join("\n");
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function percentile(values, fraction) {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.ceil((sorted.length - 1) * fraction)];
}
