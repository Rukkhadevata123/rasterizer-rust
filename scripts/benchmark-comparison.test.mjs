import assert from "node:assert/strict";
import test from "node:test";
import {
  DEFAULT_REGRESSION_THRESHOLD_PERCENT,
  MINIMUM_MEASURED_SAMPLES,
  compareBenchmarkRuns,
} from "./benchmark-comparison.mjs";
import {
  BENCHMARK_V2_COLUMNS,
  formatCsvRow,
  parseBenchmarkCsv,
} from "./benchmark-csv.mjs";

const metadata = Object.freeze({
  schemaVersion: 2,
  generatedAt: "2026-08-30T00:00:00.000Z",
  platform: "win32",
  architecture: "x64",
  cpu: "fixture cpu",
  availableParallelism: 16,
  benchmarkThreads: 12,
  measuredFrames: 5,
  warmupFrames: 3,
  rustc: "rustc fixture",
});

test("full-frame mean changes at or below five percent pass", () => {
  const comparison = compare([100, 100, 100, 100, 100], [105, 105, 105, 105, 105]);

  assert.equal(comparison.thresholdPercent, DEFAULT_REGRESSION_THRESHOLD_PERCENT);
  assert.equal(comparison.minimumSamples, MINIMUM_MEASURED_SAMPLES);
  assert.equal(comparison.passed, true);
  assert.equal(comparison.results[0].status, "pass");
  assert.equal(comparison.results[0].meanChangePercent, 5);
});

test("full-frame mean regressions above five percent fail", () => {
  const comparison = compare([100, 100, 100, 100, 100], [106, 106, 106, 106, 106]);

  assert.equal(comparison.passed, false);
  assert.equal(comparison.results[0].status, "regression");
});

test("scenario exceptions require a higher threshold and rationale", () => {
  const accepted = compare([100, 100, 100, 100, 100], [106, 106, 106, 106, 106], {
    exceptions: [
      {
        scenario: "fixture-threads-12",
        thresholdPercent: 7,
        rationale: "Reviewed tradeoff for the fixture.",
      },
    ],
  });
  assert.equal(accepted.passed, true);
  assert.equal(accepted.results[0].status, "exception");

  assert.throws(
    () =>
      compare([100, 100, 100, 100, 100], [106, 106, 106, 106, 106], {
        exceptions: [
          {
            scenario: "fixture-threads-12",
            thresholdPercent: 7,
            rationale: " ",
          },
        ],
      }),
    /requires a rationale/,
  );
});

test("comparison rejects mismatched environments, hashes, and sample counts", () => {
  assert.throws(
    () =>
      compare([100, 100, 100, 100, 100], [100, 100, 100, 100, 100], {
        candidateMetadata: { ...metadata, cpu: "different cpu" },
      }),
    /metadata field 'cpu' differs/,
  );

  assert.throws(
    () =>
      compare([100, 100, 100, 100, 100], [100, 100, 100, 100, 100], {
        candidateHash: "different",
      }),
    /output hash changed/,
  );

  const fourSampleMetadata = { ...metadata, measuredFrames: 4 };
  assert.throws(
    () =>
      compare([100, 100, 100, 100], [100, 100, 100, 100], {
        baselineMetadata: fourSampleMetadata,
        candidateMetadata: fourSampleMetadata,
      }),
    /at least 5 are required/,
  );
});

function compare(baselineFrames, candidateFrames, options = {}) {
  return compareBenchmarkRuns({
    baseline: benchmarkCsv(baselineFrames, "stable"),
    candidate: benchmarkCsv(candidateFrames, options.candidateHash ?? "stable"),
    baselineMetadata: options.baselineMetadata ?? metadata,
    candidateMetadata: options.candidateMetadata ?? metadata,
    exceptions: options.exceptions,
  });
}

function benchmarkCsv(frameTimes, outputHash) {
  const rows = frameTimes.map((completeFrame, frame) => {
    const values = Object.fromEntries(BENCHMARK_V2_COLUMNS.map((column) => [column, "0"]));
    Object.assign(values, {
      schema_version: "2",
      scenario: "fixture-threads-12",
      frame: String(frame),
      width: "640",
      height: "360",
      supersample_scale: "1",
      shadows: "false",
      rayon_threads: "12",
      warmup_frames: "3",
      complete_frame_ms: String(completeFrame),
      output_hash: outputHash,
    });
    return formatCsvRow(BENCHMARK_V2_COLUMNS.map((column) => values[column]));
  });
  return parseBenchmarkCsv(`${formatCsvRow(BENCHMARK_V2_COLUMNS)}\n${rows.join("\n")}\n`);
}
