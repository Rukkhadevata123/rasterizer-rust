import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { compareBenchmarkRuns } from "./benchmark-comparison.mjs";
import { parseBenchmarkCsv } from "./benchmark-csv.mjs";

try {
  const [baselineArgument, candidateArgument, exceptionsArgument] = process.argv.slice(2);
  if (!baselineArgument || !candidateArgument || process.argv.length > 5) {
    throw new Error(
      "usage: node scripts/compare-benchmarks.mjs <baseline-directory> <candidate-directory> [exceptions.json]",
    );
  }

  const baselineDirectory = resolve(baselineArgument);
  const candidateDirectory = resolve(candidateArgument);
  const baseline = readRun(baselineDirectory, "baseline");
  const candidate = readRun(candidateDirectory, "candidate");
  const exceptions = exceptionsArgument ? readExceptions(resolve(exceptionsArgument)) : [];
  const comparison = compareBenchmarkRuns({
    baseline: baseline.csv,
    candidate: candidate.csv,
    baselineMetadata: baseline.metadata,
    candidateMetadata: candidate.metadata,
    exceptions,
  });

  process.stdout.write(formatComparison(comparison));
  if (!comparison.passed) process.exitCode = 1;
} catch (error) {
  process.stderr.write(`${error.message}\n`);
  process.exitCode = 1;
}

function readRun(directory, label) {
  const csvPath = join(directory, "baseline.csv");
  const metadataPath = join(directory, "metadata.json");
  return {
    csv: parseBenchmarkCsv(readFileSync(csvPath, "utf8"), {
      sourceLabel: `${label} ${csvPath}`,
      supportedVersions: ["2"],
    }),
    metadata: JSON.parse(readFileSync(metadataPath, "utf8")),
  };
}

function readExceptions(path) {
  const document = JSON.parse(readFileSync(path, "utf8"));
  if (document.schemaVersion !== 1) {
    throw new Error("benchmark exception file must declare schemaVersion 1");
  }
  return document.exceptions;
}

function formatComparison(comparison) {
  const lines = [
    `Performance gate: full-frame mean <= ${comparison.thresholdPercent.toFixed(1)}% regression; minimum ${comparison.minimumSamples} same-machine samples`,
    "",
    "| Scenario | Samples | Baseline mean ms | Candidate mean ms | Mean change | p95 change | Allowed | Status |",
    "|:--|--:|--:|--:|--:|--:|--:|:--|",
  ];
  for (const result of comparison.results) {
    lines.push(
      `| ${escapeTable(result.scenario)} | ${result.samples} | ${result.baselineMean.toFixed(3)} | ${result.candidateMean.toFixed(3)} | ${signedPercent(result.meanChangePercent)} | ${signedPercent(result.p95ChangePercent)} | ${result.allowedPercent.toFixed(1)}% | ${result.status} |`,
    );
  }
  const accepted = comparison.results.filter((result) => result.status === "exception");
  if (accepted.length !== 0) {
    lines.push("", "Accepted exceptions:");
    for (const result of accepted) {
      lines.push(`- ${result.scenario}: ${result.rationale}`);
    }
  }
  lines.push("", comparison.passed ? "Performance gate passed." : "Performance gate failed.", "");
  return lines.join("\n");
}

function signedPercent(value) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function escapeTable(value) {
  return value.replaceAll("|", "\\|");
}
