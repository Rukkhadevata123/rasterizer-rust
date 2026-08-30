export const DEFAULT_REGRESSION_THRESHOLD_PERCENT = 5;
export const MINIMUM_MEASURED_SAMPLES = 5;

const MATCHING_METADATA_FIELDS = Object.freeze([
  "schemaVersion",
  "platform",
  "architecture",
  "cpu",
  "availableParallelism",
  "benchmarkThreads",
  "measuredFrames",
  "warmupFrames",
  "rustc",
]);

const MATCHING_SCENARIO_FIELDS = Object.freeze([
  "width",
  "height",
  "supersample_scale",
  "shadows",
  "rayon_threads",
  "warmup_frames",
]);

export function compareBenchmarkRuns({
  baseline,
  candidate,
  baselineMetadata,
  candidateMetadata,
  exceptions = [],
  thresholdPercent = DEFAULT_REGRESSION_THRESHOLD_PERCENT,
  minimumSamples = MINIMUM_MEASURED_SAMPLES,
}) {
  requireFinitePositive(thresholdPercent, "regression threshold");
  if (!Number.isSafeInteger(minimumSamples) || minimumSamples < 2) {
    throw new Error("minimum measured samples must be an integer of at least 2");
  }
  if (baseline.schemaVersion !== "2" || candidate.schemaVersion !== "2") {
    throw new Error("performance comparisons require benchmark schema version '2'");
  }

  validateMatchingMetadata(baselineMetadata, candidateMetadata);
  const baselineGroups = groupScenarios(baseline, baselineMetadata, "baseline", minimumSamples);
  const candidateGroups = groupScenarios(candidate, candidateMetadata, "candidate", minimumSamples);
  validateScenarioSets(baselineGroups, candidateGroups);
  const exceptionMap = validateExceptions(exceptions, baselineGroups, thresholdPercent);

  const results = [];
  for (const [scenario, baselineSamples] of baselineGroups) {
    const candidateSamples = candidateGroups.get(scenario);
    validateScenarioContract(scenario, baselineSamples, candidateSamples);

    const baselineHash = stableHash(scenario, baselineSamples, "baseline");
    const candidateHash = stableHash(scenario, candidateSamples, "candidate");
    if (baselineHash !== candidateHash) {
      throw new Error(
        `scenario '${scenario}' output hash changed: ${baselineHash} != ${candidateHash}`,
      );
    }

    const baselineFrames = numericFrames(scenario, baselineSamples, "baseline");
    const candidateFrames = numericFrames(scenario, candidateSamples, "candidate");
    const baselineMean = mean(baselineFrames);
    const candidateMean = mean(candidateFrames);
    const meanChangePercent = percentageChange(baselineMean, candidateMean);
    const p95ChangePercent = percentageChange(
      percentile(baselineFrames, 0.95),
      percentile(candidateFrames, 0.95),
    );
    const exception = exceptionMap.get(scenario);
    const allowedPercent = exception?.thresholdPercent ?? thresholdPercent;
    const withinThreshold = meanChangePercent <= allowedPercent;

    results.push({
      scenario,
      samples: baselineSamples.length,
      baselineMean,
      candidateMean,
      meanChangePercent,
      p95ChangePercent,
      allowedPercent,
      status: withinThreshold ? (exception ? "exception" : "pass") : "regression",
      rationale: exception?.rationale,
    });
  }

  return {
    passed: results.every((result) => result.status !== "regression"),
    thresholdPercent,
    minimumSamples,
    results,
  };
}

function validateMatchingMetadata(baseline, candidate) {
  for (const field of MATCHING_METADATA_FIELDS) {
    if (baseline?.[field] === undefined || candidate?.[field] === undefined) {
      throw new Error(`benchmark metadata is missing required field '${field}'`);
    }
    if (baseline[field] !== candidate[field]) {
      throw new Error(
        `benchmark metadata field '${field}' differs: ${JSON.stringify(baseline[field])} != ${JSON.stringify(candidate[field])}`,
      );
    }
  }
  if (baseline.schemaVersion !== 2) {
    throw new Error("benchmark metadata must declare schemaVersion 2");
  }
}

function groupScenarios(parsed, metadata, label, minimumSamples) {
  const groups = new Map();
  for (const row of parsed.rows) {
    const scenario = row.values.scenario;
    if (scenario.length === 0) {
      throw new Error(`${label} contains an empty scenario name`);
    }
    const samples = groups.get(scenario) ?? [];
    samples.push(row.values);
    groups.set(scenario, samples);
  }
  for (const [scenario, samples] of groups) {
    if (samples.length < minimumSamples) {
      throw new Error(
        `${label} scenario '${scenario}' has ${samples.length} samples; at least ${minimumSamples} are required`,
      );
    }
    if (samples.length !== metadata.measuredFrames) {
      throw new Error(
        `${label} scenario '${scenario}' has ${samples.length} samples; metadata declares ${metadata.measuredFrames}`,
      );
    }
    const frames = new Set(samples.map((sample) => sample.frame));
    if (frames.size !== samples.length) {
      throw new Error(`${label} scenario '${scenario}' contains duplicate frame indexes`);
    }
  }
  return groups;
}

function validateScenarioSets(baseline, candidate) {
  const missing = [...baseline.keys()].filter((scenario) => !candidate.has(scenario));
  const unexpected = [...candidate.keys()].filter((scenario) => !baseline.has(scenario));
  if (missing.length !== 0 || unexpected.length !== 0) {
    throw new Error(
      `benchmark scenario sets differ; missing=[${missing.join(", ")}], unexpected=[${unexpected.join(", ")}]`,
    );
  }
}

function validateScenarioContract(scenario, baseline, candidate) {
  for (const field of MATCHING_SCENARIO_FIELDS) {
    const baselineValues = new Set(baseline.map((sample) => sample[field]));
    const candidateValues = new Set(candidate.map((sample) => sample[field]));
    if (baselineValues.size !== 1 || candidateValues.size !== 1) {
      throw new Error(`scenario '${scenario}' does not have stable '${field}' metadata`);
    }
    const baselineValue = baselineValues.values().next().value;
    const candidateValue = candidateValues.values().next().value;
    if (baselineValue !== candidateValue) {
      throw new Error(
        `scenario '${scenario}' field '${field}' differs: ${baselineValue} != ${candidateValue}`,
      );
    }
  }
}

function validateExceptions(exceptions, scenarios, defaultThreshold) {
  if (!Array.isArray(exceptions)) {
    throw new Error("benchmark exceptions must be an array");
  }
  const exceptionMap = new Map();
  for (const exception of exceptions) {
    const scenario = exception?.scenario;
    const thresholdPercent = exception?.thresholdPercent;
    const rationale = exception?.rationale?.trim();
    if (typeof scenario !== "string" || !scenarios.has(scenario)) {
      throw new Error(`benchmark exception names unknown scenario '${scenario}'`);
    }
    if (exceptionMap.has(scenario)) {
      throw new Error(`benchmark exception duplicates scenario '${scenario}'`);
    }
    requireFinitePositive(thresholdPercent, `exception threshold for '${scenario}'`);
    if (thresholdPercent <= defaultThreshold) {
      throw new Error(
        `exception threshold for '${scenario}' must exceed the default ${defaultThreshold}%`,
      );
    }
    if (!rationale) {
      throw new Error(`benchmark exception for '${scenario}' requires a rationale`);
    }
    exceptionMap.set(scenario, { thresholdPercent, rationale });
  }
  return exceptionMap;
}

function stableHash(scenario, samples, label) {
  const hashes = new Set(samples.map((sample) => sample.output_hash));
  if (hashes.size !== 1) {
    throw new Error(`${label} scenario '${scenario}' does not have a stable output hash`);
  }
  return hashes.values().next().value;
}

function numericFrames(scenario, samples, label) {
  return samples.map((sample) => {
    const value = Number(sample.complete_frame_ms);
    if (!Number.isFinite(value) || value <= 0) {
      throw new Error(
        `${label} scenario '${scenario}' contains invalid complete_frame_ms '${sample.complete_frame_ms}'`,
      );
    }
    return value;
  });
}

function requireFinitePositive(value, label) {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${label} must be a finite positive number`);
  }
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function percentile(values, fraction) {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.ceil((sorted.length - 1) * fraction)];
}

function percentageChange(baseline, candidate) {
  return ((candidate - baseline) / baseline) * 100;
}
