import assert from "node:assert/strict";
import test from "node:test";
import {
  BENCHMARK_V1_COLUMNS,
  formatCsvRow,
  parseBenchmarkCsv,
} from "./benchmark-csv.mjs";

const sampleValues = Object.fromEntries(
  BENCHMARK_V1_COLUMNS.map((column, index) => [column, String(index)]),
);
sampleValues.scenario = 'scenario,with "quotes"';
sampleValues.output_hash = "0123456789abcdef";

test("benchmark columns are read by name instead of position", () => {
  const columns = [...BENCHMARK_V1_COLUMNS].reverse();
  const csv = `${formatCsvRow(columns)}\n${formatCsvRow(
    columns.map((column) => sampleValues[column]),
  )}\n`;

  const parsed = parseBenchmarkCsv(csv);

  assert.equal(parsed.schemaVersion, "1");
  assert.equal(parsed.rows[0].values.scenario, sampleValues.scenario);
  assert.equal(parsed.rows[0].values.output_hash, sampleValues.output_hash);
});

test("missing and duplicate benchmark columns are rejected", () => {
  const missing = BENCHMARK_V1_COLUMNS.filter((column) => column !== "output_hash");
  assert.throws(
    () => parseBenchmarkCsv(`${formatCsvRow(missing)}\n${formatCsvRow(missing)}\n`),
    /missing required column 'output_hash'/,
  );

  const duplicate = [...BENCHMARK_V1_COLUMNS, "scenario"];
  assert.throws(
    () => parseBenchmarkCsv(`${formatCsvRow(duplicate)}\n${formatCsvRow(duplicate)}\n`),
    /duplicate column 'scenario'/,
  );
});

test("benchmark rows must match the declared width", () => {
  const fields = BENCHMARK_V1_COLUMNS.slice(1).map((column) => sampleValues[column]);
  const csv = `${formatCsvRow(BENCHMARK_V1_COLUMNS)}\n${formatCsvRow(fields)}\n`;

  assert.throws(() => parseBenchmarkCsv(csv), /has 16 fields; expected 17/);
});

test("unsupported explicit benchmark schema versions are rejected", () => {
  const columns = ["schema_version", ...BENCHMARK_V1_COLUMNS];
  const fields = ["2", ...BENCHMARK_V1_COLUMNS.map((column) => sampleValues[column])];
  const csv = `${formatCsvRow(columns)}\n${formatCsvRow(fields)}\n`;

  assert.throws(() => parseBenchmarkCsv(csv), /unsupported schema version '2'/);
});
