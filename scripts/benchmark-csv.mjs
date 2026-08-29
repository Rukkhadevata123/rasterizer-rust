export const BENCHMARK_V1_COLUMNS = Object.freeze([
  "scenario",
  "frame",
  "width",
  "height",
  "supersample_scale",
  "shadows",
  "rayon_threads",
  "warmup_frames",
  "scene_loading_ms",
  "shadow_preparation_ms",
  "shadow_rasterization_ms",
  "main_preparation_ms",
  "opaque_masked_rasterization_ms",
  "transparent_rasterization_ms",
  "post_processing_ms",
  "complete_frame_ms",
  "output_hash",
]);

export const BENCHMARK_V2_COLUMNS = Object.freeze([
  "schema_version",
  "scenario",
  "frame",
  "width",
  "height",
  "supersample_scale",
  "shadows",
  "rayon_threads",
  "warmup_frames",
  "scene_loading_ms",
  "shadow_pass_setup_ms",
  "shadow_recording_ms",
  "shadow_attachment_processing_ms",
  "shadow_backend_preparation_ms",
  "shadow_rasterization_ms",
  "shadow_submission_total_ms",
  "main_pass_setup_ms",
  "main_recording_ms",
  "main_attachment_processing_ms",
  "main_backend_preparation_ms",
  "main_rasterization_ms",
  "main_opaque_masked_rasterization_ms",
  "main_transparent_rasterization_ms",
  "main_submission_total_ms",
  "post_processing_ms",
  "complete_frame_ms",
  "output_hash",
]);

const BENCHMARK_COLUMNS_BY_VERSION = new Map([
  ["1", BENCHMARK_V1_COLUMNS],
  ["2", BENCHMARK_V2_COLUMNS],
]);

export function parseBenchmarkCsv(
  source,
  { sourceLabel = "benchmark CSV", supportedVersions = ["1", "2"] } = {},
) {
  const content = source.trimEnd();
  if (content.length === 0) {
    throw new Error(`${sourceLabel} is empty`);
  }

  const lines = content.split(/\r?\n/);
  const columns = parseCsvRow(lines[0], `${sourceLabel} header`);
  const columnIndexes = new Map();
  for (const [index, column] of columns.entries()) {
    if (column.length === 0) {
      throw new Error(`${sourceLabel} contains an empty column name`);
    }
    if (columnIndexes.has(column)) {
      throw new Error(`${sourceLabel} contains duplicate column '${column}'`);
    }
    columnIndexes.set(column, index);
  }

  if (lines.length === 1) {
    throw new Error(`${sourceLabel} contains no data rows`);
  }

  let schemaVersion;
  const rows = lines.slice(1).map((line, rowOffset) => {
    const rowNumber = rowOffset + 2;
    const fields = parseCsvRow(line, `${sourceLabel} row ${rowNumber}`);
    if (fields.length !== columns.length) {
      throw new Error(
        `${sourceLabel} row ${rowNumber} has ${fields.length} fields; expected ${columns.length}`,
      );
    }

    const values = Object.fromEntries(columns.map((column, index) => [column, fields[index]]));
    const rowSchemaVersion = values.schema_version ?? "1";
    if (schemaVersion === undefined) {
      schemaVersion = rowSchemaVersion;
    } else if (schemaVersion !== rowSchemaVersion) {
      throw new Error(
        `${sourceLabel} mixes schema versions '${schemaVersion}' and '${rowSchemaVersion}'`,
      );
    }
    return { values };
  });

  if (!supportedVersions.includes(schemaVersion)) {
    throw new Error(`${sourceLabel} uses unsupported schema version '${schemaVersion}'`);
  }

  const requiredColumns = BENCHMARK_COLUMNS_BY_VERSION.get(schemaVersion);
  if (requiredColumns === undefined) {
    throw new Error(`${sourceLabel} has no column contract for schema version '${schemaVersion}'`);
  }
  for (const column of requiredColumns) {
    if (!columnIndexes.has(column)) {
      throw new Error(
        `${sourceLabel} schema version '${schemaVersion}' is missing required column '${column}'`,
      );
    }
  }

  return { schemaVersion, columns, columnIndexes, rows };
}

export function formatCsvRow(fields) {
  return fields.map(formatCsvField).join(",");
}

function parseCsvRow(line, label) {
  const fields = [];
  let field = "";
  let quoted = false;
  let closedQuote = false;

  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (quoted) {
      if (character === '"') {
        if (line[index + 1] === '"') {
          field += '"';
          index += 1;
        } else {
          quoted = false;
          closedQuote = true;
        }
      } else {
        field += character;
      }
    } else if (closedQuote) {
      if (character !== ",") {
        throw new Error(`${label} contains characters after a closing quote`);
      }
      fields.push(field);
      field = "";
      closedQuote = false;
    } else if (character === ",") {
      fields.push(field);
      field = "";
    } else if (character === '"') {
      if (field.length !== 0) {
        throw new Error(`${label} contains a quote inside an unquoted field`);
      }
      quoted = true;
    } else {
      field += character;
    }
  }

  if (quoted) {
    throw new Error(`${label} contains an unterminated quoted field`);
  }
  fields.push(field);
  return fields;
}

function formatCsvField(value) {
  const field = String(value);
  if (/[",\r\n]/.test(field)) {
    return `"${field.replaceAll('"', '""')}"`;
  }
  return field;
}
