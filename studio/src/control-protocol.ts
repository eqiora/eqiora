import { z } from "zod";
import compileV2SchemaDocument from "../../crates/eqiora-api/schemas/compile-v2.schema.json";

export const CONTROL_PROTOCOL_V2 = "eqiora.control/v2" as const;
export const COMPILE_COMMAND_V1 = "model.compile-check/v1" as const;
export const CURRENT_MODEL_SCHEMA = "eqiora.model-envelope/v27" as const;
export const CURRENT_MODEL_TRANSACTION_SCHEMA = "eqiora.model-transaction-envelope/v27" as const;

const requestDefinition = compileV2SchemaDocument.$defs.request;
const diagnosticDefinition = compileV2SchemaDocument.$defs.diagnostic;
const sourceSpanDefinition = compileV2SchemaDocument.$defs.sourceSpan;
const patchDefinition = compileV2SchemaDocument.$defs.patch;
const rejectedOutcomeDefinition = compileV2SchemaDocument.$defs.rejectedOutcome;
const graphPathArrayDefinition = z
  .object({
    maxItems: z.number().int(),
    items: z.object({
      maxLength: z.number().int(),
      "x-eqiora-maxUtf8Bytes": z.number().int(),
    }),
  })
  .parse(diagnosticDefinition.properties.graphPath.oneOf[0]);

const MAX_ENCODED_BYTES = requestDefinition["x-eqiora-maxEncodedUtf8Bytes"];
const MAX_FILENAME_CHARACTERS = requestDefinition.properties.filename.maxLength;
const MAX_FILENAME_UTF8_BYTES = requestDefinition.properties.filename["x-eqiora-maxUtf8Bytes"];
const MAX_SOURCE_CHARACTERS = requestDefinition.properties.source.maxLength;
const MAX_SOURCE_UTF8_BYTES = requestDefinition.properties.source["x-eqiora-maxUtf8Bytes"];
const MAX_DIAGNOSTIC_MESSAGE_CHARACTERS = diagnosticDefinition.properties.message.maxLength;
const MAX_DIAGNOSTIC_MESSAGE_UTF8_BYTES =
  diagnosticDefinition.properties.message["x-eqiora-maxUtf8Bytes"];
const MAX_GRAPH_PATH_SEGMENTS = graphPathArrayDefinition.maxItems;
const MAX_TEXT_MEMBER_CHARACTERS = graphPathArrayDefinition.items.maxLength;
const MAX_TEXT_MEMBER_UTF8_BYTES = graphPathArrayDefinition.items["x-eqiora-maxUtf8Bytes"];
const MAX_DIAGNOSTICS = rejectedOutcomeDefinition.properties.diagnostics.maxItems;
const utf8 = new TextEncoder();

function characterCount(value: string): number {
  return [...value].length;
}

function utf8Bytes(value: string): number {
  return utf8.encode(value).length;
}

function boundedText(
  minimumCharacters: number,
  maximumCharacters: number,
  maximumUtf8Bytes: number,
) {
  return z
    .string()
    .refine(
      (value) =>
        characterCount(value) >= minimumCharacters &&
        characterCount(value) <= maximumCharacters &&
        utf8Bytes(value) <= maximumUtf8Bytes,
    );
}

function characterBoundedText(minimumCharacters: number, maximumCharacters: number) {
  return z.string().refine((value) => {
    const characters = characterCount(value);
    return characters >= minimumCharacters && characters <= maximumCharacters;
  });
}

function encodedBytes(value: unknown): number {
  return utf8Bytes(JSON.stringify(value));
}

const requestIdSchema = z
  .string()
  .min(1)
  .max(128)
  .regex(/^[A-Za-z0-9._:-]+$/);

const filenameSchema = boundedText(1, MAX_FILENAME_CHARACTERS, MAX_FILENAME_UTF8_BYTES).refine(
  (filename) => !/\p{Cc}/u.test(filename),
  { message: "Filename cannot contain control characters." },
);

const sourceSchema = boundedText(0, MAX_SOURCE_CHARACTERS, MAX_SOURCE_UTF8_BYTES);

export const compileRequestV2Schema = z
  .object({
    protocol: z.literal(CONTROL_PROTOCOL_V2),
    command: z.literal(COMPILE_COMMAND_V1),
    requestId: requestIdSchema,
    filename: filenameSchema,
    source: sourceSchema,
  })
  .strict()
  .refine((request) => encodedBytes(request) <= MAX_ENCODED_BYTES, {
    message: "Compile/check request exceeds the encoded UTF-8 limit.",
  });

export type CompileRequestV2 = z.infer<typeof compileRequestV2Schema>;

const sourceSpanFileSchema = boundedText(
  0,
  sourceSpanDefinition.properties.file.maxLength,
  sourceSpanDefinition.properties.file["x-eqiora-maxUtf8Bytes"],
);

const controlSourceSpanV2Schema = z
  .object({
    file: sourceSpanFileSchema,
    start: z.number().int().min(0).max(4_294_967_295),
    end: z.number().int().min(0).max(4_294_967_295),
  })
  .strict()
  .refine((span) => span.end >= span.start, {
    message: "Source span end must not precede its start.",
  });

const graphPathSegmentSchema = boundedText(
  1,
  MAX_TEXT_MEMBER_CHARACTERS,
  MAX_TEXT_MEMBER_UTF8_BYTES,
);

const patchSummarySchema = boundedText(
  1,
  patchDefinition.properties.summary.maxLength,
  patchDefinition.properties.summary["x-eqiora-maxUtf8Bytes"],
);

export const controlDiagnosticV2Schema = z
  .object({
    source: z.enum(["control", "kernel"]),
    severity: z.enum(["error", "warning", "note"]),
    code: z.string().regex(/^[A-Z]{2}[0-9]{4}$/),
    message: boundedText(1, MAX_DIAGNOSTIC_MESSAGE_CHARACTERS, MAX_DIAGNOSTIC_MESSAGE_UTF8_BYTES),
    graphPath: z.array(graphPathSegmentSchema).max(MAX_GRAPH_PATH_SEGMENTS).nullable(),
    span: controlSourceSpanV2Schema.nullable(),
    patch: z.object({ summary: patchSummarySchema }).strict().nullable(),
  })
  .strict();

export type ControlDiagnosticV2 = z.infer<typeof controlDiagnosticV2Schema>;

const compileModelDescriptorV2Schema = z
  .object({
    schema: z.literal(CURRENT_MODEL_SCHEMA),
    transactionSchema: z.literal(CURRENT_MODEL_TRANSACTION_SCHEMA),
    digest: z.string().regex(/^[0-9a-f]{64}$/),
    modelId: characterBoundedText(1, 128),
    semanticRevision: z.number().int().nonnegative(),
  })
  .strict();

const compileOutcomeV2Schema = z.discriminatedUnion("status", [
  z.object({ status: z.literal("accepted"), model: compileModelDescriptorV2Schema }).strict(),
  z
    .object({
      status: z.literal("rejected"),
      diagnostics: z.array(controlDiagnosticV2Schema).min(1).max(MAX_DIAGNOSTICS),
    })
    .strict(),
]);

export const compileResponseV2Schema = z
  .object({
    protocol: z.literal(CONTROL_PROTOCOL_V2),
    command: z.literal(COMPILE_COMMAND_V1),
    requestId: requestIdSchema,
    outcome: compileOutcomeV2Schema,
  })
  .strict()
  .refine((response) => encodedBytes(response) <= MAX_ENCODED_BYTES, {
    message: "Compile/check response exceeds the encoded UTF-8 limit.",
  });

export type CompileResponseV2 = z.infer<typeof compileResponseV2Schema>;

export function studioCompileRequest(requestId: string, filename: string, source: string) {
  return compileRequestV2Schema.parse({
    protocol: CONTROL_PROTOCOL_V2,
    command: COMPILE_COMMAND_V1,
    requestId,
    filename,
    source,
  });
}

export function compileResponseMatchesRequest(
  request: CompileRequestV2,
  response: CompileResponseV2,
): boolean {
  return (
    response.protocol === request.protocol &&
    response.command === request.command &&
    response.requestId === request.requestId
  );
}

// The committed JSON Schema is the language-neutral source of this wire. This
// assertion rejects accidental replacement or opening of either top-level DTO.
const schemaIdentity = z
  .object({
    $id: z.literal("urn:eqiora:schema:control:compile-v2"),
    $defs: z.object({
      request: z.object({ additionalProperties: z.literal(false) }).passthrough(),
      response: z.object({ additionalProperties: z.literal(false) }).passthrough(),
      model: z.object({ additionalProperties: z.literal(false) }).passthrough(),
      diagnostic: z.object({ additionalProperties: z.literal(false) }).passthrough(),
    }),
  })
  .passthrough();

export const COMPILE_V2_SCHEMA_DOCUMENT = schemaIdentity.parse(compileV2SchemaDocument);
