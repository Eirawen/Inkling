import { GoogleGenAI, type GenerateContentResponse } from "@google/genai";
import * as THREE from "three";
import type { EditOperation, RefinementResponse, SDFShapeConfig, VoxelCell } from "./types";

const GEMINI_MODEL = "gemini-3-flash-preview";
const OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses";
const OPENAI_FALLBACK_MODEL =
  String(import.meta.env.VITE_OPENAI_MODEL ?? "gpt-5.2").trim() || "gpt-5.2";
const MAX_RETRIES = 1;
const MAX_TOKENS = 4096;
const MARKDOWN_JSON_REGEX = /```json?\s*([\s\S]*?)```/i;
let pendingSecondaryScreenshotBase64: string | null = null;
let providerPreference: "gemini" | "openai" = "gemini";

const ACTIONS = new Set<EditOperation["action"]>([
  "delete",
  "recolor",
  "tint",
  "light",
  "darken",
  "atmosphere",
]);
const BLEND_MODES = new Set<EditOperation["blendMode"]>([
  "MULTIPLY",
  "SET_RGB",
  "ADD_RGBA",
]);
const SHAPE_TYPES = new Set<SDFShapeConfig["type"]>([
  "SPHERE",
  "BOX",
  "ELLIPSOID",
  "CYLINDER",
  "CAPSULE",
  "PLANE",
  "INFINITE_CONE",
  "ALL",
]);

const SYSTEM_PROMPT = `You are a spatial editing assistant for 3D gaussian splat worlds rendered with Spark.

You output structured edit operations for Spark SplatEdit SDF.

Coordinate system:
- Right-handed, Y-up.
- All positions, scale values, and softEdge values are world-space units.
- Use click position and provided bounds/cell dimensions as the anchor for shape placement and sizing.

Available SDF shape types and usage:
1) SPHERE
- Best for point deletions, trees, bushes, round objects.
- Required params: position, radius.

2) BOX
- Best for buildings, walls, vehicles, road segments, rectangular regions.
- Required params: position, scale.
- scale is half-extents in world units [xHalf, yHalf, zHalf].

3) ELLIPSOID
- Best for elongated organic forms, non-uniform rounded objects.
- Required params: position, scale.
- scale is radii per axis [rx, ry, rz].

4) CYLINDER
- Best for poles, trunks, pillars, columns.
- Required params: position, scale.
- Use scale.x and scale.z as radius controls, scale.y as half-height.

5) CAPSULE
- Best for rounded cylinders and trunk-like forms with smooth ends.
- Params: position, scale, optional radius.

6) PLANE
- Best for ground/sky splits and wide directional effects.
- Params: position, rotation.
- rotation quaternion defines normal direction.

7) INFINITE_CONE
- Best for spotlights and directional cone effects.
- Params: position, rotation, radius.
- radius controls cone angle/falloff.

8) ALL
- Best for global full-scene effects.
- Params: position (use [0,0,0] unless a specific anchor is provided).

Actions and blend modes:
- action "delete": MULTIPLY + opacity: 0 => splats become invisible
- action "darken": MULTIPLY + color [0.3, 0.3, 0.3] => darken region
- action "tint": MULTIPLY + color => COLOR TINT that preserves texture, shadows, and highlights.
  The color channels act as multipliers: [0.3, 0.3, 1.0] keeps blue, crushes red/green → blue tint.
  [1.0, 0.8, 0.5] → warm golden tint. [0.4, 1.0, 0.4] → green tint.
  Always keep the target channel near 1.0 and suppress others. Preserve luminance variation.
  This is the DEFAULT for color-change commands like "make this blue", "paint it red", "autumn colors".
- action "recolor": SET_RGB + color => FLAT COLOR OVERRIDE. Replaces all color with a uniform value.
  Destroys texture and shading. Only use when the user explicitly wants a flat/uniform/solid color
  (e.g. "make this solid red", "flat white", "uniform black").
- action "light": ADD_RGBA + color => additive light/glow
- action "atmosphere": ADD_RGBA on ALL shape => global tint

softEdge:
- Feathers shape boundaries in world-space units.
- Always use softEdge > 0 for natural results.
- Typical range: 0.05 to 0.3.
- Use around 0.1 for local deletes/recolors.
- Use around 0.2 for lighting/atmosphere.

sdfSmooth:
- Smoothly blends multiple SDF shapes in a single edit.
- Use for compound objects (example: tree canopy + trunk).

extractAsset rule:
- For every delete action, always set extractAsset: true.
- Also include a descriptive assetLabel.

Geometry selection heuristics:
- Use bounding box dimensions and color data to size and choose shapes.
- If bounding box is taller than wide, prefer CYLINDER or ELLIPSOID over SPHERE.
- For buildings/structures, prefer BOX.
- For organic shapes (trees/bushes), prefer SPHERE or SPHERE+CYLINDER compound.
- For flat/wide vehicles, roads, facades, prefer BOX.
- Size shapes to object dimensions from context. Do not under-size edits.
- When a "Local high-resolution grid" is provided, prefer it over the global grid for object boundary detection near the click.
- Local grid cells are finer — use their positions and sizes for precise shape placement and sizing.

Response format requirements:
- Return ONLY a JSON array of EditOperation objects.
- No prose, no markdown, no explanations, no code fences.
- Compound targets should use multiple shapes in one operation when needed.

Examples (command -> JSON):

Example A: "Remove this tree" with click [3.2,1.0,-2.1], bbox approx width 2.2, height 3.0, depth 2.0
[
  {
    "action": "delete",
    "blendMode": "MULTIPLY",
    "softEdge": 0.1,
    "extractAsset": true,
    "assetLabel": "tree",
    "shapes": [
      { "type": "SPHERE", "position": [3.2, 2.1, -2.1], "radius": 1.5, "opacity": 0.0 }
    ]
  }
]

Example B: "Make this building red" with click [8.0,2.5,4.0], bbox approx 6x5x4
[
  {
    "action": "tint",
    "blendMode": "MULTIPLY",
    "softEdge": 0.1,
    "shapes": [
      {
        "type": "BOX",
        "position": [8.0, 2.5, 4.0],
        "scale": [3.0, 2.5, 2.0],
        "color": [1.0, 0.25, 0.2],
        "opacity": 1.0
      }
    ]
  }
]

Example C: "Add warm sunset lighting" (global)
[
  {
    "action": "light",
    "blendMode": "ADD_RGBA",
    "softEdge": 0.25,
    "shapes": [
      { "type": "PLANE", "position": [0, 1.0, 0], "rotation": [0, 0, 0, 1], "color": [0.24, 0.14, 0.06], "opacity": 0.0 },
      { "type": "SPHERE", "position": [0, 5.0, -8.0], "radius": 20.0, "color": [0.28, 0.16, 0.08], "opacity": 0.0 }
    ]
  }
]

Example D: "Create a spotlight on the fountain" with click [1.5,0.8,3.2]
[
  {
    "action": "light",
    "blendMode": "ADD_RGBA",
    "softEdge": 0.22,
    "shapes": [
      {
        "type": "INFINITE_CONE",
        "position": [1.5, 4.5, 3.2],
        "rotation": [0.7071, 0, 0, 0.7071],
        "radius": 0.45,
        "color": [0.22, 0.19, 0.12],
        "opacity": 0.0
      }
    ]
  }
]

Example E: "Make the shadows deeper"
[
  {
    "action": "darken",
    "blendMode": "MULTIPLY",
    "softEdge": 0.2,
    "shapes": [
      {
        "type": "PLANE",
        "position": [0, 0.3, 0],
        "rotation": [0, 0, 0, 1],
        "color": [0.35, 0.35, 0.4],
        "opacity": 1.0
      }
    ]
  }
]

Example F: "Autumn foliage"
[
  {
    "action": "tint",
    "blendMode": "MULTIPLY",
    "softEdge": 0.14,
    "shapes": [
      { "type": "ELLIPSOID", "position": [4.0, 2.2, -1.5], "scale": [1.8, 1.6, 1.4], "color": [1.0, 0.5, 0.15], "opacity": 1.0 },
      { "type": "ELLIPSOID", "position": [7.2, 2.6, -3.1], "scale": [2.0, 1.8, 1.6], "color": [0.9, 0.35, 0.15], "opacity": 1.0 }
    ]
  }
]

Example G: "Remove the car" with click [6.0,0.9,-4.0], bbox approx width 3.8, height 1.4, depth 1.7
[
  {
    "action": "delete",
    "blendMode": "MULTIPLY",
    "softEdge": 0.08,
    "extractAsset": true,
    "assetLabel": "car",
    "shapes": [
      { "type": "BOX", "position": [6.0, 0.9, -4.0], "scale": [1.9, 0.7, 0.85], "opacity": 0.0 }
    ]
  }
]

Example H: "Make everything foggy"
[
  {
    "action": "atmosphere",
    "blendMode": "ADD_RGBA",
    "softEdge": 0.3,
    "shapes": [
      { "type": "ALL", "position": [0, 0, 0], "color": [0.08, 0.08, 0.08], "opacity": 0.0 }
    ]
  }
]

Example I: "Remove that large tree" with click [2.8,1.1,-6.2], bbox approx width 3.0, height 6.0, depth 2.8
[
  {
    "action": "delete",
    "blendMode": "MULTIPLY",
    "softEdge": 0.1,
    "sdfSmooth": 0.2,
    "extractAsset": true,
    "assetLabel": "large tree",
    "shapes": [
      { "type": "SPHERE", "position": [2.8, 3.7, -6.2], "radius": 1.6, "opacity": 0.0 },
      { "type": "CYLINDER", "position": [2.8, 1.6, -6.2], "scale": [0.35, 1.6, 0.35], "opacity": 0.0 }
    ]
  }
]

Example J: "Clear the road" with road center [0,0.2,5], road size approx length 20, width 3, thickness 0.5
[
  {
    "action": "delete",
    "blendMode": "MULTIPLY",
    "softEdge": 0.08,
    "extractAsset": true,
    "assetLabel": "road clutter",
    "shapes": [
      { "type": "BOX", "position": [0, 0.2, 5], "scale": [10.0, 0.25, 1.5], "opacity": 0.0 }
    ]
  }
]`;

export function buildClickContext(
  clickPos: THREE.Vector3,
  cell: VoxelCell | null,
  neighbors: VoxelCell[]
): string {
  const lines: string[] = [];
  lines.push(`Click position: ${formatVec3(clickPos)}`);

  if (!cell) {
    lines.push("Primary cell: none (click outside indexed occupied cells).");
    if (neighbors.length > 0) {
      lines.push(
        `Nearby occupied cells available: ${neighbors.length} (primary cell missing).`
      );
    } else {
      lines.push("Nearby occupied cells: none.");
    }
    return lines.join("\n");
  }

  const boundsSize = new THREE.Vector3();
  cell.worldBounds.getSize(boundsSize);

  lines.push(
    `Primary cell: key=${gridPosKey(cell.gridPos)} center=${formatVec3(cell.worldCenter)} splats=${cell.splatCount} density=${cell.density.toFixed(3)}`
  );
  lines.push(
    `Primary cell bounds: min=${formatVec3(cell.worldBounds.min)} max=${formatVec3(cell.worldBounds.max)} size=${formatVec3(boundsSize)}`
  );
  lines.push(
    `Primary color stats: avg=${formatColor(cell.avgColor)} variance=${cell.colorVariance.toFixed(4)}`
  );

  const nearestNeighbors = neighbors
    .filter((neighbor) => gridPosKey(neighbor.gridPos) !== gridPosKey(cell.gridPos))
    .map((neighbor) => ({
      cell: neighbor,
      dist: neighbor.worldCenter.distanceTo(clickPos),
    }))
    .sort((a, b) => a.dist - b.dist)
    .slice(0, 6);

  if (nearestNeighbors.length === 0) {
    lines.push("Neighbor cells: none.");
    return lines.join("\n");
  }

  lines.push("Neighbor cells (nearest first):");
  for (const { cell: neighbor, dist } of nearestNeighbors) {
    const size = new THREE.Vector3();
    neighbor.worldBounds.getSize(size);
    lines.push(
      `- key=${gridPosKey(neighbor.gridPos)} dist=${dist.toFixed(2)} center=${formatVec3(neighbor.worldCenter)} size=${formatVec3(size)} splats=${neighbor.splatCount} avg=${formatColor(neighbor.avgColor)}`
    );
  }

  return lines.join("\n");
}

export async function processCommand(
  command: string,
  clickPosition: THREE.Vector3 | null,
  voxelContext: string | null,
  manifestSummary: string | null,
  screenshotBase64: string | null,
  apiKey: string
): Promise<EditOperation[]> {
  const trimmedCommand = command.trim();
  if (!trimmedCommand) {
    throw new Error("[agent] ERROR: command is empty");
  }
  const geminiApiKey = apiKey.trim();
  const openAIApiKey = readOpenAIApiKey();
  if (!geminiApiKey && !openAIApiKey) {
    throw new Error("[agent] ERROR: missing Gemini/OpenAI API key");
  }

  const secondaryScreenshotBase64 = pendingSecondaryScreenshotBase64;
  pendingSecondaryScreenshotBase64 = null;
  console.log(
    `[agent] processCommand start command="${trimmedCommand}" click=${clickPosition ? formatVec3(clickPosition) : "null"} voxelChars=${voxelContext?.length ?? 0} manifestChars=${manifestSummary?.length ?? 0} screenshotBytes=${screenshotBase64?.length ?? 0} secondaryScreenshotBytes=${secondaryScreenshotBase64?.length ?? 0} geminiKey=${Boolean(geminiApiKey)} openaiKey=${Boolean(openAIApiKey)}`
  );

  let lastError: Error | null = null;

  const preferOpenAI = providerPreference === "openai";
  if (geminiApiKey && !preferOpenAI) {
    const ai = new GoogleGenAI({ apiKey: geminiApiKey });
    for (let attempt = 0; attempt <= MAX_RETRIES; attempt += 1) {
      const retrying = attempt > 0;
      try {
        const userText = buildUserText(
          trimmedCommand,
          clickPosition,
          voxelContext,
          manifestSummary,
          retrying,
          Boolean(secondaryScreenshotBase64)
        );
        const contents = buildContents(
          userText,
          screenshotBase64,
          secondaryScreenshotBase64
        );
        const imageParts =
          contents[0]?.parts.filter((part) => "inlineData" in part).length ?? 0;
        console.log(
          `[agent] Gemini request attempt=${attempt + 1}/${MAX_RETRIES + 1} model=${GEMINI_MODEL} imageParts=${imageParts} promptChars=${userText.length}`
        );

        const response = await ai.models.generateContent({
          model: GEMINI_MODEL,
          contents,
          config: {
            temperature: 0,
            maxOutputTokens: MAX_TOKENS,
            systemInstruction: SYSTEM_PROMPT,
          },
        });
        console.log(
          `[agent] Gemini response received candidates=${response.candidates?.length ?? 0} textChars=${response.text?.length ?? 0}`
        );

        const text = extractTextFromGeminiResponse(response);
        console.log(`[agent] Extracted response text chars=${text.length}`);
        const operations = parseAndValidateOperations(text, trimmedCommand);

        console.log(`[agent] Command: "${trimmedCommand}" → ${operations.length} operations`);
        return operations;
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        lastError = new Error(message);

        if (attempt < MAX_RETRIES) {
          console.warn(
            `[agent] Retry ${attempt + 1}/${MAX_RETRIES} for command "${trimmedCommand}" due to: ${message}`
          );
          continue;
        }

        console.error(`[agent] Gemini error after retries: ${message}`);
      }
    }
  } else {
    console.warn("[agent] Gemini key missing; skipping Gemini and using OpenAI fallback");
  }

  const shouldFallback = shouldUseOpenAIFallback(geminiApiKey, lastError);
  if (openAIApiKey && (preferOpenAI || shouldFallback)) {
    console.warn(
      preferOpenAI
        ? `[agent] Using OpenAI by preference model=${OPENAI_FALLBACK_MODEL}`
        : `[agent] Falling back to OpenAI model=${OPENAI_FALLBACK_MODEL} after Gemini failure`
    );
    try {
      const userText = buildUserText(
        trimmedCommand,
        clickPosition,
        voxelContext,
        manifestSummary,
        false,
        Boolean(secondaryScreenshotBase64)
      );
      const text = await requestOpenAIText(
        openAIApiKey,
        userText,
        screenshotBase64,
        secondaryScreenshotBase64
      );
      const operations = parseAndValidateOperations(text, trimmedCommand);

      console.log(
        `[agent] Command: "${trimmedCommand}" → ${operations.length} operations (OpenAI fallback)`
      );
      return operations;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      lastError = new Error(message);
      console.error(`[agent] OpenAI fallback failed: ${message}`);
    }
  } else if (openAIApiKey && geminiApiKey && lastError && !preferOpenAI) {
    console.warn(
      `[agent] OpenAI fallback skipped: non-transient Gemini failure (${lastError.message})`
    );
  }

  throw lastError ?? new Error("[agent] ERROR: unknown processing failure");
}

function buildContents(
  userText: string,
  screenshotBase64: string | null,
  secondaryScreenshotBase64: string | null
): Array<{
  role: "user";
  parts: Array<
    | { text: string }
    | { inlineData: { mimeType: "image/png"; data: string } }
  >;
}> {
  const parts: Array<
    | { text: string }
    | { inlineData: { mimeType: "image/png"; data: string } }
  > = [];
  const normalizedImage = normalizeScreenshotBase64(screenshotBase64);
  if (normalizedImage) {
    parts.push({
      inlineData: {
        mimeType: "image/png",
        data: normalizedImage,
      },
    });
  }
  const normalizedSecondaryImage = normalizeScreenshotBase64(secondaryScreenshotBase64);
  if (normalizedSecondaryImage) {
    parts.push({
      inlineData: {
        mimeType: "image/png",
        data: normalizedSecondaryImage,
      },
    });
  }
  console.log(
    `[agent] buildContents primaryImageIncluded=${Boolean(normalizedImage)} primaryImageBytes=${normalizedImage?.length ?? 0} secondaryImageIncluded=${Boolean(normalizedSecondaryImage)} secondaryImageBytes=${normalizedSecondaryImage?.length ?? 0}`
  );
  parts.push({ text: userText });
  return [{ role: "user", parts }];
}

function buildUserText(
  command: string,
  clickPosition: THREE.Vector3 | null,
  voxelContext: string | null,
  manifestSummary: string | null,
  simplifyForRetry: boolean,
  hasSecondaryImage: boolean
): string {
  const lines: string[] = [];

  if (manifestSummary?.trim()) {
    lines.push("Scene summary:");
    lines.push(manifestSummary.trim());
    lines.push("");
  }

  if (clickPosition) {
    lines.push(`Click world position: ${formatVec3(clickPosition)}`);
    if (voxelContext?.trim()) {
      lines.push("Voxel context near click:");
      lines.push(voxelContext.trim());
    } else {
      lines.push("Voxel context near click: not provided.");
    }
    lines.push("");
  } else if (voxelContext?.trim()) {
    lines.push("Voxel context:");
    lines.push(voxelContext.trim());
    lines.push("");
  }

  lines.push("Output requirements:");
  lines.push("- Return only a valid JSON array of EditOperation objects.");
  lines.push("- No markdown, no prose.");
  if (hasSecondaryImage) {
    lines.push(
      "- The second image is a click-centered crop around the selected target. Prefer it for local object boundaries."
    );
  }
  if (simplifyForRetry) {
    lines.push(
      "- Return only valid JSON array of EditOperation objects. No markdown."
    );
  }
  lines.push("");
  lines.push(`User command: ${command}`);

  return lines.join("\n");
}

export function setSecondaryScreenshotForNextCommand(
  screenshotBase64: string | null
): void {
  pendingSecondaryScreenshotBase64 = normalizeScreenshotBase64(screenshotBase64);
  console.log(
    `[agent] setSecondaryScreenshotForNextCommand bytes=${pendingSecondaryScreenshotBase64?.length ?? 0}`
  );
}

export function setProviderPreference(preference: "gemini" | "openai"): void {
  providerPreference = preference;
  console.log(`[agent] Provider preference set to ${providerPreference}`);
}

async function requestOpenAIText(
  apiKey: string,
  userText: string,
  screenshotBase64: string | null,
  secondaryScreenshotBase64: string | null
): Promise<string> {
  const content: Array<
    | { type: "input_text"; text: string }
    | { type: "input_image"; image_url: string }
  > = [];

  const normalizedImage = normalizeScreenshotBase64(screenshotBase64);
  if (normalizedImage) {
    content.push({
      type: "input_image",
      image_url: `data:image/png;base64,${normalizedImage}`,
    });
  }

  const normalizedSecondaryImage = normalizeScreenshotBase64(secondaryScreenshotBase64);
  if (normalizedSecondaryImage) {
    content.push({
      type: "input_image",
      image_url: `data:image/png;base64,${normalizedSecondaryImage}`,
    });
  }

  content.push({
    type: "input_text",
    text: userText,
  });

  const imageParts = content.filter((item) => item.type === "input_image").length;
  console.log(
    `[agent] OpenAI request model=${OPENAI_FALLBACK_MODEL} imageParts=${imageParts} promptChars=${userText.length}`
  );

  const response = await fetch(OPENAI_RESPONSES_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: OPENAI_FALLBACK_MODEL,
      instructions: SYSTEM_PROMPT,
      temperature: 0,
      max_output_tokens: MAX_TOKENS,
      input: [
        {
          role: "user",
          content,
        },
      ],
    }),
  });

  if (!response.ok) {
    const errorText = await extractOpenAIError(response);
    throw new Error(
      `OpenAI request failed (${response.status} ${response.statusText}): ${errorText}`
    );
  }

  const payload = (await response.json()) as OpenAIResponsePayload;
  const text = extractTextFromOpenAIResponse(payload);
  console.log(`[agent] OpenAI response text chars=${text.length}`);
  return text;
}

async function requestOpenAIRefinementText(
  apiKey: string,
  userText: string,
  preEditBase64: string | null,
  preEditCropBase64: string | null,
  postEditBase64: string | null,
  postEditCropBase64: string | null
): Promise<string> {
  const content: Array<
    | { type: "input_text"; text: string }
    | { type: "input_image"; image_url: string }
  > = [];

  for (const raw of [preEditBase64, preEditCropBase64, postEditBase64, postEditCropBase64]) {
    const normalized = normalizeScreenshotBase64(raw);
    if (normalized) {
      content.push({
        type: "input_image",
        image_url: `data:image/png;base64,${normalized}`,
      });
    }
  }

  content.push({ type: "input_text", text: userText });

  const imageParts = content.filter((item) => item.type === "input_image").length;
  console.log(
    `[agent] OpenAI refinement request model=${OPENAI_FALLBACK_MODEL} imageParts=${imageParts} promptChars=${userText.length}`
  );

  const response = await fetch(OPENAI_RESPONSES_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: OPENAI_FALLBACK_MODEL,
      instructions: REFINEMENT_SYSTEM_PROMPT,
      temperature: 0,
      max_output_tokens: 2048,
      input: [{ role: "user", content }],
    }),
  });

  if (!response.ok) {
    const errorText = await extractOpenAIError(response);
    throw new Error(
      `OpenAI refinement request failed (${response.status} ${response.statusText}): ${errorText}`
    );
  }

  const payload = (await response.json()) as OpenAIResponsePayload;
  const text = extractTextFromOpenAIResponse(payload);
  console.log(`[agent] OpenAI refinement response text chars=${text.length}`);
  return text;
}

function extractTextFromOpenAIResponse(payload: OpenAIResponsePayload): string {
  if (typeof payload.output_text === "string" && payload.output_text.trim()) {
    return payload.output_text.trim();
  }

  const textParts: string[] = [];
  for (const item of payload.output ?? []) {
    if (!item || item.type !== "message" || !Array.isArray(item.content)) {
      continue;
    }

    for (const part of item.content) {
      if (
        (part.type === "output_text" || part.type === "text") &&
        typeof part.text === "string" &&
        part.text.trim()
      ) {
        textParts.push(part.text.trim());
      }
    }
  }

  const joined = textParts.join("\n").trim();
  if (!joined) {
    throw new Error("OpenAI response contained no text content.");
  }
  return joined;
}

async function extractOpenAIError(response: Response): Promise<string> {
  const raw = await response.text();
  if (!raw) {
    return "No error body";
  }
  try {
    const parsed = JSON.parse(raw) as {
      error?: { message?: string; code?: string; type?: string };
    };
    if (typeof parsed.error?.message === "string" && parsed.error.message.trim()) {
      return parsed.error.message.trim();
    }
  } catch {
    // Use raw text if JSON parse fails.
  }
  return raw.slice(0, 400);
}

function readOpenAIApiKey(): string {
  return String(import.meta.env.VITE_OPENAI_API_KEY ?? "").trim();
}

function shouldUseOpenAIFallback(
  geminiApiKey: string,
  lastError: Error | null
): boolean {
  if (!geminiApiKey) {
    return true;
  }
  if (!lastError) {
    return false;
  }

  const message = lastError.message.toLowerCase();
  return (
    message.includes("503") ||
    message.includes("status\":\"unavailable\"") ||
    message.includes("high demand") ||
    message.includes("429") ||
    message.includes("rate limit") ||
    message.includes("timeout") ||
    message.includes("network") ||
    message.includes("fetch failed")
  );
}

type OpenAIResponsePayload = {
  output_text?: string;
  output?: Array<{
    type?: string;
    content?: Array<{ type?: string; text?: string }>;
  }>;
};

function extractTextFromGeminiResponse(payload: GenerateContentResponse): string {
  if (typeof payload.text === "string" && payload.text.trim()) {
    console.log("[agent] extractTextFromGeminiResponse using payload.text");
    return payload.text.trim();
  }

  console.log("[agent] extractTextFromGeminiResponse falling back to candidate parts");
  const candidateTexts: string[] = [];
  for (const candidate of payload.candidates ?? []) {
    for (const part of candidate.content?.parts ?? []) {
      if (typeof part.text === "string" && part.text.trim()) {
        candidateTexts.push(part.text.trim());
      }
    }
  }

  const joined = candidateTexts.join("\n").trim();
  if (!joined) {
    throw new Error("Gemini response contained no text content.");
  }

  return joined;
}

function parseAndValidateOperations(
  responseText: string,
  command: string
): EditOperation[] {
  const candidates: string[] = [];
  const trimmed = responseText.trim();
  if (trimmed) {
    candidates.push(trimmed);
  }

  const markdownMatch = trimmed.match(MARKDOWN_JSON_REGEX);
  if (markdownMatch?.[1]) {
    const stripped = markdownMatch[1].trim();
    if (stripped && stripped !== trimmed) {
      candidates.push(stripped);
    }
  }
  console.log(`[agent] parseAndValidateOperations parseCandidates=${candidates.length}`);

  let lastParseError: Error | null = null;
  for (const candidate of dedupeStrings(candidates)) {
    try {
      console.log(`[agent] Attempting JSON parse candidateChars=${candidate.length}`);
      const parsed = JSON.parse(candidate) as unknown;
      return validateOperations(parsed, command);
    } catch (error) {
      lastParseError =
        error instanceof Error ? error : new Error(String(error));
      console.warn(`[agent] JSON parse candidate failed: ${lastParseError.message}`);
    }
  }

  throw new Error(
    `Unable to parse Gemini JSON response. ${lastParseError?.message ?? "No parse attempts succeeded."}`
  );
}

function validateOperations(raw: unknown, command: string): EditOperation[] {
  if (!Array.isArray(raw)) {
    throw new Error("Gemini output is not a JSON array.");
  }
  console.log(`[agent] Validating ${raw.length} operation(s)`);

  return raw.map((entry, index) => validateOperation(entry, index, command));
}

function validateOperation(
  raw: unknown,
  index: number,
  command: string
): EditOperation {
  if (!isRecord(raw)) {
    throw new Error(`Operation ${index} is not an object.`);
  }

  const action = raw.action;
  if (typeof action !== "string" || !ACTIONS.has(action as EditOperation["action"])) {
    throw new Error(`Operation ${index} has invalid action.`);
  }

  const blendMode = raw.blendMode;
  if (
    typeof blendMode !== "string" ||
    !BLEND_MODES.has(blendMode as EditOperation["blendMode"])
  ) {
    throw new Error(`Operation ${index} has invalid blendMode.`);
  }

  const rawShapes = raw.shapes;
  if (!Array.isArray(rawShapes) || rawShapes.length === 0) {
    throw new Error(`Operation ${index} must include non-empty shapes array.`);
  }
  console.log(
    `[agent] validateOperation #${index + 1}: action=${action} blend=${blendMode} shapes=${rawShapes.length}`
  );

  const requestedBlendMode = blendMode as EditOperation["blendMode"];
  const normalizedBlendMode = canonicalBlendModeForAction(
    action as EditOperation["action"],
    requestedBlendMode
  );
  if (normalizedBlendMode !== requestedBlendMode) {
    console.warn(
      `[agent] Adjusted blendMode for action=${action} from ${requestedBlendMode} to ${normalizedBlendMode}`
    );
  }

  const normalized: EditOperation = {
    action: action as EditOperation["action"],
    blendMode: normalizedBlendMode,
    shapes: rawShapes.map((shape, shapeIndex) =>
      validateShape(shape, index, shapeIndex)
    ),
  };

  const softEdgeDefault =
    normalized.action === "light" || normalized.action === "atmosphere"
      ? 0.2
      : 0.1;
  normalized.softEdge = clampNumber(raw.softEdge, 0.001, 10, softEdgeDefault);
  normalized.sdfSmooth = clampOptionalNumber(raw.sdfSmooth, 0, 10);
  normalized.invert = typeof raw.invert === "boolean" ? raw.invert : undefined;

  if (normalized.action === "delete") {
    normalized.invert = false;
    for (const shape of normalized.shapes) {
      shape.opacity = 0;
    }
    normalized.extractAsset = true;
    normalized.assetLabel = getDeleteAssetLabel(raw.assetLabel, command);
  } else {
    normalized.extractAsset =
      typeof raw.extractAsset === "boolean" ? raw.extractAsset : undefined;
    normalized.assetLabel =
      typeof raw.assetLabel === "string" && raw.assetLabel.trim()
        ? raw.assetLabel.trim()
        : undefined;
  }

  return normalized;
}

function validateShape(
  raw: unknown,
  opIndex: number,
  shapeIndex: number
): SDFShapeConfig {
  if (!isRecord(raw)) {
    throw new Error(`Operation ${opIndex} shape ${shapeIndex} is not an object.`);
  }

  const type = raw.type;
  if (typeof type !== "string" || !SHAPE_TYPES.has(type as SDFShapeConfig["type"])) {
    throw new Error(`Operation ${opIndex} shape ${shapeIndex} has invalid type.`);
  }

  const position = toNumberTuple(raw.position, 3);
  if (!position) {
    throw new Error(`Operation ${opIndex} shape ${shapeIndex} has invalid position.`);
  }

  const shape: SDFShapeConfig = {
    type: type as SDFShapeConfig["type"],
    position: position as [number, number, number],
  };

  const rotation = toNumberTuple(raw.rotation, 4);
  if (rotation) {
    shape.rotation = rotation as [number, number, number, number];
  }

  if (typeof raw.radius === "number" && Number.isFinite(raw.radius)) {
    shape.radius = Math.max(0, raw.radius);
  }

  const scale = toNumberTuple(raw.scale, 3);
  if (scale) {
    shape.scale = [
      Math.max(0, scale[0]),
      Math.max(0, scale[1]),
      Math.max(0, scale[2]),
    ];
  }

  const color = toNumberTuple(raw.color, 3);
  if (color) {
    shape.color = [
      clamp(color[0], 0, 1),
      clamp(color[1], 0, 1),
      clamp(color[2], 0, 1),
    ];
  }

  if (typeof raw.opacity === "number" && Number.isFinite(raw.opacity)) {
    shape.opacity = clamp(raw.opacity, 0, 1);
  }

  const displace = toNumberTuple(raw.displace, 3);
  if (displace) {
    shape.displace = displace as [number, number, number];
  }

  return shape;
}

function getDeleteAssetLabel(rawLabel: unknown, command: string): string {
  if (typeof rawLabel === "string" && rawLabel.trim()) {
    return rawLabel.trim();
  }
  const compact = command
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, "")
    .trim()
    .replace(/\s+/g, " ");
  if (!compact) {
    return "extracted asset";
  }
  return compact.length > 48 ? `${compact.slice(0, 45)}...` : compact;
}

function toNumberTuple(value: unknown, length: number): number[] | null {
  if (!Array.isArray(value) || value.length !== length) {
    return null;
  }
  const out: number[] = [];
  for (const item of value) {
    if (typeof item !== "number" || !Number.isFinite(item)) {
      return null;
    }
    out.push(item);
  }
  return out;
}

function clampOptionalNumber(
  value: unknown,
  min: number,
  max: number
): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return undefined;
  }
  return clamp(value, min, max);
}

function clampNumber(
  value: unknown,
  min: number,
  max: number,
  fallback: number
): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return fallback;
  }
  return clamp(value, min, max);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function dedupeStrings(values: string[]): string[] {
  return Array.from(new Set(values));
}

function canonicalBlendModeForAction(
  action: EditOperation["action"],
  requestedBlendMode: EditOperation["blendMode"]
): EditOperation["blendMode"] {
  if (action === "delete") {
    return "MULTIPLY";
  }
  if (action === "tint") {
    return "MULTIPLY";
  }
  if (action === "recolor") {
    return "SET_RGB";
  }
  if (action === "darken") {
    return "MULTIPLY";
  }
  if (action === "light") {
    return "ADD_RGBA";
  }
  return requestedBlendMode;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function normalizeScreenshotBase64(value: string | null): string | null {
  if (!value) {
    return null;
  }

  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }

  const dataUrlMatch = trimmed.match(/^data:image\/(?:png|jpeg|jpg|webp);base64,(.+)$/i);
  const base64 = dataUrlMatch ? dataUrlMatch[1] : trimmed;
  const compact = base64.replace(/\s+/g, "");
  return compact || null;
}

function gridPosKey(gridPos: [number, number, number]): string {
  return `${gridPos[0]},${gridPos[1]},${gridPos[2]}`;
}

function formatVec3(vec: THREE.Vector3): string {
  return `[${vec.x.toFixed(3)}, ${vec.y.toFixed(3)}, ${vec.z.toFixed(3)}]`;
}

function formatColor(color: THREE.Color): string {
  return `[${color.r.toFixed(3)}, ${color.g.toFixed(3)}, ${color.b.toFixed(3)}]`;
}

const REFINEMENT_SYSTEM_PROMPT = `You are a visual QA assistant for 3D gaussian splat edits.

You receive:
1. The original user command
2. A summary of the edit operations that were applied
3. Images in order: pre-edit scene, pre-edit crop (around click), post-edit scene, post-edit crop. Some may be absent.
4. Optionally, spatial context from a click-selected cluster (bounds, center, dimensions)

Compare the before and after images to evaluate whether the edit achieved the user's intent.

For delete operations:
- The target region should be fully invisible/removed with no visible remnants within shape bounds.
- If remnants are visible at edges, suggest enlarging radius/scale by 1.2-1.5x.
- If the wrong area was deleted, suggest shifting position toward the intended target.

For recolor operations:
- The target region should uniformly show the new color with no spillover to adjacent objects.
- If partial coverage, suggest expanding the shape or shifting position.

Use the spatial context (cluster bounds, center, dimensions) when provided to judge whether the shape covers the intended region.

Respond with a JSON object (no markdown, no prose):
{
  "accept": true/false,
  "confidence": 0.0-1.0,
  "reason": "brief explanation",
  "corrections": []
}

If accept is false, provide corrections as an array of EditOperation objects (same format as the original system).
Corrections should adjust position, scale, or radius — not start from scratch.
Common fixes: enlarge radius/scale if remnants visible, shift position if off-center.
Keep corrections minimal — one shape adjustment is better than a full rewrite.`;

export async function processRefinement(
  command: string,
  opsSummary: string,
  preEditScreenshotBase64: string | null,
  preEditCropBase64: string | null,
  postEditScreenshotBase64: string | null,
  postEditCropBase64: string | null,
  selectionHint: string | null,
  apiKey: string
): Promise<RefinementResponse> {
  const geminiApiKey = apiKey.trim();
  const openAIApiKey = readOpenAIApiKey();
  if (!geminiApiKey && !openAIApiKey) {
    throw new Error("[agent] ERROR: missing API key for refinement");
  }

  const lines: string[] = [
    `Original command: ${command}`,
    "",
    "Applied operations:",
    opsSummary,
    "",
    "Images provided in order: (1) pre-edit scene, (2) pre-edit crop around target, (3) post-edit scene, (4) post-edit crop around target.",
    "Compare before and after to evaluate whether the edit achieved the user's intent.",
  ];

  if (selectionHint) {
    lines.push("");
    lines.push("Spatial context from click selection:");
    lines.push(selectionHint);
  }

  lines.push("");
  lines.push("Evaluate the result. Return only a JSON object with accept, confidence, reason, and corrections.");

  const userText = lines.join("\n");

  console.log(`[agent] processRefinement start command="${command}" opsChars=${opsSummary.length} hasPreEdit=${Boolean(preEditScreenshotBase64)} hasSelection=${Boolean(selectionHint)}`);

  const preferOpenAI = providerPreference === "openai";

  if (geminiApiKey && !preferOpenAI) {
    try {
      const ai = new GoogleGenAI({ apiKey: geminiApiKey });
      const contents = buildRefinementContents(
        userText,
        preEditScreenshotBase64,
        preEditCropBase64,
        postEditScreenshotBase64,
        postEditCropBase64
      );
      const response = await ai.models.generateContent({
        model: GEMINI_MODEL,
        contents,
        config: {
          temperature: 0,
          maxOutputTokens: 2048,
          systemInstruction: REFINEMENT_SYSTEM_PROMPT,
        },
      });
      const text = extractTextFromGeminiResponse(response);
      return parseRefinementResponse(text);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.warn(`[agent] Gemini refinement failed: ${message}`);
      if (!openAIApiKey) throw error;
    }
  }

  if (openAIApiKey) {
    const text = await requestOpenAIRefinementText(
      openAIApiKey,
      userText,
      preEditScreenshotBase64,
      preEditCropBase64,
      postEditScreenshotBase64,
      postEditCropBase64
    );
    return parseRefinementResponse(text);
  }

  throw new Error("[agent] No provider available for refinement");
}

function buildRefinementContents(
  userText: string,
  preEditBase64: string | null,
  preEditCropBase64: string | null,
  postEditBase64: string | null,
  postEditCropBase64: string | null
): Array<{
  role: "user";
  parts: Array<
    | { text: string }
    | { inlineData: { mimeType: "image/png"; data: string } }
  >;
}> {
  const parts: Array<
    | { text: string }
    | { inlineData: { mimeType: "image/png"; data: string } }
  > = [];

  // Images in order: pre-edit full, pre-edit crop, post-edit full, post-edit crop
  for (const raw of [preEditBase64, preEditCropBase64, postEditBase64, postEditCropBase64]) {
    const normalized = normalizeScreenshotBase64(raw);
    if (normalized) {
      parts.push({ inlineData: { mimeType: "image/png", data: normalized } });
    }
  }

  const imageParts = parts.length;
  console.log(`[agent] buildRefinementContents images=${imageParts} textChars=${userText.length}`);

  parts.push({ text: userText });
  return [{ role: "user", parts }];
}

function parseRefinementResponse(responseText: string): RefinementResponse {
  const trimmed = responseText.trim();
  const candidates: string[] = [trimmed];
  const markdownMatch = trimmed.match(MARKDOWN_JSON_REGEX);
  if (markdownMatch?.[1]) {
    candidates.push(markdownMatch[1].trim());
  }

  for (const candidate of dedupeStrings(candidates)) {
    try {
      const parsed = JSON.parse(candidate) as Record<string, unknown>;
      const accept = typeof parsed.accept === "boolean" ? parsed.accept : true;
      const confidence = typeof parsed.confidence === "number" ? clamp(parsed.confidence, 0, 1) : 0.5;
      const reason = typeof parsed.reason === "string" ? parsed.reason : "";
      let corrections: EditOperation[] = [];
      if (Array.isArray(parsed.corrections) && parsed.corrections.length > 0) {
        try {
          corrections = validateOperations(parsed.corrections, "refinement");
        } catch {
          console.warn("[agent] Refinement corrections failed validation; ignoring corrections");
          corrections = [];
        }
      }
      return { accept, confidence, reason, corrections };
    } catch {
      continue;
    }
  }

  console.warn("[agent] Could not parse refinement response; defaulting to accept");
  return { accept: true, confidence: 0.5, reason: "parse failure", corrections: [] };
}

export { SYSTEM_PROMPT };
