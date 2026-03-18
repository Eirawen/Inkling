import type { SplatEdit } from "@sparkjsdev/spark";
import type * as THREE from "three";
import type { EditOperation, RefinementResponse } from "./types";

export interface RefinementDependencies {
  processRefinement: (
    command: string,
    opsSummary: string,
    preEditScreenshotBase64: string | null,
    preEditCropBase64: string | null,
    postEditScreenshotBase64: string | null,
    postEditCropBase64: string | null,
    selectionHint: string | null,
    apiKey: string
  ) => Promise<RefinementResponse>;
  executeOperations: (ops: EditOperation[], parent: THREE.Object3D) => SplatEdit[];
  undoN: (count: number) => number;
  getScreenshot: () => string;
  getScreenshotCropAroundPoint?: (point: THREE.Vector3, sizePx?: number) => string | null;
  getApiKey: () => string;
}

export interface RefinementConfig {
  maxIterations: number;
  acceptConfidence: number;
  renderDelayMs: number;
  cropSizePx: number;
}

export interface RefinementResult {
  finalOps: EditOperation[];
  iterations: number;
  totalEditsApplied: number;
  accepted: boolean;
  reason: string;
}

const DEFAULT_CONFIG: RefinementConfig = {
  maxIterations: 2,
  acceptConfidence: 0.85,
  renderDelayMs: 150,
  cropSizePx: 320,
};

export async function refineEdit(
  initialOps: EditOperation[],
  appliedEdits: SplatEdit[],
  command: string,
  clickPosition: THREE.Vector3 | null,
  editParent: THREE.Object3D,
  preEditScreenshotBase64: string | null,
  preEditCropBase64: string | null,
  selectionHint: string | null,
  deps: RefinementDependencies,
  config?: Partial<RefinementConfig>
): Promise<RefinementResult> {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  let currentOps = initialOps;
  let totalEditsApplied = appliedEdits.length;
  let lastReason = "initial";

  console.log(
    `[refine] Starting refinement loop maxIter=${cfg.maxIterations} threshold=${cfg.acceptConfidence}`
  );

  for (let iteration = 1; iteration <= cfg.maxIterations; iteration++) {
    // Wait for GPU to render the current state
    await delay(cfg.renderDelayMs);

    // Capture post-edit screenshots
    const screenshot = normalizeBase64(deps.getScreenshot());
    const crop =
      clickPosition && deps.getScreenshotCropAroundPoint
        ? normalizeBase64(deps.getScreenshotCropAroundPoint(clickPosition, cfg.cropSizePx) ?? "")
        : null;

    const opsSummary = summarizeOps(currentOps);
    const apiKey = deps.getApiKey();

    console.log(`[refine] Iteration ${iteration}/${cfg.maxIterations} opsCount=${currentOps.length}`);

    let refinement: RefinementResponse;
    try {
      refinement = await deps.processRefinement(
        command,
        opsSummary,
        preEditScreenshotBase64,
        preEditCropBase64,
        screenshot,
        crop,
        selectionHint,
        apiKey
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.warn(`[refine] Refinement call failed: ${message}; accepting current state`);
      return {
        finalOps: currentOps,
        iterations: iteration,
        totalEditsApplied,
        accepted: true,
        reason: `refinement error: ${message}`,
      };
    }

    console.log(
      `[refine] Iteration ${iteration} result: accept=${refinement.accept} confidence=${refinement.confidence.toFixed(2)} reason="${refinement.reason}" corrections=${refinement.corrections.length}`
    );

    if (refinement.accept || refinement.confidence >= cfg.acceptConfidence) {
      lastReason = refinement.reason || "accepted";
      return {
        finalOps: currentOps,
        iterations: iteration,
        totalEditsApplied,
        accepted: true,
        reason: lastReason,
      };
    }

    if (refinement.corrections.length === 0) {
      lastReason = refinement.reason || "no corrections provided";
      console.log(`[refine] Rejected but no corrections; accepting as-is`);
      return {
        finalOps: currentOps,
        iterations: iteration,
        totalEditsApplied,
        accepted: false,
        reason: lastReason,
      };
    }

    // Undo current edits and apply corrections
    console.log(`[refine] Undoing ${totalEditsApplied} edits and applying ${refinement.corrections.length} corrections`);
    deps.undoN(totalEditsApplied);

    // Merge: use corrections as the new ops
    currentOps = refinement.corrections;
    const newEdits = deps.executeOperations(currentOps, editParent);
    totalEditsApplied = newEdits.length;
    lastReason = refinement.reason;
  }

  console.log(`[refine] Max iterations reached; accepting current state`);
  return {
    finalOps: currentOps,
    iterations: cfg.maxIterations,
    totalEditsApplied,
    accepted: false,
    reason: lastReason || "max iterations reached",
  };
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function normalizeBase64(dataUrl: string): string {
  const trimmed = dataUrl.trim();
  if (!trimmed) return trimmed;
  const match = trimmed.match(/^data:image\/(?:png|jpeg|jpg|webp);base64,(.+)$/i);
  return match ? match[1] : trimmed;
}

function summarizeOps(ops: EditOperation[]): string {
  return ops
    .map((op, i) => {
      const shapes = op.shapes
        .map((s) => {
          const parts = [`type=${s.type}`, `pos=[${s.position.join(",")}]`];
          if (s.radius !== undefined) parts.push(`r=${s.radius}`);
          if (s.scale) parts.push(`scale=[${s.scale.join(",")}]`);
          return parts.join(" ");
        })
        .join("; ");
      return `Op${i + 1}: ${op.action} blend=${op.blendMode} softEdge=${op.softEdge ?? "?"} shapes=[${shapes}]`;
    })
    .join("\n");
}
