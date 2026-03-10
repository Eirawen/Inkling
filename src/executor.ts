import {
  SplatEdit,
  SplatEditSdf,
  SplatEditSdfType,
  SplatEditRgbaBlendMode,
} from "@sparkjsdev/spark";
import * as THREE from "three";
import type { EditOperation, SDFShapeConfig } from "./types";

type HistoryEntry = {
  edit: SplatEdit;
  addedParent: THREE.Object3D;
};

type AssetExtractionHandler = (op: EditOperation, parent: THREE.Object3D) => void;

const history: HistoryEntry[] = [];
let assetExtractionHandler: AssetExtractionHandler | null = null;

const BLEND_MODE_MAP: Record<EditOperation["blendMode"], SplatEditRgbaBlendMode> = {
  MULTIPLY: SplatEditRgbaBlendMode.MULTIPLY,
  SET_RGB: SplatEditRgbaBlendMode.SET_RGB,
  ADD_RGBA: SplatEditRgbaBlendMode.ADD_RGBA,
};

const SHAPE_TYPE_MAP: Record<SDFShapeConfig["type"], SplatEditSdfType> = {
  SPHERE: SplatEditSdfType.SPHERE,
  BOX: SplatEditSdfType.BOX,
  ELLIPSOID: SplatEditSdfType.ELLIPSOID,
  CYLINDER: SplatEditSdfType.CYLINDER,
  CAPSULE: SplatEditSdfType.CAPSULE,
  PLANE: SplatEditSdfType.PLANE,
  INFINITE_CONE: SplatEditSdfType.INFINITE_CONE,
  ALL: SplatEditSdfType.ALL,
};

export function executeOperations(
  ops: EditOperation[],
  parent: THREE.Object3D
): SplatEdit[] {
  const applied: SplatEdit[] = [];
  console.log(`[executor] executeOperations called with ${ops.length} op(s)`);

  for (const [opIndex, op] of ops.entries()) {
    if (op.action === "delete" && op.extractAsset && assetExtractionHandler) {
      try {
        console.log(
          `[executor] Running asset extraction hook for delete op ${opIndex + 1}/${ops.length}`
        );
        assetExtractionHandler(op, parent);
      } catch (error) {
        console.error("[executor] Asset extraction hook failed", error);
      }
    }

    console.log(
      `[executor] Op ${opIndex + 1}/${ops.length}: action=${op.action} blend=${op.blendMode} shapes=${op.shapes.length} softEdge=${op.softEdge ?? "default"} sdfSmooth=${op.sdfSmooth ?? "default"} invert=${op.invert ?? false}`
    );
    const edit = new SplatEdit({
      rgbaBlendMode: BLEND_MODE_MAP[op.blendMode],
      softEdge: op.softEdge,
      sdfSmooth: op.sdfSmooth,
      invert: op.invert === true ? true : false,
    });

    for (const [shapeIndex, shape] of op.shapes.entries()) {
      console.log(
        `[executor]   Shape ${shapeIndex + 1}/${op.shapes.length}: type=${shape.type} pos=[${shape.position.join(", ")}] radius=${shape.radius ?? "-"} opacity=${shape.opacity ?? "-"} scale=${shape.scale ? `[${shape.scale.join(", ")}]` : "-"}`
      );
      const sdf = new SplatEditSdf({
        type: SHAPE_TYPE_MAP[shape.type],
      });

      sdf.position.set(shape.position[0], shape.position[1], shape.position[2]);

      if (shape.radius !== undefined) {
        sdf.radius = shape.radius;
      }
      if (shape.color) {
        if (sdf.color) {
          sdf.color.setRGB(shape.color[0], shape.color[1], shape.color[2]);
        } else {
          sdf.color = new THREE.Color(shape.color[0], shape.color[1], shape.color[2]);
        }
      }
      if (shape.opacity !== undefined) {
        sdf.opacity = shape.opacity;
      }
      if (shape.scale) {
        sdf.scale.set(shape.scale[0], shape.scale[1], shape.scale[2]);
      }
      if (shape.rotation) {
        sdf.quaternion.set(
          shape.rotation[0],
          shape.rotation[1],
          shape.rotation[2],
          shape.rotation[3]
        );
      }
      if (shape.displace) {
        sdf.displace?.set(shape.displace[0], shape.displace[1], shape.displace[2]);
      }

      edit.addSdf(sdf);
    }

    const targetParent = resolveParentForOperation(op, parent);
    targetParent.add(edit);
    console.log(
      `[executor] Added edit to parent=${targetParent.type} (isScene=${targetParent instanceof THREE.Scene})`
    );

    history.push({ edit, addedParent: targetParent });
    applied.push(edit);

    console.log(`[executor] Applied ${op.action} with ${op.shapes.length} shapes`);
  }

  return applied;
}

export function setAssetExtractionHandler(
  handler: AssetExtractionHandler | null
): void {
  assetExtractionHandler = handler;
  console.log(
    `[executor] Asset extraction handler ${handler ? "registered" : "cleared"}`
  );
}

export function undoLastEdit(): boolean {
  const last = history.pop();
  if (!last) {
    console.log("[executor] undoLastEdit called with empty history");
    return false;
  }

  const removalParent = last.edit.parent ?? last.addedParent;
  console.log(
    `[executor] undoLastEdit removing from parent=${removalParent.type} historyBefore=${history.length + 1}`
  );
  removalParent.remove(last.edit);

  const maybeDisposable = last.edit as unknown as { dispose?: () => void };
  if (typeof maybeDisposable.dispose === "function") {
    maybeDisposable.dispose();
  }

  console.log(`[executor] Undid last edit (${history.length} remaining)`);
  return true;
}

export function undoN(count: number): number {
  const toUndo = Math.min(count, history.length);
  console.log(`[executor] undoN requested=${count} actual=${toUndo} historyBefore=${history.length}`);
  let undone = 0;
  for (let i = 0; i < toUndo; i++) {
    const entry = history.pop();
    if (!entry) break;
    const removalParent = entry.edit.parent ?? entry.addedParent;
    removalParent.remove(entry.edit);
    const maybeDisposable = entry.edit as unknown as { dispose?: () => void };
    if (typeof maybeDisposable.dispose === "function") {
      maybeDisposable.dispose();
    }
    undone++;
  }
  console.log(`[executor] undoN undone=${undone} historyAfter=${history.length}`);
  return undone;
}

export function undoAllEdits(): void {
  console.log(`[executor] undoAllEdits called for ${history.length} edit(s)`);
  while (history.length > 0) {
    const entry = history.pop()!;
    const removalParent = entry.edit.parent ?? entry.addedParent;
    removalParent.remove(entry.edit);

    const maybeDisposable = entry.edit as unknown as { dispose?: () => void };
    if (typeof maybeDisposable.dispose === "function") {
      maybeDisposable.dispose();
    }
  }

  console.log("[executor] Cleared all edits (0 remaining)");
}

export function getEditHistory(): readonly SplatEdit[] {
  return history.map((entry) => entry.edit);
}

function resolveParentForOperation(
  op: EditOperation,
  defaultParent: THREE.Object3D
): THREE.Object3D {
  const isGlobalLight =
    op.action === "light" && op.shapes.some((shape) => shape.type === "ALL");
  const shouldUseScene = op.action === "atmosphere" || isGlobalLight;

  if (!shouldUseScene) {
    console.log("[executor] Parent resolution: using provided parent (scoped edit)");
    return defaultParent;
  }

  if (defaultParent instanceof THREE.Scene) {
    console.log("[executor] Parent resolution: provided parent is scene (global edit)");
    return defaultParent;
  }

  const sceneAncestor = findSceneAncestor(defaultParent);
  if (sceneAncestor) {
    console.log("[executor] Parent resolution: found scene ancestor for global edit");
    return sceneAncestor;
  }

  console.warn(
    "[executor] Global operation requested, but scene parent was unavailable. Falling back to provided parent."
  );
  return defaultParent;
}

function findSceneAncestor(node: THREE.Object3D): THREE.Scene | null {
  let current: THREE.Object3D | null = node;
  while (current) {
    if (current instanceof THREE.Scene) {
      return current;
    }
    current = current.parent;
  }
  return null;
}
