import {
  SplatEdit,
  SplatEditSdf,
  SplatEditSdfType,
  SplatEditRgbaBlendMode,
  type SplatMesh,
} from "@sparkjsdev/spark";
import * as THREE from "three";
import type { EditOperation, InfillConfig, SDFShapeConfig } from "./types";
import type { InfillResult } from "./infill";

type HistoryEntry = {
  edit: SplatEdit;
  addedParent: THREE.Object3D;
  fillMesh?: SplatMesh;
  fillMeshParent?: THREE.Object3D;
};

type AssetExtractionHandler = (op: EditOperation, parent: THREE.Object3D) => void;
type InfillHandler = (shapes: SDFShapeConfig[], opts: InfillConfig) => InfillResult | null;

const history: HistoryEntry[] = [];
let assetExtractionHandler: AssetExtractionHandler | null = null;
let infillHandler: InfillHandler | null = null;

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
    // For deletes: hard cutoff (no softEdge blur) and expand shapes slightly
    // to consume degraded boundary splats that were captured at grazing angles
    const isDelete = op.action === "delete";
    const effectiveSoftEdge = isDelete ? 0 : op.softEdge;
    const DELETE_OVERSIZE = 1.12; // 12% expansion eats boundary artifacts

    const edit = new SplatEdit({
      rgbaBlendMode: BLEND_MODE_MAP[op.blendMode],
      softEdge: effectiveSoftEdge,
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
        sdf.radius = isDelete ? shape.radius * DELETE_OVERSIZE : shape.radius;
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
        const sm = isDelete ? DELETE_OVERSIZE : 1;
        sdf.scale.set(shape.scale[0] * sm, shape.scale[1] * sm, shape.scale[2] * sm);
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

    const entry: HistoryEntry = { edit, addedParent: targetParent };

    // Trigger infill after delete operations
    if (op.action === "delete" && infillHandler) {
      try {
        const result = infillHandler(op.shapes, {
          softEdge: op.softEdge,
          sdfSmooth: op.sdfSmooth,
        });
        if (result) {
          const sceneParent = findSceneAncestor(targetParent) ?? targetParent;
          sceneParent.add(result.mesh);
          entry.fillMesh = result.mesh;
          entry.fillMeshParent = sceneParent;
          console.log(`[executor] Infill: added fill mesh with ${result.splatCount} splats across ${result.surfaceCount} surfaces`);
        }
      } catch (error) {
        console.error("[executor] Infill failed", error);
      }
    }

    history.push(entry);
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

export function setInfillHandler(handler: InfillHandler | null): void {
  infillHandler = handler;
  console.log(`[executor] Infill handler ${handler ? "registered" : "cleared"}`);
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

  removeFillMesh(last);

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
    removeFillMesh(entry);
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

    removeFillMesh(entry);
  }

  console.log("[executor] Cleared all edits (0 remaining)");
}

function removeFillMesh(entry: HistoryEntry): void {
  if (!entry.fillMesh) return;
  const parent = entry.fillMeshParent ?? entry.addedParent;
  parent.remove(entry.fillMesh);
  const disposable = entry.fillMesh as unknown as { dispose?: () => void };
  if (typeof disposable.dispose === "function") {
    disposable.dispose();
  }
}

export function getEditHistory(): readonly SplatEdit[] {
  return history.map((entry) => entry.edit);
}

export function transformOperations(
  ops: EditOperation[],
  transform: {
    scaleMultiplier?: number;
    positionOffset?: [number, number, number];
  }
): EditOperation[] {
  return ops.map((op) => {
    const newShapes = op.shapes.map((shape) => {
      const s: SDFShapeConfig = {
        ...shape,
        position: [...shape.position] as [number, number, number],
        ...(shape.rotation ? { rotation: [...shape.rotation] as [number, number, number, number] } : {}),
        ...(shape.scale ? { scale: [...shape.scale] as [number, number, number] } : {}),
        ...(shape.color ? { color: [...shape.color] as [number, number, number] } : {}),
        ...(shape.displace ? { displace: [...shape.displace] as [number, number, number] } : {}),
      };

      if (transform.scaleMultiplier !== undefined) {
        const m = transform.scaleMultiplier;
        if (s.radius !== undefined) s.radius *= m;
        if (s.scale) s.scale = [s.scale[0] * m, s.scale[1] * m, s.scale[2] * m];
      }

      if (transform.positionOffset) {
        const [dx, dy, dz] = transform.positionOffset;
        s.position = [s.position[0] + dx, s.position[1] + dy, s.position[2] + dz];
      }

      return s;
    });

    const newOp: EditOperation = { ...op, shapes: newShapes };
    if (transform.scaleMultiplier !== undefined && newOp.softEdge !== undefined) {
      newOp.softEdge *= transform.scaleMultiplier;
    }
    return newOp;
  });
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
