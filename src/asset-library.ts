import { PackedSplats, SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";
import type { AssetEntry, EditOperation, SDFShapeConfig } from "./types";

type CapturedSplat = {
  center: THREE.Vector3;
  scales: THREE.Vector3;
  quaternion: THREE.Quaternion;
  opacity: number;
  color: THREE.Color;
};

type CompiledShape = {
  type: "SPHERE" | "ELLIPSOID" | "BOX" | "CYLINDER" | "CAPSULE";
  position: THREE.Vector3;
  inverseRotation: THREE.Quaternion;
  radius: number;
  scale: THREE.Vector3;
};

const assets: AssetEntry[] = [];
const BUILTIN_ASSET_PREFIX = "builtin_";
const PRELOADED_ASSET_MANIFEST_URL = "/scenes/preloaded-assets/index.json";
const LEGACY_FALLBACK_PRELOADED: Array<{ url: string; label: string }> = [
  { url: "/scenes/butterfly.spz", label: "butterfly" },
];
let defaultAssetSeeded = false;

const MIN_ASSET_SPLATS = 1;
const MAX_EXTRACTED_SPLATS = 120_000;
const MIN_OPACITY = 0.02;
const EPSILON = 1e-6;

const tmpLocal = new THREE.Vector3();

export function extractAssetFromDeleteOperation(
  op: EditOperation,
  splatMesh: SplatMesh,
  sourceScene: string
): AssetEntry | null {
  if (op.action !== "delete" || !op.extractAsset) {
    return null;
  }

  const startMs = nowMs();
  const compiledShapes = compileSupportedShapes(op.shapes, op.softEdge ?? 0.1);
  if (compiledShapes.length === 0) {
    console.warn(
      `[asset-library] No supported delete shapes found for extraction. opShapes=${op.shapes
        .map((shape) => shape.type)
        .join(",")}`
    );
    return null;
  }

  const candidates: CapturedSplat[] = [];
  let insideRegionCount = 0;
  let filteredLowOpacity = 0;
  const world = getSplatMeshWorldTransform(splatMesh);

  splatMesh.forEachSplat((_, center, scales, quaternion, opacity, color) => {
    const worldCenter = toWorldCenter(center, world.matrixWorld);
    if (!pointInsideAnyCompiledShape(worldCenter, compiledShapes)) {
      return;
    }

    insideRegionCount += 1;
    if (opacity < MIN_OPACITY) {
      filteredLowOpacity += 1;
      return;
    }

    candidates.push({
      center: worldCenter,
      scales: toWorldScales(scales, world.uniformScale),
      quaternion: toWorldQuaternion(quaternion, world.worldQuaternion),
      opacity,
      color: new THREE.Color().copy(color),
    });
  });

  let working = candidates;
  if (working.length < MIN_ASSET_SPLATS) {
    console.warn(
      `[asset-library] Too few strict candidates (inside=${insideRegionCount}, kept=${working.length}). Retrying with relaxed capture.`
    );
    working = collectRelaxedCandidates(splatMesh, compiledShapes);
  }
  if (working.length < MIN_ASSET_SPLATS) {
    console.warn(
      `[asset-library] Too few relaxed candidates (${working.length}). Retrying with proximity capture.`
    );
    working = collectProximityCandidates(splatMesh, compiledShapes);
  }
  if (working.length === 0) {
    console.warn(
      "[asset-library] No splats captured from region. Generating synthetic fallback asset."
    );
    working = buildSyntheticFallbackCandidates(op.shapes);
  }
  if (working.length === 0) {
    console.warn("[asset-library] Synthetic fallback failed; skipping asset extraction");
    return null;
  }

  const kept = working;

  // Sort by opacity descending so truncation keeps most visually important splats
  kept.sort((a, b) => b.opacity - a.opacity);
  const truncated = kept.slice(0, MAX_EXTRACTED_SPLATS);
  const centroid = computeCentroid(truncated);
  const bounds = computeBounds(truncated);

  const packedSplats = new PackedSplats({ maxSplats: truncated.length + 1 });
  for (const splat of truncated) {
    const localCenter = new THREE.Vector3().copy(splat.center).sub(centroid);
    packedSplats.pushSplat(
      localCenter,
      splat.scales,
      splat.quaternion,
      splat.opacity,
      splat.color
    );
  }
  packedSplats.needsUpdate = true;

  const entry: AssetEntry = {
    id: buildAssetId(),
    label: normalizeAssetLabel(op.assetLabel),
    sourceScene,
    extractedAt: new Date(),
    splats: packedSplats,
    thumbnailDataUrl: buildPlaceholderThumbnail(normalizeAssetLabel(op.assetLabel)),
    originalPosition: centroid,
    bounds,
    splatCount: truncated.length,
  };

  const elapsedMs = nowMs() - startMs;
  console.log(
    `[asset-library] Extracted asset "${entry.label}" splats=${entry.splatCount} inside=${insideRegionCount} filteredOpacity=${filteredLowOpacity} kept=${kept.length} ms=${elapsedMs.toFixed(1)}`
  );

  return entry;
}

export function addAsset(entry: AssetEntry): void {
  assets.unshift(entry);
  console.log(
    `[asset-library] Added asset id=${entry.id} label="${entry.label}" total=${assets.length}`
  );
}

export function listAssets(): readonly AssetEntry[] {
  const builtin = assets.filter((asset) => asset.id.startsWith(BUILTIN_ASSET_PREFIX));
  const regular = assets.filter((asset) => !asset.id.startsWith(BUILTIN_ASSET_PREFIX));
  return [...builtin, ...regular];
}

export function getAssetById(id: string): AssetEntry | undefined {
  return assets.find((asset) => asset.id === id);
}

export function createPlacedAssetMesh(
  asset: AssetEntry,
  worldPos: THREE.Vector3
): SplatMesh {
  const mesh = new SplatMesh({ packedSplats: asset.splats });
  mesh.maxSh = 0;
  mesh.quaternion.set(1, 0, 0, 0);
  mesh.position.copy(worldPos);
  console.log(
    `[asset-library] Created placed mesh for "${asset.label}" at [${worldPos.x.toFixed(3)}, ${worldPos.y.toFixed(3)}, ${worldPos.z.toFixed(3)}]`
  );
  return mesh;
}

export async function ensureDefaultLibraryAsset(): Promise<void> {
  if (defaultAssetSeeded) {
    return;
  }

  defaultAssetSeeded = true;

  const preloadedSpecs = await loadPreloadedAssetSpecs();
  if (preloadedSpecs.length === 0) {
    console.warn("[asset-library] No preloaded assets configured");
    return;
  }

  for (const spec of preloadedSpecs) {
    await seedPreloadedAsset(spec.url, spec.label);
  }
}

async function seedPreloadedAsset(url: string, label: string): Promise<void> {
  const id = `${BUILTIN_ASSET_PREFIX}${slugifyLabel(label)}`;
  if (assets.some((asset) => asset.id === id)) {
    return;
  }

  try {
    console.log(`[asset-library] Loading preloaded asset "${label}" from ${url}`);
    const packedSplats = new PackedSplats({ url });
    await packedSplats.initialized;

    const captured: CapturedSplat[] = [];
    packedSplats.forEachSplat((_, center, scales, quaternion, opacity, color) => {
      captured.push({
        center: new THREE.Vector3().copy(center),
        scales: new THREE.Vector3().copy(scales),
        quaternion: new THREE.Quaternion().copy(quaternion),
        opacity,
        color: new THREE.Color().copy(color),
      });
    });

    if (captured.length === 0) {
      console.warn(`[asset-library] Preloaded asset "${label}" has zero splats, skipping`);
      return;
    }

    const centroid = computeCentroid(captured);
    const bounds = computeBounds(captured);

    const normalizedPackedSplats = new PackedSplats({ maxSplats: captured.length + 1 });
    for (const splat of captured) {
      const localCenter = new THREE.Vector3().copy(splat.center).sub(centroid);
      normalizedPackedSplats.pushSplat(
        localCenter,
        splat.scales,
        splat.quaternion,
        splat.opacity,
        splat.color
      );
    }
    normalizedPackedSplats.needsUpdate = true;

    const entry: AssetEntry = {
      id,
      label,
      sourceScene: "builtin",
      extractedAt: new Date(),
      splats: normalizedPackedSplats,
      thumbnailDataUrl: buildPlaceholderThumbnail(label),
      originalPosition: centroid,
      bounds,
      splatCount: captured.length,
    };

    assets.push(entry);
    console.log(
      `[asset-library] Builtin asset ready id=${entry.id} label="${entry.label}" splats=${entry.splatCount}`
    );
  } catch (error) {
    console.warn(
      `[asset-library] Failed to load preloaded asset "${label}" from ${url}`,
      error
    );
  }
}

async function loadPreloadedAssetSpecs(): Promise<Array<{ url: string; label: string }>> {
  try {
    const response = await fetch(PRELOADED_ASSET_MANIFEST_URL);
    if (!response.ok) {
      console.warn(
        `[asset-library] Preloaded manifest missing (${response.status}). Using legacy fallback list.`
      );
      return LEGACY_FALLBACK_PRELOADED;
    }
    const payload = (await response.json()) as unknown;
    const parsed = normalizePreloadedManifest(payload);
    if (parsed.length > 0) {
      return parsed;
    }
    console.warn("[asset-library] Preloaded manifest empty/invalid. Using legacy fallback list.");
    return LEGACY_FALLBACK_PRELOADED;
  } catch (error) {
    console.warn(
      "[asset-library] Failed to load preloaded asset manifest. Using legacy fallback list.",
      error
    );
    return LEGACY_FALLBACK_PRELOADED;
  }
}

function normalizePreloadedManifest(
  payload: unknown
): Array<{ url: string; label: string }> {
  if (!Array.isArray(payload)) {
    return [];
  }

  const out: Array<{ url: string; label: string }> = [];
  for (const item of payload) {
    if (typeof item === "string") {
      const url = item.trim();
      if (!url) {
        continue;
      }
      out.push({ url, label: inferLabelFromUrl(url) });
      continue;
    }

    if (!item || typeof item !== "object") {
      continue;
    }

    const record = item as { url?: unknown; label?: unknown };
    const url = typeof record.url === "string" ? record.url.trim() : "";
    if (!url) {
      continue;
    }

    const label =
      typeof record.label === "string" && record.label.trim()
        ? record.label.trim()
        : inferLabelFromUrl(url);
    out.push({ url, label });
  }

  return out;
}

function inferLabelFromUrl(url: string): string {
  const clean = url.split("?")[0] ?? url;
  const tail = clean.split("/").pop() ?? clean;
  return tail.replace(/\.spz$/i, "").replace(/[_-]+/g, " ").trim() || "preloaded asset";
}

function slugifyLabel(label: string): string {
  const base = label
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return base || "asset";
}

function compileSupportedShapes(shapes: SDFShapeConfig[], softEdge: number = 0): CompiledShape[] {
  const compiled: CompiledShape[] = [];

  for (const shape of shapes) {
    if (
      shape.type !== "SPHERE" &&
      shape.type !== "ELLIPSOID" &&
      shape.type !== "BOX" &&
      shape.type !== "CYLINDER" &&
      shape.type !== "CAPSULE"
    ) {
      console.warn(`[asset-library] Skipping unsupported extraction shape type=${shape.type}`);
      continue;
    }

    const position = new THREE.Vector3(
      shape.position[0],
      shape.position[1],
      shape.position[2]
    );
    const rotation = shape.rotation
      ? new THREE.Quaternion(
          shape.rotation[0],
          shape.rotation[1],
          shape.rotation[2],
          shape.rotation[3]
        )
      : new THREE.Quaternion(0, 0, 0, 1);

    const scale = shape.scale
      ? new THREE.Vector3(shape.scale[0], shape.scale[1], shape.scale[2])
      : new THREE.Vector3(shape.radius ?? 0.5, shape.radius ?? 0.5, shape.radius ?? 0.5);

    let radius = shape.radius ?? Math.max(scale.x, scale.y, scale.z);

    // Expand shapes by softEdge to match GPU SplatEdit fade zone
    if (softEdge > 0) {
      switch (shape.type) {
        case "SPHERE":
          radius += softEdge;
          break;
        case "BOX":
        case "ELLIPSOID":
          scale.addScalar(softEdge);
          break;
        case "CYLINDER":
          radius += softEdge;
          scale.y += softEdge;
          break;
        case "CAPSULE":
          radius += softEdge;
          break;
      }
    }

    compiled.push({
      type: shape.type,
      position,
      inverseRotation: rotation.clone().invert(),
      radius,
      scale,
    });
  }

  return compiled;
}

function pointInsideAnyCompiledShape(
  point: THREE.Vector3,
  compiledShapes: CompiledShape[]
): boolean {
  for (const shape of compiledShapes) {
    if (pointInsideCompiledShape(point, shape)) {
      return true;
    }
  }
  return false;
}

function pointInsideCompiledShape(point: THREE.Vector3, shape: CompiledShape): boolean {
  tmpLocal.copy(point).sub(shape.position);
  tmpLocal.applyQuaternion(shape.inverseRotation);

  switch (shape.type) {
    case "SPHERE": {
      const radius = Math.max(shape.radius, EPSILON);
      return tmpLocal.lengthSq() <= radius * radius;
    }
    case "ELLIPSOID": {
      const rx = Math.max(shape.scale.x, EPSILON);
      const ry = Math.max(shape.scale.y, EPSILON);
      const rz = Math.max(shape.scale.z, EPSILON);
      const normalized =
        (tmpLocal.x * tmpLocal.x) / (rx * rx) +
        (tmpLocal.y * tmpLocal.y) / (ry * ry) +
        (tmpLocal.z * tmpLocal.z) / (rz * rz);
      return normalized <= 1;
    }
    case "BOX": {
      return (
        Math.abs(tmpLocal.x) <= Math.max(shape.scale.x, EPSILON) &&
        Math.abs(tmpLocal.y) <= Math.max(shape.scale.y, EPSILON) &&
        Math.abs(tmpLocal.z) <= Math.max(shape.scale.z, EPSILON)
      );
    }
    case "CYLINDER": {
      const radius = Math.max(shape.radius, shape.scale.x, shape.scale.z, EPSILON);
      const halfHeight = Math.max(shape.scale.y, EPSILON);
      const radial = Math.sqrt(tmpLocal.x * tmpLocal.x + tmpLocal.z * tmpLocal.z);
      return radial <= radius && Math.abs(tmpLocal.y) <= halfHeight;
    }
    case "CAPSULE": {
      const radius = Math.max(shape.radius, shape.scale.x, shape.scale.z, EPSILON);
      const halfSegment = Math.max(shape.scale.y - radius, 0);
      const closestY = clamp(tmpLocal.y, -halfSegment, halfSegment);
      const dx = tmpLocal.x;
      const dy = tmpLocal.y - closestY;
      const dz = tmpLocal.z;
      return dx * dx + dy * dy + dz * dz <= radius * radius;
    }
    default:
      return false;
  }
}

function computeCentroid(splats: CapturedSplat[]): THREE.Vector3 {
  const centroid = new THREE.Vector3();
  if (splats.length === 0) {
    return centroid;
  }

  for (const splat of splats) {
    centroid.add(splat.center);
  }

  centroid.multiplyScalar(1 / splats.length);
  return centroid;
}

function computeBounds(splats: CapturedSplat[]): THREE.Box3 {
  const bounds = new THREE.Box3();
  if (splats.length === 0) {
    bounds.set(
      new THREE.Vector3(-0.001, -0.001, -0.001),
      new THREE.Vector3(0.001, 0.001, 0.001)
    );
    return bounds;
  }

  for (const splat of splats) {
    bounds.expandByPoint(splat.center);
  }

  return bounds;
}

function normalizeAssetLabel(label: string | undefined): string {
  const trimmed = label?.trim();
  return trimmed && trimmed.length > 0 ? trimmed : "extracted asset";
}

function buildAssetId(): string {
  return `asset_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function buildPlaceholderThumbnail(label: string): string {
  const safeLabel = label.replace(/[<>&]/g, "");
  const svg = `
<svg xmlns="http://www.w3.org/2000/svg" width="256" height="144" viewBox="0 0 256 144">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#23303a"/>
      <stop offset="100%" stop-color="#11161b"/>
    </linearGradient>
  </defs>
  <rect width="256" height="144" fill="url(#g)"/>
  <text x="12" y="76" fill="#f3f8ff" font-family="sans-serif" font-size="16">${safeLabel}</text>
</svg>`.trim();

  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

function nowMs(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function collectRelaxedCandidates(
  splatMesh: SplatMesh,
  compiledShapes: CompiledShape[]
): CapturedSplat[] {
  const relaxed: CapturedSplat[] = [];
  const world = getSplatMeshWorldTransform(splatMesh);
  splatMesh.forEachSplat((_, center, scales, quaternion, opacity, color) => {
    const worldCenter = toWorldCenter(center, world.matrixWorld);
    if (!pointInsideAnyCompiledShape(worldCenter, compiledShapes)) {
      return;
    }
    relaxed.push({
      center: worldCenter,
      scales: toWorldScales(scales, world.uniformScale),
      quaternion: toWorldQuaternion(quaternion, world.worldQuaternion),
      opacity,
      color: new THREE.Color().copy(color),
    });
  });
  return relaxed;
}

function collectProximityCandidates(
  splatMesh: SplatMesh,
  compiledShapes: CompiledShape[]
): CapturedSplat[] {
  const proximity: CapturedSplat[] = [];
  const world = getSplatMeshWorldTransform(splatMesh);
  const centers = compiledShapes.map((shape) => shape.position);
  const radiusHint =
    compiledShapes.reduce((max, shape) => {
      const size = Math.max(shape.radius, shape.scale.x, shape.scale.y, shape.scale.z);
      return Math.max(max, size);
    }, 0) || 0.25;
  const captureRadius = Math.max(0.35, radiusHint * 2.8);

  splatMesh.forEachSplat((_, center, scales, quaternion, opacity, color) => {
    const worldCenter = toWorldCenter(center, world.matrixWorld);
    let minDistanceSq = Number.POSITIVE_INFINITY;
    for (const c of centers) {
      const dx = worldCenter.x - c.x;
      const dy = worldCenter.y - c.y;
      const dz = worldCenter.z - c.z;
      const distSq = dx * dx + dy * dy + dz * dz;
      if (distSq < minDistanceSq) {
        minDistanceSq = distSq;
      }
    }

    if (minDistanceSq > captureRadius * captureRadius) {
      return;
    }

    proximity.push({
      center: worldCenter,
      scales: toWorldScales(scales, world.uniformScale),
      quaternion: toWorldQuaternion(quaternion, world.worldQuaternion),
      opacity,
      color: new THREE.Color().copy(color),
    });
  });

  console.log(
    `[asset-library] Proximity capture kept=${proximity.length} radius=${captureRadius.toFixed(3)}`
  );
  return proximity;
}

function buildSyntheticFallbackCandidates(shapes: SDFShapeConfig[]): CapturedSplat[] {
  const out: CapturedSplat[] = [];
  for (const shape of shapes) {
    const position = new THREE.Vector3(
      shape.position[0],
      shape.position[1],
      shape.position[2]
    );
    const color = shape.color
      ? new THREE.Color(shape.color[0], shape.color[1], shape.color[2])
      : new THREE.Color(0.6, 0.6, 0.6);
    const baseRadius = Math.max(0.02, shape.radius ?? 0.12);
    out.push({
      center: position,
      scales: new THREE.Vector3(baseRadius, baseRadius, baseRadius),
      quaternion: new THREE.Quaternion(0, 0, 0, 1),
      opacity: 1,
      color,
    });
  }
  return out;
}

type SplatMeshWorldTransform = {
  matrixWorld: THREE.Matrix4;
  worldQuaternion: THREE.Quaternion;
  uniformScale: number;
};

function getSplatMeshWorldTransform(splatMesh: SplatMesh): SplatMeshWorldTransform {
  if (typeof splatMesh.updateMatrixWorld === "function") {
    splatMesh.updateMatrixWorld(true);
  }

  const matrixWorld =
    (splatMesh as unknown as { matrixWorld?: THREE.Matrix4 }).matrixWorld?.clone() ??
    new THREE.Matrix4().identity();

  const worldQuaternion = new THREE.Quaternion();
  if (typeof splatMesh.getWorldQuaternion === "function") {
    splatMesh.getWorldQuaternion(worldQuaternion);
  } else {
    worldQuaternion.set(0, 0, 0, 1);
  }

  const worldScale = new THREE.Vector3(1, 1, 1);
  if (typeof splatMesh.getWorldScale === "function") {
    splatMesh.getWorldScale(worldScale);
  }
  const uniformScale = (Math.abs(worldScale.x) + Math.abs(worldScale.y) + Math.abs(worldScale.z)) / 3;

  return { matrixWorld, worldQuaternion, uniformScale };
}

function toWorldCenter(center: THREE.Vector3, matrixWorld: THREE.Matrix4): THREE.Vector3 {
  return new THREE.Vector3().copy(center).applyMatrix4(matrixWorld);
}

function toWorldScales(scales: THREE.Vector3, uniformScale: number): THREE.Vector3 {
  const scale = Number.isFinite(uniformScale) && uniformScale > 0 ? uniformScale : 1;
  return new THREE.Vector3().copy(scales).multiplyScalar(scale);
}

function toWorldQuaternion(
  localQuaternion: THREE.Quaternion,
  meshWorldQuaternion: THREE.Quaternion
): THREE.Quaternion {
  return new THREE.Quaternion()
    .copy(meshWorldQuaternion)
    .multiply(localQuaternion)
    .normalize();
}
