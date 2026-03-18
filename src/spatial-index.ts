import type { SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";
import type { SpatialGrid, VoxelCell } from "./types";

export interface SpatialIndexOptions {
  resolution: [number, number, number];
  cropYFraction: [number, number];
  logPrefix: string;
}

export interface CroppedBoundsResult {
  bounds: THREE.Box3;
  usedFallback: boolean;
}

type VoxelAccumulator = {
  gridPos: [number, number, number];
  count: number;
  sumPos: THREE.Vector3;
  sumColor: THREE.Vector3;
  sumColorSq: THREE.Vector3;
  min: THREE.Vector3;
  max: THREE.Vector3;
  splatIndices: number[];
};

const DEFAULT_OPTIONS: SpatialIndexOptions = {
  resolution: [20, 20, 20],
  cropYFraction: [0.1, 0.1],
  logPrefix: "[spatial]",
};

// Cached world-space positions and colors from the last buildSpatialGrid call.
// Populated once at startup, used by buildLocalGrid for fine-grained re-binning.
let cachedWorldPositions: Float32Array | null = null;
let cachedWorldColors: Float32Array | null = null;
let cachedQuaternions: Float32Array | null = null;   // x,y,z,w per splat (4 floats)
let cachedScales: Float32Array | null = null;         // x,y,z per splat (3 floats)
let cachedOpacities: Float32Array | null = null;      // 1 float per splat
let cachedSplatCount = 0;

export function gridKey(x: number, y: number, z: number): string {
  return `${x},${y},${z}`;
}

export interface CachedSplatData {
  positions: Float32Array;
  colors: Float32Array;
  quaternions: Float32Array;
  scales: Float32Array;
  opacities: Float32Array;
  count: number;
}

export function getCachedSplatData(): CachedSplatData | null {
  if (!cachedWorldPositions || !cachedWorldColors || !cachedQuaternions || !cachedScales || !cachedOpacities || cachedSplatCount === 0) {
    return null;
  }
  return {
    positions: cachedWorldPositions,
    colors: cachedWorldColors,
    quaternions: cachedQuaternions,
    scales: cachedScales,
    opacities: cachedOpacities,
    count: cachedSplatCount,
  };
}

export function computeNominalCellSize(
  bounds: THREE.Box3,
  resolution: [number, number, number]
): THREE.Vector3 {
  const size = new THREE.Vector3();
  bounds.getSize(size);

  return new THREE.Vector3(
    safeDiv(size.x, resolution[0]),
    safeDiv(size.y, resolution[1]),
    safeDiv(size.z, resolution[2])
  );
}

export function computeCroppedBounds(
  rawBounds: THREE.Box3,
  cropFractions: [number, number] = [0.1, 0.1]
): CroppedBoundsResult {
  const bounds = rawBounds.clone();
  const size = new THREE.Vector3();
  bounds.getSize(size);

  const height = size.y;
  if (!Number.isFinite(height) || height <= 0) {
    return { bounds, usedFallback: true };
  }

  const [cropBottom, cropTop] = cropFractions;
  const minY = rawBounds.min.y + height * cropBottom;
  const maxY = rawBounds.max.y - height * cropTop;

  if (!(maxY > minY)) {
    return { bounds, usedFallback: true };
  }

  bounds.min.y = minY;
  bounds.max.y = maxY;
  return { bounds, usedFallback: false };
}

export function transformBoundsToWorld(
  localBounds: THREE.Box3,
  matrixWorld: THREE.Matrix4
): THREE.Box3 {
  const worldBounds = new THREE.Box3();
  const corners: THREE.Vector3[] = [
    new THREE.Vector3(localBounds.min.x, localBounds.min.y, localBounds.min.z),
    new THREE.Vector3(localBounds.min.x, localBounds.min.y, localBounds.max.z),
    new THREE.Vector3(localBounds.min.x, localBounds.max.y, localBounds.min.z),
    new THREE.Vector3(localBounds.min.x, localBounds.max.y, localBounds.max.z),
    new THREE.Vector3(localBounds.max.x, localBounds.min.y, localBounds.min.z),
    new THREE.Vector3(localBounds.max.x, localBounds.min.y, localBounds.max.z),
    new THREE.Vector3(localBounds.max.x, localBounds.max.y, localBounds.min.z),
    new THREE.Vector3(localBounds.max.x, localBounds.max.y, localBounds.max.z),
  ];

  for (const corner of corners) {
    corner.applyMatrix4(matrixWorld);
    worldBounds.expandByPoint(corner);
  }

  return worldBounds;
}

export function worldPosToGridCoord(
  worldPos: THREE.Vector3,
  gridBounds: THREE.Box3,
  resolution: [number, number, number],
  clamp: boolean = false
): [number, number, number] | null {
  if (!gridBounds.containsPoint(worldPos)) {
    if (!clamp) return null;
  }

  const size = new THREE.Vector3();
  gridBounds.getSize(size);

  const nx = normalizeAxis(worldPos.x, gridBounds.min.x, size.x);
  const ny = normalizeAxis(worldPos.y, gridBounds.min.y, size.y);
  const nz = normalizeAxis(worldPos.z, gridBounds.min.z, size.z);

  return [
    normalizedToIndex(nx, resolution[0]),
    normalizedToIndex(ny, resolution[1]),
    normalizedToIndex(nz, resolution[2]),
  ];
}

export function buildSpatialGrid(
  splatMesh: SplatMesh,
  options: Partial<SpatialIndexOptions> = {}
): SpatialGrid {
  const config: SpatialIndexOptions = {
    ...DEFAULT_OPTIONS,
    ...options,
  };
  const [rx, ry, rz] = config.resolution;
  const logPrefix = config.logPrefix;
  if (typeof splatMesh.updateMatrixWorld === "function") {
    splatMesh.updateMatrixWorld(true);
  }
  const matrixWorld =
    (splatMesh as unknown as { matrixWorld?: THREE.Matrix4 }).matrixWorld?.clone() ??
    new THREE.Matrix4().identity();
  const localBounds = splatMesh.getBoundingBox();
  const rawBounds = transformBoundsToWorld(localBounds, matrixWorld);

  if (isDegenerateBounds(rawBounds)) {
    console.warn(`${logPrefix} Degenerate bounding box; returning empty grid`);
    return {
      resolution: config.resolution,
      worldBounds: rawBounds.clone(),
      cellSize: new THREE.Vector3(0, 0, 0),
      cells: new Map<string, VoxelCell>(),
    };
  }

  const { bounds: croppedBounds, usedFallback } = computeCroppedBounds(
    rawBounds,
    config.cropYFraction
  );
  if (usedFallback) {
    console.warn(`${logPrefix} Y-crop fallback to raw bounds (scene too small/degenerate)`);
  }
  console.log(
    `${logPrefix} Grid worldBounds: min=${croppedBounds.min.toArray()} max=${croppedBounds.max.toArray()}`
  );

  const cellSize = computeNominalCellSize(croppedBounds, config.resolution);
  const nominalCellVolume = Math.max(cellSize.x * cellSize.y * cellSize.z, 1e-9);

  const accumulators = new Map<string, VoxelAccumulator>();
  let totalSplats = 0;
  let indexedSplats = 0;
  const start = nowMs();
  const worldCenter = new THREE.Vector3();

  // First pass: count splats to allocate cache arrays
  let splatCountEstimate = 0;
  splatMesh.forEachSplat(() => { splatCountEstimate += 1; });

  // Allocate cache arrays for local grid re-binning and infill
  const positions = new Float32Array(splatCountEstimate * 3);
  const colors = new Float32Array(splatCountEstimate * 3);
  const quaternions = new Float32Array(splatCountEstimate * 4);
  const scales = new Float32Array(splatCountEstimate * 3);
  const opacities = new Float32Array(splatCountEstimate);

  splatMesh.forEachSplat((index, center, splatScales, quat, opacity, color) => {
    totalSplats += 1;
    worldCenter.copy(center).applyMatrix4(matrixWorld);

    // Cache world position and color for later local grid building
    const i3 = index * 3;
    positions[i3] = worldCenter.x;
    positions[i3 + 1] = worldCenter.y;
    positions[i3 + 2] = worldCenter.z;
    colors[i3] = color.r;
    colors[i3 + 1] = color.g;
    colors[i3 + 2] = color.b;

    // Cache quaternion, scale, opacity for infill
    const i4 = index * 4;
    quaternions[i4] = quat.x;
    quaternions[i4 + 1] = quat.y;
    quaternions[i4 + 2] = quat.z;
    quaternions[i4 + 3] = quat.w;
    scales[i3] = splatScales.x;
    scales[i3 + 1] = splatScales.y;
    scales[i3 + 2] = splatScales.z;
    opacities[index] = opacity;

    const coord = worldPosToGridCoord(worldCenter, croppedBounds, config.resolution);
    if (!coord) {
      return;
    }

    const [gx, gy, gz] = coord;
    const key = gridKey(gx, gy, gz);
    let acc = accumulators.get(key);
    if (!acc) {
      acc = {
        gridPos: [gx, gy, gz],
        count: 0,
        sumPos: new THREE.Vector3(),
        sumColor: new THREE.Vector3(),
        sumColorSq: new THREE.Vector3(),
        min: new THREE.Vector3(worldCenter.x, worldCenter.y, worldCenter.z),
        max: new THREE.Vector3(worldCenter.x, worldCenter.y, worldCenter.z),
        splatIndices: [],
      };
      accumulators.set(key, acc);
    }

    indexedSplats += 1;
    acc.count += 1;
    acc.sumPos.add(worldCenter);
    acc.sumColor.x += color.r;
    acc.sumColor.y += color.g;
    acc.sumColor.z += color.b;
    acc.sumColorSq.x += color.r * color.r;
    acc.sumColorSq.y += color.g * color.g;
    acc.sumColorSq.z += color.b * color.b;
    acc.min.min(worldCenter);
    acc.max.max(worldCenter);
    acc.splatIndices.push(index);
  });

  // Store cached data for buildLocalGrid and infill
  cachedWorldPositions = positions;
  cachedWorldColors = colors;
  cachedQuaternions = quaternions;
  cachedScales = scales;
  cachedOpacities = opacities;
  cachedSplatCount = totalSplats;
  const totalBytes = positions.byteLength + colors.byteLength + quaternions.byteLength + scales.byteLength + opacities.byteLength;
  console.log(`${logPrefix} Cached ${totalSplats} splat data (${(totalBytes / 1024 / 1024).toFixed(1)}MB)`);

  const cells = new Map<string, VoxelCell>();
  for (const [key, acc] of accumulators) {
    cells.set(key, finalizeVoxelCell(acc, nominalCellVolume));
  }
  const firstCell = cells.values().next().value as VoxelCell | undefined;
  if (firstCell) {
    console.log(`${logPrefix} Sample cell center: ${firstCell.worldCenter.toArray()}`);
  }

  const elapsedMs = nowMs() - start;
  console.log(
    `${logPrefix} Built grid ${rx}x${ry}x${rz}: occupied=${cells.size}, indexedSplats=${indexedSplats}/${totalSplats}, elapsed=${elapsedMs.toFixed(1)}ms`
  );

  return {
    resolution: config.resolution,
    worldBounds: croppedBounds,
    cellSize,
    cells,
  };
}

export function getCellAtWorldPos(
  grid: SpatialGrid,
  pos: THREE.Vector3
): VoxelCell | null {
  // Try exact match first
  const coord = worldPosToGridCoord(pos, grid.worldBounds, grid.resolution);
  if (coord) {
    const cell = grid.cells.get(gridKey(coord[0], coord[1], coord[2]));
    if (cell) return cell;
  }

  // If outside bounds or cell empty, clamp to nearest occupied cell
  const clamped = worldPosToGridCoord(pos, grid.worldBounds, grid.resolution, true);
  if (!clamped) return null;

  const cell = grid.cells.get(gridKey(clamped[0], clamped[1], clamped[2]));
  if (cell) return cell;

  // Clamped cell is unoccupied — search expanding rings for nearest by world distance.
  // Don't early-exit on first ring hit: a diagonal cell at r=1 can be farther
  // than an axial cell at r=2, so keep expanding until the ring's minimum
  // possible distance exceeds our best candidate.
  const [cx, cy, cz] = clamped;
  const [rx, ry, rz] = grid.resolution;
  let nearest: VoxelCell | null = null;
  let bestDist = Infinity;

  for (let r = 1; r <= 5; r++) {
    // Minimum possible world distance for any cell in this ring
    const minRingDist = Math.min(grid.cellSize.x, grid.cellSize.y, grid.cellSize.z) * (r - 1);
    if (minRingDist * minRingDist > bestDist) break; // can't beat current best

    for (let x = Math.max(0, cx - r); x <= Math.min(rx - 1, cx + r); x++) {
      for (let y = Math.max(0, cy - r); y <= Math.min(ry - 1, cy + r); y++) {
        for (let z = Math.max(0, cz - r); z <= Math.min(rz - 1, cz + r); z++) {
          const neighbor = grid.cells.get(gridKey(x, y, z));
          if (neighbor) {
            const dist = neighbor.worldCenter.distanceToSquared(pos);
            if (dist < bestDist) {
              bestDist = dist;
              nearest = neighbor;
            }
          }
        }
      }
    }
  }

  return nearest;
}

export function getNeighborCells(
  grid: SpatialGrid,
  cell: VoxelCell,
  radius: number = 1
): VoxelCell[] {
  const out: VoxelCell[] = [];
  const [cx, cy, cz] = cell.gridPos;
  const [rx, ry, rz] = grid.resolution;
  const r = Math.max(0, Math.floor(radius));

  for (let x = Math.max(0, cx - r); x <= Math.min(rx - 1, cx + r); x += 1) {
    for (let y = Math.max(0, cy - r); y <= Math.min(ry - 1, cy + r); y += 1) {
      for (let z = Math.max(0, cz - r); z <= Math.min(rz - 1, cz + r); z += 1) {
        const neighbor = grid.cells.get(gridKey(x, y, z));
        if (neighbor) {
          out.push(neighbor);
        }
      }
    }
  }

  return out;
}

export function serializeSpatialGridForLLM(
  grid: SpatialGrid,
  options: { maxCells?: number; minSplats?: number } = {}
): string {
  const minSplats = options.minSplats ?? 10;
  const maxCells = options.maxCells ?? 600;
  let cells = Array.from(grid.cells.entries()).filter(
    ([, cell]) => cell.splatCount >= minSplats
  );

  if (cells.length > maxCells) {
    cells = cells
      .sort((a, b) => b[1].splatCount - a[1].splatCount)
      .slice(0, maxCells);
    console.log(`[spatial] serializeSpatialGridForLLM truncated to ${cells.length} cells (from ${grid.cells.size})`);
  }

  const r2 = (n: number) => Math.round(n * 100) / 100;

  const payload = {
    resolution: grid.resolution,
    worldBounds: {
      min: grid.worldBounds.min.toArray().map(r2),
      max: grid.worldBounds.max.toArray().map(r2),
    },
    cellSize: grid.cellSize.toArray().map(r2),
    cells: cells.map(([, cell]) => {
      const bSize = new THREE.Vector3();
      cell.worldBounds.getSize(bSize);
      return {
        g: cell.gridPos,
        c: cell.worldCenter.toArray().map(r2),
        d: [r2(bSize.x), r2(bSize.y), r2(bSize.z)],
        n: cell.splatCount,
        col: "#" + cell.avgColor.getHexString(),
        cv: r2(cell.colorVariance),
        den: r2(cell.density),
      };
    }),
  };

  return JSON.stringify(payload);
}

export function buildLocalGrid(
  globalGrid: SpatialGrid,
  center: THREE.Vector3,
  extent: THREE.Vector3,
  resolution: [number, number, number] = [10, 10, 10]
): SpatialGrid | null {
  if (!cachedWorldPositions || !cachedWorldColors || cachedSplatCount === 0) {
    console.warn("[spatial] buildLocalGrid: no cached splat data available");
    return null;
  }

  const localBounds = new THREE.Box3(
    new THREE.Vector3(center.x - extent.x, center.y - extent.y, center.z - extent.z),
    new THREE.Vector3(center.x + extent.x, center.y + extent.y, center.z + extent.z)
  );

  // Collect splat indices from coarse cells that overlap the local bounds
  const candidateIndices: number[] = [];
  for (const cell of globalGrid.cells.values()) {
    if (localBounds.intersectsBox(cell.worldBounds)) {
      for (const idx of cell.splatIndices) {
        candidateIndices.push(idx);
      }
    }
  }

  if (candidateIndices.length === 0) {
    console.log("[spatial] buildLocalGrid: no splats in local region");
    return null;
  }

  const [rx, ry, rz] = resolution;
  const cellSize = computeNominalCellSize(localBounds, resolution);
  const nominalCellVolume = Math.max(cellSize.x * cellSize.y * cellSize.z, 1e-9);
  const accumulators = new Map<string, VoxelAccumulator>();
  const pos = cachedWorldPositions;
  const col = cachedWorldColors;
  const worldPos = new THREE.Vector3();
  let binned = 0;

  for (const idx of candidateIndices) {
    const i3 = idx * 3;
    if (i3 + 2 >= pos.length) continue;

    worldPos.set(pos[i3], pos[i3 + 1], pos[i3 + 2]);
    if (!localBounds.containsPoint(worldPos)) continue;

    const coord = worldPosToGridCoord(worldPos, localBounds, resolution);
    if (!coord) continue;

    const [gx, gy, gz] = coord;
    const key = gridKey(gx, gy, gz);
    let acc = accumulators.get(key);
    if (!acc) {
      acc = {
        gridPos: [gx, gy, gz],
        count: 0,
        sumPos: new THREE.Vector3(),
        sumColor: new THREE.Vector3(),
        sumColorSq: new THREE.Vector3(),
        min: new THREE.Vector3(worldPos.x, worldPos.y, worldPos.z),
        max: new THREE.Vector3(worldPos.x, worldPos.y, worldPos.z),
        splatIndices: [],
      };
      accumulators.set(key, acc);
    }

    const r = col[i3], g = col[i3 + 1], b = col[i3 + 2];
    acc.count += 1;
    acc.sumPos.add(worldPos);
    acc.sumColor.x += r;
    acc.sumColor.y += g;
    acc.sumColor.z += b;
    acc.sumColorSq.x += r * r;
    acc.sumColorSq.y += g * g;
    acc.sumColorSq.z += b * b;
    acc.min.min(worldPos);
    acc.max.max(worldPos);
    acc.splatIndices.push(idx);
    binned += 1;
  }

  const cells = new Map<string, VoxelCell>();
  for (const [key, acc] of accumulators) {
    cells.set(key, finalizeVoxelCell(acc, nominalCellVolume));
  }

  console.log(
    `[spatial] Built local grid ${rx}x${ry}x${rz}: occupied=${cells.size} binnedSplats=${binned} candidates=${candidateIndices.length}`
  );

  return {
    resolution,
    worldBounds: localBounds,
    cellSize,
    cells,
  };
}

export function serializeLocalGridForLLM(
  grid: SpatialGrid,
  options: { maxCells?: number; minSplats?: number } = {}
): string {
  return serializeSpatialGridForLLM(grid, {
    maxCells: options.maxCells ?? 200,
    minSplats: options.minSplats ?? 3,
  });
}

function finalizeVoxelCell(acc: VoxelAccumulator, nominalCellVolume: number): VoxelCell {
  const invCount = 1 / acc.count;
  const center = acc.sumPos.clone().multiplyScalar(invCount);
  const avgR = acc.sumColor.x * invCount;
  const avgG = acc.sumColor.y * invCount;
  const avgB = acc.sumColor.z * invCount;
  const varianceR = Math.max(0, acc.sumColorSq.x * invCount - avgR * avgR);
  const varianceG = Math.max(0, acc.sumColorSq.y * invCount - avgG * avgG);
  const varianceB = Math.max(0, acc.sumColorSq.z * invCount - avgB * avgB);
  const colorVariance = (varianceR + varianceG + varianceB) / 3;

  return {
    gridPos: acc.gridPos,
    worldCenter: center,
    worldBounds: new THREE.Box3(acc.min.clone(), acc.max.clone()),
    splatCount: acc.count,
    avgColor: new THREE.Color(avgR, avgG, avgB),
    colorVariance,
    density: acc.count / nominalCellVolume,
    splatIndices: acc.splatIndices,
  };
}

function isDegenerateBounds(bounds: THREE.Box3): boolean {
  const size = new THREE.Vector3();
  bounds.getSize(size);
  return (
    !Number.isFinite(bounds.min.x) ||
    !Number.isFinite(bounds.min.y) ||
    !Number.isFinite(bounds.min.z) ||
    !Number.isFinite(bounds.max.x) ||
    !Number.isFinite(bounds.max.y) ||
    !Number.isFinite(bounds.max.z) ||
    size.x <= 0 ||
    size.y <= 0 ||
    size.z <= 0
  );
}

function normalizeAxis(value: number, min: number, size: number): number {
  if (!Number.isFinite(size) || size <= 0) {
    return 0;
  }
  return THREE.MathUtils.clamp((value - min) / size, 0, 1);
}

function normalizedToIndex(normalized: number, resolution: number): number {
  if (resolution <= 1) {
    return 0;
  }
  if (normalized >= 1) {
    return resolution - 1;
  }
  return THREE.MathUtils.clamp(Math.floor(normalized * resolution), 0, resolution - 1);
}

function safeDiv(n: number, d: number): number {
  return d > 0 ? n / d : 0;
}

function nowMs(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}
