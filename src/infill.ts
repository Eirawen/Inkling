import { PackedSplats, SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";
import type { CachedSplatData } from "./spatial-index";
import { getCachedSplatData, gridKey, worldPosToGridCoord } from "./spatial-index";
import type { InfillConfig, SDFShapeConfig, SpatialGrid } from "./types";

// ─── Internal Types ────────────────────────────────────────────────

interface BoundarySplat {
  index: number;
  position: THREE.Vector3;
  normal: THREE.Vector3;
  color: THREE.Color;
  scale: THREE.Vector3;
  opacity: number;
  sdfDist: number; // distance from deletion surface (higher = more likely background)
}

interface SurfaceCluster {
  splats: BoundarySplat[];
  avgNormal: THREE.Vector3;
  avgColor: THREE.Color;
  avgDensity: number;
  bounds: THREE.Box3;
}

interface FittedPlane {
  normal: THREE.Vector3;
  d: number;
  inlierRatio: number;
}

interface GeneratedSplat {
  position: THREE.Vector3;
  color: THREE.Color;
  scale: THREE.Vector3;
  quaternion: THREE.Quaternion;
  opacity: number;
}

export interface InfillResult {
  mesh: SplatMesh;
  splatCount: number;
  surfaceCount: number;
  fillBounds: THREE.Box3;
}

// ─── Phase 2: SDF Evaluation ──────────────────────────────────────

const _localPt = new THREE.Vector3();
const _invQuat = new THREE.Quaternion();

function rotatePointToLocal(
  point: THREE.Vector3,
  center: [number, number, number],
  rotation: [number, number, number, number] | undefined,
  out: THREE.Vector3
): THREE.Vector3 {
  out.set(point.x - center[0], point.y - center[1], point.z - center[2]);
  if (rotation) {
    _invQuat.set(rotation[0], rotation[1], rotation[2], rotation[3]).invert();
    out.applyQuaternion(_invQuat);
  }
  return out;
}

function sdfSphere(
  point: THREE.Vector3,
  shape: SDFShapeConfig
): number {
  const r = shape.radius ?? 1;
  const dx = point.x - shape.position[0];
  const dy = point.y - shape.position[1];
  const dz = point.z - shape.position[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz) - r;
}

function sdfBox(
  point: THREE.Vector3,
  shape: SDFShapeConfig
): number {
  const s = shape.scale ?? [1, 1, 1];
  rotatePointToLocal(point, shape.position, shape.rotation, _localPt);
  const qx = Math.abs(_localPt.x) - s[0];
  const qy = Math.abs(_localPt.y) - s[1];
  const qz = Math.abs(_localPt.z) - s[2];
  const outside = Math.sqrt(
    Math.max(qx, 0) ** 2 + Math.max(qy, 0) ** 2 + Math.max(qz, 0) ** 2
  );
  const inside = Math.min(Math.max(qx, qy, qz), 0);
  return outside + inside;
}

function sdfEllipsoid(
  point: THREE.Vector3,
  shape: SDFShapeConfig
): number {
  const s = shape.scale ?? [1, 1, 1];
  rotatePointToLocal(point, shape.position, shape.rotation, _localPt);
  // Normalize by scale
  const nx = _localPt.x / s[0];
  const ny = _localPt.y / s[1];
  const nz = _localPt.z / s[2];
  const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
  const minScale = Math.min(s[0], s[1], s[2]);
  return (len - 1) * minScale;
}

function sdfCylinder(
  point: THREE.Vector3,
  shape: SDFShapeConfig
): number {
  const r = shape.radius ?? 1;
  const s = shape.scale ?? [1, 1, 1];
  const halfH = s[1];
  rotatePointToLocal(point, shape.position, shape.rotation, _localPt);
  const radialDist = Math.sqrt(_localPt.x ** 2 + _localPt.z ** 2) - r;
  const verticalDist = Math.abs(_localPt.y) - halfH;
  const outside = Math.sqrt(Math.max(radialDist, 0) ** 2 + Math.max(verticalDist, 0) ** 2);
  const inside = Math.min(Math.max(radialDist, verticalDist), 0);
  return outside + inside;
}

function sdfCapsule(
  point: THREE.Vector3,
  shape: SDFShapeConfig
): number {
  const r = shape.radius ?? 1;
  const s = shape.scale ?? [1, 1, 1];
  const halfH = s[1];
  rotatePointToLocal(point, shape.position, shape.rotation, _localPt);
  _localPt.y -= THREE.MathUtils.clamp(_localPt.y, -halfH, halfH);
  return _localPt.length() - r;
}

function sdfPlane(
  point: THREE.Vector3,
  shape: SDFShapeConfig
): number {
  // Plane normal is the local Y axis rotated by the shape's quaternion
  const normal = new THREE.Vector3(0, 1, 0);
  if (shape.rotation) {
    normal.applyQuaternion(
      new THREE.Quaternion(shape.rotation[0], shape.rotation[1], shape.rotation[2], shape.rotation[3])
    );
  }
  return (
    (point.x - shape.position[0]) * normal.x +
    (point.y - shape.position[1]) * normal.y +
    (point.z - shape.position[2]) * normal.z
  );
}

function sdfShape(point: THREE.Vector3, shape: SDFShapeConfig): number {
  switch (shape.type) {
    case "SPHERE": return sdfSphere(point, shape);
    case "BOX": return sdfBox(point, shape);
    case "ELLIPSOID": return sdfEllipsoid(point, shape);
    case "CYLINDER": return sdfCylinder(point, shape);
    case "CAPSULE": return sdfCapsule(point, shape);
    case "PLANE": return sdfPlane(point, shape);
    case "ALL": return -Infinity;
    case "INFINITE_CONE": return sdfSphere(point, shape); // approximate
    default: return Infinity;
  }
}

function evaluateSDF(
  point: THREE.Vector3,
  shapes: SDFShapeConfig[],
  smooth?: number
): number {
  if (shapes.length === 0) return Infinity;
  if (shapes.length === 1) return sdfShape(point, shapes[0]);

  let d = sdfShape(point, shapes[0]);
  const k = smooth ?? 0;

  for (let i = 1; i < shapes.length; i++) {
    const di = sdfShape(point, shapes[i]);
    if (k > 0) {
      // Smooth union (polynomial smooth min)
      const h = Math.max(k - Math.abs(d - di), 0) / k;
      d = Math.min(d, di) - h * h * k * 0.25;
    } else {
      d = Math.min(d, di);
    }
  }

  return d;
}

// ─── Phase 3: Boundary Identification & Surface Segmentation ──────

const MAX_BOUNDARY_SPLATS = 2000;

function computeAdaptiveBoundaryWidth(
  shapes: SDFShapeConfig[],
  softEdge?: number
): number {
  // Derive boundary width from the smallest shape dimension
  let minDim = Infinity;
  for (const shape of shapes) {
    if (shape.scale) {
      minDim = Math.min(minDim, shape.scale[0], shape.scale[1], shape.scale[2]);
    }
    if (shape.radius !== undefined) {
      minDim = Math.min(minDim, shape.radius);
    }
  }
  if (!Number.isFinite(minDim)) minDim = 0.5;

  // Boundary shell = 20% of smallest shape dimension, clamped
  const fromShape = minDim * 0.2;
  const fromEdge = (softEdge ?? 0.1) * 1.2;
  return Math.max(0.02, Math.min(fromShape, fromEdge, 0.15));
}

function subsampleBoundary(splats: BoundarySplat[], maxCount: number): BoundarySplat[] {
  if (splats.length <= maxCount) return splats;
  // Deterministic stride-based subsampling
  const stride = splats.length / maxCount;
  const result: BoundarySplat[] = [];
  for (let i = 0; i < maxCount; i++) {
    result.push(splats[Math.floor(i * stride)]);
  }
  return result;
}

function identifyBoundarySplats(
  shapes: SDFShapeConfig[],
  splatData: CachedSplatData,
  grid: SpatialGrid,
  options: { softEdge?: number; sdfSmooth?: number; boundaryWidth?: number }
): BoundarySplat[] {
  const boundaryWidth = options.boundaryWidth ?? computeAdaptiveBoundaryWidth(shapes, options.softEdge);
  const smooth = options.sdfSmooth;
  const boundary: BoundarySplat[] = [];

  // Compute SDF bounding box expanded by boundaryWidth to find candidate cells
  const sdfBounds = new THREE.Box3();
  for (const shape of shapes) {
    const pos = new THREE.Vector3(shape.position[0], shape.position[1], shape.position[2]);
    const extent = shape.radius ?? 1;
    const scaleMax = shape.scale ? Math.max(shape.scale[0], shape.scale[1], shape.scale[2]) : extent;
    const r = Math.max(extent, scaleMax) + boundaryWidth;
    sdfBounds.expandByPoint(pos.clone().addScalar(r));
    sdfBounds.expandByPoint(pos.clone().subScalar(r));
  }

  // Collect candidate splat indices from cells overlapping the expanded SDF bounds
  const candidateIndices = new Set<number>();
  for (const cell of grid.cells.values()) {
    const expandedCellBounds = cell.worldBounds.clone().expandByScalar(boundaryWidth);
    if (sdfBounds.intersectsBox(expandedCellBounds)) {
      for (const idx of cell.splatIndices) {
        candidateIndices.add(idx);
      }
    }
  }

  const pos = splatData.positions;
  const col = splatData.colors;
  const quat = splatData.quaternions;
  const scl = splatData.scales;
  const opa = splatData.opacities;
  const testPoint = new THREE.Vector3();

  for (const idx of candidateIndices) {
    const i3 = idx * 3;
    const i4 = idx * 4;
    testPoint.set(pos[i3], pos[i3 + 1], pos[i3 + 2]);

    const dist = evaluateSDF(testPoint, shapes, smooth);
    // Boundary: outside SDF but close to the surface
    if (dist > 0 && dist < boundaryWidth) {
      // Extract normal from quaternion (smallest scale axis = surface normal)
      const sx = scl[i3], sy = scl[i3 + 1], sz = scl[i3 + 2];
      const normal = extractNormalFromQuaternion(
        quat[i4], quat[i4 + 1], quat[i4 + 2], quat[i4 + 3],
        sx, sy, sz
      );

      boundary.push({
        index: idx,
        position: new THREE.Vector3(pos[i3], pos[i3 + 1], pos[i3 + 2]),
        normal,
        color: new THREE.Color(col[i3], col[i3 + 1], col[i3 + 2]),
        scale: new THREE.Vector3(sx, sy, sz),
        opacity: opa[idx],
        sdfDist: dist,
      });
    }
  }

  return boundary;
}

function extractNormalFromQuaternion(
  qx: number, qy: number, qz: number, qw: number,
  sx: number, sy: number, sz: number
): THREE.Vector3 {
  // The surface normal is the direction of the smallest scale axis
  let axis: THREE.Vector3;
  const absSx = Math.abs(sx), absSy = Math.abs(sy), absSz = Math.abs(sz);

  if (absSx <= absSy && absSx <= absSz) {
    axis = new THREE.Vector3(1, 0, 0);
  } else if (absSy <= absSx && absSy <= absSz) {
    axis = new THREE.Vector3(0, 1, 0);
  } else {
    axis = new THREE.Vector3(0, 0, 1);
  }

  // Rotate axis by quaternion
  const q = new THREE.Quaternion(qx, qy, qz, qw);
  axis.applyQuaternion(q);
  return axis.normalize();
}

function clusterBySurfaceNormal(
  splats: BoundarySplat[],
  maxClusters: number = 4
): SurfaceCluster[] {
  if (splats.length === 0) return [];
  if (splats.length < 3) {
    return [buildCluster(splats)];
  }

  // Try k=2 through maxClusters, pick best via elbow method
  let bestClusters: SurfaceCluster[] = [buildCluster(splats)];
  let bestScore = Infinity;

  for (let k = 2; k <= Math.min(maxClusters, splats.length); k++) {
    const assignments = kmeansNormals(splats, k, 20);
    const clusters: SurfaceCluster[] = [];
    let totalVariance = 0;

    for (let c = 0; c < k; c++) {
      const members = splats.filter((_, i) => assignments[i] === c);
      if (members.length === 0) continue;
      const cluster = buildCluster(members);
      clusters.push(cluster);

      // Within-cluster angular variance
      for (const s of members) {
        const dot = Math.abs(s.normal.dot(cluster.avgNormal));
        totalVariance += Math.acos(Math.min(dot, 1)) ** 2;
      }
    }

    // Elbow: penalize more clusters
    const score = totalVariance / splats.length + k * 0.1;
    if (score < bestScore) {
      bestScore = score;
      bestClusters = clusters;
    }
  }

  return bestClusters;
}

function kmeansNormals(
  splats: BoundarySplat[],
  k: number,
  maxIter: number
): Int32Array {
  const n = splats.length;
  const assignments = new Int32Array(n);
  const centroids: THREE.Vector3[] = [];

  // Initialize centroids: pick k evenly spaced splats
  for (let i = 0; i < k; i++) {
    const idx = Math.floor(i * n / k);
    centroids.push(splats[idx].normal.clone());
  }

  for (let iter = 0; iter < maxIter; iter++) {
    let changed = false;

    // Assign
    for (let i = 0; i < n; i++) {
      let bestDist = -Infinity;
      let bestC = 0;
      for (let c = 0; c < k; c++) {
        // Angular similarity: abs dot product (handles flipped normals)
        const dot = Math.abs(splats[i].normal.dot(centroids[c]));
        if (dot > bestDist) {
          bestDist = dot;
          bestC = c;
        }
      }
      if (assignments[i] !== bestC) {
        assignments[i] = bestC;
        changed = true;
      }
    }

    if (!changed) break;

    // Update centroids
    for (let c = 0; c < k; c++) {
      const sum = new THREE.Vector3();
      let count = 0;
      for (let i = 0; i < n; i++) {
        if (assignments[i] === c) {
          // Align normals to same hemisphere before averaging
          const dot = splats[i].normal.dot(centroids[c]);
          if (dot >= 0) {
            sum.add(splats[i].normal);
          } else {
            sum.sub(splats[i].normal);
          }
          count++;
        }
      }
      if (count > 0) {
        sum.divideScalar(count).normalize();
        if (sum.lengthSq() > 0.01) {
          centroids[c].copy(sum);
        }
      }
    }
  }

  return assignments;
}

function buildCluster(splats: BoundarySplat[]): SurfaceCluster {
  const avgNormal = new THREE.Vector3();
  const avgColor = new THREE.Color(0, 0, 0);
  const bounds = new THREE.Box3();
  let totalDensity = 0;

  // Use first normal as reference for hemisphere alignment
  const ref = splats[0].normal;
  for (const s of splats) {
    const dot = s.normal.dot(ref);
    if (dot >= 0) {
      avgNormal.add(s.normal);
    } else {
      avgNormal.sub(s.normal);
    }
    avgColor.r += s.color.r;
    avgColor.g += s.color.g;
    avgColor.b += s.color.b;
    bounds.expandByPoint(s.position);
    totalDensity += s.opacity;
  }

  const inv = 1 / splats.length;
  avgNormal.multiplyScalar(inv).normalize();
  avgColor.r *= inv;
  avgColor.g *= inv;
  avgColor.b *= inv;

  return {
    splats,
    avgNormal: avgNormal.lengthSq() > 0.001 ? avgNormal : new THREE.Vector3(0, 1, 0),
    avgColor,
    avgDensity: totalDensity * inv,
    bounds,
  };
}

// ─── Phase 4: RANSAC Plane Fitting ────────────────────────────────

function fitPlaneRANSAC(
  points: THREE.Vector3[],
  normals: THREE.Vector3[],
  clusterNormal: THREE.Vector3,
  iterations: number = 100,
  threshold: number = 0.05
): FittedPlane | null {
  if (points.length < 3) return null;

  let bestNormal = new THREE.Vector3();
  let bestD = 0;
  let bestInliers = 0;

  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  const candidate = new THREE.Vector3();

  for (let iter = 0; iter < iterations; iter++) {
    // Sample 3 random points
    const i0 = Math.floor(Math.random() * points.length);
    let i1 = Math.floor(Math.random() * points.length);
    let i2 = Math.floor(Math.random() * points.length);
    if (i1 === i0) i1 = (i0 + 1) % points.length;
    if (i2 === i0 || i2 === i1) i2 = (i0 + 2) % points.length;

    ab.subVectors(points[i1], points[i0]);
    ac.subVectors(points[i2], points[i0]);
    candidate.crossVectors(ab, ac);

    if (candidate.lengthSq() < 1e-12) continue;
    candidate.normalize();

    // Check normal consistency: reject if too far from cluster normal (> 45 deg)
    if (Math.abs(candidate.dot(clusterNormal)) < 0.707) continue;

    // Ensure consistent orientation with cluster normal
    if (candidate.dot(clusterNormal) < 0) candidate.negate();

    const d = candidate.dot(points[i0]);

    // Count inliers
    let inliers = 0;
    for (const p of points) {
      if (Math.abs(candidate.dot(p) - d) < threshold) {
        inliers++;
      }
    }

    if (inliers > bestInliers) {
      bestInliers = inliers;
      bestNormal.copy(candidate);
      bestD = d;
    }
  }

  if (bestInliers < 3) return null;

  return {
    normal: bestNormal.clone(),
    d: bestD,
    inlierRatio: bestInliers / points.length,
  };
}

// ─── Phase 5: Fill Position Sampling ──────────────────────────────

function estimateDensity(splats: BoundarySplat[]): number {
  if (splats.length < 2) return 100; // default

  // Average nearest-neighbor distance
  let totalDist = 0;
  const sampleSize = Math.min(splats.length, 50);
  for (let i = 0; i < sampleSize; i++) {
    let minDist = Infinity;
    for (let j = 0; j < splats.length; j++) {
      if (i === j) continue;
      const d = splats[i].position.distanceToSquared(splats[j].position);
      if (d < minDist) minDist = d;
    }
    totalDist += Math.sqrt(minDist);
  }

  const avgSpacing = totalDist / sampleSize;
  if (avgSpacing < 1e-6) return 10000;
  return 1 / (avgSpacing * avgSpacing);
}

function sampleFillPositions(
  plane: FittedPlane,
  cluster: SurfaceCluster,
  sdfShapes: SDFShapeConfig[],
  targetDensity: number,
  sdfSmooth?: number,
  maxFillSplats?: number
): THREE.Vector3[] {
  const maxSplats = maxFillSplats ?? 5000;
  const spacing = 1 / Math.sqrt(Math.max(targetDensity, 1));

  // Build two tangent vectors on the plane
  const n = plane.normal;
  const tangentU = new THREE.Vector3();
  if (Math.abs(n.y) < 0.9) {
    tangentU.crossVectors(n, new THREE.Vector3(0, 1, 0)).normalize();
  } else {
    tangentU.crossVectors(n, new THREE.Vector3(1, 0, 0)).normalize();
  }
  const tangentV = new THREE.Vector3().crossVectors(n, tangentU).normalize();

  // Project SDF bounds onto plane to determine fill extent
  const center = new THREE.Vector3();
  cluster.bounds.getCenter(center);

  // Project center onto plane
  const distToPlane = center.dot(n) - plane.d;
  const projCenter = center.clone().addScaledVector(n, -distToPlane);

  // Determine extent from cluster bounds
  const size = new THREE.Vector3();
  cluster.bounds.getSize(size);
  const extent = Math.max(size.x, size.y, size.z) * 0.75;

  const gridSteps = Math.ceil(extent / spacing);
  const capped = Math.min(gridSteps, Math.ceil(Math.sqrt(maxSplats)));
  const jitter = spacing * 0.6; // 60% jitter to break visible grid pattern
  const offPlaneJitter = spacing * 0.15; // slight depth variation
  const positions: THREE.Vector3[] = [];
  const testPoint = new THREE.Vector3();

  for (let u = -capped; u <= capped; u++) {
    for (let v = -capped; v <= capped; v++) {
      if (positions.length >= maxSplats) break;

      testPoint.copy(projCenter)
        .addScaledVector(tangentU, u * spacing + (Math.random() - 0.5) * jitter)
        .addScaledVector(tangentV, v * spacing + (Math.random() - 0.5) * jitter)
        .addScaledVector(n, (Math.random() - 0.5) * offPlaneJitter);

      // Must be inside the deletion SDF
      const sdfDist = evaluateSDF(testPoint, sdfShapes, sdfSmooth);
      if (sdfDist < 0) {
        positions.push(testPoint.clone());
      }
    }
    if (positions.length >= maxSplats) break;
  }

  return positions;
}

// ─── Phase 6: Property Assignment ─────────────────────────────────

function assignFillProperties(
  fillPositions: THREE.Vector3[],
  boundarySplats: BoundarySplat[],
  cluster: SurfaceCluster,
  plane: FittedPlane,
  sdfShapes: SDFShapeConfig[],
  sdfSmooth?: number
): GeneratedSplat[] {
  if (boundarySplats.length === 0 || fillPositions.length === 0) return [];

  const K = Math.min(6, boundarySplats.length);
  const result: GeneratedSplat[] = [];

  // Precompute average boundary scale and opacity
  const avgScale = new THREE.Vector3();
  let avgOpacity = 0;
  for (const s of boundarySplats) {
    avgScale.add(s.scale);
    avgOpacity += s.opacity;
  }
  avgScale.divideScalar(boundarySplats.length);
  avgOpacity /= boundarySplats.length;

  // Compute fill splat scale: ensure overlap by sizing relative to fill spacing.
  // The two axes parallel to the surface should be large enough that adjacent
  // splats visually overlap. The normal axis stays thin (flat disk on surface).
  const density = estimateDensity(boundarySplats);
  const fillSpacing = 1 / Math.sqrt(Math.max(density, 1));
  // Each splat needs to cover ~1.5x the spacing to guarantee overlap
  const coverageScale = fillSpacing * 1.5;
  const fillScale = new THREE.Vector3(
    Math.max(avgScale.x, coverageScale),
    Math.max(avgScale.y, coverageScale),
    Math.max(avgScale.z, coverageScale)
  );
  // Flatten along the plane normal (thin disk, not a ball)
  const nAbs = new THREE.Vector3(
    Math.abs(plane.normal.x),
    Math.abs(plane.normal.y),
    Math.abs(plane.normal.z)
  );
  // Shrink the axis most aligned with the plane normal
  if (nAbs.x >= nAbs.y && nAbs.x >= nAbs.z) {
    fillScale.x = Math.min(fillScale.x, avgScale.x);
  } else if (nAbs.y >= nAbs.x && nAbs.y >= nAbs.z) {
    fillScale.y = Math.min(fillScale.y, avgScale.y);
  } else {
    fillScale.z = Math.min(fillScale.z, avgScale.z);
  }

  // Compute max boundary distance for blend interpolation
  let maxBoundaryDist = 0;
  for (const pos of fillPositions) {
    const d = evaluateSDF(pos, sdfShapes, sdfSmooth);
    maxBoundaryDist = Math.max(maxBoundaryDist, Math.abs(d));
  }
  if (maxBoundaryDist < 1e-6) maxBoundaryDist = 1;

  // Build quaternion from plane normal (orient splat disk parallel to surface)
  const fillQuat = quaternionFromNormal(plane.normal);

  // Find max sdfDist in boundary for normalization
  let maxSdfDist = 0;
  for (const s of boundarySplats) {
    if (s.sdfDist > maxSdfDist) maxSdfDist = s.sdfDist;
  }
  if (maxSdfDist < 1e-6) maxSdfDist = 1;

  for (const pos of fillPositions) {
    // Find K nearest boundary splats, weighted by SDF distance
    // (prefer outer boundary splats — more likely background, less object contamination)
    const nearest = findKNearest(pos, boundarySplats, K);

    const blendColor = new THREE.Color(0, 0, 0);
    let totalWeight = 0;
    let blendOpacity = 0;

    for (const { splat, dist } of nearest) {
      // Spatial proximity weight
      const spatialW = 1 / (dist * dist + 1e-6);
      // SDF distance weight: splats further from deletion surface are more likely background
      const sdfW = splat.sdfDist / maxSdfDist;
      // Combined: heavily prefer outer boundary splats
      const w = spatialW * (0.1 + sdfW * sdfW);

      blendColor.r += splat.color.r * w;
      blendColor.g += splat.color.g * w;
      blendColor.b += splat.color.b * w;
      blendOpacity += splat.opacity * w;
      totalWeight += w;
    }

    if (totalWeight > 0) {
      blendColor.r /= totalWeight;
      blendColor.g /= totalWeight;
      blendColor.b /= totalWeight;
      blendOpacity /= totalWeight;
    } else {
      blendColor.copy(cluster.avgColor);
      blendOpacity = avgOpacity;
    }

    // Small color variation to break uniformity
    const variation = (Math.random() - 0.5) * 0.04;
    blendColor.r = Math.max(0, Math.min(1, blendColor.r + variation));
    blendColor.g = Math.max(0, Math.min(1, blendColor.g + variation));
    blendColor.b = Math.max(0, Math.min(1, blendColor.b + variation));

    result.push({
      position: pos.clone(),
      color: blendColor,
      scale: fillScale.clone(),
      quaternion: fillQuat.clone(),
      opacity: Math.min(blendOpacity, avgOpacity),
    });
  }

  return result;
}

function findKNearest(
  pos: THREE.Vector3,
  splats: BoundarySplat[],
  k: number
): Array<{ splat: BoundarySplat; dist: number }> {
  const scored = splats.map(s => ({
    splat: s,
    dist: pos.distanceTo(s.position),
  }));
  scored.sort((a, b) => a.dist - b.dist);
  return scored.slice(0, k);
}

function quaternionFromNormal(normal: THREE.Vector3): THREE.Quaternion {
  // Create a quaternion that rotates (0,0,1) to the given normal
  const q = new THREE.Quaternion();
  q.setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal);
  return q;
}

// ─── Phase 7: Public API ──────────────────────────────────────────

export function generateInfill(
  deletionShapes: SDFShapeConfig[],
  grid: SpatialGrid,
  options?: InfillConfig
): InfillResult | null {
  const startMs = typeof performance !== "undefined" ? performance.now() : Date.now();

  // Skip ALL-type shapes (global operations, no hole to fill)
  if (deletionShapes.some(s => s.type === "ALL")) {
    console.log("[infill] Skipping: ALL-type shape (no localized hole)");
    return null;
  }

  const splatData = getCachedSplatData();
  if (!splatData) {
    console.warn("[infill] No cached splat data available");
    return null;
  }

  const softEdge = options?.softEdge ?? 0.15;
  const sdfSmooth = options?.sdfSmooth;
  const boundaryWidth = options?.boundaryWidth ?? computeAdaptiveBoundaryWidth(deletionShapes, softEdge);
  const maxFillSplats = options?.maxFillSplats ?? 5000;
  const maxClusters = options?.maxClusters ?? 4;

  // Phase 3a: Identify boundary splats
  const rawBoundary = identifyBoundarySplats(
    deletionShapes, splatData, grid,
    { softEdge, sdfSmooth, boundaryWidth }
  );
  console.log(`[infill] Found ${rawBoundary.length} boundary splats (width=${boundaryWidth.toFixed(4)})`);

  if (rawBoundary.length < 5) {
    console.log("[infill] Too few boundary splats for infill");
    return null;
  }

  const boundarySplats = subsampleBoundary(rawBoundary, MAX_BOUNDARY_SPLATS);
  if (boundarySplats.length < rawBoundary.length) {
    console.log(`[infill] Subsampled boundary: ${rawBoundary.length} → ${boundarySplats.length}`);
  }

  // Phase 3b-c: Cluster by surface normal
  const clusters = clusterBySurfaceNormal(boundarySplats, maxClusters);
  console.log(`[infill] Segmented into ${clusters.length} surfaces`);

  // Phase 4: Fit planes + Phase 5-6: Sample & assign
  const allGeneratedSplats: GeneratedSplat[] = [];
  const inlierRatios: string[] = [];
  const cellSize = Math.max(grid.cellSize.x, grid.cellSize.y, grid.cellSize.z);
  const planeThreshold = cellSize * 0.5 || 0.05;

  // Find the largest cluster (dominant surface)
  const largestClusterSize = Math.max(...clusters.map(c => c.splats.length));
  const MIN_CLUSTER_FRACTION = 0.15; // skip clusters with <15% of the largest
  const MIN_INLIER_RATIO = 0.5;     // skip poorly-fitted planes

  for (const cluster of clusters) {
    if (cluster.splats.length < 3) continue;

    // Skip tiny clusters — likely noise from mixed normals
    if (cluster.splats.length < largestClusterSize * MIN_CLUSTER_FRACTION) {
      console.log(`[infill] Skipping small cluster (${cluster.splats.length}/${largestClusterSize} splats)`);
      inlierRatios.push("skipped:small");
      continue;
    }

    const points = cluster.splats.map(s => s.position);
    const normals = cluster.splats.map(s => s.normal);

    const plane = fitPlaneRANSAC(points, normals, cluster.avgNormal, 100, planeThreshold);
    if (!plane) {
      console.log(`[infill] Skipping cluster (RANSAC failed, ${cluster.splats.length} splats)`);
      inlierRatios.push("skipped:nofit");
      continue;
    }

    if (plane.inlierRatio < MIN_INLIER_RATIO) {
      console.log(`[infill] Skipping cluster (inlier ratio ${plane.inlierRatio.toFixed(2)} < ${MIN_INLIER_RATIO})`);
      inlierRatios.push(`skipped:${plane.inlierRatio.toFixed(2)}`);
      continue;
    }

    inlierRatios.push(plane.inlierRatio.toFixed(2));

    const density = estimateDensity(cluster.splats);
    const perClusterMax = Math.floor(maxFillSplats / clusters.length);
    const positions = sampleFillPositions(
      plane, cluster, deletionShapes, density, sdfSmooth, perClusterMax
    );
    const generated = assignFillProperties(
      positions, cluster.splats, cluster, plane, deletionShapes, sdfSmooth
    );
    allGeneratedSplats.push(...generated);
  }

  console.log(`[infill] Fitted ${clusters.length} planes (inlier ratios: ${inlierRatios.join(", ")})`);
  console.log(`[infill] Generated ${allGeneratedSplats.length} fill splats`);

  if (allGeneratedSplats.length < 10) {
    console.log("[infill] Too few fill splats generated (<10), skipping");
    return null;
  }

  // Cap total
  const capped = allGeneratedSplats.slice(0, maxFillSplats);

  // Phase 7: Create mesh
  const packedSplats = new PackedSplats({ maxSplats: capped.length });

  const fillBounds = new THREE.Box3();
  const tmpCenter = new THREE.Vector3();
  const tmpScales = new THREE.Vector3();
  const tmpQuat = new THREE.Quaternion();
  const tmpColor = new THREE.Color();

  for (const splat of capped) {
    tmpCenter.copy(splat.position);
    tmpScales.copy(splat.scale);
    tmpQuat.copy(splat.quaternion);
    tmpColor.copy(splat.color);

    packedSplats.pushSplat(tmpCenter, tmpScales, tmpQuat, splat.opacity, tmpColor);
    fillBounds.expandByPoint(splat.position);
  }

  packedSplats.needsUpdate = true;

  const mesh = new SplatMesh({ packedSplats });
  mesh.maxSh = 0;
  // Do NOT apply OpenCV quaternion fix — cached positions are already world-space

  const elapsedMs = (typeof performance !== "undefined" ? performance.now() : Date.now()) - startMs;
  console.log(`[infill] Created fill mesh with ${capped.length} splats in ${elapsedMs.toFixed(1)}ms`);

  return {
    mesh,
    splatCount: capped.length,
    surfaceCount: clusters.length,
    fillBounds,
  };
}
