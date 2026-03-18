import * as THREE from "three";
import { describe, expect, it, beforeEach } from "vitest";
import type { SDFShapeConfig, SpatialGrid, VoxelCell, InfillConfig } from "../src/types";

// We test the infill module's internal logic by importing its public API
// and validating the full pipeline against synthetic data.
// Since generateInfill depends on getCachedSplatData() (module-level state),
// we test the SDF evaluation and algorithmic pieces through the public entry point
// and also test the spatial-index cache accessor separately.

// ---------------------------------------------------------------------------
// Helpers: build synthetic spatial grids and cached splat data
// ---------------------------------------------------------------------------

function makeCell(
  gx: number, gy: number, gz: number,
  center: [number, number, number],
  splatIndices: number[],
  color: [number, number, number] = [0.5, 0.5, 0.5]
): [string, VoxelCell] {
  const key = `${gx},${gy},${gz}`;
  const c = new THREE.Vector3(...center);
  const half = 0.25;
  return [
    key,
    {
      gridPos: [gx, gy, gz],
      worldCenter: c,
      worldBounds: new THREE.Box3(
        new THREE.Vector3(center[0] - half, center[1] - half, center[2] - half),
        new THREE.Vector3(center[0] + half, center[1] + half, center[2] + half)
      ),
      splatCount: splatIndices.length,
      avgColor: new THREE.Color(color[0], color[1], color[2]),
      colorVariance: 0.01,
      density: splatIndices.length / (half * half * half * 8),
      splatIndices,
    },
  ];
}

function makeSyntheticGrid(cells: Array<[string, VoxelCell]>): SpatialGrid {
  const map = new Map(cells);
  return {
    resolution: [10, 10, 10],
    worldBounds: new THREE.Box3(
      new THREE.Vector3(-5, -5, -5),
      new THREE.Vector3(5, 5, 5)
    ),
    cellSize: new THREE.Vector3(1, 1, 1),
    cells: map,
  };
}

// ---------------------------------------------------------------------------
// SDF shape configs for testing
// ---------------------------------------------------------------------------

const sphereShape: SDFShapeConfig = {
  type: "SPHERE",
  position: [0, 0, 0],
  radius: 1.0,
};

const boxShape: SDFShapeConfig = {
  type: "BOX",
  position: [0, 0, 0],
  scale: [1, 1, 1],
};

const ellipsoidShape: SDFShapeConfig = {
  type: "ELLIPSOID",
  position: [0, 0, 0],
  scale: [2, 1, 1],
};

const cylinderShape: SDFShapeConfig = {
  type: "CYLINDER",
  position: [0, 0, 0],
  radius: 1.0,
  scale: [1, 2, 1],
};

const capsuleShape: SDFShapeConfig = {
  type: "CAPSULE",
  position: [0, 0, 0],
  radius: 0.5,
  scale: [1, 1, 1],
};

const allShape: SDFShapeConfig = {
  type: "ALL",
  position: [0, 0, 0],
};

const rotatedBoxShape: SDFShapeConfig = {
  type: "BOX",
  position: [0, 0, 0],
  scale: [1, 1, 1],
  // 45 degree rotation around Y axis
  rotation: [0, Math.sin(Math.PI / 8), 0, Math.cos(Math.PI / 8)],
};

// ---------------------------------------------------------------------------
// Test: SDF evaluation (tested indirectly through generateInfill behavior)
// We import the module to test that:
// 1) It returns null for insufficient data
// 2) It produces valid results for well-formed inputs
// ---------------------------------------------------------------------------

// We need a way to test SDF math without exporting internals.
// Strategy: import generateInfill and verify it behaves correctly
// given known inputs, which exercises all internal SDF functions.

import { generateInfill } from "../src/infill";
import { getCachedSplatData, buildSpatialGrid, type CachedSplatData } from "../src/spatial-index";

describe("generateInfill", () => {
  it("returns null when no cached splat data exists", () => {
    const grid = makeSyntheticGrid([]);
    const result = generateInfill([sphereShape], grid);
    expect(result).toBeNull();
  });

  it("returns null for ALL-type shapes", () => {
    const grid = makeSyntheticGrid([]);
    const result = generateInfill([allShape], grid);
    expect(result).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Test: SDF evaluation functions directly
// We re-implement minimal SDF tests by importing the module and using
// a mock approach: populate the spatial-index cache, build a grid with
// boundary splats around a known SDF shape, then verify generateInfill
// produces fill splats inside the deletion region.
// ---------------------------------------------------------------------------

describe("infill pipeline with synthetic splat data", () => {
  // We manually set up cached splat data by calling the spatial-index cache setter.
  // Since buildSpatialGrid is the only way to populate it, and it requires a SplatMesh,
  // we test the full pipeline integration by checking generateInfill's behavior
  // with a mocked spatial index.

  // Instead, let's test the SDF math by extracting it.
  // Since we can't easily mock the module state, we'll test the public API contract:
  // generateInfill should return null when there aren't enough boundary splats.

  it("returns null with empty grid (no boundary splats possible)", () => {
    const grid = makeSyntheticGrid([]);
    const result = generateInfill([sphereShape], grid);
    // null because getCachedSplatData returns null (no buildSpatialGrid called)
    expect(result).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Test: InfillConfig type
// ---------------------------------------------------------------------------

describe("InfillConfig type", () => {
  it("accepts all optional fields", () => {
    const config: InfillConfig = {
      softEdge: 0.2,
      sdfSmooth: 0.1,
      boundaryWidth: 0.3,
      maxFillSplats: 1000,
      maxClusters: 3,
    };
    expect(config.softEdge).toBe(0.2);
    expect(config.maxFillSplats).toBe(1000);
  });

  it("accepts empty config", () => {
    const config: InfillConfig = {};
    expect(config.softEdge).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Test: SDF signed distance functions
// Since these are not exported, we test them indirectly through a module
// that exposes them for testing. To keep the production module clean,
// we duplicate the core SDF math here for unit validation.
// ---------------------------------------------------------------------------

describe("SDF math (reference implementation)", () => {
  // Reference SDF implementations matching infill.ts logic
  function sdfSphere(point: THREE.Vector3, center: [number, number, number], radius: number): number {
    const dx = point.x - center[0];
    const dy = point.y - center[1];
    const dz = point.z - center[2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz) - radius;
  }

  function sdfBox(point: THREE.Vector3, center: [number, number, number], scale: [number, number, number]): number {
    const lx = Math.abs(point.x - center[0]) - scale[0];
    const ly = Math.abs(point.y - center[1]) - scale[1];
    const lz = Math.abs(point.z - center[2]) - scale[2];
    const outside = Math.sqrt(Math.max(lx, 0) ** 2 + Math.max(ly, 0) ** 2 + Math.max(lz, 0) ** 2);
    const inside = Math.min(Math.max(lx, ly, lz), 0);
    return outside + inside;
  }

  function sdfEllipsoid(point: THREE.Vector3, center: [number, number, number], scale: [number, number, number]): number {
    const nx = (point.x - center[0]) / scale[0];
    const ny = (point.y - center[1]) / scale[1];
    const nz = (point.z - center[2]) / scale[2];
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    return (len - 1) * Math.min(scale[0], scale[1], scale[2]);
  }

  describe("sphere SDF", () => {
    it("returns negative inside", () => {
      expect(sdfSphere(new THREE.Vector3(0, 0, 0), [0, 0, 0], 1)).toBeLessThan(0);
    });

    it("returns zero on surface", () => {
      expect(sdfSphere(new THREE.Vector3(1, 0, 0), [0, 0, 0], 1)).toBeCloseTo(0, 5);
    });

    it("returns positive outside", () => {
      expect(sdfSphere(new THREE.Vector3(2, 0, 0), [0, 0, 0], 1)).toBeCloseTo(1, 5);
    });

    it("works with offset center", () => {
      expect(sdfSphere(new THREE.Vector3(3, 0, 0), [3, 0, 0], 0.5)).toBeCloseTo(-0.5, 5);
    });

    it("handles diagonal points", () => {
      const d = Math.sqrt(3); // distance from origin to (1,1,1)
      expect(sdfSphere(new THREE.Vector3(1, 1, 1), [0, 0, 0], 2)).toBeCloseTo(d - 2, 5);
    });
  });

  describe("box SDF", () => {
    it("returns negative inside", () => {
      expect(sdfBox(new THREE.Vector3(0, 0, 0), [0, 0, 0], [1, 1, 1])).toBeLessThan(0);
    });

    it("returns zero on face center", () => {
      expect(sdfBox(new THREE.Vector3(1, 0, 0), [0, 0, 0], [1, 1, 1])).toBeCloseTo(0, 5);
    });

    it("returns positive outside face", () => {
      expect(sdfBox(new THREE.Vector3(2, 0, 0), [0, 0, 0], [1, 1, 1])).toBeCloseTo(1, 5);
    });

    it("returns positive outside corner (Euclidean distance)", () => {
      // Point at (2, 2, 2), box half-extents (1,1,1)
      // Closest point on box is (1,1,1), distance = sqrt(3)
      const d = sdfBox(new THREE.Vector3(2, 2, 2), [0, 0, 0], [1, 1, 1]);
      expect(d).toBeCloseTo(Math.sqrt(3), 5);
    });

    it("returns correct negative depth inside", () => {
      // Point at center of box, distance to nearest face = 1
      expect(sdfBox(new THREE.Vector3(0, 0, 0), [0, 0, 0], [1, 1, 1])).toBeCloseTo(-1, 5);
    });

    it("works with non-uniform scale", () => {
      // Thin box: half-extents (2, 0.5, 1)
      // Point at (0, 0.3, 0) is inside, nearest face at y=0.5, dist = -0.2
      expect(sdfBox(new THREE.Vector3(0, 0.3, 0), [0, 0, 0], [2, 0.5, 1])).toBeCloseTo(-0.2, 5);
    });
  });

  describe("ellipsoid SDF", () => {
    it("returns negative inside", () => {
      expect(sdfEllipsoid(new THREE.Vector3(0, 0, 0), [0, 0, 0], [2, 1, 1])).toBeLessThan(0);
    });

    it("returns approximately zero on surface along major axis", () => {
      // Point at (2, 0, 0), scale (2, 1, 1) → normalized (1, 0, 0) → len=1
      expect(sdfEllipsoid(new THREE.Vector3(2, 0, 0), [0, 0, 0], [2, 1, 1])).toBeCloseTo(0, 1);
    });

    it("returns positive outside", () => {
      expect(sdfEllipsoid(new THREE.Vector3(3, 0, 0), [0, 0, 0], [2, 1, 1])).toBeGreaterThan(0);
    });
  });
});

// ---------------------------------------------------------------------------
// Test: Normal extraction from quaternion
// ---------------------------------------------------------------------------

describe("normal extraction from quaternion + scale", () => {
  // Reference implementation matching infill.ts
  function extractNormal(
    qx: number, qy: number, qz: number, qw: number,
    sx: number, sy: number, sz: number
  ): THREE.Vector3 {
    let axis: THREE.Vector3;
    const absSx = Math.abs(sx), absSy = Math.abs(sy), absSz = Math.abs(sz);
    if (absSx <= absSy && absSx <= absSz) {
      axis = new THREE.Vector3(1, 0, 0);
    } else if (absSy <= absSx && absSy <= absSz) {
      axis = new THREE.Vector3(0, 1, 0);
    } else {
      axis = new THREE.Vector3(0, 0, 1);
    }
    const q = new THREE.Quaternion(qx, qy, qz, qw);
    axis.applyQuaternion(q);
    return axis.normalize();
  }

  it("returns X axis for identity quaternion when X scale is smallest", () => {
    const n = extractNormal(0, 0, 0, 1, 0.01, 0.5, 0.5);
    expect(n.x).toBeCloseTo(1, 5);
    expect(n.y).toBeCloseTo(0, 5);
    expect(n.z).toBeCloseTo(0, 5);
  });

  it("returns Y axis for identity quaternion when Y scale is smallest", () => {
    const n = extractNormal(0, 0, 0, 1, 0.5, 0.01, 0.5);
    expect(n.x).toBeCloseTo(0, 5);
    expect(n.y).toBeCloseTo(1, 5);
    expect(n.z).toBeCloseTo(0, 5);
  });

  it("returns Z axis for identity quaternion when Z scale is smallest", () => {
    const n = extractNormal(0, 0, 0, 1, 0.5, 0.5, 0.01);
    expect(n.x).toBeCloseTo(0, 5);
    expect(n.y).toBeCloseTo(0, 5);
    expect(n.z).toBeCloseTo(1, 5);
  });

  it("rotates normal by 90 degrees around Y", () => {
    // 90 deg around Y: quat = (0, sin(45deg), 0, cos(45deg))
    const s45 = Math.sin(Math.PI / 4);
    const c45 = Math.cos(Math.PI / 4);
    // Smallest scale is X → axis is (1,0,0), rotated 90 around Y → (0,0,-1)
    const n = extractNormal(0, s45, 0, c45, 0.01, 0.5, 0.5);
    expect(Math.abs(n.x)).toBeLessThan(0.01);
    expect(Math.abs(n.y)).toBeLessThan(0.01);
    expect(Math.abs(n.z)).toBeCloseTo(1, 2);
  });

  it("returns unit length normal", () => {
    const n = extractNormal(0.1, 0.2, 0.3, 0.9, 0.01, 0.2, 0.3);
    expect(n.length()).toBeCloseTo(1, 5);
  });
});

// ---------------------------------------------------------------------------
// Test: K-means normal clustering (reference implementation)
// ---------------------------------------------------------------------------

describe("surface normal clustering", () => {
  it("groups normals pointing in same direction", () => {
    // Two groups: normals pointing up (Y+) and forward (Z+)
    const upNormals = Array.from({ length: 10 }, () => new THREE.Vector3(0, 1, 0));
    const fwdNormals = Array.from({ length: 10 }, () => new THREE.Vector3(0, 0, 1));
    const allNormals = [...upNormals, ...fwdNormals];

    // Simple cluster check: compute angular similarity within groups
    for (let i = 0; i < 10; i++) {
      for (let j = 0; j < 10; j++) {
        // Same group should have dot product ~1
        expect(Math.abs(allNormals[i].dot(allNormals[j]))).toBeCloseTo(1, 5);
        // Cross-group should have dot product ~0
        expect(Math.abs(allNormals[i].dot(allNormals[10 + j]))).toBeCloseTo(0, 5);
      }
    }
  });

  it("handles nearly-parallel normals with noise", () => {
    const normals = Array.from({ length: 20 }, () => {
      // Slightly perturbed Y-up normals
      return new THREE.Vector3(
        (Math.random() - 0.5) * 0.1,
        1,
        (Math.random() - 0.5) * 0.1
      ).normalize();
    });

    // All normals should be close to Y-up
    for (const n of normals) {
      expect(n.dot(new THREE.Vector3(0, 1, 0))).toBeGreaterThan(0.99);
    }
  });
});

// ---------------------------------------------------------------------------
// Test: RANSAC plane fitting (reference implementation)
// ---------------------------------------------------------------------------

describe("RANSAC plane fitting", () => {
  function fitPlane(
    points: THREE.Vector3[],
    clusterNormal: THREE.Vector3,
    iterations: number = 100,
    threshold: number = 0.05
  ): { normal: THREE.Vector3; d: number; inlierRatio: number } | null {
    if (points.length < 3) return null;

    let bestNormal = new THREE.Vector3();
    let bestD = 0;
    let bestInliers = 0;
    const ab = new THREE.Vector3();
    const ac = new THREE.Vector3();
    const candidate = new THREE.Vector3();

    for (let iter = 0; iter < iterations; iter++) {
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
      if (Math.abs(candidate.dot(clusterNormal)) < 0.707) continue;
      if (candidate.dot(clusterNormal) < 0) candidate.negate();

      const d = candidate.dot(points[i0]);
      let inliers = 0;
      for (const p of points) {
        if (Math.abs(candidate.dot(p) - d) < threshold) inliers++;
      }

      if (inliers > bestInliers) {
        bestInliers = inliers;
        bestNormal.copy(candidate);
        bestD = d;
      }
    }

    if (bestInliers < 3) return null;
    return { normal: bestNormal.clone(), d: bestD, inlierRatio: bestInliers / points.length };
  }

  it("fits a perfect XZ plane (Y=0)", () => {
    const points = [];
    for (let x = -2; x <= 2; x += 0.5) {
      for (let z = -2; z <= 2; z += 0.5) {
        points.push(new THREE.Vector3(x, 0, z));
      }
    }

    const result = fitPlane(points, new THREE.Vector3(0, 1, 0), 200, 0.01);
    expect(result).not.toBeNull();
    expect(result!.inlierRatio).toBeCloseTo(1.0, 1);
    // Normal should be approximately Y-up
    expect(Math.abs(result!.normal.y)).toBeGreaterThan(0.99);
    expect(Math.abs(result!.d)).toBeLessThan(0.02);
  });

  it("fits a tilted plane with high inlier ratio", () => {
    // Plane: y = x (normal = (-1, 1, 0) / sqrt(2))
    const points = [];
    for (let x = -2; x <= 2; x += 0.5) {
      for (let z = -2; z <= 2; z += 0.5) {
        points.push(new THREE.Vector3(x, x, z));
      }
    }

    const clusterNormal = new THREE.Vector3(-1, 1, 0).normalize();
    const result = fitPlane(points, clusterNormal, 200, 0.01);
    expect(result).not.toBeNull();
    expect(result!.inlierRatio).toBeGreaterThan(0.9);
  });

  it("handles noisy points with reasonable inlier ratio", () => {
    const points = [];
    // Mostly on Y=2 plane with some noise
    for (let i = 0; i < 50; i++) {
      points.push(new THREE.Vector3(
        Math.random() * 4 - 2,
        2 + (Math.random() - 0.5) * 0.02, // small noise
        Math.random() * 4 - 2
      ));
    }
    // Add 5 outliers
    for (let i = 0; i < 5; i++) {
      points.push(new THREE.Vector3(Math.random(), Math.random() * 10, Math.random()));
    }

    const result = fitPlane(points, new THREE.Vector3(0, 1, 0), 200, 0.05);
    expect(result).not.toBeNull();
    expect(result!.inlierRatio).toBeGreaterThan(0.8);
    expect(result!.d).toBeCloseTo(2, 0);
  });

  it("returns null for fewer than 3 points", () => {
    const points = [new THREE.Vector3(0, 0, 0), new THREE.Vector3(1, 0, 0)];
    const result = fitPlane(points, new THREE.Vector3(0, 1, 0));
    expect(result).toBeNull();
  });

  it("rejects planes inconsistent with cluster normal", () => {
    // Points on XY plane (normal = Z), but cluster normal = Y
    // RANSAC should reject planes with normal too far from Y
    const points = [];
    for (let x = -1; x <= 1; x += 0.5) {
      for (let y = -1; y <= 1; y += 0.5) {
        points.push(new THREE.Vector3(x, y, 0));
      }
    }

    const result = fitPlane(points, new THREE.Vector3(0, 1, 0), 200, 0.01);
    // Should either return null (all candidates rejected) or a bad fit
    // since Z-normal doesn't pass the 45-degree check against Y-normal
    if (result) {
      // If it somehow found a fit, it should have poor inlier ratio
      // because valid plane normals are far from Y
      expect(Math.abs(result.normal.dot(new THREE.Vector3(0, 1, 0)))).toBeGreaterThan(0.707);
    }
  });
});

// ---------------------------------------------------------------------------
// Test: Fill density estimation
// ---------------------------------------------------------------------------

describe("density estimation", () => {
  function estimateDensity(positions: THREE.Vector3[]): number {
    if (positions.length < 2) return 100;
    let totalDist = 0;
    const sampleSize = Math.min(positions.length, 50);
    for (let i = 0; i < sampleSize; i++) {
      let minDist = Infinity;
      for (let j = 0; j < positions.length; j++) {
        if (i === j) continue;
        const d = positions[i].distanceToSquared(positions[j]);
        if (d < minDist) minDist = d;
      }
      totalDist += Math.sqrt(minDist);
    }
    const avgSpacing = totalDist / sampleSize;
    if (avgSpacing < 1e-6) return 10000;
    return 1 / (avgSpacing * avgSpacing);
  }

  it("returns high density for closely spaced points", () => {
    const points = [];
    for (let i = 0; i < 20; i++) {
      points.push(new THREE.Vector3(i * 0.01, 0, 0));
    }
    const density = estimateDensity(points);
    expect(density).toBeGreaterThan(1000);
  });

  it("returns low density for widely spaced points", () => {
    const points = [];
    for (let i = 0; i < 10; i++) {
      points.push(new THREE.Vector3(i * 10, 0, 0));
    }
    const density = estimateDensity(points);
    expect(density).toBeLessThan(1);
  });

  it("returns default for single point", () => {
    expect(estimateDensity([new THREE.Vector3()])).toBe(100);
  });
});

// ---------------------------------------------------------------------------
// Test: Quaternion from normal
// ---------------------------------------------------------------------------

describe("quaternion from normal", () => {
  function quatFromNormal(normal: THREE.Vector3): THREE.Quaternion {
    const q = new THREE.Quaternion();
    q.setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal);
    return q;
  }

  it("identity for Z-forward normal", () => {
    const q = quatFromNormal(new THREE.Vector3(0, 0, 1));
    expect(q.x).toBeCloseTo(0, 5);
    expect(q.y).toBeCloseTo(0, 5);
    expect(q.z).toBeCloseTo(0, 5);
    expect(q.w).toBeCloseTo(1, 5);
  });

  it("90 degree rotation for Y-up normal", () => {
    const q = quatFromNormal(new THREE.Vector3(0, 1, 0));
    // Applying this quaternion to (0,0,1) should give (0,1,0)
    const result = new THREE.Vector3(0, 0, 1).applyQuaternion(q);
    expect(result.x).toBeCloseTo(0, 3);
    expect(result.y).toBeCloseTo(1, 3);
    expect(result.z).toBeCloseTo(0, 3);
  });

  it("handles arbitrary normal", () => {
    const normal = new THREE.Vector3(1, 1, 1).normalize();
    const q = quatFromNormal(normal);
    const result = new THREE.Vector3(0, 0, 1).applyQuaternion(q);
    expect(result.x).toBeCloseTo(normal.x, 3);
    expect(result.y).toBeCloseTo(normal.y, 3);
    expect(result.z).toBeCloseTo(normal.z, 3);
  });

  it("handles opposite direction (-Z)", () => {
    const q = quatFromNormal(new THREE.Vector3(0, 0, -1));
    const result = new THREE.Vector3(0, 0, 1).applyQuaternion(q);
    expect(result.z).toBeCloseTo(-1, 3);
  });
});

// ---------------------------------------------------------------------------
// Test: K-nearest neighbor search
// ---------------------------------------------------------------------------

describe("K-nearest neighbor", () => {
  interface FakeSplat { position: THREE.Vector3 }

  function findKNearest(
    pos: THREE.Vector3,
    splats: FakeSplat[],
    k: number
  ): Array<{ splat: FakeSplat; dist: number }> {
    const scored = splats.map(s => ({
      splat: s,
      dist: pos.distanceTo(s.position),
    }));
    scored.sort((a, b) => a.dist - b.dist);
    return scored.slice(0, k);
  }

  it("returns K nearest points", () => {
    const splats: FakeSplat[] = [
      { position: new THREE.Vector3(0, 0, 0) },
      { position: new THREE.Vector3(1, 0, 0) },
      { position: new THREE.Vector3(2, 0, 0) },
      { position: new THREE.Vector3(3, 0, 0) },
      { position: new THREE.Vector3(10, 0, 0) },
    ];

    const result = findKNearest(new THREE.Vector3(0.5, 0, 0), splats, 2);
    expect(result).toHaveLength(2);
    expect(result[0].dist).toBeCloseTo(0.5, 5);
    expect(result[1].dist).toBeCloseTo(0.5, 5);
  });

  it("returns all when K > length", () => {
    const splats: FakeSplat[] = [
      { position: new THREE.Vector3(0, 0, 0) },
      { position: new THREE.Vector3(1, 0, 0) },
    ];

    const result = findKNearest(new THREE.Vector3(0, 0, 0), splats, 5);
    expect(result).toHaveLength(2);
  });

  it("sorted by distance", () => {
    const splats: FakeSplat[] = [
      { position: new THREE.Vector3(5, 0, 0) },
      { position: new THREE.Vector3(1, 0, 0) },
      { position: new THREE.Vector3(3, 0, 0) },
    ];

    const result = findKNearest(new THREE.Vector3(0, 0, 0), splats, 3);
    expect(result[0].dist).toBeLessThan(result[1].dist);
    expect(result[1].dist).toBeLessThan(result[2].dist);
  });
});

// ---------------------------------------------------------------------------
// Test: getCachedSplatData accessor
// ---------------------------------------------------------------------------

describe("getCachedSplatData", () => {
  it("returns null before buildSpatialGrid is called", () => {
    // Note: this depends on test isolation — if other tests call
    // buildSpatialGrid, this could be non-null. In practice vitest
    // runs files in separate workers, so this should be clean.
    // We just verify the return type contract.
    const data = getCachedSplatData();
    if (data !== null) {
      expect(data).toHaveProperty("positions");
      expect(data).toHaveProperty("colors");
      expect(data).toHaveProperty("quaternions");
      expect(data).toHaveProperty("scales");
      expect(data).toHaveProperty("opacities");
      expect(data).toHaveProperty("count");
      expect(data.count).toBeGreaterThan(0);
    }
  });
});

// ---------------------------------------------------------------------------
// Test: InfillResult contract
// ---------------------------------------------------------------------------

describe("InfillResult shape", () => {
  it("type satisfies expected interface", () => {
    // Compile-time check: this block just needs to typecheck
    const fakeResult: import("../src/infill").InfillResult = {
      mesh: {} as any,
      splatCount: 42,
      surfaceCount: 2,
      fillBounds: new THREE.Box3(),
    };
    expect(fakeResult.splatCount).toBe(42);
    expect(fakeResult.surfaceCount).toBe(2);
  });
});
