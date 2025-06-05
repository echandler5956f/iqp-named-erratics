// frontend/src/services/tspService.js
// Stub TSP service – replace `solveTsp` implementation with your algorithm.
// The service is kept synchronous at the interface level but returns a Promise so you can
// plug in Web-Worker or async algorithmic code without touching the callers.

/**
 * Point object used by the solver.
 * @typedef {{ id: string|number, lat: number, lng: number, name?: string }} TspPoint
 */

// ------------------------------------------------------------
//  Core TSP algorithm – Nearest-Neighbour + 2-opt heuristic
// ------------------------------------------------------------

/**
 * Build symmetric distance matrix (in km) for the supplied points.
 * @param {TspPoint[]} pts
 * @returns {number[][]}
 */
function buildDistanceMatrix(pts) {
  const n = pts.length;
  const dm = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = haversineKm([pts[i].lat, pts[i].lng], [pts[j].lat, pts[j].lng]);
      dm[i][j] = dm[j][i] = d;
    }
  }
  return dm;
}

/**
 * Nearest-neighbour constructive heuristic.
 * @param {number[][]} dm distance matrix
 * @param {number} startIdx starting city index
 * @returns {number[]} tour as array of indices
 */
function nearestNeighbour(dm, startIdx = 0) {
  const n = dm.length;
  const unvisited = new Set(Array.from({ length: n }, (_, i) => i));
  unvisited.delete(startIdx);
  const tour = [startIdx];
  let current = startIdx;
  while (unvisited.size) {
    let next = null;
    let bestDist = Infinity;
    unvisited.forEach((idx) => {
      const d = dm[current][idx];
      if (d < bestDist) {
        bestDist = d;
        next = idx;
      }
    });
    unvisited.delete(next);
    tour.push(next);
    current = next;
  }
  return tour;
}

/**
 * Two-opt improvement until no better swap found.
 * @param {number[]} tour index tour (not closed)
 * @param {number[][]} dm distance matrix
 * @returns {number[]} improved tour
 */
function twoOpt(tour, dm) {
  const n = tour.length;
  let improved = true;
  while (improved) {
    improved = false;
    outer: for (let i = 1; i < n - 2; i++) {
      const a = tour[i - 1];
      const b = tour[i];
      for (let j = i + 1; j < n; j++) {
        const c = tour[j - 1];
        const d = tour[j % n];
        const delta = dm[a][b] + dm[c][d] - (dm[a][c] + dm[b][d]);
        if (delta > 1e-9) {
          // Reverse segment (i .. j-1)
          for (let k = 0, lo = i, hi = j - 1; k < (j - i) / 2; k++, lo++, hi--) {
            [tour[lo], tour[hi]] = [tour[hi], tour[lo]];
          }
          improved = true;
          break outer; // restart outer loop after any improvement
        }
      }
    }
  }
  return tour;
}

/**
 * Solve TSP using NN + 2-opt (open path, not returning to start).
 * @param {TspPoint[]} points
 * @returns {Promise<TspPoint[]>}
 */
export async function solveTsp(points) {
  if (!Array.isArray(points) || points.length < 2) return points ?? [];

  // Ensure user point (id==='user') is first for distance-matrix start index 0
  const pts = [...points];
  const userIdx = pts.findIndex((p) => p.id === 'user');
  if (userIdx > 0) {
    const [u] = pts.splice(userIdx, 1);
    pts.unshift(u);
  }

  const dm = buildDistanceMatrix(pts);
  let tourIdx = nearestNeighbour(dm, 0);
  if (pts.length >= 4) {
    tourIdx = twoOpt(tourIdx, dm);
  }

  // map indices back to points order returned
  const ordered = tourIdx.map((idx) => pts[idx]);
  return ordered;
}

// ------------------------------------------------------------
// Utility helpers – these are independent of the solver
// ------------------------------------------------------------

/**
 * Compute haversine distance (in kilometres) between two lat/lng pairs.
 * @param {[number, number]} a – [lat, lng]
 * @param {[number, number]} b – [lat, lng]
 * @returns {number}
 */
function haversineKm(a, b) {
  const toRad = (deg) => (deg * Math.PI) / 180;
  const R = 6371; // Earth radius km
  const dLat = toRad(b[0] - a[0]);
  const dLon = toRad(b[1] - a[1]);
  const lat1 = toRad(a[0]);
  const lat2 = toRad(b[0]);
  const sinLat = Math.sin(dLat / 2);
  const sinLon = Math.sin(dLon / 2);
  const aa = sinLat * sinLat + Math.cos(lat1) * Math.cos(lat2) * sinLon * sinLon;
  return 2 * R * Math.atan2(Math.sqrt(aa), Math.sqrt(1 - aa));
}

/**
 * Calculate total length of a TSP path (closed path not assumed).
 * @param {TspPoint[]} orderedPoints 
 * @returns {number} – distance in kilometres.
 */
export function calculatePathDistance(orderedPoints) {
  if (!Array.isArray(orderedPoints) || orderedPoints.length < 2) return 0;
  let km = 0;
  for (let i = 0; i < orderedPoints.length - 1; i++) {
    km += haversineKm([
      orderedPoints[i].lat,
      orderedPoints[i].lng,
    ], [
      orderedPoints[i + 1].lat,
      orderedPoints[i + 1].lng,
    ]);
  }
  return km;
} 