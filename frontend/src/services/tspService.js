// frontend/src/services/tspService.js
// Stub TSP service – replace `solveTsp` implementation with your algorithm.
// The service is kept synchronous at the interface level but returns a Promise so you can
// plug in Web-Worker or async algorithmic code without touching the callers.

/**
 * Point object used by the solver.
 * @typedef {{ id: string|number, lat: number, lng: number, name?: string }} TspPoint
 */

/**
 * Solve a Traveling Salesman Problem for the supplied points.
 * The first point in the returned path should be the user location (id === 'user')
 * if present.  The algorithm is currently a stub returning the points in the
 * original order; replace with a real TSP implementation.
 *
 * @param {TspPoint[]} points – Array of points to visit.
 * @returns {Promise<TspPoint[]>} – Ordered path array visiting every point once.
 */
export async function solveTsp(points) {
  if (!Array.isArray(points) || points.length === 0) return [];
  // ★★★★★  REPLACE THIS SECTION WITH YOUR OPTIMAL / NEAR-OPTIMAL SOLVER ★★★★★
  // For now we simply return the points in their existing order, keeping the
  // 'user' point (if present) first.
  const cloned = [...points];
  const userIdx = cloned.findIndex(p => p.id === 'user');
  if (userIdx > 0) {
    const [userPoint] = cloned.splice(userIdx, 1);
    cloned.unshift(userPoint);
  }
  return cloned;
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