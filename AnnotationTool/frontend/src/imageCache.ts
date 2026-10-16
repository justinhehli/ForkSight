import { getImageUrl, getMaskUrl, getNeighborImageUrl } from "./api";

// "low" deprioritizes the request (in browsers that support the Fetch
// Priority API) so it doesn't compete with a fetch the user is actively
// waiting on - e.g. surrounding-tile/neighbor prefetches shouldn't slow
// down loading the tile currently on screen.
const warm = (url: string, priority?: "low"): void => {
  const img = new Image();
  if (priority) img.fetchPriority = priority;
  img.src = url;
};

export const prefetchImage = (project: string, imageId: string, priority?: "low"): void => {
  warm(getImageUrl(project, imageId), priority);
};

export const prefetchMask = (project: string, imageId: string, priority?: "low"): void => {
  warm(getMaskUrl(project, imageId), priority);
};

// The 8 tile positions directly surrounding a tile, as (dRow, dCol) offsets.
export const NEIGHBOR_OFFSETS: [number, number][] = [
  [-1, -1], [-1, 0], [-1, 1],
  [0, -1], [0, 1],
  [1, -1], [1, 0], [1, 1],
];

// Warms the browser cache for all 8 neighbor tiles so they're already loaded
// by the time the user toggles them on. Missing neighbors (tile at the edge
// of its 4x4 grid) 404 and are simply ignored here - the <img> that actually
// displays them handles that the same way.
export const prefetchNeighbors = (project: string, imageId: string): void => {
  for (const [dRow, dCol] of NEIGHBOR_OFFSETS) {
    warm(getNeighborImageUrl(project, imageId, dRow, dCol), "low");
  }
};
