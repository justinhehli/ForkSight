import { getImageUrl, getMaskUrl, getNeighborImageUrl } from "./api";

const warm = (url: string): void => {
  const img = new Image();
  img.src = url;
};

export const prefetchImage = (project: string, imageId: string): void => {
  warm(getImageUrl(project, imageId));
};

export const prefetchMask = (project: string, imageId: string): void => {
  warm(getMaskUrl(project, imageId));
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
    warm(getNeighborImageUrl(project, imageId, dRow, dCol));
  }
};
