import { getImageUrl, getMaskUrl } from "./api";

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
