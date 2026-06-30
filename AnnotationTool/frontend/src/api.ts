import { ImageAnnotations, ImageMeta, ProjectAnnotations } from "./types";

const BASE = "http://localhost:8000";

export const getImageUrl = (project: string, image: string) =>
  `${BASE}/projects/${encodeURIComponent(project)}/images/${encodeURIComponent(image)}`;

const request = async <T>(url: string, init?: RequestInit): Promise<T> => {
  const response = await fetch(url, init);
  if (!response.ok) {
    throw new Error(`${response.status} ${await response.text()}`);
  }

  return response.json() as Promise<T>;
}

export const getProjects = () => request<string[]>(`${BASE}/projects`);

export const getImages = (project: string) =>
  request<ImageMeta[]>(`${BASE}/projects/${encodeURIComponent(project)}/images`);

export const getAnnotations = (project: string) =>
  request<ProjectAnnotations>(`${BASE}/projects/${encodeURIComponent(project)}/annotations`);

export const saveImageAnnotations = async (project: string, image: string, data: ImageAnnotations): Promise<void> => {
  const response = await fetch(
    `${BASE}/projects/${encodeURIComponent(project)}/annotations/${encodeURIComponent(image)}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    },
  );

  if (!response.ok) {
    throw new Error(`${response.status} ${await response.text()}`);
  }
}

export const exportProject = async (project: string): Promise<Blob> => {
  const response = await fetch(`${BASE}/projects/${encodeURIComponent(project)}/export`, { method: "POST" });
  if (!response.ok) {
    throw new Error(`${response.status} ${await response.text()}`);
  }

  return response.blob();
}

export const runJunctionDetection = (project: string) =>
  request<{ status: string; project: string }>(
    `${BASE}/projects/${encodeURIComponent(project)}/run-junction-detection`,
    {
      method: "POST",
    },
  );
