import { ImageAnnotations, ImageMeta, ProjectAnnotations, ProjectCandidate } from "./types";

const BASE = "http://localhost:8000";

const encodeProjectPath = (project: string) => project.split("/").map(encodeURIComponent).join("/");

export const getImageUrl = (project: string, imageId: string) =>
  `${BASE}/projects/${encodeProjectPath(project)}/images/${encodeURIComponent(imageId)}`;

export const getMaskUrl = (project: string, imageId: string) =>
  `${BASE}/projects/${encodeProjectPath(project)}/images/${encodeURIComponent(imageId)}/mask`;

const request = async <T>(url: string, init?: RequestInit): Promise<T> => {
  const response = await fetch(url, init);
  if (!response.ok) {
    throw new Error(`${response.status} ${await response.text()}`);
  }

  return response.json() as Promise<T>;
};

export const getProjects = () => request<string[]>(`${BASE}/projects`);

// liveness check used while waiting for the backend to restart
// (registering a new project restarts the sandboxed backend)
export const pingBackend = async (timeoutMs = 1500): Promise<boolean> => {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${BASE}/projects`, { signal: controller.signal });
    return response.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
};

export const getImages = (project: string) =>
  request<ImageMeta[]>(`${BASE}/projects/${encodeProjectPath(project)}/images`);

export const getAnnotations = (project: string) =>
  request<ProjectAnnotations>(`${BASE}/projects/${encodeProjectPath(project)}/annotations`);

export const saveImageAnnotations = async (project: string, imageId: string, data: ImageAnnotations): Promise<void> => {
  const response = await fetch(
    `${BASE}/projects/${encodeProjectPath(project)}/annotations/${encodeURIComponent(imageId)}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    },
  );

  if (!response.ok) {
    throw new Error(`${response.status} ${await response.text()}`);
  }
};

export const exportProject = async (project: string): Promise<Blob> => {
  const response = await fetch(`${BASE}/projects/${encodeProjectPath(project)}/export`, { method: "POST" });
  if (!response.ok) {
    throw new Error(`${response.status} ${await response.text()}`);
  }

  return response.blob();
};

export const exportProjectExcel = async (project: string): Promise<Blob> => {
  const response = await fetch(`${BASE}/projects/${encodeProjectPath(project)}/export-excel`, { method: "POST" });
  if (!response.ok) {
    throw new Error(`${response.status} ${await response.text()}`);
  }

  return response.blob();
};

export const runJunctionDetection = (project: string) =>
  request<{ status: string; project: string }>(
    `${BASE}/projects/${encodeProjectPath(project)}/run-junction-detection`,
    {
      method: "POST",
    },
  );

export const stopJunctionDetection = (project: string) =>
  request<{ status: string; project: string }>(
    `${BASE}/projects/${encodeProjectPath(project)}/stop-junction-detection`,
    {
      method: "POST",
    },
  );

export const getPipelineStatus = () => request<{ running_project: string | null }>(`${BASE}/pipeline-status`);

export const getPipelineLog = async (project: string): Promise<string> => {
  const response = await fetch(`${BASE}/projects/${encodeProjectPath(project)}/pipeline-log`);
  if (!response.ok) {
    throw new Error(`${response.status} ${await response.text()}`);
  }

  return response.text();
};

export const addCustomLabel = (project: string, label: string) =>
  request<string[]>(`${BASE}/projects/${encodeProjectPath(project)}/labels`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label }),
  });

export const deleteCustomLabel = (project: string, label: string) =>
  request<string[]>(`${BASE}/projects/${encodeProjectPath(project)}/labels/${encodeURIComponent(label)}`, {
    method: "DELETE",
  });

export const getProjectCandidates = () => request<ProjectCandidate[]>(`${BASE}/project-candidates`);

export const getProjectFolderPath = (name: string) =>
  request<{ path: string }>(`${BASE}/project-candidates/${encodeProjectPath(name)}/path`);

export const setRegisteredProjects = (names: string[]) =>
  request<ProjectCandidate[]>(`${BASE}/project-candidates`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ names }),
  });
