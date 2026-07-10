export enum JunctionType {
  ReplicationFork = "Replication Fork",
  ReversedFork = "Reversed Fork",
}

export enum PipelineStatus {
  Idle = "Idle",
  Running = "Running",
  Done = "Done",
  Failed = "Failed",
}

export interface Point {
  id: string;
  x: number;
  y: number;
  label: JunctionType;
}

export interface ImageAnnotations {
  processed: boolean;
  points: Point[];
}

export interface ProjectAnnotations {
  junction_detection_pipeline_status: PipelineStatus;
  pipeline_error?: string | null;
  images: Record<string, ImageAnnotations>;
}

export interface ImageMeta {
  id: string;
  name: string;
  processed: boolean;
}

export interface ProjectCandidate {
  name: string;
  valid: boolean;
  registered: boolean;
}
