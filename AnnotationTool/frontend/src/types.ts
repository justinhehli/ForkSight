export enum JunctionType {
  ReplicationFork50 = "Replication Fork 50%",
  ReplicationFork100 = "Replication Fork 100%",
  ReversedFork50 = "Reversed Fork 50%",
  ReversedFork100 = "Reversed Fork 100%",
}

// Weight that each fork label contributes towards the fork-ratio calculation.
export const FORK_WEIGHTS: Record<string, number> = {
  [JunctionType.ReplicationFork50]: 0.5,
  [JunctionType.ReplicationFork100]: 1,
  [JunctionType.ReversedFork50]: 0.5,
  [JunctionType.ReversedFork100]: 1,
};

export interface ForkGroup {
  name: string;
  fifty: JunctionType;
  hundred: JunctionType;
  color: string;
}

export const FORK_GROUPS: ForkGroup[] = [
  {
    name: "Replication Fork",
    fifty: JunctionType.ReplicationFork50,
    hundred: JunctionType.ReplicationFork100,
    color: "#4caf50",
  },
  {
    name: "Reversed Fork",
    fifty: JunctionType.ReversedFork50,
    hundred: JunctionType.ReversedFork100,
    color: "#f44336",
  },
];

const FORK_LABEL_SET = new Set<string>(FORK_GROUPS.flatMap((g) => [g.fifty, g.hundred]));

// Fork labels (if set) always come first, followed by any additional labels alphabetically.
export const sortLabelsForDisplay = (labels: string[]): string[] => {
  const forkLabels = labels.filter((l) => FORK_LABEL_SET.has(l));
  const otherLabels = labels.filter((l) => !FORK_LABEL_SET.has(l)).sort((a, b) => a.localeCompare(b));
  return [...forkLabels, ...otherLabels];
};

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
  labels: string[];
}

export interface ImageAnnotations {
  processed: boolean;
  points: Point[];
}

export interface ProjectAnnotations {
  junction_detection_pipeline_status: PipelineStatus;
  pipeline_error?: string | null;
  additional_labels: string[];
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

export interface PipelineProgress {
  stage: "preprocessing" | "segmentation" | "detection" | null;
  completed: number;
  total: number;
}
