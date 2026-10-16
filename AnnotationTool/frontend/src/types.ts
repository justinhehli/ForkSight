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

export enum PipelineMode {
  Sequential = "sequential",
  Staged = "staged",
}

export type DiscoveryConditionType = "file" | "dir";

export interface DiscoveryCondition {
  type: DiscoveryConditionType;
  pattern: string;
}

// A candidate directory qualifies as a project if it satisfies ANY rule,
// where a rule is a list of conditions that must ALL be met.
export type ProjectDiscoveryRule = DiscoveryCondition[];

export interface PipelineSettings {
  // Applies globally, to every project.
  pipeline_mode: PipelineMode;
  sequential_target_junction_count: number;
  staged_sample_count: number;
  tile_glob_patterns: string[];
  project_discovery_rules: ProjectDiscoveryRule[];
}

export interface ProjectTileSettings {
  // Per-project override of the global tile_glob_patterns; null means the
  // project just uses the global default.
  tile_glob_patterns_override: string[] | null;
  effective_tile_glob_patterns: string[];
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
  archived?: boolean;
}

export interface ProjectAnnotations {
  junction_detection_pipeline_status: PipelineStatus;
  pipeline_error?: string | null;
  pipeline_mode?: PipelineMode | null;
  additional_labels: string[];
  images: Record<string, ImageAnnotations>;
}

export interface ImageMeta {
  id: string;
  name: string;
  processed: boolean;
  archived: boolean;
}

export interface ProjectCandidate {
  name: string;
  valid: boolean;
  registered: boolean;
}

export interface PipelineProgress {
  stage: "preprocessing" | "segmentation" | "detection" | "sequential" | null;
  completed: number;
  total: number;
}
