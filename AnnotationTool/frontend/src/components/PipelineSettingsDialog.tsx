import { useEffect, useState } from "react";
import {
  Add as AddIcon,
  Close as CloseIcon,
} from "@mui/icons-material";
import {
  Alert,
  Box,
  Button,
  Checkbox,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  FormControl,
  FormControlLabel,
  IconButton,
  MenuItem,
  Paper,
  Radio,
  RadioGroup,
  Select,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import { getPipelineSettings, getProjectTileSettings, setPipelineSettings, setProjectTileSettings } from "../api";
import { PipelineMode } from "../types";
import type { DiscoveryCondition, PipelineSettings, ProjectDiscoveryRule } from "../types";

interface Props {
  open: boolean;
  onClose: () => void;
  isTrainEnv: boolean;
  // Currently selected project, or "" if none is selected yet.
  project: string;
}

const emptyCondition = (): DiscoveryCondition => ({ type: "file", pattern: "" });
const emptyRule = (): ProjectDiscoveryRule => [emptyCondition()];

const nonBlank = (patterns: string[]) => patterns.map((p) => p.trim()).filter(Boolean);

// Editable list of glob-pattern strings, used for both the global default tile
// patterns and (when overriding) a single project's own patterns.
const GlobPatternListEditor = ({
  patterns,
  onChange,
}: {
  patterns: string[];
  onChange: (patterns: string[]) => void;
}) => (
  <Stack spacing={1} sx={{ mt: 1 }}>
    {patterns.map((pattern, i) => (
      <Box key={i} sx={{ display: "flex", alignItems: "center", gap: 1 }}>
        <TextField
          size="small"
          fullWidth
          placeholder="e.g. LayersData/highmag/Tile Set (*)/*.tif"
          value={pattern}
          onChange={(e) => onChange(patterns.map((p, j) => (j === i ? e.target.value : p)))}
          error={!pattern.trim()}
        />
        <IconButton
          size="small"
          onClick={() => onChange(patterns.filter((_, j) => j !== i))}
          disabled={patterns.length <= 1}
        >
          <CloseIcon fontSize="small" />
        </IconButton>
      </Box>
    ))}
    <Button size="small" startIcon={<AddIcon />} onClick={() => onChange([...patterns, ""])} sx={{ alignSelf: "flex-start" }}>
      Add pattern
    </Button>
  </Stack>
);

// A folder is recognized as a project if it matches ANY of these rules; a
// rule matches if ALL of its conditions are met. That OR-of-ANDs structure is
// implicit in the UI: users just add rules and, within each, conditions.
const ProjectDiscoveryRulesEditor = ({
  rules,
  onChange,
}: {
  rules: ProjectDiscoveryRule[];
  onChange: (rules: ProjectDiscoveryRule[]) => void;
}) => {
  const updateRule = (ruleIdx: number, rule: ProjectDiscoveryRule) =>
    onChange(rules.map((r, i) => (i === ruleIdx ? rule : r)));

  return (
    <Stack spacing={1.5} sx={{ mt: 1 }}>
      {rules.map((rule, ruleIdx) => (
        <Paper key={ruleIdx} variant="outlined" sx={{ p: 1.5 }}>
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1 }}>
            <Typography variant="caption" color="text.secondary">
              All of the following must exist in the folder:
            </Typography>
            <IconButton size="small" onClick={() => onChange(rules.filter((_, i) => i !== ruleIdx))} disabled={rules.length <= 1}>
              <CloseIcon fontSize="small" />
            </IconButton>
          </Box>
          <Stack spacing={1}>
            {rule.map((condition, condIdx) => (
              <Box key={condIdx} sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <Select
                  size="small"
                  value={condition.type}
                  onChange={(e) =>
                    updateRule(
                      ruleIdx,
                      rule.map((c, j) => (j === condIdx ? { ...c, type: e.target.value as "file" | "dir" } : c)),
                    )
                  }
                  sx={{ width: 110 }}
                >
                  <MenuItem value="file">File</MenuItem>
                  <MenuItem value="dir">Folder</MenuItem>
                </Select>
                <TextField
                  size="small"
                  fullWidth
                  placeholder={condition.type === "file" ? "e.g. *.mapsxml" : "e.g. LayersData/highmag"}
                  value={condition.pattern}
                  onChange={(e) =>
                    updateRule(
                      ruleIdx,
                      rule.map((c, j) => (j === condIdx ? { ...c, pattern: e.target.value } : c)),
                    )
                  }
                  error={!condition.pattern.trim()}
                />
                <IconButton
                  size="small"
                  onClick={() => updateRule(ruleIdx, rule.filter((_, j) => j !== condIdx))}
                  disabled={rule.length <= 1}
                >
                  <CloseIcon fontSize="small" />
                </IconButton>
              </Box>
            ))}
            <Button
              size="small"
              startIcon={<AddIcon />}
              onClick={() => updateRule(ruleIdx, [...rule, emptyCondition()])}
              sx={{ alignSelf: "flex-start" }}
            >
              Add condition
            </Button>
          </Stack>
        </Paper>
      ))}
      <Button size="small" startIcon={<AddIcon />} onClick={() => onChange([...rules, emptyRule()])} sx={{ alignSelf: "flex-start" }}>
        Add rule
      </Button>
    </Stack>
  );
};

const PipelineSettingsDialog = ({ open, onClose, isTrainEnv, project }: Props) => {
  const [settings, setSettings] = useState<PipelineSettings | null>(null);
  const [projectOverrideEnabled, setProjectOverrideEnabled] = useState(false);
  const [projectPatterns, setProjectPatterns] = useState<string[]>([""]);
  const [globalDefaultPatterns, setGlobalDefaultPatterns] = useState<string[]>([]);

  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const showProjectSection = !isTrainEnv && !!project;

  useEffect(() => {
    if (!open) return;

    setLoading(true);
    setError(null);

    Promise.all([
      getPipelineSettings(),
      showProjectSection ? getProjectTileSettings(project) : Promise.resolve(null),
    ])
      .then(([globalSettings, projectTileSettings]) => {
        setSettings(globalSettings);
        setGlobalDefaultPatterns(globalSettings.tile_glob_patterns);
        if (projectTileSettings) {
          setProjectOverrideEnabled(projectTileSettings.tile_glob_patterns_override !== null);
          setProjectPatterns(
            projectTileSettings.tile_glob_patterns_override ?? projectTileSettings.effective_tile_glob_patterns,
          );
        } else {
          setProjectOverrideEnabled(false);
          setProjectPatterns([""]);
        }
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, project, showProjectSection]);

  const handleSave = async () => {
    if (!settings) return;

    setSaving(true);
    setError(null);
    try {
      await setPipelineSettings(settings);
      if (showProjectSection) {
        await setProjectTileSettings(project, projectOverrideEnabled ? nonBlank(projectPatterns) : null);
      }
      onClose();
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  };

  const globalTilePatternsValid = !!settings && nonBlank(settings.tile_glob_patterns).length > 0;
  const discoveryRulesValid =
    !!settings &&
    settings.project_discovery_rules.length > 0 &&
    settings.project_discovery_rules.every((rule) => rule.length > 0 && rule.every((c) => c.pattern.trim()));
  const projectPatternsValid = !projectOverrideEnabled || nonBlank(projectPatterns).length > 0;

  const invalid =
    !settings ||
    (settings.pipeline_mode === PipelineMode.Sequential && !(settings.sequential_target_junction_count > 0)) ||
    (settings.pipeline_mode === PipelineMode.Staged && !(settings.staged_sample_count > 0)) ||
    !globalTilePatternsValid ||
    (!isTrainEnv && !discoveryRulesValid) ||
    !projectPatternsValid;

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Settings</DialogTitle>
      <DialogContent>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        {loading && (
          <Box sx={{ display: "flex", justifyContent: "center", py: 3 }}>
            <CircularProgress size={20} />
          </Box>
        )}
        {!loading && settings && (
          <Stack spacing={2.5}>
            <Box>
              <Typography variant="overline" color="text.secondary">
                Global settings — apply to every project
              </Typography>

              <Typography variant="subtitle2" sx={{ mt: 1.5 }}>
                Detection pipeline
              </Typography>
              <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                Default mode used when a project's fork detection starts. Applies the next time detection is run.
              </Typography>
              <FormControl fullWidth>
                <RadioGroup
                  value={settings.pipeline_mode}
                  onChange={(e) => setSettings({ ...settings, pipeline_mode: e.target.value as PipelineMode })}
                >
                  <Box sx={{ mb: 1 }}>
                    <FormControlLabel value={PipelineMode.Sequential} control={<Radio size="small" />} label="Sequential" />
                    <Typography variant="caption" color="text.secondary" display="block" sx={{ ml: 4, mb: 1.5 }}>
                      Tiles are randomly sampled and processed sequentially one at a time until a pre-defined number total
                      forks have been found.
                    </Typography>
                    <TextField
                      label="Target fork number to be found"
                      type="number"
                      size="small"
                      sx={{ ml: 4, width: 280 }}
                      value={settings.sequential_target_junction_count}
                      onChange={(e) =>
                        setSettings({ ...settings, sequential_target_junction_count: Number(e.target.value) })
                      }
                      error={!(settings.sequential_target_junction_count > 0)}
                      inputProps={{ min: 1 }}
                    />
                  </Box>

                  <Box>
                    <FormControlLabel value={PipelineMode.Staged} control={<Radio size="small" />} label="Staged" />
                    <Typography variant="caption" color="text.secondary" display="block" sx={{ ml: 4, mb: 1.5 }}>
                      A random subsample of tiles is processed in full, regardless of the number of found forks.
                    </Typography>
                    <TextField
                      label="Subsample size (number of tiles)"
                      type="number"
                      size="small"
                      sx={{ ml: 4, width: 280 }}
                      value={settings.staged_sample_count}
                      onChange={(e) => setSettings({ ...settings, staged_sample_count: Number(e.target.value) })}
                      error={!(settings.staged_sample_count > 0)}
                      inputProps={{ min: 1 }}
                    />
                  </Box>
                </RadioGroup>
              </FormControl>

              <Divider sx={{ my: 2 }} />

              <Typography variant="subtitle2">Tile discovery (default)</Typography>
              <Typography variant="caption" color="text.secondary" display="block">
                Where to look for tiles inside a project's folder, relative to that folder. A project's tiles are the
                union of everything matched by these patterns. Can be overridden for the currently selected project
                below.
              </Typography>
              <GlobPatternListEditor
                patterns={settings.tile_glob_patterns}
                onChange={(tile_glob_patterns) => setSettings({ ...settings, tile_glob_patterns })}
              />

              {!isTrainEnv && (
                <>
                  <Divider sx={{ my: 2 }} />

                  <Typography variant="subtitle2">Project discovery</Typography>
                  <Typography variant="caption" color="text.secondary" display="block">
                    Controls which folders show up as project candidates in "Manage projects". A folder is recognized
                    as a project if it matches any one of these rules.
                  </Typography>
                  <ProjectDiscoveryRulesEditor
                    rules={settings.project_discovery_rules}
                    onChange={(project_discovery_rules) => setSettings({ ...settings, project_discovery_rules })}
                  />
                </>
              )}
            </Box>

            {showProjectSection && (
              <Box>
                <Divider sx={{ mb: 2 }} />
                <Typography variant="overline" color="text.secondary">
                  This project — {project}
                </Typography>

                <Box sx={{ mt: 1 }}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        size="small"
                        checked={projectOverrideEnabled}
                        onChange={(e) => {
                          setProjectOverrideEnabled(e.target.checked);
                          if (e.target.checked) setProjectPatterns((prev) => (prev.some((p) => p.trim()) ? prev : globalDefaultPatterns));
                        }}
                      />
                    }
                    label="Override tile discovery patterns for this project"
                  />
                  {projectOverrideEnabled ? (
                    <GlobPatternListEditor patterns={projectPatterns} onChange={setProjectPatterns} />
                  ) : (
                    <Typography variant="caption" color="text.secondary" display="block" sx={{ ml: 4 }}>
                      Currently using the global default: {globalDefaultPatterns.join(", ") || "—"}
                    </Typography>
                  )}
                </Box>
              </Box>
            )}
          </Stack>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={saving}>
          Cancel
        </Button>
        <Button
          variant="contained"
          onClick={handleSave}
          disabled={loading || saving || invalid}
          startIcon={saving ? <CircularProgress size={16} color="inherit" /> : undefined}
        >
          {saving ? "Saving…" : "Save"}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default PipelineSettingsDialog;
