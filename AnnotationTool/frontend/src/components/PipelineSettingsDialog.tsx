import { useEffect, useState } from "react";
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControl,
  FormControlLabel,
  Radio,
  RadioGroup,
  TextField,
  Typography,
} from "@mui/material";
import { getPipelineSettings, setPipelineSettings } from "../api";
import { PipelineMode } from "../types";
import type { PipelineSettings } from "../types";

interface Props {
  open: boolean;
  onClose: () => void;
}

const PipelineSettingsDialog = ({ open, onClose }: Props) => {
  const [settings, setSettings] = useState<PipelineSettings | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;

    setLoading(true);
    setError(null);

    getPipelineSettings()
      .then(setSettings)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [open]);

  const handleSave = async () => {
    if (!settings) return;

    setSaving(true);
    setError(null);
    try {
      await setPipelineSettings(settings);
      onClose();
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  };

  const invalid =
    !settings ||
    (settings.pipeline_mode === PipelineMode.Sequential && !(settings.sequential_target_junction_count > 0)) ||
    (settings.pipeline_mode === PipelineMode.Staged &&
      !(settings.staged_sample_percentage > 0 && settings.staged_sample_percentage <= 100));

  return (
    <Dialog open={open} onClose={onClose} maxWidth="xs" fullWidth>
      <DialogTitle>Fork detection settings</DialogTitle>
      <DialogContent dividers>
        <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 2 }}>
          Choose how the automatic fork detection pipeline processes a project's tiles. Applies the
          next time detection is run, for any project.
        </Typography>
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
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
            <FormControl>
              <RadioGroup
                value={settings.pipeline_mode}
                onChange={(e) => setSettings({ ...settings, pipeline_mode: e.target.value as PipelineMode })}
              >
                <FormControlLabel
                  value={PipelineMode.Sequential}
                  control={<Radio size="small" />}
                  label="Sequential"
                />
                <Typography variant="caption" color="text.secondary" sx={{ ml: 4, mt: -1, mb: 1 }}>
                  Tiles are randomly sampled and processed one at a time (segmented, detected,
                  saved) until enough total junctions have been found.
                </Typography>
                <FormControlLabel
                  value={PipelineMode.Staged}
                  control={<Radio size="small" />}
                  label="Staged"
                />
                <Typography variant="caption" color="text.secondary" sx={{ ml: 4, mt: -1 }}>
                  A random subsample of tiles is fully segmented first, then junctions are detected
                  in all of them.
                </Typography>
              </RadioGroup>
            </FormControl>

            {settings.pipeline_mode === PipelineMode.Sequential && (
              <TextField
                label="Target total junction count"
                type="number"
                size="small"
                value={settings.sequential_target_junction_count}
                onChange={(e) =>
                  setSettings({ ...settings, sequential_target_junction_count: Number(e.target.value) })
                }
                error={!(settings.sequential_target_junction_count > 0)}
                helperText="Stop once this many replication + reversed forks have been found in total"
                inputProps={{ min: 1 }}
              />
            )}

            {settings.pipeline_mode === PipelineMode.Staged && (
              <TextField
                label="Tile sample percentage"
                type="number"
                size="small"
                value={settings.staged_sample_percentage}
                onChange={(e) =>
                  setSettings({ ...settings, staged_sample_percentage: Number(e.target.value) })
                }
                error={
                  !(settings.staged_sample_percentage > 0 && settings.staged_sample_percentage <= 100)
                }
                helperText="Percentage of not-yet-processed tiles to randomly sample and process"
                inputProps={{ min: 1, max: 100 }}
              />
            )}
          </Box>
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
