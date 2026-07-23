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
  InputAdornment,
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
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Fork detection settings</DialogTitle>
      <DialogContent dividers>
        <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
          Choose how the automatic fork detection pipeline processes project tiles. Applies the next time detection is
          run.
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
                  label="Subsample size (percentage of all tiles)"
                  type="number"
                  size="small"
                  sx={{ ml: 4, width: 280 }}
                  value={settings.staged_sample_percentage}
                  onChange={(e) => setSettings({ ...settings, staged_sample_percentage: Number(e.target.value) })}
                  error={!(settings.staged_sample_percentage > 0 && settings.staged_sample_percentage <= 100)}
                  inputProps={{ min: 1, max: 100 }}
                  InputProps={{ endAdornment: <InputAdornment position="end">%</InputAdornment> }}
                />
              </Box>
            </RadioGroup>
          </FormControl>
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
