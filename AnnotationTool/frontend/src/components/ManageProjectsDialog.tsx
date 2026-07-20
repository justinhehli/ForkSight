import { useEffect, useState } from "react";
import { ContentCopy as ContentCopyIcon } from "@mui/icons-material";
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
  FormControlLabel,
  IconButton,
  List,
  ListItem,
  Tooltip,
  Typography,
} from "@mui/material";
import { getProjectCandidates, getProjectFolderPath, setRegisteredProjects } from "../api";
import type { ProjectCandidate } from "../types";

interface Props {
  open: boolean;
  onClose: () => void;
  // `mightRestart` is true if this save registered a project that wasn't previously
  // registered - the backend may need to restart itself to gain write access to it.
  onSaved: (mightRestart: boolean) => void;
}

const ManageProjectsDialog = ({ open, onClose, onSaved }: Props) => {
  const [candidates, setCandidates] = useState<ProjectCandidate[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copiedName, setCopiedName] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;

    setLoading(true);
    setError(null);

    getProjectCandidates()
      .then((cs) => {
        setCandidates(cs);
        setSelected(new Set(cs.filter((c) => c.registered).map((c) => c.name)));
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [open]);

  const toggle = (name: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(name)) {
        next.delete(name);
      } else {
        next.add(name);
      }

      return next;
    });
  };

  const handleCopyPath = async (name: string) => {
    try {
      const { path } = await getProjectFolderPath(name);
      await navigator.clipboard.writeText(path);
      setCopiedName(name);
      setTimeout(() => setCopiedName((prev) => (prev === name ? null : prev)), 1500);
    } catch (e) {
      setError(String(e));
    }
  };

  const hasNewSelection = candidates.some((c) => selected.has(c.name) && !c.registered);

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    const mightRestart = hasNewSelection;

    try {
      await setRegisteredProjects(Array.from(selected));
      onSaved(mightRestart);
      onClose();
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Manage projects</DialogTitle>
      <DialogContent dividers sx={{ maxHeight: 420 }}>
        <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
          Select which directories should show up as projects. Directories with a <code>*.mapsxml</code>{" "} 
          file and a <code>./LayersData/highmag</code>{" "} sub-directory are considered as candidates.
        </Typography>
        {error && (
          <Alert severity="error" sx={{ mb: 1 }}>
            {error}
          </Alert>
        )}
        {hasNewSelection && (
          <Alert severity="warning" sx={{ mb: 1 }}>
            Adding a new project restarts the backend so it can gain write access to it.
            The app will be briefly unavailable while it restarts.
          </Alert>
        )}
        {loading && (
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 1.5, py: 4 }}>
            <CircularProgress size={20} />
            <Typography variant="body2" color="text.secondary">
              Scanning for project directories…
            </Typography>
          </Box>
        )}
        {!loading && candidates.length === 0 && !error && (
          <Typography variant="body2" color="text.secondary">
            No directories found.
          </Typography>
        )}
        {!loading && (
          <List dense disablePadding>
            {candidates.map((c) => (
              <ListItem key={c.name} disablePadding>
                <Box sx={{ display: "flex", alignItems: "center", width: "100%" }}>
                  <Tooltip title={c.valid ? "" : "No LayersData/highmag folder found in this directory"}>
                    <span>
                      <FormControlLabel
                        control={
                          <Checkbox
                            size="small"
                            checked={selected.has(c.name)}
                            disabled={!c.valid}
                            onChange={() => toggle(c.name)}
                          />
                        }
                        label={c.name}
                      />
                    </span>
                  </Tooltip>
                  <Tooltip title={copiedName === c.name ? "Copied!" : "Copy directory path"}>
                    <IconButton size="small" sx={{ ml: "auto" }} onClick={() => handleCopyPath(c.name)}>
                      <ContentCopyIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Box>
              </ListItem>
            ))}
          </List>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={saving}>
          Cancel
        </Button>
        <Button
          variant="contained"
          onClick={handleSave}
          disabled={loading || saving}
          startIcon={saving ? <CircularProgress size={16} color="inherit" /> : undefined}
        >
          {saving ? "Saving…" : "Save"}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ManageProjectsDialog;
