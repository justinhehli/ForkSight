import { useEffect, useState } from "react";
import {
  Alert,
  Button,
  Checkbox,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControlLabel,
  List,
  ListItem,
  Tooltip,
  Typography,
} from "@mui/material";
import { getProjectCandidates, setRegisteredProjects } from "../api";
import type { ProjectCandidate } from "../types";

interface Props {
  open: boolean;
  onClose: () => void;
  onSaved: () => void;
}

const ManageProjectsDialog = ({ open, onClose, onSaved }: Props) => {
  const [candidates, setCandidates] = useState<ProjectCandidate[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    
    try {
      await setRegisteredProjects(Array.from(selected));
      onSaved();
      onClose();
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="xs" fullWidth>
      <DialogTitle>Manage projects</DialogTitle>
      <DialogContent dividers sx={{ maxHeight: 420 }}>
        <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
          Select which directories should show up as projects.<br />Directories with a <code>*.mapsxml</code>{" "} 
          file and a <code>LayersData/highmag</code>{" "} sub-directory are considered as candidates.
        </Typography>
        {error && (
          <Alert severity="error" sx={{ mb: 1 }}>
            {error}
          </Alert>
        )}
        {!loading && candidates.length === 0 && !error && (
          <Typography variant="body2" color="text.secondary">
            No directories found.
          </Typography>
        )}
        <List dense disablePadding>
          {candidates.map((c) => (
            <ListItem key={c.name} disablePadding>
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
            </ListItem>
          ))}
        </List>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={saving}>
          Cancel
        </Button>
        <Button variant="contained" onClick={handleSave} disabled={loading || saving}>
          Save
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ManageProjectsDialog;
