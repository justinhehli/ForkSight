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
  TextField,
  Typography,
} from "@mui/material";
import { getPipelineSettings } from "../api";
import { PipelineMode } from "../types";

interface Props {
  open: boolean;
  mode: PipelineMode | null | undefined;
  onClose: () => void;
  onConfirm: (amount: number) => void;
}

const AdditionalDetectionDialog = ({ open, mode, onClose, onConfirm }: Props) => {
  const [amount, setAmount] = useState<number | "">("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isStaged = mode === PipelineMode.Staged;

  useEffect(() => {
    if (!open) return;

    setLoading(true);
    setError(null);

    getPipelineSettings()
      .then((s) => setAmount(isStaged ? s.staged_sample_percentage : s.sequential_target_junction_count))
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const valid = typeof amount === "number" && amount > 0 && (!isStaged || amount <= 100);

  return (
    <Dialog open={open} onClose={onClose} maxWidth="xs" fullWidth>
      <DialogTitle>Run Additional Automatic Fork Detection</DialogTitle>
      <DialogContent dividers>
        <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 2 }}>
          {isStaged
            ? "Detect junctions in another random sample of tiles."
            : "Keep sampling new tiles until a pre-defined number of additional junctions have been found."}
        </Typography>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        {loading ? (
          <Box sx={{ display: "flex", justifyContent: "center", py: 3 }}>
            <CircularProgress size={20} />
          </Box>
        ) : (
          <TextField
            autoFocus
            label={isStaged ? "Percentage of total images to process" : "Additional junctions to find"}
            type="number"
            size="small"
            fullWidth
            value={amount}
            onChange={(e) => setAmount(e.target.value === "" ? "" : Number(e.target.value))}
            error={amount !== "" && !valid}
            inputProps={isStaged ? { min: 1, max: 100 } : { min: 1 }}
          />
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button variant="contained" disabled={loading || !valid} onClick={() => onConfirm(amount as number)}>
          Run
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default AdditionalDetectionDialog;
