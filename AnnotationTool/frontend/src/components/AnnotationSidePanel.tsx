import { memo, useState } from "react";
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  ExpandLess as ExpandLessIcon,
  ExpandMore as ExpandMoreIcon,
} from "@mui/icons-material";
import {
  Box,
  Button,
  Chip,
  Divider,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import { labelColor } from "./ImageAnnotator";
import { FORK_GROUPS, sortLabelsForDisplay } from "../types";
import type { ForkGroup, Point } from "../types";
import React from "react";

interface RowProps {
  point: Point;
  index: number;
  selected: boolean;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

// Memoized per-row: editing one point only replaces that point's own object in the array
const PointListRow = memo(function PointListRow({ point, index, selected, onSelect, onDelete }: RowProps) {
  const sortedLabels = sortLabelsForDisplay(point.labels);
  return (
    <ListItem
      disablePadding
      secondaryAction={
        <IconButton edge="end" size="small" onClick={() => onDelete(point.id)} tabIndex={-1}>
          <DeleteIcon sx={{ fontSize: 15 }} />
        </IconButton>
      }
    >
      <ListItemButton selected={selected} onClick={() => onSelect(point.id)} sx={{ py: 0.25, pl: 1, pr: 4 }}>
        <ListItemIcon sx={{ minWidth: 22 }}>
          <Box sx={{ display: "flex", gap: 0.25, flexWrap: "wrap", maxWidth: 14 }}>
            {(sortedLabels.length > 0 ? sortedLabels : ["#9e9e9e"]).map((l, li) => (
              <Box
                key={li}
                sx={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  bgcolor: sortedLabels.length > 0 ? labelColor(l) : l,
                  flexShrink: 0,
                }}
              />
            ))}
          </Box>
        </ListItemIcon>
        <Tooltip title={sortedLabels.length > 0 ? sortedLabels.join(", ") : "(unlabeled)"} placement="top">
          <ListItemText
            primary={`${index + 1}. ${sortedLabels.length > 0 ? sortedLabels.join(", ") : "(unlabeled)"}`}
            secondary={`${Math.round(point.x)}, ${Math.round(point.y)}`}
            primaryTypographyProps={{ variant: "body2", fontSize: 12, noWrap: true }}
            secondaryTypographyProps={{ variant: "caption", fontFamily: "monospace", fontSize: 10 }}
            sx={{ minWidth: 0 }}
          />
        </Tooltip>
      </ListItemButton>
    </ListItem>
  );
});

interface Props {
  points: Point[];
  selectedPointId: string | null;
  additionalLabels: string[];
  onTogglePoint: (id: string) => void;
  onDeletePoint: (id: string) => void;
  onCycleForkGroup: (id: string, group: ForkGroup) => void;
  onToggleLabel: (id: string, label: string) => void;
  onAddCustomLabel: (label: string) => void;
  onDeleteCustomLabel: (label: string) => void;
}

// Owns its own text-input and collapsible-instructions state so typing a new label only re-renders this side panel
const AnnotationSidePanel = memo(function AnnotationSidePanel({
  points,
  selectedPointId,
  additionalLabels,
  onTogglePoint,
  onDeletePoint,
  onCycleForkGroup,
  onToggleLabel,
  onAddCustomLabel,
  onDeleteCustomLabel,
}: Props) {
  const [newLabelInput, setNewLabelInput] = useState("");
  const [showShortcuts, setShowShortcuts] = useState(true);
  const selectedPoint = points.find((p) => p.id === selectedPointId);

  const handleAddCustomLabel = () => {
    const label = newLabelInput.trim();
    if (!label) return;
    onAddCustomLabel(label);
    setNewLabelInput("");
  };

  return (
    <>
      <Box
        sx={{ px: 1.5, py: 1, borderBottom: 1, borderColor: "divider", display: "flex", alignItems: "center", gap: 1 }}
      >
        <Typography variant="subtitle2">Annotations</Typography>
        <Chip label={points.length} size="small" />
      </Box>

      <List dense disablePadding sx={{ flex: 1, minHeight: 48, overflowY: "auto" }}>
        {points.map((p, i) => (
          <PointListRow
            key={p.id}
            point={p}
            index={i}
            selected={p.id === selectedPointId}
            onSelect={onTogglePoint}
            onDelete={onDeletePoint}
          />
        ))}
      </List>

      {selectedPoint && (
        <React.Fragment>
          <Divider />
          <Box sx={{ p: 1.5, display: "flex", flexDirection: "column", minHeight: 0 }}>
            <Box sx={{ minHeight: 0, overflowY: "auto" }}>
              <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5, mb: 1 }}>
                {FORK_GROUPS.map((group) => {
                  const active = selectedPoint.labels.includes(group.hundred)
                    ? group.hundred
                    : selectedPoint.labels.includes(group.fifty)
                      ? group.fifty
                      : null;
                  const confidence = active === group.hundred ? "100%" : active === group.fifty ? "50%" : null;
                  // 50% confidence uses the same lighter shade as the point marker on the canvas
                  const activeColor = active ? labelColor(active) : group.color;
                  return (
                    <Button
                      key={group.name}
                      size="small"
                      variant={active ? "contained" : "outlined"}
                      onClick={() => onCycleForkGroup(selectedPoint.id, group)}
                      sx={{
                        textTransform: "none",
                        justifyContent: "flex-start",
                        fontSize: 12,
                        bgcolor: active ? activeColor : undefined,
                        borderColor: activeColor,
                        color: active ? "#fff" : activeColor,
                        "&:hover": { bgcolor: activeColor, borderColor: activeColor, color: "#fff" },
                      }}
                    >
                      <Box
                        sx={{
                          width: 10,
                          height: 10,
                          borderRadius: "50%",
                          bgcolor: active ? "#fff" : activeColor,
                          mr: 1,
                          flexShrink: 0,
                        }}
                      />
                      {group.name}
                      {confidence ? ` (${confidence})` : ""}
                    </Button>
                  );
                })}
              </Box>

              {additionalLabels.length > 0 && (
                <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5, mb: 1 }}>
                  {additionalLabels.map((l) => {
                    const active = selectedPoint.labels.includes(l);
                    const color = labelColor(l);
                    return (
                      <Box key={l} sx={{ display: "flex", alignItems: "center", gap: 0.25, minWidth: 0 }}>
                        <Tooltip title={l} placement="top">
                          <Button
                            size="small"
                            variant={active ? "contained" : "outlined"}
                            onClick={() => onToggleLabel(selectedPoint.id, l)}
                            sx={{
                              flex: 1,
                              minWidth: 0,
                              textTransform: "none",
                              justifyContent: "flex-start",
                              fontSize: 12,
                              bgcolor: active ? color : undefined,
                              borderColor: color,
                              color: active ? "#fff" : color,
                              "&:hover": { bgcolor: color, borderColor: color, color: "#fff" },
                            }}
                          >
                            <Box
                              sx={{
                                width: 10,
                                height: 10,
                                borderRadius: "50%",
                                bgcolor: active ? "#fff" : color,
                                mr: 1,
                                flexShrink: 0,
                              }}
                            />
                            <Box
                              component="span"
                              sx={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                            >
                              {l}
                            </Box>
                          </Button>
                        </Tooltip>
                        <Tooltip title="Delete label from project">
                          <IconButton size="small" onClick={() => onDeleteCustomLabel(l)}>
                            <DeleteIcon sx={{ fontSize: 15 }} />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    );
                  })}
                </Box>
              )}
            </Box>

            <Box sx={{ display: "flex", gap: 0.5, flexShrink: 0 }}>
              <TextField
                size="small"
                placeholder="New label"
                value={newLabelInput}
                onChange={(e) => setNewLabelInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleAddCustomLabel();
                }}
                sx={{ flex: 1 }}
                inputProps={{ style: { fontSize: 12, padding: "4px 8px" } }}
              />
              <Tooltip title="Add project-wide custom label">
                <span>
                  <IconButton size="small" onClick={handleAddCustomLabel} disabled={!newLabelInput.trim()}>
                    <AddIcon fontSize="small" />
                  </IconButton>
                </span>
              </Tooltip>
            </Box>
          </Box>
        </React.Fragment>
      )}

      <Divider />
      <Box
        sx={{
          px: 1.5,
          py: 0.75,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          cursor: "pointer",
          flexShrink: 0,
        }}
        onClick={() => setShowShortcuts((s) => !s)}
      >
        <Typography variant="caption" color="text.secondary" fontWeight={600}>
          Instructions
        </Typography>
        <IconButton size="small" sx={{ p: 0.25 }}>
          {showShortcuts ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
        </IconButton>
      </Box>
      {showShortcuts && (
        <Box sx={{ px: 1.5, pb: 1.5, flexShrink: 0 }}>
          <Typography variant="caption" color="text.secondary" lineHeight={1.6}>
            <b>Click</b> image: add point
            <br />
            <b>Click / drag point</b>: select / move
            <br />
            <b>Drag</b>: pan &nbsp;|&nbsp; <b>Scroll</b>: zoom
            <br />
            <b>←/→</b>: navigate &nbsp;|&nbsp; <b>P</b>: set processed
            <br />
            <b>D / Delete</b>: delete selected point
            <br />
            <b>M</b>: toggle segmentation mask
            <br />
            <b>Hold H</b>: Hide points
          </Typography>
        </Box>
      )}
    </>
  );
});

export default AnnotationSidePanel;
