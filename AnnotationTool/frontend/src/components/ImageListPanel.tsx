import { memo, useMemo, useState } from "react";
import {
  CheckCircle as CheckCircleIcon,
  Circle as CircleIcon,
  DeleteOutline as DeleteOutlineIcon,
  HighlightOff as HighlightOffIcon,
  RadioButtonUnchecked as RadioButtonUncheckedIcon,
} from "@mui/icons-material";
import {
  Box,
  Chip,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Typography,
} from "@mui/material";
import { FORK_GROUPS } from "../types";
import type { ImageAnnotations, ImageMeta } from "../types";

const REPLICATION_FORK_GROUP = FORK_GROUPS.find((g) => g.name === "Replication Fork")!;
const REVERSED_FORK_GROUP = FORK_GROUPS.find((g) => g.name === "Reversed Fork")!;

interface RowProps {
  img: ImageMeta;
  idx: number;
  selected: boolean;
  imgAnnotations: ImageAnnotations | undefined;
  disabled: boolean;
  onSelect: (idx: number) => void;
  onArchive: (imageId: string) => void;
}

// Memoized per-row: editing the currently-open image only changes that image's entry in
// `annotations.images` (see handleAnnotationsChange in App.tsx, which spreads the other
// entries unchanged), so every other row here keeps the exact same props and skips
// re-rendering. Without this, every point drag/add/delete re-rendered the whole list.
const ImageListRow = memo(function ImageListRow({ img, idx, selected, imgAnnotations, disabled, onSelect, onArchive }: RowProps) {
  let hasReplication = false;
  let hasReversed = false;
  const points = imgAnnotations?.points ?? [];
  for (const p of points) {
    if (p.labels.includes(REPLICATION_FORK_GROUP.fifty) || p.labels.includes(REPLICATION_FORK_GROUP.hundred)) {
      hasReplication = true;
    }
    if (p.labels.includes(REVERSED_FORK_GROUP.fifty) || p.labels.includes(REVERSED_FORK_GROUP.hundred)) {
      hasReversed = true;
    }
  }
  const hasAnyAnnotations = points.length > 0;

  return (
    <ListItem
      disablePadding
      secondaryAction={
        <Tooltip title="Archive image (can be restored later)">
          <span>
            <IconButton
              edge="end"
              size="small"
              tabIndex={-1}
              disabled={disabled}
              onClick={(e) => {
                e.stopPropagation();
                onArchive(img.id);
              }}
            >
              <DeleteOutlineIcon sx={{ fontSize: 15 }} />
            </IconButton>
          </span>
        </Tooltip>
      }
    >
      <ListItemButton selected={selected} onClick={() => onSelect(idx)} sx={{ py: 0.25, pr: 4.5 }} disabled={disabled}>
        <ListItemIcon sx={{ minWidth: 28 }}>
          {img.processed ? (
            <CheckCircleIcon sx={{ fontSize: 16 }} color="success" />
          ) : (
            <RadioButtonUncheckedIcon sx={{ fontSize: 16 }} color="disabled" />
          )}
        </ListItemIcon>
        <ListItemText primary={img.name} primaryTypographyProps={{ variant: "body2", noWrap: true, fontSize: 12 }} />
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.25, ml: 0.5, flexShrink: 0 }}>
          {!hasAnyAnnotations && (
            <Tooltip title="No annotations">
              <HighlightOffIcon sx={{ fontSize: 14 }} color="disabled" />
            </Tooltip>
          )}
          {hasReplication && (
            <Tooltip title="Has replication forks">
              <CircleIcon sx={{ fontSize: 11, color: REPLICATION_FORK_GROUP.color }} />
            </Tooltip>
          )}
          {hasReversed && (
            <Tooltip title="Has reversed forks">
              <CircleIcon sx={{ fontSize: 11, color: REVERSED_FORK_GROUP.color }} />
            </Tooltip>
          )}
        </Box>
      </ListItemButton>
    </ListItem>
  );
});

interface Props {
  images: ImageMeta[];
  archivedCount: number;
  imageAnnotations: Record<string, ImageAnnotations>;
  imageIdx: number;
  processedCount: number;
  disabled: boolean;
  onSelectImage: (idx: number) => void;
  onArchiveImage: (imageId: string) => void;
  onShowArchived: () => void;
}

type ProcessedFilter = "all" | "unprocessed" | "processed";

const ImageListPanel = memo(function ImageListPanel({
  images,
  archivedCount,
  imageAnnotations,
  imageIdx,
  processedCount,
  disabled,
  onSelectImage,
  onArchiveImage,
  onShowArchived,
}: Props) {
  const [filter, setFilter] = useState<ProcessedFilter>("all");

  const visibleImages = useMemo(
    () =>
      images
        .map((img, idx) => ({ img, idx }))
        .filter(({ img }) => filter === "all" || (filter === "processed" ? img.processed : !img.processed)),
    [images, filter],
  );

  return (
    <>
      <Box sx={{ px: 1.5, pt: 1, display: "flex", alignItems: "center", gap: 1 }}>
        <Typography variant="subtitle2">Images</Typography>
        <Chip label={`${processedCount}/${images.length}`} size="small" color="primary" />
        {archivedCount > 0 && (
          <Tooltip title="View / restore archived images">
            <Chip
              label={`${archivedCount} archived`}
              size="small"
              variant="outlined"
              onClick={onShowArchived}
              sx={{ cursor: "pointer" }}
            />
          </Tooltip>
        )}
      </Box>
      <Box sx={{ px: 1.5, pt: 0.75 }}>
        <ToggleButtonGroup
          value={filter}
          exclusive
          size="small"
          onChange={(_e, value: ProcessedFilter | null) => value && setFilter(value)}
          sx={{ "& .MuiToggleButton-root": { py: 0.25, px: 1, fontSize: 11, textTransform: "none" } }}
        >
          <ToggleButton value="all">All</ToggleButton>
          <ToggleButton value="unprocessed">Unprocessed</ToggleButton>
          <ToggleButton value="processed">Processed</ToggleButton>
        </ToggleButtonGroup>
      </Box>
      <List dense disablePadding sx={{ flex: 1, overflowY: "auto", mt: 0.5 }}>
        {visibleImages.length === 0 && (
          <Typography variant="body2" color="text.secondary" sx={{ px: 1.5, py: 1 }}>
            No images match this filter.
          </Typography>
        )}
        {visibleImages.map(({ img, idx }) => (
          <ImageListRow
            key={img.id}
            img={img}
            idx={idx}
            selected={idx === imageIdx}
            imgAnnotations={imageAnnotations[img.id]}
            disabled={disabled}
            onSelect={onSelectImage}
            onArchive={onArchiveImage}
          />
        ))}
      </List>
    </>
  );
});

export default ImageListPanel;
