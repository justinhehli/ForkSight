import { memo, useCallback, useMemo, useState } from "react";
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
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Typography,
} from "@mui/material";
import { List as VirtualList, type RowComponentProps } from "react-window";
import { FORK_GROUPS } from "../types";
import type { ImageAnnotations, ImageMeta } from "../types";

const REPLICATION_FORK_GROUP = FORK_GROUPS.find((g) => g.name === "Replication Fork")!;
const REVERSED_FORK_GROUP = FORK_GROUPS.find((g) => g.name === "Reversed Fork")!;

const ROW_HEIGHT = 28;

interface VisibleImage {
  img: ImageMeta;
  idx: number;
}

interface RowData {
  visibleImages: VisibleImage[];
  imageIdx: number;
  imageAnnotations: Record<string, ImageAnnotations>;
  disabled: boolean;
  onSelectImage: (idx: number) => void;
  onArchiveImage: (imageId: string) => void;
}

// Only the rows actually scrolled into view are ever mounted (see the virtualized List
// below) - real projects here can have 700+ tiles, and mounting a MUI ListItemButton per
// row for all of them (e.g. switching the "Annotated" filter to "All") used to cause
// multi-second UI freezes that had nothing to do with image loading.
function ImageListRow({
  index,
  style,
  visibleImages,
  imageIdx,
  imageAnnotations,
  disabled,
  onSelectImage,
  onArchiveImage,
}: RowComponentProps<RowData>) {
  const { img, idx } = visibleImages[index];
  const selected = idx === imageIdx;
  const imgAnnotations = imageAnnotations[img.id];

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
      style={style}
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
                onArchiveImage(img.id);
              }}
            >
              <DeleteOutlineIcon sx={{ fontSize: 15 }} />
            </IconButton>
          </span>
        </Tooltip>
      }
    >
      <ListItemButton
        dense
        selected={selected}
        onClick={() => onSelectImage(idx)}
        sx={{ py: 0.25, pr: 4.5 }}
        disabled={disabled}
      >
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
}

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

type ImageFilter = "all" | "unprocessed" | "processed" | "annotated";

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
  const [filter, setFilter] = useState<ImageFilter>("all");

  const visibleImages = useMemo(
    () =>
      images.map((img, idx) => ({ img, idx })).filter(({ img }) => {
        switch (filter) {
          case "processed":
            return img.processed;
          case "unprocessed":
            return !img.processed;
          case "annotated":
            return (imageAnnotations[img.id]?.points.length ?? 0) > 0;
          default:
            return true;
        }
      }),
    [images, filter, imageAnnotations],
  );

  const rowProps = useMemo<RowData>(
    () => ({ visibleImages, imageIdx, imageAnnotations, disabled, onSelectImage, onArchiveImage }),
    [visibleImages, imageIdx, imageAnnotations, disabled, onSelectImage, onArchiveImage],
  );

  // stable identity required by react-window - keying by image id (rather than the
  // default row index) keeps a row's DOM/ripple state tied to the actual image across
  // filter changes, instead of getting reused for whatever image lands on that index
  const rowKey = useCallback((index: number, data: RowData) => data.visibleImages[index].img.id, []);

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
      <Box sx={{ px: 1.5, pt: 0.75, display: "flex", flexDirection: "column", gap: 0.5 }}>
        <ToggleButtonGroup
          value={filter}
          exclusive
          fullWidth
          size="small"
          onChange={(_e, value: ImageFilter | null) => value && setFilter(value)}
          sx={{ "& .MuiToggleButton-root": { py: 0.25, px: 1, fontSize: 11, textTransform: "none" } }}
        >
          <ToggleButton value="all">All</ToggleButton>
          <ToggleButton value="unprocessed">Unprocessed</ToggleButton>
        </ToggleButtonGroup>
        <ToggleButtonGroup
          value={filter}
          exclusive
          fullWidth
          size="small"
          onChange={(_e, value: ImageFilter | null) => value && setFilter(value)}
          sx={{ "& .MuiToggleButton-root": { py: 0.25, px: 1, fontSize: 11, textTransform: "none" } }}
        >
          <ToggleButton value="processed">Processed</ToggleButton>
          <ToggleButton value="annotated">Annotated</ToggleButton>
        </ToggleButtonGroup>
      </Box>
      <Box sx={{ flex: 1, minHeight: 0, mt: 0.5 }}>
        {visibleImages.length === 0 ? (
          <Typography variant="body2" color="text.secondary" sx={{ px: 1.5, py: 1 }}>
            No images match this filter.
          </Typography>
        ) : (
          <VirtualList
            rowComponent={ImageListRow}
            rowCount={visibleImages.length}
            rowHeight={ROW_HEIGHT}
            rowProps={rowProps}
            rowKey={rowKey}
            style={{ height: "100%" }}
          />
        )}
      </Box>
    </>
  );
});

export default ImageListPanel;
