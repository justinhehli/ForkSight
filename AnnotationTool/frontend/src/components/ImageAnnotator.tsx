import { useCallback, useEffect, useRef, useState } from "react";
import { Box, IconButton, Tooltip, Typography } from "@mui/material";
import FitScreenIcon from "@mui/icons-material/FitScreen";
import ZoomInIcon from "@mui/icons-material/ZoomIn";
import ZoomOutIcon from "@mui/icons-material/ZoomOut";
import VisibilityIcon from "@mui/icons-material/Visibility";
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff";
import { v4 as uuidv4 } from "uuid";
import { getImageUrl, getMaskUrl } from "../api";
import { JunctionType, sortLabelsForDisplay } from "../types";
import type { ImageAnnotations, Point } from "../types";
import React from "react";

const LABEL_COLORS: Record<string, string> = {
  [JunctionType.ReplicationFork50]: "#a5d6a7",
  [JunctionType.ReplicationFork100]: "#4caf50",
  [JunctionType.ReversedFork50]: "#ef9a9a",
  [JunctionType.ReversedFork100]: "#f44336",
};
// Stable, distinguishable colors for user-created custom labels not in LABEL_COLORS.
const CUSTOM_LABEL_PALETTE = ["#3f51b5", "#9c27b0", "#ff9800", "#009688", "#795548", "#607d8b", "#e91e63", "#00bcd4"];
const hashString = (s: string): number => {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  return Math.abs(h);
};
export const labelColor = (label: string): string =>
  LABEL_COLORS[label] ?? CUSTOM_LABEL_PALETTE[hashString(label) % CUSTOM_LABEL_PALETTE.length];

interface View {
  panX: number;
  panY: number;
  zoom: number;
}

interface Props {
  project: string;
  imageId: string;
  imageName: string;
  annotations: ImageAnnotations;
  onAnnotationsChange: (a: ImageAnnotations) => void;
  selectedPointId: string | null;
  onSelectPoint: (id: string | null) => void;
}

// Memoized so re-renders elsewhere in App don't force this canvas to redraw
const ImageAnnotatorComponent = ({
  project,
  imageId,
  imageName,
  annotations,
  onAnnotationsChange,
  selectedPointId,
  onSelectPoint,
}: Props) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [view, setView] = useState<View>({ panX: 0, panY: 0, zoom: 1 });
  const [natSize, setNatSize] = useState({ w: 1, h: 1 });
  const [showMask, setShowMask] = useState(true);
  const [maskAvailable, setMaskAvailable] = useState(true);

  useEffect(() => {
    setMaskAvailable(true);
  }, [imageId]);

  // keep refs in sync so event-listener closures read fresh values
  const viewRef = useRef(view);
  useEffect(() => {
    viewRef.current = view;
  }, [view]);

  const natRef = useRef(natSize);
  useEffect(() => {
    natRef.current = natSize;
  }, [natSize]);

  const annRef = useRef(annotations);
  useEffect(() => {
    annRef.current = annotations;
  }, [annotations]);

  const onChangeRef = useRef(onAnnotationsChange);
  useEffect(() => {
    onChangeRef.current = onAnnotationsChange;
  }, [onAnnotationsChange]);

  const onSelectRef = useRef(onSelectPoint);
  useEffect(() => {
    onSelectRef.current = onSelectPoint;
  }, [onSelectPoint]);

  // zoom via wheel (non-passive)
  // ====================
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const handler = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const { panX, panY, zoom } = viewRef.current;
      const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
      const newZoom = Math.min(Math.max(zoom * factor, 0.04), 30);
      const ix = (mx - panX) / zoom;
      const iy = (my - panY) / zoom;
      setView({ zoom: newZoom, panX: mx - ix * newZoom, panY: my - iy * newZoom });
    };
    el.addEventListener("wheel", handler, { passive: false });
    return () => el.removeEventListener("wheel", handler);
  }, []);

  // pan / point-drag / click
  // ====================
  type DragMode = "none" | "pan" | "point";
  const drag = useRef<{
    mode: DragMode;
    moved: boolean;
    sx: number;
    sy: number;
    spx: number;
    spy: number;
    pointId: string | null;
  }>({ mode: "none", moved: false, sx: 0, sy: 0, spx: 0, spy: 0, pointId: null });
  const [cursor, setCursor] = useState<string>("crosshair");
  const [draggingPointId, setDraggingPointId] = useState<string | null>(null);

  const [dragPreview, setDragPreview] = useState<{ id: string; x: number; y: number } | null>(null);
  const dragPreviewRef = useRef(dragPreview);
  useEffect(() => {
    dragPreviewRef.current = dragPreview;
  }, [dragPreview]);

  const commitDragPreview = useCallback(() => {
    const preview = dragPreviewRef.current;
    if (!preview) return;
    const cur = annRef.current;
    onChangeRef.current({
      ...cur,
      points: cur.points.map((p) => (p.id === preview.id ? { ...p, x: preview.x, y: preview.y } : p)),
    });
    setDragPreview(null);
  }, []);

  const hitPoint = (clientX: number, clientY: number) => {
    const container = containerRef.current;
    if (!container) return null;
    const rect = container.getBoundingClientRect();
    const sx = clientX - rect.left;
    const sy = clientY - rect.top;
    const { panX, panY, zoom } = viewRef.current;
    return (
      annRef.current.points.find((p) => Math.hypot(sx - (p.x * zoom + panX), sy - (p.y * zoom + panY)) < 10) ?? null
    );
  };

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    const hit = hitPoint(e.clientX, e.clientY);
    if (hit) {
      drag.current = { mode: "point", moved: false, sx: e.clientX, sy: e.clientY, spx: 0, spy: 0, pointId: hit.id };
      setCursor("grabbing");
    } else {
      drag.current = {
        mode: "pan",
        moved: false,
        sx: e.clientX,
        sy: e.clientY,
        spx: viewRef.current.panX,
        spy: viewRef.current.panY,
        pointId: null,
      };
    }
  }, []);

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    const { mode } = drag.current;
    if (mode === "none") {
      // update hover cursor
      setCursor(hitPoint(e.clientX, e.clientY) ? "grab" : "crosshair");
      return;
    }
    const dx = e.clientX - drag.current.sx;
    const dy = e.clientY - drag.current.sy;
    if (!drag.current.moved && (Math.abs(dx) > 4 || Math.abs(dy) > 4)) {
      drag.current.moved = true;
      if (drag.current.mode === "point") setDraggingPointId(drag.current.pointId);
    }
    if (!drag.current.moved) return;

    if (mode === "pan") {
      setView((v) => ({ ...v, panX: drag.current.spx + dx, panY: drag.current.spy + dy }));
    } else if (mode === "point" && drag.current.pointId) {
      const container = containerRef.current;
      if (!container) return;
      const rect = container.getBoundingClientRect();
      const { panX, panY, zoom } = viewRef.current;
      const { w, h } = natRef.current;
      const nx = Math.round(Math.max(0, Math.min(w, (e.clientX - rect.left - panX) / zoom)));
      const ny = Math.round(Math.max(0, Math.min(h, (e.clientY - rect.top - panY) / zoom)));
      setDragPreview({ id: drag.current.pointId, x: nx, y: ny });
    }
  }, []);

  const onMouseUp = useCallback((e: React.MouseEvent) => {
    const { mode, moved, pointId } = drag.current;
    drag.current.mode = "none";
    setDraggingPointId(null);
    setCursor(hitPoint(e.clientX, e.clientY) ? "grab" : "crosshair");

    if (mode === "point") {
      commitDragPreview();
      onSelectRef.current(pointId);
      return;
    }

    if (mode === "pan" && !moved) {
      // bare click on empty space → add point
      const container = containerRef.current;
      if (!container) return;
      const rect = container.getBoundingClientRect();
      const { panX, panY, zoom } = viewRef.current;
      const nx = (e.clientX - rect.left - panX) / zoom;
      const ny = (e.clientY - rect.top - panY) / zoom;
      if (nx < 0 || nx > natRef.current.w || ny < 0 || ny > natRef.current.h) return;
      const newPt: Point = { id: uuidv4(), x: Math.round(nx), y: Math.round(ny), labels: [] };
      const cur = annRef.current;
      onChangeRef.current({ ...cur, points: [...cur.points, newPt] });
      onSelectRef.current(newPt.id);
    }
  }, []);

  // fit to container
  // ====================
  const fitToContainer = useCallback(() => {
    const c = containerRef.current;
    const img = imgRef.current;
    if (!c || !img || !img.naturalWidth) return;
    const { clientWidth: cw, clientHeight: ch } = c;
    const nw = img.naturalWidth,
      nh = img.naturalHeight;
    const scale = Math.min(cw / nw, ch / nh) * 0.95;
    setView({ zoom: scale, panX: (cw - nw * scale) / 2, panY: (ch - nh * scale) / 2 });
  }, []);

  const onImageLoad = () => {
    const img = imgRef.current;
    if (!img) return;
    setNatSize({ w: img.naturalWidth, h: img.naturalHeight });
    fitToContainer();
  };

  // reset when image changes
  useEffect(() => {
    fitToContainer();
  }, [imageId, fitToContainer]);

  // delete selected point with Delete key
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      if ((e.key === "Delete" || e.key === "d" || e.key === "D") && selectedPointId) {
        const cur = annRef.current;
        onChangeRef.current({ ...cur, points: cur.points.filter((p) => p.id !== selectedPointId) });
        onSelectRef.current(null);
      }
      if (e.key === "m" || e.key === "M") setShowMask((s) => !s);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [selectedPointId]);

  const { panX, panY, zoom } = view;
  const { w: nw, h: nh } = natSize;

  return (
    <Box
      ref={containerRef}
      sx={{
        position: "relative",
        width: "100%",
        height: "100%",
        overflow: "hidden",
        bgcolor: "action.disabledBackground",
        cursor,
        userSelect: "none",
      }}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={() => {
        commitDragPreview();
        drag.current.mode = "none";
        setDraggingPointId(null);
        setCursor("crosshair");
      }}
    >
      {/* Image */}
      <img
        ref={imgRef}
        key={imageId}
        src={getImageUrl(project, imageId)}
        onLoad={onImageLoad}
        draggable={false}
        alt={imageName}
        style={{
          position: "absolute",
          left: panX,
          top: panY,
          width: nw * zoom,
          height: nh * zoom,
          imageRendering: zoom > 3 ? "pixelated" : "auto",
          pointerEvents: "none",
        }}
      />

      {/* Predicted segmentation mask overlay */}
      <img
        key={`${imageId}-mask`}
        src={getMaskUrl(project, imageId)}
        onLoad={() => setMaskAvailable(true)}
        onError={() => setMaskAvailable(false)}
        draggable={false}
        alt=""
        style={{
          position: "absolute",
          left: panX,
          top: panY,
          width: nw * zoom,
          height: nh * zoom,
          imageRendering: zoom > 3 ? "pixelated" : "auto",
          pointerEvents: "none",
          opacity: showMask && maskAvailable ? 1 : 0,
        }}
      />

      {/* SVG annotation overlay — fixed to container, points computed in screen space */}
      <svg
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
          overflow: "visible",
          pointerEvents: "none",
        }}
      >
        {annotations.points.map((p) => {
          const usePreview = dragPreview !== null && dragPreview.id === p.id;
          const px = usePreview ? dragPreview.x : p.x;
          const py = usePreview ? dragPreview.y : p.y;
          const cx = px * zoom + panX;
          const cy = py * zoom + panY;
          const selected = p.id === selectedPointId && p.id !== draggingPointId;
          // multiple labels are drawn as a small cluster of dots around the point
          const labels = p.labels.length > 0 ? sortLabelsForDisplay(p.labels) : [""];
          const n = labels.length;
          return (
            <g key={p.id}>
              {labels.map((l, i) => {
                const angle = n > 1 ? (2 * Math.PI * i) / n - Math.PI / 2 : 0;
                const offset = n > 1 ? 5 : 0;
                const dx = cx + Math.cos(angle) * offset;
                const dy = cy + Math.sin(angle) * offset;
                const color = l ? labelColor(l) : "#9e9e9e";
                return (
                  <circle
                    key={i}
                    cx={dx}
                    cy={dy}
                    r={n > 1 ? 5 : 7}
                    fill={color}
                    fillOpacity={0.85}
                    stroke="#fff"
                    strokeWidth={1.5}
                  />
                );
              })}
              {/* outer ring when selected */}
              {selected && (
                <React.Fragment>
                  <circle cx={cx} cy={cy} r={13} fill="none" stroke="#fff" strokeWidth={2} />
                  <text
                    x={cx}
                    y={cy - 11}
                    fontSize={11}
                    fontFamily="sans-serif"
                    fontWeight={600}
                    fill="#fff"
                    stroke="#000"
                    strokeWidth={2.5}
                    paintOrder="stroke"
                    textAnchor="middle"
                    style={{ pointerEvents: "none", userSelect: "none" }}
                  >
                    {p.labels.length > 0 ? sortLabelsForDisplay(p.labels).join(", ") : "unlabeled"}
                  </text>
                </React.Fragment>
              )}
            </g>
          );
        })}
      </svg>

      {/* Zoom controls */}
      <Box
        // stop mousedown/up from reaching the container's pan/click handlers - otherwise a
        // click here can register as a click on the image underneath and drop a new point
        onMouseDown={(e) => e.stopPropagation()}
        onMouseUp={(e) => e.stopPropagation()}
        sx={{
          position: "absolute",
          bottom: 12,
          right: 12,
          display: "flex",
          flexDirection: "column",
          gap: 0.5,
          alignItems: "center",
        }}
      >
        <Tooltip
          title={
            maskAvailable ? `${showMask ? "Hide" : "Show"} segmentation mask (M)` : "No segmentation mask available"
          }
          placement="left"
        >
          <span>
            <IconButton
              size="small"
              onClick={() => setShowMask((s) => !s)}
              disabled={!maskAvailable}
              sx={{ bgcolor: "rgba(0,0,0,0.55)", color: "#fff", "&:hover": { bgcolor: "rgba(0,0,0,0.8)" } }}
            >
              {showMask ? <VisibilityIcon fontSize="small" /> : <VisibilityOffIcon fontSize="small" />}
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title="Zoom in" placement="left">
          <IconButton
            size="small"
            onClick={() => setView((v) => ({ ...v, zoom: v.zoom * 1.2 }))}
            sx={{ bgcolor: "rgba(0,0,0,0.55)", color: "#fff", "&:hover": { bgcolor: "rgba(0,0,0,0.8)" } }}
          >
            <ZoomInIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title="Zoom out" placement="left">
          <IconButton
            size="small"
            onClick={() => setView((v) => ({ ...v, zoom: v.zoom / 1.2 }))}
            sx={{ bgcolor: "rgba(0,0,0,0.55)", color: "#fff", "&:hover": { bgcolor: "rgba(0,0,0,0.8)" } }}
          >
            <ZoomOutIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title="Fit to view" placement="left">
          <IconButton
            size="small"
            onClick={fitToContainer}
            sx={{ bgcolor: "rgba(0,0,0,0.55)", color: "#fff", "&:hover": { bgcolor: "rgba(0,0,0,0.8)" } }}
          >
            <FitScreenIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Typography
          variant="caption"
          sx={{
            bgcolor: "rgba(0,0,0,0.55)",
            color: "#fff",
            px: 0.75,
            borderRadius: 0.5,
            lineHeight: "20px",
            fontSize: 11,
          }}
        >
          {Math.round(zoom * 100)}%
        </Typography>
      </Box>
    </Box>
  );
};

export default React.memo(ImageAnnotatorComponent);
