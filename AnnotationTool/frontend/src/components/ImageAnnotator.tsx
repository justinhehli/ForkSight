import { useCallback, useEffect, useRef, useState } from "react";
import { Box, IconButton, Tooltip, Typography } from "@mui/material";
import FitScreenIcon from "@mui/icons-material/FitScreen";
import ZoomInIcon from "@mui/icons-material/ZoomIn";
import ZoomOutIcon from "@mui/icons-material/ZoomOut";
import VisibilityIcon from "@mui/icons-material/Visibility";
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff";
import { v4 as uuidv4 } from "uuid";
import { getImageUrl, getMaskUrl } from "../api";
import { JunctionType } from "../types";
import type { ImageAnnotations, Point } from "../types";
import React from "react";

const LABEL_COLORS: Record<JunctionType, string> = {
  [JunctionType.ReplicationFork]: "#4caf50",
  [JunctionType.ReversedFork]: "#f44336",
};
export const labelColor = (label: JunctionType): string => LABEL_COLORS[label] ?? "#9e9e9e";

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
  selectedLabel: JunctionType;
  selectedPointId: string | null;
  onSelectPoint: (id: string | null) => void;
}

const ImageAnnotator = ({
  project,
  imageId,
  imageName,
  annotations,
  onAnnotationsChange,
  selectedLabel,
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

  const labelRef = useRef(selectedLabel);
  useEffect(() => {
    labelRef.current = selectedLabel;
  }, [selectedLabel]);

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
      const cur = annRef.current;
      onChangeRef.current({
        ...cur,
        points: cur.points.map((p) => (p.id === drag.current.pointId ? { ...p, x: nx, y: ny } : p)),
      });
    }
  }, []);

  const onMouseUp = useCallback((e: React.MouseEvent) => {
    const { mode, moved, pointId } = drag.current;
    drag.current.mode = "none";
    setDraggingPointId(null);
    setCursor(hitPoint(e.clientX, e.clientY) ? "grab" : "crosshair");

    if (mode === "point") {
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
      const newPt: Point = { id: uuidv4(), x: Math.round(nx), y: Math.round(ny), label: labelRef.current };
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
          const cx = p.x * zoom + panX;
          const cy = p.y * zoom + panY;
          const selected = p.id === selectedPointId && p.id !== draggingPointId;
          const color = labelColor(p.label);
          return (
            <g key={p.id}>
              {/* outer ring when selected */}
              <circle cx={cx} cy={cy} r={7} fill={color} fillOpacity={0.85} stroke="#fff" strokeWidth={1.5} />
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
                    {p.label}
                  </text>
                </React.Fragment>
              )}
            </g>
          );
        })}
      </svg>

      {/* Zoom controls */}
      <Box
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

export default ImageAnnotator;
