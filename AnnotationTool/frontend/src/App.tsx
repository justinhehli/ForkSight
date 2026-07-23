import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Add as AddIcon,
  ArrowDropDown as ArrowDropDownIcon,
  BarChart as BarChartIcon,
  CheckCircle as CheckCircleIcon,
  DarkMode as DarkModeIcon,
  Delete as DeleteIcon,
  ExpandLess as ExpandLessIcon,
  ExpandMore as ExpandMoreIcon,
  ContentCopy as ContentCopyIcon,
  FileDownload as FileDownloadIcon,
  FolderShared as FolderSharedIcon,
  LightMode as LightModeIcon,
  NavigateBefore,
  NavigateNext,
  DataObject as DataObjectIcon,
  ListAlt as ListAltIcon,
  PlayArrow as PlayArrowIcon,
  RadioButtonUnchecked as RadioButtonUncheckedIcon,
  Settings as SettingsIcon,
  TaskAlt as TaskAltIcon,
} from "@mui/icons-material";
import {
  Alert,
  Backdrop,
  Box,
  Button,
  ButtonGroup,
  Chip,
  CircularProgress,
  CssBaseline,
  Dialog,
  DialogContent,
  DialogTitle,
  Divider,
  Drawer,
  FormControl,
  IconButton,
  InputLabel,
  LinearProgress,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Menu,
  MenuItem,
  Select,
  Snackbar,
  TextField,
  ThemeProvider,
  Toolbar,
  Tooltip,
  Typography,
  createTheme,
} from "@mui/material";
import {
  addCustomLabel,
  deleteCustomLabel,
  exportProject,
  exportProjectExcel,
  getAnnotations,
  getImages,
  getPipelineLog,
  getPipelineProgress,
  getPipelineStatus,
  getProjectFolderPath,
  getProjects,
  pingBackend,
  runJunctionDetection,
  saveImageAnnotations,
  stopJunctionDetection,
} from "./api";
import AdditionalDetectionDialog from "./components/AdditionalDetectionDialog";
import ImageAnnotator, { labelColor } from "./components/ImageAnnotator";
import ManageProjectsDialog from "./components/ManageProjectsDialog";
import PipelineSettingsDialog from "./components/PipelineSettingsDialog";
import { FORK_GROUPS, FORK_WEIGHTS, PipelineStatus, sortLabelsForDisplay } from "./types";
import type { ImageAnnotations, ImageMeta, PipelineProgress, ProjectAnnotations } from "./types";
import React from "react";

const DRAWER_WIDTH = 270;

const PIPELINE_STAGE_LABELS: Record<NonNullable<PipelineProgress["stage"]>, string> = {
  preprocessing: "Preprocessing",
  segmentation: "Segmentation",
  detection: "Junction detection",
  sequential: "Junctions found",
};

const EMPTY_IMG_ANNOTATIONS: ImageAnnotations = { processed: false, points: [] };
const EMPTY_PROJECT: ProjectAnnotations = {
  junction_detection_pipeline_status: PipelineStatus.Idle,
  additional_labels: [],
  images: {},
};

const App = () => {
  // state
  // ====================
  const [mode, setMode] = useState<"light" | "dark">(
    () => (localStorage.getItem("colorMode") as "light" | "dark") ?? "dark",
  );
  const theme = useMemo(() => createTheme({ palette: { mode } }), [mode]);
  const toggleMode = () =>
    setMode((m) => {
      const next = m === "dark" ? "light" : "dark";
      localStorage.setItem("colorMode", next);
      return next;
    });

  const [projects, setProjects] = useState<string[]>([]);
  const [selectedProject, setSelectedProject] = useState("");
  const [images, setImages] = useState<ImageMeta[]>([]);
  const [imageIdx, setImageIdx] = useState(0);
  const [annotations, setAnnotations] = useState<ProjectAnnotations>(EMPTY_PROJECT);
  const [selectedPointId, setSelectedPointId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [exportAnchor, setExportAnchor] = useState<HTMLElement | null>(null);
  const [overviewOpen, setOverviewOpen] = useState(false);
  const [manageProjectsOpen, setManageProjectsOpen] = useState(false);
  const [pipelineSettingsOpen, setPipelineSettingsOpen] = useState(false);
  const [additionalRunOpen, setAdditionalRunOpen] = useState(false);
  const [pathCopied, setPathCopied] = useState(false);
  const [logDialogOpen, setLogDialogOpen] = useState(false);
  const [logText, setLogText] = useState("");
  const [runningProject, setRunningProject] = useState<string | null>(null);
  const [newLabelInput, setNewLabelInput] = useState("");
  const [showShortcuts, setShowShortcuts] = useState(true);
  const [backendRestarting, setBackendRestarting] = useState(false);
  const [backendRestartTimedOut, setBackendRestartTimedOut] = useState(false);
  const [pipelineProgress, setPipelineProgress] = useState<PipelineProgress | null>(null);

  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const logContainerRef = useRef<HTMLPreElement | null>(null);

  // derived values
  // ====================
  const pipelineStatus = annotations.junction_detection_pipeline_status;
  const blockedByOtherProject = runningProject !== null && runningProject !== selectedProject;
  const currentImageId = images[imageIdx]?.id ?? "";
  const currentImageName = images[imageIdx]?.name ?? "";
  const currentAnnotations: ImageAnnotations = annotations.images[currentImageId] ?? EMPTY_IMG_ANNOTATIONS;
  const processedCount = images.filter((i) => i.processed).length;
  const selectedPoint = currentAnnotations.points.find((p) => p.id === selectedPointId);

  const stats = useMemo(() => {
    const allPoints = Object.values(annotations.images).flatMap((a) => a.points);
    const replicationGroup = FORK_GROUPS.find((g) => g.name === "Replication Fork")!;
    const reversedGroup = FORK_GROUPS.find((g) => g.name === "Reversed Fork")!;
    let replicationForks = 0;
    let reversedForks = 0;
    for (const p of allPoints) {
      for (const l of p.labels) {
        if (l === replicationGroup.fifty || l === replicationGroup.hundred) {
          replicationForks += FORK_WEIGHTS[l];
        } else if (l === reversedGroup.fifty || l === reversedGroup.hundred) {
          reversedForks += FORK_WEIGHTS[l];
        }
      }
    }
    const ratio = reversedForks > 0 ? (replicationForks / reversedForks).toFixed(2) : "—";
    return { replicationForks, reversedForks, ratio };
  }, [annotations.images]);

  // bootstrap - load projects
  // ====================
  useEffect(() => {
    getProjects()
      .then((ps) => {
        setProjects(ps);
      })
      .catch((e) => setError(String(e)));
  }, []);

  // wait out a backend restart (triggered by registering a new project)
  // ====================
  const RESTART_GRACE_MS = 1500; // covers the backend's own delay before it actually exits
  const RESTART_TIMEOUT_MS = 30000;
  const RESTART_POLL_INTERVAL_MS = 500;

  const waitForBackendRestart = useCallback(async () => {
    setBackendRestarting(true);
    setBackendRestartTimedOut(false);

    // give the backend a moment to actually go down before polling, so a fast
    // request that beats the restart doesn't conclude too early
    await new Promise((resolve) => setTimeout(resolve, RESTART_GRACE_MS));

    const deadline = Date.now() + RESTART_TIMEOUT_MS;
    let alive = await pingBackend();
    while (!alive && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, RESTART_POLL_INTERVAL_MS));
      alive = await pingBackend();
    }

    setBackendRestarting(false);
    if (!alive) {
      setBackendRestartTimedOut(true);
      return;
    }

    getProjects()
      .then(setProjects)
      .catch((e) => setError(String(e)));
  }, []);

  // load selected project data
  // ====================
  const loadSelectedProject = useCallback(() => {
    if (!selectedProject) return;

    setLoading(true);
    setError(null);
    setSelectedPointId(null);

    Promise.all([getImages(selectedProject), getAnnotations(selectedProject)])
      .then(([imgs, ann]) => {
        setImages(imgs);
        setAnnotations(ann);
        const firstUnprocessed = imgs.findIndex((img) => !img.processed);
        setImageIdx(firstUnprocessed >= 0 ? firstUnprocessed : 0);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [selectedProject]);

  useEffect(() => {
    loadSelectedProject();
  }, [loadSelectedProject]);

  // auto-save
  // ====================
  const scheduleSave = useCallback(
    (imageId: string, data: ImageAnnotations) => {
      if (saveTimerRef.current) {
        clearTimeout(saveTimerRef.current);
      }

      saveTimerRef.current = setTimeout(() => {
        saveImageAnnotations(selectedProject, imageId, data).catch((e) => console.error("Save failed:", e));

        // update image meta data (used in sidebar) processed label
        setImages((prev) => prev.map((img) => (img.id === imageId ? { ...img, processed: data.processed } : img)));
      }, 600);
    },
    [selectedProject],
  );

  const handleAnnotationsChange = useCallback(
    (updated: ImageAnnotations) => {
      const rounded: ImageAnnotations = {
        ...updated,
        points: updated.points.map((p) => ({ ...p, x: Math.round(p.x), y: Math.round(p.y) })),
      };
      setAnnotations((prev) => ({ ...prev, images: { ...prev.images, [currentImageId]: rounded } }));
      scheduleSave(currentImageId, rounded);
    },
    [currentImageId, scheduleSave],
  );

  // actions
  // ====================
  const toggleImageProcessed = useCallback(() => {
    handleAnnotationsChange({ ...currentAnnotations, processed: !currentAnnotations.processed });
  }, [currentAnnotations, handleAnnotationsChange]);

  const deletePoint = (id: string) => {
    handleAnnotationsChange({ ...currentAnnotations, points: currentAnnotations.points.filter((p) => p.id !== id) });
    if (selectedPointId === id) {
      setSelectedPointId(null);
    }
  };

  const cycleForkGroup = (id: string, group: (typeof FORK_GROUPS)[number]) => {
    const point = currentAnnotations.points.find((p) => p.id === id);
    if (!point) return;

    const has50 = point.labels.includes(group.fifty);
    const has100 = point.labels.includes(group.hundred);
    const forkLabels = new Set<string>(FORK_GROUPS.flatMap((g) => [g.fifty, g.hundred]));
    const rest = point.labels.filter((l) => !forkLabels.has(l));
    const labels = has100 ? rest : has50 ? [...rest, group.hundred] : [...rest, group.fifty];
    handleAnnotationsChange({
      ...currentAnnotations,
      points: currentAnnotations.points.map((p) => (p.id === id ? { ...p, labels } : p)),
    });
  };

  const toggleLabel = (id: string, label: string) => {
    const point = currentAnnotations.points.find((p) => p.id === id);
    if (!point) return;
    const labels = point.labels.includes(label) ? point.labels.filter((l) => l !== label) : [...point.labels, label];
    handleAnnotationsChange({
      ...currentAnnotations,
      points: currentAnnotations.points.map((p) => (p.id === id ? { ...p, labels } : p)),
    });
  };

  const handleAddCustomLabel = async () => {
    const label = newLabelInput.trim();
    if (!label) return;
    try {
      const updated = await addCustomLabel(selectedProject, label);
      setAnnotations((prev) => ({ ...prev, additional_labels: updated }));
      setNewLabelInput("");
    } catch (e) {
      setError(String(e));
    }
  };

  const handleDeleteCustomLabel = async (label: string) => {
    try {
      const updated = await deleteCustomLabel(selectedProject, label);
      setAnnotations((prev) => ({
        ...prev,
        additional_labels: updated,
        images: Object.fromEntries(
          Object.entries(prev.images).map(([imgId, img]) => [
            imgId,
            { ...img, points: img.points.map((p) => ({ ...p, labels: p.labels.filter((l) => l !== label) })) },
          ]),
        ),
      }));
    } catch (e) {
      setError(String(e));
    }
  };

  const handleDownloadJson = async () => {
    setExportAnchor(null);

    try {
      const blob = await exportProject(selectedProject);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${selectedProject}_annotations.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(String(e));
    }
  };

  const handleDownloadExcel = async () => {
    setExportAnchor(null);

    try {
      const blob = await exportProjectExcel(selectedProject);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${selectedProject}_annotations.xlsx`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(String(e));
    }
  };

  const handleCopyProjectPath = async () => {
    try {
      const { path } = await getProjectFolderPath(selectedProject);
      await navigator.clipboard.writeText(path);
      setPathCopied(true);
      setTimeout(() => setPathCopied(false), 1500);
    } catch (e) {
      setError(String(e));
    }
  };

  const handleRunJunctionDetection = async (amount?: number) => {
    setAnnotations((prev) => ({ ...prev, junction_detection_pipeline_status: PipelineStatus.Running }));
    try {
      await runJunctionDetection(selectedProject, amount);
    } catch (e) {
      setError(String(e));
      loadSelectedProject();
    }
  };

  const handleConfirmAdditionalRun = (amount: number) => {
    setAdditionalRunOpen(false);
    handleRunJunctionDetection(amount);
  };

  const handleStopJunctionDetection = async () => {
    try {
      await stopJunctionDetection(selectedProject);
    } catch (e) {
      setError(String(e));
    } finally {
      loadSelectedProject();
    }
  };

  // global poll: which project (if any) has a pipeline running
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const { running_project } = await getPipelineStatus();
        if (!cancelled) {
          setRunningProject(running_project);
        }
      } catch {
        // ignore transient errors, keep showing the last known state
      }
    };

    poll();

    const timer = setInterval(poll, 3000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  // while the selected project's pipeline is running, poll its annotations
  // and stage/image progress so completion/failure is reflected here instead
  // of spinning forever
  useEffect(() => {
    if (pipelineStatus !== PipelineStatus.Running || !selectedProject) {
      setPipelineProgress(null);
      return;
    }

    const timer = setInterval(async () => {
      try {
        const latest = await getAnnotations(selectedProject);
        setAnnotations(latest);
        if (latest.junction_detection_pipeline_status === PipelineStatus.Done) {
          loadSelectedProject();
        }
      } catch {
        // ignore transient errors, keep showing the last known state
      }
      try {
        setPipelineProgress(await getPipelineProgress(selectedProject));
      } catch {
        // ignore transient errors, keep showing the last known state
      }
    }, 3000);
    return () => clearInterval(timer);
  }, [pipelineStatus, selectedProject, loadSelectedProject]);

  const handleOpenLogDialog = async () => {
    setLogDialogOpen(true);
    try {
      setLogText(await getPipelineLog(selectedProject));
    } catch (e) {
      setLogText(`Failed to load pipeline log: ${e}`);
    }
  };

  // poll the pipeline log while the dialog is open and the pipeline is running
  useEffect(() => {
    if (!logDialogOpen || pipelineStatus !== PipelineStatus.Running) return;
    const timer = setInterval(async () => {
      try {
        setLogText(await getPipelineLog(selectedProject));
      } catch {
        // keep showing the last successfully fetched log on transient errors
      }
    }, 2000);
    return () => clearInterval(timer);
  }, [logDialogOpen, pipelineStatus, selectedProject]);

  // auto-scroll the log view to the bottom as new content arrives
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logText]);

  // keyboard shortcuts
  // ====================
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      if (e.key === "ArrowRight") setImageIdx((i) => Math.min(images.length - 1, i + 1));
      if (e.key === "ArrowLeft") setImageIdx((i) => Math.max(0, i - 1));
      if ((e.key === "p" || e.key === "P") && !e.ctrlKey) toggleImageProcessed();
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [images.length, toggleImageProcessed]);

  // render
  // ====================
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: "flex", height: "100vh", overflow: "hidden" }}>
        {/* ── Left drawer ─────────────────────────────────────────────────── */}
        <Drawer
          variant="permanent"
          sx={{
            width: DRAWER_WIDTH,
            flexShrink: 0,
            "& .MuiDrawer-paper": {
              width: DRAWER_WIDTH,
              boxSizing: "border-box",
              display: "flex",
              flexDirection: "column",
            },
          }}
        >
          <Toolbar variant="dense" sx={{ height: 48, minHeight: 48, borderBottom: 1, borderColor: "divider" }}>
            <Typography variant="h6" noWrap fontWeight={700}>
              DNA Fork Annotator
            </Typography>
          </Toolbar>

          {/* Project selector */}
          <Box sx={{ p: 1.5, pb: selectedProject ? 0.5 : 1.5, display: "flex", gap: 1 }}>
            <FormControl fullWidth size="small">
              <InputLabel>Project</InputLabel>
              <Select
                value={selectedProject}
                label="Project"
                onChange={(e) => {
                  setSelectedProject(e.target.value);
                  setImages([]);
                  setAnnotations(EMPTY_PROJECT);
                  setImageIdx(0);
                  setSelectedPointId(null);
                  setError(null);
                }}
              >
                {projects.map((p) => (
                  <MenuItem key={p} value={p}>
                    {p}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <Tooltip title="Manage projects">
              <IconButton size="small" onClick={() => setManageProjectsOpen(true)}>
                <FolderSharedIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
          {selectedProject && (
            <Box sx={{ px: 1.5, pb: 1, display: "flex", alignItems: "center", gap: 0.5, minWidth: 0 }}>
              <Typography
                variant="caption"
                color="text.secondary"
                noWrap
                title={selectedProject}
                sx={{ flex: 1, minWidth: 0 }}
              >
                {selectedProject}
              </Typography>
              <Tooltip title={pathCopied ? "Copied!" : "Copy directory path"}>
                <IconButton size="small" onClick={handleCopyProjectPath}>
                  <ContentCopyIcon sx={{ fontSize: 14 }} />
                </IconButton>
              </Tooltip>
            </Box>
          )}
          <Divider />

          {/* Image list */}
          <Box sx={{ px: 1.5, pt: 1, display: "flex", alignItems: "center", gap: 1 }}>
            <Typography variant="subtitle2">Images</Typography>
            <Chip label={`${processedCount}/${images.length}`} size="small" color="primary" />
          </Box>
          <List dense disablePadding sx={{ flex: 1, overflowY: "auto", mt: 0.5 }}>
            {images.map((img, idx) => (
              <ListItemButton
                key={img.id}
                selected={idx === imageIdx}
                onClick={() => {
                  setImageIdx(idx);
                  setSelectedPointId(null);
                }}
                sx={{ py: 0.25 }}
                disabled={!!error || pipelineStatus !== PipelineStatus.Done}
              >
                <ListItemIcon sx={{ minWidth: 28 }}>
                  {img.processed ? (
                    <CheckCircleIcon sx={{ fontSize: 16 }} color="success" />
                  ) : (
                    <RadioButtonUncheckedIcon sx={{ fontSize: 16 }} color="disabled" />
                  )}
                </ListItemIcon>
                <ListItemText
                  primary={img.name}
                  primaryTypographyProps={{ variant: "body2", noWrap: true, fontSize: 12 }}
                />
              </ListItemButton>
            ))}
          </List>
          <Divider />
        </Drawer>

        {/* ── Main content ────────────────────────────────────────────────── */}
        <Box sx={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, overflow: "hidden" }}>
          {/* Toolbar */}
          <Toolbar
            variant="dense"
            sx={{ height: 48, minHeight: 48, borderBottom: 1, borderColor: "divider", gap: 1, flexShrink: 0 }}
          >
            <Tooltip title="Previous (←)">
              <span>
                <IconButton
                  size="small"
                  onClick={() => setImageIdx((i) => Math.max(0, i - 1))}
                  disabled={!!error || imageIdx === 0 || pipelineStatus !== PipelineStatus.Done}
                >
                  <NavigateBefore />
                </IconButton>
              </span>
            </Tooltip>

            <Typography variant="body2" noWrap sx={{ flex: 1, fontFamily: "monospace" }}>
              {selectedProject === "" || currentImageName == null || pipelineStatus !== PipelineStatus.Done
                ? ""
                : currentImageName}
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ flexShrink: 0 }}>
              {images.length > 0 && pipelineStatus === PipelineStatus.Done ? `${imageIdx + 1} / ${images.length}` : ""}
            </Typography>

            <Tooltip title="Next (→)">
              <span>
                <IconButton
                  size="small"
                  onClick={() => setImageIdx((i) => Math.min(images.length - 1, i + 1))}
                  disabled={!!error || imageIdx >= images.length - 1 || pipelineStatus !== PipelineStatus.Done}
                >
                  <NavigateNext />
                </IconButton>
              </span>
            </Tooltip>

            <Tooltip title="Toggle processed (P)">
              <Button
                size="small"
                variant={currentAnnotations.processed ? "contained" : "outlined"}
                color="success"
                startIcon={<TaskAltIcon sx={{ fontSize: 16 }} />}
                onClick={toggleImageProcessed}
                sx={{ textTransform: "none", fontSize: 12, minWidth: 130 }}
                disabled={!!error || !currentImageName || pipelineStatus !== PipelineStatus.Done}
              >
                {currentAnnotations.processed ? "Processed ✓" : "Mark processed"}
              </Button>
            </Tooltip>

            <ButtonGroup
              size="small"
              variant="outlined"
              disabled={!!error || !selectedProject || pipelineStatus !== PipelineStatus.Done}
            >
              <Button
                startIcon={<FileDownloadIcon sx={{ fontSize: 16 }} />}
                onClick={handleDownloadExcel}
                sx={{ textTransform: "none", fontSize: 12 }}
              >
                Export
              </Button>
              <Button sx={{ px: 0.5 }} onClick={(e) => setExportAnchor(e.currentTarget)}>
                <ArrowDropDownIcon fontSize="small" />
              </Button>
            </ButtonGroup>
            <Menu
              anchorEl={exportAnchor}
              open={Boolean(exportAnchor)}
              onClose={() => setExportAnchor(null)}
              anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
              transformOrigin={{ vertical: "top", horizontal: "right" }}
            >
              <MenuItem dense onClick={handleDownloadExcel}>
                <ListItemIcon>
                  <ListAltIcon fontSize="small" />
                </ListItemIcon>
                Download Excel
              </MenuItem>
              <MenuItem dense onClick={handleDownloadJson}>
                <ListItemIcon>
                  <DataObjectIcon fontSize="small" />
                </ListItemIcon>
                Download JSON
              </MenuItem>
            </Menu>

            <Tooltip title="Run additional automatic fork detection">
              <span>
                <IconButton
                  size="small"
                  onClick={() => setAdditionalRunOpen(true)}
                  disabled={
                    !!error || !selectedProject || pipelineStatus !== PipelineStatus.Done || blockedByOtherProject
                  }
                >
                  <PlayArrowIcon fontSize="small" />
                </IconButton>
              </span>
            </Tooltip>

            <Tooltip title="Project overview">
              <span>
                <IconButton
                  size="small"
                  onClick={() => setOverviewOpen(true)}
                  disabled={!!error || !selectedProject || pipelineStatus !== PipelineStatus.Done}
                >
                  <BarChartIcon fontSize="small" />
                </IconButton>
              </span>
            </Tooltip>

            <Divider orientation="vertical" flexItem sx={{ my: 1 }} />

            <Tooltip title="Settings">
              <IconButton size="small" onClick={() => setPipelineSettingsOpen(true)}>
                <SettingsIcon fontSize="small" />
              </IconButton>
            </Tooltip>

            <Tooltip title={mode === "dark" ? "Switch to light mode" : "Switch to dark mode"}>
              <IconButton size="small" onClick={toggleMode}>
                {mode === "dark" ? <LightModeIcon fontSize="small" /> : <DarkModeIcon fontSize="small" />}
              </IconButton>
            </Tooltip>
          </Toolbar>

          {/* Annotator + right panel */}
          <Box sx={{ flex: 1, display: "flex", overflow: "hidden" }}>
            {/* Centre: annotator or junction-detection gate */}
            <Box sx={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
              {error && (
                <Alert severity="error" onClose={() => setError(null)} sx={{ m: 1, flexShrink: 0 }}>
                  {error}
                </Alert>
              )}
              {!selectedProject && (
                <Alert severity="info" variant="outlined" sx={{ m: 1, flexShrink: 0 }}>
                  Select a project to begin annotating
                </Alert>
              )}
              <Box sx={{ flex: 1, overflow: "hidden", position: "relative" }}>
                {loading && (
                  <Box
                    sx={{
                      position: "absolute",
                      inset: 0,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      zIndex: 1,
                    }}
                  >
                    <CircularProgress />
                  </Box>
                )}

                {!loading && !error && selectedProject && pipelineStatus !== PipelineStatus.Done && (
                  <Box
                    sx={{
                      height: "100%",
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      justifyContent: "center",
                      gap: 2,
                      p: 4,
                    }}
                  >
                    {pipelineStatus === PipelineStatus.Idle && (
                      <React.Fragment>
                        <Typography variant="h6" color="text.secondary">
                          Fork detection not yet run
                        </Typography>
                        <Typography variant="body2" color="text.secondary" textAlign="center">
                          {blockedByOtherProject
                            ? `Fork detection is currently running for project "${runningProject}". Please wait until it finishes.`
                            : "Run the automatic fork detection pipeline before reviewing images."}
                        </Typography>
                        <Button
                          variant="contained"
                          size="large"
                          startIcon={<PlayArrowIcon />}
                          onClick={() => handleRunJunctionDetection()}
                          disabled={blockedByOtherProject}
                        >
                          Run fork detection
                        </Button>
                      </React.Fragment>
                    )}
                    {pipelineStatus === PipelineStatus.Running && (
                      <React.Fragment>
                        <CircularProgress />
                        <Typography variant="h6" color="text.secondary">
                          Automatic fork detection is running…
                        </Typography>
                        <Typography variant="body2" color="text.secondary" textAlign="center">
                          This project reloads automatically once the pipeline completes.
                        </Typography>
                        {pipelineProgress?.stage && (
                          <Box sx={{ width: 260, display: "flex", flexDirection: "column", gap: 0.5 }}>
                            <Typography variant="body2" color="text.secondary" textAlign="center">
                              {PIPELINE_STAGE_LABELS[pipelineProgress.stage]}: {pipelineProgress.completed} /{" "}
                              {pipelineProgress.total}
                              {pipelineProgress.stage === "sequential"
                                ? " junctions"
                                : ` image${pipelineProgress.total === 1 ? "" : "s"}`}
                            </Typography>
                            <LinearProgress
                              variant={pipelineProgress.total > 0 ? "determinate" : "indeterminate"}
                              value={
                                pipelineProgress.total > 0
                                  ? (pipelineProgress.completed / pipelineProgress.total) * 100
                                  : undefined
                              }
                            />
                          </Box>
                        )}
                        <Box sx={{ display: "flex", gap: 1 }}>
                          <Button variant="outlined" size="small" onClick={handleOpenLogDialog}>
                            View log
                          </Button>
                          <Button variant="outlined" size="small" color="error" onClick={handleStopJunctionDetection}>
                            Stop pipeline
                          </Button>
                        </Box>
                      </React.Fragment>
                    )}
                    {pipelineStatus === PipelineStatus.Failed && (
                      <React.Fragment>
                        <Typography variant="h6" color="error">
                          Fork detection failed
                        </Typography>
                        <Typography variant="body2" color="text.secondary" textAlign="center">
                          {blockedByOtherProject
                            ? `Fork detection is currently running for project "${runningProject}". Please wait until it finishes.`
                            : (annotations.pipeline_error ??
                              "The automatic fork detection pipeline encountered an error. Check the logs and try again.")}
                        </Typography>
                        <Box sx={{ display: "flex", gap: 1 }}>
                          <Button variant="outlined" size="small" onClick={handleOpenLogDialog}>
                            View log
                          </Button>
                          <Button
                            variant="contained"
                            size="large"
                            color="error"
                            startIcon={<PlayArrowIcon />}
                            onClick={() => setAdditionalRunOpen(true)}
                            disabled={blockedByOtherProject}
                          >
                            Re-run fork detection
                          </Button>
                        </Box>
                      </React.Fragment>
                    )}
                  </Box>
                )}

                {!loading && pipelineStatus === PipelineStatus.Done && currentImageId && (
                  <ImageAnnotator
                    project={selectedProject}
                    imageId={currentImageId}
                    imageName={currentImageName}
                    annotations={currentAnnotations}
                    onAnnotationsChange={handleAnnotationsChange}
                    selectedPointId={selectedPointId}
                    onSelectPoint={setSelectedPointId}
                  />
                )}
              </Box>
            </Box>

            {/* Right panel: annotation list */}
            <Box
              sx={{
                width: 260,
                flexShrink: 0,
                borderLeft: 1,
                borderColor: "divider",
                display: "flex",
                flexDirection: "column",
                overflow: "hidden",
                opacity: !!error || pipelineStatus !== PipelineStatus.Done ? 0.4 : 1,
                pointerEvents: !!error || pipelineStatus !== PipelineStatus.Done ? "none" : "auto",
              }}
            >
              <Box
                sx={{
                  px: 1.5,
                  py: 1,
                  borderBottom: 1,
                  borderColor: "divider",
                  display: "flex",
                  alignItems: "center",
                  gap: 1,
                }}
              >
                <Typography variant="subtitle2">Annotations</Typography>
                <Chip label={currentAnnotations.points.length} size="small" />
              </Box>

              <List dense disablePadding sx={{ flex: 1, minHeight: 48, overflowY: "auto" }}>
                {currentAnnotations.points.map((p, i) => {
                  const sortedLabels = sortLabelsForDisplay(p.labels);
                  return (
                    <ListItem
                      key={p.id}
                      disablePadding
                      secondaryAction={
                        <IconButton edge="end" size="small" onClick={() => deletePoint(p.id)} tabIndex={-1}>
                          <DeleteIcon sx={{ fontSize: 15 }} />
                        </IconButton>
                      }
                    >
                      <ListItemButton
                        selected={p.id === selectedPointId}
                        onClick={() => setSelectedPointId((prev) => (prev === p.id ? null : p.id))}
                        sx={{ py: 0.25, pl: 1, pr: 4 }}
                      >
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
                        <Tooltip
                          title={sortedLabels.length > 0 ? sortedLabels.join(", ") : "(unlabeled)"}
                          placement="top"
                        >
                          <ListItemText
                            primary={`${i + 1}. ${sortedLabels.length > 0 ? sortedLabels.join(", ") : "(unlabeled)"}`}
                            secondary={`${Math.round(p.x)}, ${Math.round(p.y)}`}
                            primaryTypographyProps={{ variant: "body2", fontSize: 12, noWrap: true }}
                            secondaryTypographyProps={{ variant: "caption", fontFamily: "monospace", fontSize: 10 }}
                            sx={{ minWidth: 0 }}
                          />
                        </Tooltip>
                      </ListItemButton>
                    </ListItem>
                  );
                })}
              </List>

              {/* Selected point: change labels */}
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
                              onClick={() => cycleForkGroup(selectedPoint.id, group)}
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

                      {annotations.additional_labels.length > 0 && (
                        <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5, mb: 1 }}>
                          {annotations.additional_labels.map((l) => {
                            const active = selectedPoint.labels.includes(l);
                            const color = labelColor(l);
                            return (
                              <Box key={l} sx={{ display: "flex", alignItems: "center", gap: 0.25, minWidth: 0 }}>
                                <Tooltip title={l} placement="top">
                                  <Button
                                    size="small"
                                    variant={active ? "contained" : "outlined"}
                                    onClick={() => toggleLabel(selectedPoint.id, l)}
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
                                      sx={{
                                        minWidth: 0,
                                        overflow: "hidden",
                                        textOverflow: "ellipsis",
                                        whiteSpace: "nowrap",
                                      }}
                                    >
                                      {l}
                                    </Box>
                                  </Button>
                                </Tooltip>
                                <Tooltip title="Delete label from project">
                                  <IconButton size="small" onClick={() => handleDeleteCustomLabel(l)}>
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
                  </Typography>
                </Box>
              )}
            </Box>
          </Box>
        </Box>
      </Box>
      <Dialog open={overviewOpen} onClose={() => setOverviewOpen(false)} maxWidth="xs" fullWidth>
        <DialogTitle>{selectedProject}</DialogTitle>
        <DialogContent>
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: "1fr auto",
              rowGap: 1.5,
              columnGap: 4,
              alignItems: "baseline",
              py: 1,
            }}
          >
            <Typography variant="body2" color="text.secondary">
              Total images
            </Typography>
            <Typography variant="body2" fontWeight={600} textAlign="right">
              {images.length}
            </Typography>

            <Typography variant="body2" color="text.secondary">
              Processed images
            </Typography>
            <Typography variant="body2" fontWeight={600} textAlign="right">
              {processedCount}
            </Typography>

            <Divider sx={{ gridColumn: "1 / -1" }} />

            <Typography variant="body2" color="text.secondary">
              Replication forks (weighted)
            </Typography>
            <Typography variant="body2" fontWeight={600} textAlign="right">
              {stats.replicationForks.toFixed(1)}
            </Typography>

            <Typography variant="body2" color="text.secondary">
              Reversed forks (weighted)
            </Typography>
            <Typography variant="body2" fontWeight={600} textAlign="right">
              {stats.reversedForks.toFixed(1)}
            </Typography>

            <Divider sx={{ gridColumn: "1 / -1" }} />

            <Typography variant="body2" color="text.secondary">
              Replication / reversed ratio
            </Typography>
            <Typography variant="body2" fontWeight={600} textAlign="right">
              {stats.ratio}
            </Typography>
          </Box>
        </DialogContent>
      </Dialog>
      <Dialog open={logDialogOpen} onClose={() => setLogDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Pipeline log{pipelineStatus === PipelineStatus.Running ? " (live)" : ""}</DialogTitle>
        <DialogContent>
          <Box
            component="pre"
            ref={logContainerRef}
            sx={{
              m: 0,
              p: 1.5,
              maxHeight: "60vh",
              overflow: "auto",
              bgcolor: mode === "dark" ? "grey.900" : "grey.100",
              borderRadius: 1,
              fontSize: 12,
              fontFamily: "monospace",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}
          >
            {logText || "No log output yet."}
          </Box>
        </DialogContent>
      </Dialog>
      <PipelineSettingsDialog open={pipelineSettingsOpen} onClose={() => setPipelineSettingsOpen(false)} />
      <AdditionalDetectionDialog
        open={additionalRunOpen}
        mode={annotations.pipeline_mode}
        onClose={() => setAdditionalRunOpen(false)}
        onConfirm={handleConfirmAdditionalRun}
      />
      <ManageProjectsDialog
        open={manageProjectsOpen}
        onClose={() => setManageProjectsOpen(false)}
        onSaved={(mightRestart) => {
          if (mightRestart) {
            waitForBackendRestart();
          } else {
            getProjects()
              .then(setProjects)
              .catch((e) => setError(String(e)));
          }
        }}
      />
      <Backdrop
        open={backendRestarting}
        sx={{
          // Must outrank Dialog's zIndex.modal - ManageProjectsDialog (which
          // triggers this) is still fading out via its own exit transition
          // when this opens, and would otherwise cover it while both overlap.
          zIndex: (t) => t.zIndex.modal + 1,
          color: "#fff",
          flexDirection: "column",
          gap: 2,
        }}
      >
        <CircularProgress color="inherit" />
        <Typography variant="h6">Restarting backend…</Typography>
        <Typography variant="body2">This can take a few seconds.</Typography>
      </Backdrop>
      <Snackbar
        open={backendRestartTimedOut}
        onClose={() => setBackendRestartTimedOut(false)}
        autoHideDuration={8000}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert severity="error" onClose={() => setBackendRestartTimedOut(false)}>
          The backend didn't come back after restarting. Check that it's still running.
        </Alert>
      </Snackbar>
    </ThemeProvider>
  );
};

export default App;
