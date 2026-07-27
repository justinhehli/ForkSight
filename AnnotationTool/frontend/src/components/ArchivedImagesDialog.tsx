import { Unarchive as UnarchiveIcon } from "@mui/icons-material";
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
  List,
  ListItem,
  ListItemText,
  Tooltip,
  Typography,
} from "@mui/material";
import type { ImageMeta } from "../types";

interface Props {
  open: boolean;
  images: ImageMeta[];
  onClose: () => void;
  onRestore: (imageId: string) => void;
}

const ArchivedImagesDialog = ({ open, images, onClose, onRestore }: Props) => (
  <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
    <DialogTitle>Archived images</DialogTitle>
    <DialogContent dividers sx={{ maxHeight: 420 }}>
      <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
        Archived images are hidden from the list and excluded from fork-ratio calculations and exports.
      </Typography>
      {images.length === 0 ? (
        <Typography variant="body2" color="text.secondary">
          No archived images.
        </Typography>
      ) : (
        <List dense disablePadding>
          {images.map((img) => (
            <ListItem
              key={img.id}
              disablePadding
              secondaryAction={
                <Tooltip title="Restore image">
                  <IconButton edge="end" size="small" onClick={() => onRestore(img.id)}>
                    <UnarchiveIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              }
              sx={{ py: 0.25 }}
            >
              <ListItemText
                primary={img.name}
                primaryTypographyProps={{ variant: "body2", noWrap: true, fontSize: 13 }}
              />
            </ListItem>
          ))}
        </List>
      )}
    </DialogContent>
    <DialogActions>
      <Button onClick={onClose}>Close</Button>
    </DialogActions>
  </Dialog>
);

export default ArchivedImagesDialog;
