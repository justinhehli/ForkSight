# Environment Variables

The table below lists all required environment variables to run dataset preparation, training and evaluation scripts and notebooks. Some variables may be overridden "from the outside", e.g. during hyperparameter tuning ("sweeps").

These live in `Environment/.env`, except for the `POSTPROCESSING_*` variables, which live in `Environment/shared.env` which holds only the variables read by library code shared between training/evaluation code and the deployed annotation-tool pipeline (`AnnotationTool/`)

| Parameter                        | Description                                                |
| -------------------------------- | ---------------------------------------------------------- |
| SEED                             | Random seed for reproducibility                            |
| RAW_DATA_DIR                     | root path of the raw original images                       |
| DATASETS_DIR                     | root path of processed datasets |
| MODEL_CHECKPOINTS_DIR            | directory where (foundation) model checkpoints are stored for import (e.g. SAM checkpoints)                     |
| MODEL_OUT_DIR                    | root path for storing trained model parameters        |
| WANDB_DIR                        | path to Weights & Biases metadata directory |
| LOG_DIR                          | directory for training and debug logs                      |
| EVALUATION_OUTPUT_DIR                          | directory for evaluation outputs                      |
| DATASET_NAME                     | dame of the dataset used for training        |
| HIGHRES_IMG_DIR_NAME             | Subfolder for 4096×4096 images                             |
| HIGHRES_MASK_DIR_NAME            | Subfolder for 4096×4096 masks                              |
| HIGHRES_HEATMAP_DIR_NAME            | Subfolder for 4096×4096 heatmaps for junction weighting                         |
| HIGHRES_IMG_PATCHES_DIR_NAME         | Subfolder for 1024X1024 patches taken from 4096X4096 (augmented) full images                     |
| HIGHRES_MASK_PATCHES_DIR_NAME        | Subfolder for 1024x1024 mask patches taken from 4096x4096 (augmented) full images                       |
| HIGHRES_HEATMAP_PATCHES_DIR_NAME        | Subfolder for 1024x1024 heatmaps (for junction weighting) patches taken from 4096x4096 (augmented) full images                      |
| HEATMAP_VISUALIZATION_DIR_NAME        | Subfolder for visualizations of heatmaps for junction weighting                      |
| SAM3_OUTPUT_DIR_NAME             | Subfolder containing (zero-shot) SAM3 output used for training data annotations                           |
| FINETUNED_SAM_OUTPUT_DIR_NAME             | Subfolder containing fine-tuned SAM output used for training data annotations                            |
| CVAT_DIR_NAME                    | Directory for exported CVAT projects                       |
| CVAT_BACKGROUND_COLOR            | CVAT background label RGB value                     |
| CVAT_MASK_COLOR                  | CVAT mask label RGB value                    |
| CVAT_GENERATE_BW_MASKS           | Whether to convert masks to binary black/white (0/255), instead of 0/1             |
| DATASET_GAMMA_RANGE              | Gamma augmentation range                                   |
| DATASET_MAX_DISTORT              | Maximum elastic distortion                                 |
| DATASET_DISTORT_GRID_SIZE        | Grid size for elastic distortion                           |
| DATASET_JUNCTION_COORDS_CVAT_XML_PATH     | Path to the XML file containing the coordinates of junctions in the raw images                           |
| DATASET_OVERSAMPLE_JUNCTION_PATCHES       | If > 0, for each junction (as annotated in DATASET_JUNCTION_COORDS_CVAT_XML_PATH), DATASET_OVERSAMPLE_JUNCTION_PATCHES 1024x1024 patches will be extracted and added to the dataset (per augmented image)                           |
| DATASET_VAL_SPLIT                         | Percentage-wise size of validation set relative to "base" training set                            |
| DATASET_JUNCTION_WEIGHT_SIGMA             | Sigma for gaussian junction weighting heatmaps                            |
| DATASET_JUNCTION_WEIGHT_CLIP_THRESHOLD    | Threshold for clipping gaussian junction weighting heatmaps                            |
| DATASET_JUNCTION_WEIGHT_RADIUS_MULTIPLIER | Multiplier for radius calculation for gaussian junction weighting heatmaps                           |
| DATASET_SAVE_HEATMAP_VISUALIZATIONS       | If true, save visualizations of junction weighting heatmaps as PNG images                            |
| USE_WANDB                        | Enables or disables Weights & Biases tracking              |
| WANDB_ENTITY                     | W&B team, should be EM_IMCR_BIOVSION                                      |
| WANDB_SAM_PROJECT                    | W&B project name                                           |
| WANDB_API_KEY                    | API key for W&B (keep private)                             |
| SAM_LORA_VENV                    | Virtual environment path for SAM LoRA finetuning                   |
| SAM_LORA_FINETUNE_IMAGE_ENCODER  | Whether to fine-tune the image encoder                     |
| SAM_LORA_FINETUNE_MASK_DECODER   | Whether to fine-tune the mask decoder                      |
| SAM_LORA_FINETUNE_PROMPT_ENCODER | Whether to fine-tune the prompt encoder                    |
| SAM_LORA_LR                      | Learning rate                                              |
| SAM_LORA_SCHEDULER_TYPE          | LR scheduler to use (OneCycleLR or CosineAnnealingLR)                                              |
| SAM_LORA_BATCH_SIZE              | Batch size                                                 |
| SAM_LORA_MAX_EPOCHS              | Maximum number of epochs                                   |
| SAM_LORA_MODEL_CHECKPOINT        | SAM checkpoint used for training (located in MODEL_CHECKPOINTS_DIR)                      |
| SAM_LORA_MODEL_TYPE              | SAM model type (e.g., vit_b)                               |
| SAM_LORA_RANK                    | LoRA rank value                                            |
| SAM_LORA_BCE_LOSS_WEIGHT                  | SAM fine-tuning: BCE loss term weight     |
| SAM_LORA_FOCAL_LOSS_WEIGHT                | SAM fine-tuning: focal loss term weight     |
| SAM_LORA_FOCAL_ALPHA                      | SAM fine-tuning: focal loss alpha     |
| SAM_LORA_DICE_LOSS_WEIGHT        | SAM fine-tuning: dice loss term weight                      |
| SAM_LORA_CL_DICE_LOSS_WEIGHT     | SAM fine-tuning: clDice loss term weight (should be 0.0 if SAM_LORA_SKELETON_RECALL_LOSS_WEIGHT > 0.0)                      |
| SAM_LORA_CL_DICE_SKELETONIZE_ITERATIONS        | Number of iterations for skeletonization in clDice                      |
| SAM_LORA_FOCAL_GAMMA                      | SAM fine-tuning: focal loss gamma     |
| SAM_LORA_SKELETON_RECALL_LOSS_WEIGHT      | SAM fine-tuning: skeleton recall loss term weight (should be 0.0 if SAM_LORA_CL_DICE_LOSS_WEIGHT > 0.0)     |
| SAM_LORA_JUNCTION_HEATMAP_WEIGHT_SCALE    | SAM fine-tuning: if > 0, use junction heatmap weights and scale loss at in those regions with this value     |
| SAM_LORA_JUNCTION_LOSS_TYPE               | SAM fine-tuning: loss term type for separate loss in junction regions     |
| SAM_LORA_JUNCTION_PATCH_WEIGHT            | SAM fine-tuning: weight for loss term for separate loss in junction regions     |
| EARLY_STOPPING_PATIENCE          | Number of epochs without improvement before stopping       |
| EARLY_STOPPING_DELTA             | Minimum improvement considered progress                    |
| EARLY_STOPPING_MIN_EPOCHS        | Minimum epochs before early stopping can activate          |
| HUGGINGFACE_TOKEN                | Access token for Hugging Face models (keep private)        |

`Environment/shared.env` (shared with the deployed annotation-tool pipeline, see above):

| Parameter                        | Description                                                |
| -------------------------------- | ---------------------------------------------------------- |
| POSTPROCESSING_MIN_OBJ_SIZE      | Minimum size in pixels for objects in segmentation masks, smaller objects will be removed in postprocessing. If left empty, small objects won't be removed        |
| POSTPROCESSING_CONNECT_DIAGONALLY      | A boolean that indicates whether pixels in segmentation masks that are connected diagonally are considered for connected features/elements        |
| POSTPROCESSING_SMALL_BBOX_THRESHOLD | Connected components whose bounding box width AND height are both below this threshold are excluded from junction detection. If left unset, no components are excluded this way |
