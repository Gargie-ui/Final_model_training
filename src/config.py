import os

# Dataset paths
ORIGINAL_DATASET_DIR = r"D:\Final year project\FINAL_DATASET"
NEW_DATASET_DIR = r"D:\Final year project\SELECTED_4CLASS_DATASET"

# Selected classes
CLASSES = ["Atelectasis", "Infiltration", "Lung_Tumor", "No_Finding"]

# Image and training hyperparameters
# Note: On CPU, setting IMG_SIZE = 160 (instead of 224) will significantly speed up preprocessing
# and training if needed. Default is 224 as required.
IMG_SIZE = 224 
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 3e-5
RANDOM_SEED = 42

# Imbalance handling and dataset caps (Train split only)
# Capping No_Finding in the Train split to 5000 images, keeping val/test splits uncapped.
MAX_NO_FINDING_TRAIN_IMAGES = 5000
MAX_IMAGES_PER_CLASS = None  # Optional global cap per class on Train split only (for fast iteration)

# Optimization toggles
# Disabled cached features to allow end-to-end gradient backpropagation through MobileNetV2.
USE_CACHED_FEATURES = False

# CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing toggle
USE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Number of layers at the top of the backbone to unfreeze for fine-tuning.
# With USE_CACHED_FEATURES = False, the entire MobileNetV2 model is trained end-to-end.
UNFREEZE_TOP_N_LAYERS = 0

# Feature caching folder path
FEATURES_DIR = os.path.join(NEW_DATASET_DIR, "features")

# Models and results folder path for current experiment
EXPERIMENTS_DIR = os.path.join(NEW_DATASET_DIR, "experiments")
MODELS_DIR = os.path.join(EXPERIMENTS_DIR, "exp_clahe_focal_threshold")
