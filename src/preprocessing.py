import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from src import config

def apply_clahe_np(image_np):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to a single-channel grayscale numpy array (H, W, 1) or 2D image uint8.
    """
    if image_np.ndim == 3 and image_np.shape[-1] == 1:
        img_gray = image_np[:, :, 0]
    elif image_np.ndim == 2:
        img_gray = image_np
    else:
        img_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID_SIZE
    )
    cl = clahe.apply(img_gray.astype(np.uint8))
    return cl[:, :, np.newaxis]

def apply_clahe_tf(image_tensor):
    """
    TensorFlow wrapper for OpenCV CLAHE preprocessing via tf.py_function.
    """
    clahe_tensor = tf.py_function(
        func=lambda img: apply_clahe_np(img.numpy()),
        inp=[image_tensor],
        Tout=tf.uint8
    )
    clahe_tensor.set_shape(image_tensor.shape)
    return clahe_tensor

def load_and_preprocess_image(filepath, label_idx):
    """
    Loads an image from disk, decodes as grayscale, applies CLAHE (if enabled),
    converts to 3-channel RGB, resizes to target dimensions, and normalizes
    using MobileNetV2 preprocess_input.
    """
    # 1. Read raw file bytes
    image_bytes = tf.io.read_file(filepath)
    
    # 2. Decode image as grayscale uint8 (1 channel).
    image = tf.image.decode_image(image_bytes, channels=1, expand_animations=False)
    
    # 2.5 Apply CLAHE preprocessing BEFORE resizing and BEFORE preprocess_input
    if getattr(config, "USE_CLAHE", False):
        image = apply_clahe_tf(image)
        
    # 3. Convert grayscale (H, W, 1) to RGB (H, W, 3) by replicating the single channel.
    image = tf.image.grayscale_to_rgb(image)
    
    # 4. Convert image to float32
    image = tf.cast(image, tf.float32)
    
    # 5. Resize to target dimension
    image = tf.image.resize(image, (config.IMG_SIZE, config.IMG_SIZE))
    
    # 6. Apply MobileNetV2-specific preprocessing
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    
    return image, label_idx

def build_dataset_from_csv(csv_path, shuffle=False, use_cache=False):
    """
    Reads a split CSV file and constructs an optimized tf.data.Dataset pipeline.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Split CSV file not found at: {csv_path}. Run dataset_splitter.py first.")
        
    df = pd.read_csv(csv_path)
    
    # Extract file paths and integer label indices
    filepaths = df["filepath"].values
    labels = df["label"].values
    label_indices = [config.CLASSES.index(l) for l in labels]
    
    # Convert label indices to one-hot encoding for multi-class classification
    label_indices_one_hot = tf.keras.utils.to_categorical(label_indices, num_classes=len(config.CLASSES))
    
    # Create Dataset slices
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, label_indices_one_hot))
    
    # Shuffle if required (usually only for Train)
    if shuffle:
        # Buffer size matches training size to ensure uniform shuffling
        dataset = dataset.shuffle(buffer_size=len(df), seed=config.RANDOM_SEED)
        
    # Map preprocessing function
    # On CPU, keep parallel calls reasonable to prevent memory exhaustion or high thread contention.
    num_cpus = os.cpu_count() or 4
    num_parallel = max(1, num_cpus - 1)  # Leave 1 CPU core free
    
    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=num_parallel
    )
    
    # Batch, Cache, and Prefetch for I/O performance
    # Prefetch (AUTOTUNE) overlaps dataset preprocessing and model execution.
    dataset = dataset.batch(config.BATCH_SIZE)
    if use_cache:
        # Cache in memory only if requested (can cause OOM on large datasets on CPU)
        dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset, len(df)

if __name__ == "__main__":
    # Test pipeline on a small subset if run directly
    train_csv = os.path.join(config.NEW_DATASET_DIR, "train.csv")
    if os.path.exists(train_csv):
        print("Testing preprocessing pipeline on train split...")
        dataset, count = build_dataset_from_csv(train_csv, shuffle=True)
        for images, labels in dataset.take(1):
            print(f"Batch images shape: {images.shape}")
            print(f"Batch labels shape: {labels.shape}")
            print(f"Image min/max pixel values: {tf.reduce_min(images):.4f} / {tf.reduce_max(images):.4f}")
    else:
        print(f"No train.csv found at {train_csv}. Run splitting step first to test.")
