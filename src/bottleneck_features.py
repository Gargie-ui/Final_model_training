import os
import numpy as np
import tensorflow as tf
from src import config
from src import preprocessing
from src import augmentation

def get_feature_extractor():
    """
    Instantiates the pretrained DenseNet121 base model with global average pooling.
    This model outputs a 1024-dimensional vector for each preprocessed image.
    """
    print("Loading pretrained DenseNet121 for bottleneck feature extraction...")
    base_model = tf.keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3)
    )
    base_model.trainable = False
    return base_model

def extract_and_cache_split(split_name, csv_path, base_model, force=False):
    """
    Extracts features for a single split and saves them to a compressed .npz file.
    For the train split, we dynamically perform offline data augmentation on minority classes
    (Atelectasis, Infiltration, Lung_Tumor) to balance the representation and avoid OOM.
    """
    os.makedirs(config.FEATURES_DIR, exist_ok=True)
    npz_path = os.path.join(config.FEATURES_DIR, f"{split_name}_features.npz")
    
    if os.path.exists(npz_path) and not force:
        print(f"Cached features for '{split_name}' already exist at {npz_path}. Skipping extraction.")
        return
        
    print(f"\nExtracting bottleneck features for split '{split_name}'...")
    # Load dataset WITHOUT shuffling to maintain alignment between features and labels
    dataset, count = preprocessing.build_dataset_from_csv(csv_path, shuffle=False)
    
    all_features = []
    all_labels = []
    
    # Load data augmentation layers for train split
    aug_layers = None
    if split_name == "train":
        aug_layers = augmentation.get_augmentation_layers()
        
    total_batches = int(np.ceil(count / config.BATCH_SIZE))
    
    # Iterate exactly once over the dataset to extract both features and labels
    for i, (images, labels) in enumerate(dataset):
        # 1. Forward pass on the raw batch in inference mode (training=False)
        batch_feats = base_model(images, training=False)
        all_features.append(batch_feats.numpy())
        all_labels.append(labels.numpy())
        
        # 2. Apply offline data augmentation for training split only
        if split_name == "train" and aug_layers is not None:
            labels_np = labels.numpy()
            
            # Generate augmented images for the entire batch
            # We use training=True so that RandomRotation/RandomZoom actually apply their transforms
            aug_images = aug_layers(images, training=True)
            batch_feats_aug = base_model(aug_images, training=False).numpy()
            
            # Selectively append augmented features for minority classes
            for idx in range(len(labels_np)):
                class_idx = np.argmax(labels_np[idx])
                # If class is Atelectasis (0), Infiltration (1), or Lung_Tumor (2):
                if class_idx < 3:
                    # Append first augmented instance
                    all_features.append(batch_feats_aug[idx : idx + 1])
                    all_labels.append(labels_np[idx : idx + 1])
                    
                # If class is Lung_Tumor (2), add a second augmented instance to boost it further
                if class_idx == 2:
                    single_img = tf.expand_dims(images[idx], axis=0)
                    aug_img_2 = aug_layers(single_img, training=True)
                    feat_2 = base_model(aug_img_2, training=False).numpy()
                    all_features.append(feat_2)
                    all_labels.append(labels_np[idx : idx + 1])
                    
        # Periodic console logging (carriage return to keep progress clean)
        if (i + 1) % 10 == 0 or (i + 1) == total_batches:
            print(f"  Processed batch {i + 1}/{total_batches}...", end="\r", flush=True)
            
    print() # New line after carriage return progress
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    print(f"Extraction completed. Features shape: {features.shape}, Labels shape: {labels.shape}")
    
    # Save as compressed archive
    np.savez_compressed(npz_path, features=features, labels=labels)
    print(f"Saved cached features to: {npz_path}")

def extract_and_cache_all(force=False):
    """
    Extracts and caches bottleneck features for train, val, and test splits.
    """
    if not config.USE_CACHED_FEATURES or config.UNFREEZE_TOP_N_LAYERS > 0:
        print("Feature caching is disabled by configuration (or fine-tuning is enabled). Skipping.")
        return
        
    train_csv = os.path.join(config.NEW_DATASET_DIR, "train.csv")
    val_csv = os.path.join(config.NEW_DATASET_DIR, "val.csv")
    test_csv = os.path.join(config.NEW_DATASET_DIR, "test.csv")
    
    # Check if all files exist to avoid loading the model if not needed
    splits_to_run = []
    for name, path in [("train", train_csv), ("val", val_csv), ("test", test_csv)]:
        npz_path = os.path.join(config.FEATURES_DIR, f"{name}_features.npz")
        if not os.path.exists(npz_path) or force:
            splits_to_run.append((name, path))
            
    if not splits_to_run:
        print("All bottleneck features are already cached. Ready to train.")
        return
        
    # Instantiate extractor model since we need to extract at least one split
    base_model = get_feature_extractor()
    
    for name, path in splits_to_run:
        extract_and_cache_split(name, path, base_model, force=force)
        
    print("\nAll splits successfully precomputed and cached.")

def load_cached_features(split_name):
    """
    Loads precomputed features and labels for a given split from disk.
    """
    npz_path = os.path.join(config.FEATURES_DIR, f"{split_name}_features.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Cached features for '{split_name}' not found at: {npz_path}. Run bottleneck_features.py first.")
        
    data = np.load(npz_path)
    return data["features"], data["labels"]

if __name__ == "__main__":
    # If executed directly, precompute all splits
    extract_and_cache_all()
