import os
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
import tensorflow as tf
from src import config

def get_augmentation_layers():
    """
    Returns a Sequential layer of Keras data augmentations to apply to raw train images.
    Note: Skip horizontal flip because in chest X-rays, left/right laterality is critical,
    and flipping can invert anatomical features (e.g. placing the heart on the right side),
    confusing the network and making training data unrealistic.
    """
    # Using Keras preprocessing layers for augmentation (modest and CPU-friendly)
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(
            factor=0.10,  # ±10 degrees (0.1 * 360 = 36 degrees max rotation)
            fill_mode="reflect",
            seed=config.RANDOM_SEED
        ),
        tf.keras.layers.RandomZoom(
            height_factor=0.10,
            width_factor=0.10,
            fill_mode="reflect",
            seed=config.RANDOM_SEED
        )
    ], name="data_augmentation")
    
    return augmentation

def compute_train_class_weights(train_csv_path):
    """
    Computes class weights based on the class distribution in the train split.
    Why Class Weights vs Oversampling?
    Oversampling minority classes increases the dataset size by duplicating images,
    which increases CPU forward/backward pass times and memory usage, and risks overfitting.
    Class weighting adjusts the loss function's cost penalty on a per-class basis,
    ensuring gradients from minority classes aren't drowned out, without any added compute cost.
    """
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Train split CSV not found at: {train_csv_path}")
        
    df = pd.read_csv(train_csv_path)
    labels = df["label"].values
    
    # Map label names to class index integers (0-3)
    label_indices = np.array([config.CLASSES.index(l) for l in labels])
    
    # Compute balanced class weights
    classes_present = np.unique(label_indices)
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=classes_present,
        y=label_indices
    )
    
    # Softened class weights (sqrt power of 0.5) to prevent over-penalizing No_Finding
    damped_weights = weights ** 0.5
    counts = np.bincount(label_indices)
    weighted_sum = np.sum(damped_weights * counts)
    normalized_weights = damped_weights * (len(label_indices) / weighted_sum)
    
    # Map to dictionary format {class_index: weight_value} expected by model.fit()
    class_weights_dict = {int(cls): float(w) for cls, w in zip(classes_present, normalized_weights)}
    
    print("\nComputed Class Weights for Train Split:")
    for cls_idx, weight in class_weights_dict.items():
        cls_name = config.CLASSES[cls_idx]
        print(f"  Class {cls_idx} ({cls_name}): {weight:.4f}")
        
    return class_weights_dict

class FocalLoss(tf.keras.losses.Loss):
    """
    Implements Focal Loss for multi-class classification:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        
    Comment / Rationale:
    Focal loss down-weights easy/confident examples (mostly correctly-classified No_Finding)
    dynamically per-example during training, unlike static class weights which apply a fixed
    multiplier regardless of how easy or hard each individual prediction is.
    """
    def __init__(self, alpha_weights=None, gamma=2.0, reduction=tf.keras.losses.Reduction.AUTO, name="focal_loss"):
        super(FocalLoss, self).__init__(reduction=reduction, name=name)
        self.gamma = float(gamma)
        self.alpha_weights = alpha_weights
        if alpha_weights is not None:
            self.alpha = tf.constant(alpha_weights, dtype=tf.float32)
        else:
            self.alpha = None

    def call(self, y_true, y_pred):
        # Clip predictions to prevent log(0) numerical instability
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate p_t (probability of the true class)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        
        # Calculate focal factor: (1 - p_t)^gamma
        focal_factor = tf.pow(1.0 - p_t, self.gamma)
        
        # Calculate log(p_t)
        log_p_t = tf.math.log(p_t)
        
        # Apply alpha_t class weighting term if provided
        if self.alpha is not None:
            alpha_t = tf.reduce_sum(y_true * self.alpha, axis=-1)
            loss = -alpha_t * focal_factor * log_p_t
        else:
            loss = -focal_factor * log_p_t
            
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super(FocalLoss, self).get_config()
        config.update({
            "alpha_weights": self.alpha_weights,
            "gamma": self.gamma,
        })
        return config

if __name__ == "__main__":
    # Test class weights calculation
    train_csv = os.path.join(config.NEW_DATASET_DIR, "train.csv")
    if os.path.exists(train_csv):
        compute_train_class_weights(train_csv)
    else:
        print(f"No train.csv found at {train_csv} to compute weights. Run dataset_splitter.py first.")
