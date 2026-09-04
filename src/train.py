import os
import time
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from src import config
from src import model as classifier_model
from src import preprocessing
from src import bottleneck_features
from src import augmentation

class MacroF1Callback(tf.keras.callbacks.Callback):
    """
    Computes Macro F1 score on the validation dataset at the end of each epoch
    and injects 'val_macro_f1' into logs so EarlyStopping, ReduceLROnPlateau,
    and ModelCheckpoint can monitor it.
    """
    def __init__(self, val_ds, class_names):
        super(MacroF1Callback, self).__init__()
        self.val_ds = val_ds
        self.class_names = class_names
        
        # Pre-extract validation true labels ONCE to avoid duplicate I/O iterations on every epoch
        print("Pre-extracting validation true labels for Macro F1 callback...", flush=True)
        y_true_list = []
        for _, labels in self.val_ds:
            y_true_list.append(labels.numpy())
        y_true = np.concatenate(y_true_list, axis=0)
        self.y_true_idx = np.argmax(y_true, axis=1)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        y_pred_probs = self.model.predict(self.val_ds, verbose=0)
        y_pred_idx = np.argmax(y_pred_probs, axis=1)

        report = classification_report(
            self.y_true_idx, y_pred_idx, target_names=self.class_names, output_dict=True, zero_division=0
        )
        macro_f1 = float(report["macro avg"]["f1-score"])
        logs["val_macro_f1"] = macro_f1
        print(f"\n [Validation Metric] val_macro_f1: {macro_f1:.4f}", flush=True)

class EpochTimeEstimator(tf.keras.callbacks.Callback):
    """
    A custom callback to measure the duration of the first training epoch
    and print a rough estimate of the remaining training time.
    """
    def __init__(self):
        super(EpochTimeEstimator, self).__init__()
        self.epoch_start_time = None
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            self.epoch_start_time = time.time()
            
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            duration = time.time() - self.epoch_start_time
            epochs = self.params.get('epochs', config.EPOCHS)
            estimated_total_sec = duration * epochs
            print("\n" + "="*70)
            print(f"[Time Estimate] First epoch completed in {duration:.1f} seconds ({duration/60:.2f} minutes).")
            print(f"[Time Estimate] Estimated total training time for {epochs} epochs:")
            print(f"  -> ~{estimated_total_sec:.1f} seconds (~{estimated_total_sec/60:.2f} minutes)")
            print("="*70 + "\n")

def run_training():
    """
    Orchestrates the training process based on configuration settings.
    """
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # 1. Compute class weights for loss balancing
    train_csv = os.path.join(config.NEW_DATASET_DIR, "train.csv")
    class_weights = augmentation.compute_train_class_weights(train_csv)
    
    # Save the class-to-index mapping for downstream model consumption (FastAPI / React)
    class_indices = {cls: idx for idx, cls in enumerate(config.CLASSES)}
    mapping_path = os.path.join(config.MODELS_DIR, "class_indices.json")
    with open(mapping_path, "w") as f:
        json.dump(class_indices, f, indent=4)
    print(f"Saved class-to-index mapping to: {mapping_path}")
    
    # 2. Check training mode
    is_caching_active = config.USE_CACHED_FEATURES and config.UNFREEZE_TOP_N_LAYERS == 0
    
    if is_caching_active:
        print("\n" + "="*70)
        print("TRAINING MODE: Head-only Training on Cached Bottleneck Features (CPU OPTIMIZED)")
        print("="*70)
        
        # Load cached features
        train_features, train_labels = bottleneck_features.load_cached_features("train")
        val_features, val_labels = bottleneck_features.load_cached_features("val")
        
        # Build head model
        head_model = classifier_model.build_head_model()
        head_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        # Setup callbacks
        weights_checkpoint_path = os.path.join(config.MODELS_DIR, "best_head_weights.h5")
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=2, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=weights_checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            ),
            EpochTimeEstimator()
        ]
        
        # Train head-only model
        print("Fitting classification head...")
        history = head_model.fit(
            x=train_features,
            y=train_labels,
            validation_data=(val_features, val_labels),
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Re-load best weights into head model
        print(f"Loading best head weights from: {weights_checkpoint_path}")
        head_model.load_weights(weights_checkpoint_path)
        
        # Instantiate full DenseNet121 model and sync weights
        full_model = classifier_model.build_full_model(unfreeze_top_n=0)
        classifier_model.sync_head_weights(head_model, full_model)
        
        # Compile full model so it is saved in a fully compiled/runnable state
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
    else:
        print("\n" + "="*70)
        print("TRAINING MODE: End-to-End Image pipeline (Full Forward Passes)")
        print("="*70)
        
        # Build dataset pipelines
        val_csv = os.path.join(config.NEW_DATASET_DIR, "val.csv")
        train_ds, _ = preprocessing.build_dataset_from_csv(train_csv, shuffle=True)
        val_ds, _ = preprocessing.build_dataset_from_csv(val_csv, shuffle=False)
        
        # Build base model
        full_model = classifier_model.build_full_model(unfreeze_top_n=config.UNFREEZE_TOP_N_LAYERS)
        
        # Wrap full model in data augmentation for training
        inputs = tf.keras.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3), name="input_image")
        x = augmentation.get_augmentation_layers()(inputs)
        outputs = full_model(x)
        trainable_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="trainable_model_with_aug")
        
        # Compute alpha weights vector for Focal Loss from softened class weights
        alpha_weights = [class_weights[i] for i in range(len(config.CLASSES))]
        focal_loss = augmentation.FocalLoss(alpha_weights=alpha_weights, gamma=2.0)
        
        trainable_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss=focal_loss,
            metrics=["accuracy"]
        )
        
        # Compile full model as well to ensure matching structure
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss=focal_loss,
            metrics=["accuracy"]
        )
        
        # Setup callbacks monitoring val_macro_f1 (mode="max", patience=7)
        full_model_checkpoint_path = os.path.join(config.MODELS_DIR, "best_full_model.h5")
        callbacks = [
            MacroF1Callback(val_ds, config.CLASSES),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_macro_f1", mode="max", factor=0.2, patience=2, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_macro_f1", mode="max", patience=7, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=full_model_checkpoint_path,
                monitor="val_macro_f1",
                mode="max",
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            ),
            EpochTimeEstimator()
        ]
        
        # Fit trainable model (class_weight is embedded inside FocalLoss alpha term)
        print("Fitting model with Focal Loss (gamma=2.0)...")
        history = trainable_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.EPOCHS,
            class_weight=None,
            callbacks=callbacks,
            verbose=1
        )
        
        # Load best weights back into trainable model (which syncs full_model inner weights)
        print(f"Loading best weights from: {full_model_checkpoint_path}")
        trainable_model.load_weights(full_model_checkpoint_path)
        
    # Save the final model in Keras format
    final_model_path = os.path.join(config.MODELS_DIR, "final_model.keras")
    print(f"\nSaving final trained model to: {final_model_path}")
    full_model.save(final_model_path)
    
    # Save history as JSON for inspection/plotting later
    history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    history_path = os.path.join(config.MODELS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=4)
        
    print(f"Saved training history to: {history_path}")
    print("Training process finished.")
    print("=" * 70)
    
    return history_dict

if __name__ == "__main__":
    # If run directly, run training
    run_training()
