import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from src import config
from src import preprocessing

def run_evaluation(history_dict=None):
    """
    Evaluates the final saved model on the held-out test dataset and plots
    performance metrics.
    """
    print("\n" + "="*70)
    print("EVALUATING MODEL ON HELD-OUT TEST SPLIT")
    print("="*70)
    
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # 1. Load the saved model
    final_model_path = os.path.join(config.MODELS_DIR, "final_model.keras")
    if not os.path.exists(final_model_path):
        raise FileNotFoundError(f"Trained model not found at: {final_model_path}. Run training first.")
        
    from src.augmentation import FocalLoss
    print(f"Loading final trained model from: {final_model_path}")
    model = tf.keras.models.load_model(final_model_path, custom_objects={'FocalLoss': FocalLoss}, compile=False)
    
    # 2. Load the test dataset (always run end-to-end on raw images to verify full pipeline)
    test_csv = os.path.join(config.NEW_DATASET_DIR, "test.csv")
    test_ds, test_count = preprocessing.build_dataset_from_csv(test_csv, shuffle=False)
    
    print(f"Running predictions on {test_count} test images...")
    y_pred = model.predict(test_ds, verbose=1)
    
    # 3. Extract true labels from the dataset
    all_true_labels = []
    for _, labels in test_ds:
        all_true_labels.append(labels.numpy())
    y_true = np.concatenate(all_true_labels, axis=0)
    
    y_true_idx = np.argmax(y_true, axis=1)
    y_pred_idx = np.argmax(y_pred, axis=1)
    
    # 4. Perform Joint Threshold Tuning on Validation set
    from src import threshold_tuning
    val_csv = os.path.join(config.NEW_DATASET_DIR, "val.csv")
    val_ds, val_count = preprocessing.build_dataset_from_csv(val_csv, shuffle=False)
    print(f"\nRunning predictions on {val_count:,} validation images for threshold tuning...")
    val_probs = model.predict(val_ds, verbose=1)
    
    all_val_labels = []
    for _, labels in val_ds:
        all_val_labels.append(labels.numpy())
    val_true = np.concatenate(all_val_labels, axis=0)
    val_true_idx = np.argmax(val_true, axis=1)
    
    # 1. Unconstrained Joint Threshold Search (Previous result)
    unconstrained_thresh, _, _, _, _, _, _ = threshold_tuning.grid_search_joint_thresholds(val_probs, val_true_idx, min_recall_floor=0.0)
    
    # 2. Side-by-Side Recall Floor Trade-off Analysis (40% Floor vs. 50% Floor)
    chosen_thresh, res_40, res_50 = threshold_tuning.run_floor_tradeoff_comparison(val_probs, val_true_idx)
    
    # 3. Three-Way Test Set Evaluation (Plain Argmax vs Unconstrained vs Recall-Floor Constrained)
    threshold_tuning.evaluate_three_way_comparison(y_pred, y_true_idx, unconstrained_thresh, chosen_thresh)
    
    # 5. Generate Classification Report (Plain Argmax)
    print("\nClassification Report (Plain Argmax):")
    report = classification_report(
        y_true_idx,
        y_pred_idx,
        target_names=config.CLASSES,
        digits=4
    )
    print(report)
    
    # Extract macro F1, precision, and recall to print them prominently
    report_dict = classification_report(y_true_idx, y_pred_idx, target_names=config.CLASSES, output_dict=True)
    macro_precision = report_dict["macro avg"]["precision"]
    macro_recall = report_dict["macro avg"]["recall"]
    macro_f1 = report_dict["macro avg"]["f1-score"]
    
    print("-" * 50)
    print(f"PRIMARY PERFORMANCE METRICS (Macro Averages):")
    print(f"  Macro-averaged Precision: {macro_precision:.4f}")
    print(f"  Macro-averaged Recall:    {macro_recall:.4f}")
    print(f"  Macro-averaged F1-Score:  {macro_f1:.4f}")
    print("-" * 50)
    
    # 5. Plot dual confusion matrices (Raw and Normalized) side-by-side
    cm = confusion_matrix(y_true_idx, y_pred_idx)
    # Row-wise normalization (percentages of true classes, representing sensitivity/recall)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    
    # Left subplot: Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=True,
        xticklabels=config.CLASSES, yticklabels=config.CLASSES, ax=axes[0]
    )
    axes[0].set_title("Confusion Matrix (Raw Counts)", fontsize=12, fontweight="bold", pad=10)
    axes[0].set_xlabel("Predicted Label", fontsize=10)
    axes[0].set_ylabel("True Label", fontsize=10)
    
    # Right subplot: Normalized row-wise
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2%", cmap="Blues", cbar=True,
        xticklabels=config.CLASSES, yticklabels=config.CLASSES, ax=axes[1]
    )
    axes[1].set_title("Confusion Matrix (Normalized / Recall)", fontsize=12, fontweight="bold", pad=10)
    axes[1].set_xlabel("Predicted Label", fontsize=10)
    axes[1].set_ylabel("True Label", fontsize=10)
    
    fig.suptitle("Model Classification Confusion Analysis", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    
    cm_path = os.path.join(config.MODELS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"Saved dual confusion matrix plot to: {cm_path}")
    
    # 6. Plot ROC curves (One-vs-Rest)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    fpr_dict = {}
    tpr_dict = {}
    roc_auc_dict = {}
    
    # Compute ROC and AUC per class
    for i, class_name in enumerate(config.CLASSES):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        
        ax.plot(
            fpr_dict[i], tpr_dict[i],
            label=f"ROC: {class_name} (AUC = {roc_auc_dict[i]:.4f})",
            linewidth=2
        )
        
    # Calculate Macro ROC-AUC
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(len(config.CLASSES))]))
    # Then interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(config.CLASSES)):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    # Average and compute AUC
    mean_tpr /= len(config.CLASSES)
    
    macro_auc = auc(all_fpr, mean_tpr)
    
    ax.plot(
        all_fpr, mean_tpr,
        label=f"Macro-average ROC (AUC = {macro_auc:.4f})",
        color="deeppink", linestyle=":", linewidth=3
    )
    
    ax.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.5000)", linewidth=1.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="bold")
    ax.set_title("One-vs-Rest ROC Curves", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none")
    
    roc_path = os.path.join(config.MODELS_DIR, "roc_curves.png")
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()
    print(f"Saved One-vs-Rest ROC curve plot to: {roc_path}")
    
    # 7. Plot training history (Loss & Accuracy curves)
    if history_dict is None:
        history_path = os.path.join(config.MODELS_DIR, "training_history.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history_dict = json.load(f)
                
    if history_dict is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
        
        # Loss curves
        epochs_range = range(1, len(history_dict["loss"]) + 1)
        axes[0].plot(epochs_range, history_dict["loss"], label="Train Loss", color="#4F46E5", linewidth=2)
        axes[0].plot(epochs_range, history_dict["val_loss"], label="Val Loss", color="#10B981", linewidth=2)
        axes[0].set_title("Training & Validation Loss", fontsize=12, fontweight="bold", pad=10)
        axes[0].set_xlabel("Epoch", fontsize=10)
        axes[0].set_ylabel("Loss", fontsize=10)
        axes[0].set_xticks(epochs_range)
        axes[0].legend(frameon=True, facecolor="white")
        
        # Accuracy curves
        axes[1].plot(epochs_range, history_dict["accuracy"], label="Train Acc", color="#4F46E5", linewidth=2)
        axes[1].plot(epochs_range, history_dict["val_accuracy"], label="Val Acc", color="#10B981", linewidth=2)
        axes[1].set_title("Training & Validation Accuracy", fontsize=12, fontweight="bold", pad=10)
        axes[1].set_xlabel("Epoch", fontsize=10)
        axes[1].set_ylabel("Accuracy", fontsize=10)
        axes[1].set_xticks(epochs_range)
        axes[1].legend(frameon=True, facecolor="white")
        
        fig.suptitle("Training Optimization Curves", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        
        history_plot_path = os.path.join(config.MODELS_DIR, "loss_accuracy_curves.png")
        plt.savefig(history_plot_path, bbox_inches="tight")
        plt.close()
        print(f"Saved loss & accuracy curves plot to: {history_plot_path}")
        
    print("Evaluation completed successfully.")
    print("=" * 70)

if __name__ == "__main__":
    # Run evaluation if executed directly
    run_evaluation()
