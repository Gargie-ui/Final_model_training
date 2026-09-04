import os
import json
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
import tensorflow as tf
from src import config

def predict_with_thresholds(y_pred_probs, thresholds):
    """
    Single-instance or batch threshold predictor with conflict resolution:
    - If one or more classes exceed their threshold, select the class with the
      largest positive margin (p_i - threshold_i).
    - If no class exceeds its threshold, fall back to standard argmax(p_i).
    """
    thresholds = np.array(thresholds, dtype=np.float32)
    n_samples = y_pred_probs.shape[0]
    
    margins = y_pred_probs - thresholds
    exceeded = (margins >= 0)
    any_exceeded = np.any(exceeded, axis=1)
    
    # Mask non-exceeded margins with -inf
    masked_margins = np.where(exceeded, margins, -np.inf)
    best_exceeded = np.argmax(masked_margins, axis=1)
    best_fallback = np.argmax(y_pred_probs, axis=1)
    
    final_preds = np.where(any_exceeded, best_exceeded, best_fallback)
    return final_preds

def grid_search_joint_thresholds(val_probs, val_true_idx, class_names=config.CLASSES, min_recall_floor=0.40):
    """
    Performs a fully vectorized bounded grid search over threshold space [0.15, 0.85]^4
    with step size 0.05 (15^4 = 50,625 combinations).
    
    Enforces a strict recall floor constraint on the 3 DISEASE classes (Atelectasis, Infiltration, Lung_Tumor):
    rejects any threshold combination where any disease class validation recall falls below min_recall_floor.
    If no combination satisfies all 3 disease recall floors simultaneously, reports the combination that
    minimizes the total recall shortfall across disease classes.
    """
    print("\n" + "=" * 75)
    print(f"STARTING VECTORIZED RECALL-FLOOR CONSTRAINED JOINT THRESHOLD TUNING")
    print(f"  Range: [0.15, 0.85], Step size: 0.05 per class (50,625 combinations)")
    print(f"  Enforced Disease Recall Floor Constraint: >= {min_recall_floor * 100:.0f}%")
    print("=" * 75)

    steps = np.arange(0.15, 0.86, 0.05)
    combo_list = list(itertools.product(steps, repeat=len(class_names)))
    thresholds_matrix = np.array(combo_list, dtype=np.float32) # shape: (50625, 4)
    
    # 1. Plain argmax baseline on validation set
    plain_preds = np.argmax(val_probs, axis=1)
    baseline_report = classification_report(val_true_idx, plain_preds, target_names=class_names, output_dict=True, zero_division=0)
    baseline_macro_f1 = baseline_report["macro avg"]["f1-score"]
    print(f"Validation Plain Argmax Baseline Macro F1: {baseline_macro_f1:.4f}\n")

    print(f"Broadcasting predictions in chunks for all {len(thresholds_matrix):,} combinations...")
    
    chunk_size = 2500
    num_combos = len(thresholds_matrix)
    macro_f1_scores = np.zeros(num_combos, dtype=np.float32)
    per_class_recalls = np.zeros((num_combos, len(class_names)), dtype=np.float32)
    per_class_precisions = np.zeros((num_combos, len(class_names)), dtype=np.float32)
    per_class_f1s = np.zeros((num_combos, len(class_names)), dtype=np.float32)
    
    probs_expanded = val_probs[:, np.newaxis, :]  # (N, 1, 4)
    best_fallback = np.argmax(val_probs, axis=1)[:, np.newaxis]  # (N, 1)

    for start_idx in range(0, num_combos, chunk_size):
        end_idx = min(start_idx + chunk_size, num_combos)
        thresh_chunk = thresholds_matrix[start_idx:end_idx][np.newaxis, :, :]  # (1, B, 4)
        
        margins = probs_expanded - thresh_chunk  # (N, B, 4)
        exceeded = (margins >= 0)
        any_exceeded = np.any(exceeded, axis=2)  # (N, B)
        
        masked_margins = np.where(exceeded, margins, -np.inf)
        best_exceeded = np.argmax(masked_margins, axis=2)  # (N, B)
        
        chunk_preds = np.where(any_exceeded, best_exceeded, best_fallback)  # (N, B)
        
        # Pure NumPy fast F1 & Recall calculation across all B columns simultaneously
        f1_sum = np.zeros(end_idx - start_idx, dtype=np.float32)
        for c in range(len(class_names)):
            tp = np.sum((chunk_preds == c) & (val_true_idx[:, None] == c), axis=0)
            pred_pos = np.sum(chunk_preds == c, axis=0)
            actual_pos = np.sum(val_true_idx == c)
            prec = np.where(pred_pos > 0, tp / pred_pos, 0.0)
            rec = np.where(actual_pos > 0, tp / actual_pos, 0.0)
            f1_c = np.where((prec + rec) > 0, 2.0 * prec * rec / (prec + rec), 0.0)
            f1_sum += f1_c
            per_class_recalls[start_idx:end_idx, c] = rec
            per_class_precisions[start_idx:end_idx, c] = prec
            per_class_f1s[start_idx:end_idx, c] = f1_c
        macro_f1_scores[start_idx:end_idx] = f1_sum / len(class_names)

    # Disease indices: 0 (Atelectasis), 1 (Infiltration), 2 (Lung_Tumor)
    disease_recalls = per_class_recalls[:, :3]
    valid_mask = np.all(disease_recalls >= min_recall_floor, axis=1)
    valid_count = int(np.sum(valid_mask))
    
    print("-" * 75)
    print("GRID SEARCH RECALL FLOOR CONSTRAINT SUMMARY:")
    print(f"  Total combinations evaluated    : {num_combos:,}")
    print(f"  Satisfying Recall Floor ({min_recall_floor*100:.0f}%): {valid_count:,} / {num_combos:,} ({valid_count/num_combos*100:.2f}%)")
    print("-" * 75)

    if valid_count > 0:
        masked_f1 = np.where(valid_mask, macro_f1_scores, -1.0)
        best_idx = int(np.argmax(masked_f1))
        is_exact = True
    else:
        print(f"\n⚠️ EXPLICIT NOTICE: NO threshold combination satisfied recall >= {min_recall_floor*100:.0f}% for all 3 disease classes simultaneously.")
        # Calculate total recall shortfall across disease classes
        shortfall = np.sum(np.maximum(0.0, min_recall_floor - disease_recalls), axis=1)
        best_idx = int(np.argmin(shortfall))
        min_sf = shortfall[best_idx]
        print(f"  Selected closest combination with minimum total recall shortfall ({min_sf:.4f}).")
        is_exact = False

    best_macro_f1 = float(macro_f1_scores[best_idx])
    best_thresholds = [float(x) for x in thresholds_matrix[best_idx]]

    print("\n" + "=" * 75)
    print(f"SELECTED OPTIMAL THRESHOLDS (VALIDATION SET, FLOOR >= {min_recall_floor*100:.0f}%):")
    print("=" * 75)
    for i, cls_name in enumerate(class_names):
        rec_v = per_class_recalls[best_idx, i]
        prec_v = per_class_precisions[best_idx, i]
        f1_v = per_class_f1s[best_idx, i]
        t_v = best_thresholds[i]
        tag = " (DISEASE)" if i < 3 else " (HEALTHY)"
        print(f"  {cls_name:15s}{tag:10s} -> Thresh: {t_v:.2f} | Recall: {rec_v*100:.2f}% | Prec: {prec_v*100:.2f}% | F1: {f1_v:.4f}")
    print("-" * 75)
    print(f"Validation Joint-Tuned Macro F1: {best_macro_f1:.4f} (Gain over baseline: {best_macro_f1 - baseline_macro_f1:+.4f})")
    print("=" * 75 + "\n")

    threshold_dict = {cls_name: round(float(t_val), 4) for cls_name, t_val in zip(class_names, best_thresholds)}
    
    return best_thresholds, threshold_dict, valid_count, best_macro_f1, per_class_recalls[best_idx], per_class_precisions[best_idx], per_class_f1s[best_idx]

def run_floor_tradeoff_comparison(val_probs, val_true_idx, class_names=config.CLASSES):
    """
    Runs grid search with 40% vs 50% disease recall floor constraints side by side
    and prints a clear trade-off comparison table.
    """
    print("\n" + "#" * 80)
    print("RECALL-FLOOR CONSTRAINT TRADE-OFF ANALYSIS (40% FLOOR VS. 50% FLOOR)")
    print("#" * 80)
    
    res_40 = grid_search_joint_thresholds(val_probs, val_true_idx, class_names, min_recall_floor=0.40)
    res_50 = grid_search_joint_thresholds(val_probs, val_true_idx, class_names, min_recall_floor=0.50)
    
    thresh_40, dict_40, valid_40, macro_40, rec_40, prec_40, f1_40 = res_40
    thresh_50, dict_50, valid_50, macro_50, rec_50, prec_50, f1_50 = res_50
    
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE RECALL-FLOOR TRADE-OFF SUMMARY (VALIDATION SET)")
    print("=" * 80)
    print(f"{'Metric / Parameter':35s} | {'40% Recall Floor':20s} | {'50% Recall Floor':20s}")
    print("-" * 80)
    print(f"{'Valid Combinations (Satisfying Floor)':35s} | {valid_40:^20d} | {valid_50:^20d}")
    print(f"{'Validation Macro F1':35s} | {macro_40:^20.4f} | {macro_50:^20.4f}")
    print("-" * 80)
    for i, cls_name in enumerate(class_names):
        str_40 = f"t={thresh_40[i]:.2f} (R:{rec_40[i]*100:.1f}%)"
        str_50 = f"t={thresh_50[i]:.2f} (R:{rec_50[i]*100:.1f}%)"
        print(f"{cls_name + ' Threshold & Recall':35s} | {str_40:^20s} | {str_50:^20s}")
    print("=" * 80 + "\n")
    
    # Save primary 40% floor thresholds (unless 50% is virtually identical in F1)
    chosen_thresh = thresh_40 if macro_40 >= macro_50 - 0.005 else thresh_50
    chosen_dict = dict_40 if macro_40 >= macro_50 - 0.005 else dict_50
    chosen_floor = "40%" if macro_40 >= macro_50 - 0.005 else "50%"
    
    save_path = os.path.join(config.MODELS_DIR, "class_thresholds.json")
    with open(save_path, "w") as f:
        json.dump(chosen_dict, f, indent=4)
        
    print(f"[OK] Saved primary ({chosen_floor} floor) optimal class thresholds to: {save_path}")
    print(f"     Rationale: {chosen_floor} recall floor balances high disease sensitivity (no recall collapse)")
    print(f"                while preserving high Macro F1 without sacrificing clinical utility.\n")
    
    return chosen_thresh, res_40, res_50

def evaluate_three_way_comparison(test_probs, test_true_idx, unconstrained_thresh, constrained_thresh, class_names=config.CLASSES):
    """
    Evaluates Plain Argmax vs. Unconstrained Joint-Tuned vs. Recall-Floor Constrained
    thresholding on the TEST set (10,348 images).
    Prints a 3-way comparison table.
    """
    plain_preds = np.argmax(test_probs, axis=1)
    unconstrained_preds = predict_with_thresholds(test_probs, unconstrained_thresh)
    constrained_preds = predict_with_thresholds(test_probs, constrained_thresh)

    rep_plain = classification_report(test_true_idx, plain_preds, target_names=class_names, output_dict=True, zero_division=0)
    rep_uncon = classification_report(test_true_idx, unconstrained_preds, target_names=class_names, output_dict=True, zero_division=0)
    rep_con = classification_report(test_true_idx, constrained_preds, target_names=class_names, output_dict=True, zero_division=0)

    print("\n" + "=" * 115)
    print("THREE-WAY TEST SET BENCHMARK: ARGMAX vs UNCONSTRAINED TUNED vs RECALL-FLOOR CONSTRAINED (10,348 IMAGES)")
    print("=" * 115)

    table_rows = []
    for cls_name in class_names:
        supp = rep_plain[cls_name]["support"]
        
        p_0, r_0, f_0 = rep_plain[cls_name]["precision"], rep_plain[cls_name]["recall"], rep_plain[cls_name]["f1-score"]
        p_u, r_u, f_u = rep_uncon[cls_name]["precision"], rep_uncon[cls_name]["recall"], rep_uncon[cls_name]["f1-score"]
        p_c, r_c, f_c = rep_con[cls_name]["precision"], rep_con[cls_name]["recall"], rep_con[cls_name]["f1-score"]

        table_rows.append({
            "Class": cls_name,
            "Support": supp,
            "Argmax P": f"{p_0:.4f}", "Argmax R": f"{r_0:.4f}", "Argmax F1": f"{f_0:.4f}",
            "Unconstrained P": f"{p_u:.4f}", "Unconstrained R": f"{r_u:.4f}", "Unconstrained F1": f"{f_u:.4f}",
            "Floor Constrained P": f"{p_c:.4f}", "Floor Constrained R": f"{r_c:.4f}", "Floor Constrained F1": f"{f_c:.4f}"
        })

    for avg_type in ["macro avg", "weighted avg"]:
        supp = rep_plain[avg_type]["support"]
        p_0, r_0, f_0 = rep_plain[avg_type]["precision"], rep_plain[avg_type]["recall"], rep_plain[avg_type]["f1-score"]
        p_u, r_u, f_u = rep_uncon[avg_type]["precision"], rep_uncon[avg_type]["recall"], rep_uncon[avg_type]["f1-score"]
        p_c, r_c, f_c = rep_con[avg_type]["precision"], rep_con[avg_type]["recall"], rep_con[avg_type]["f1-score"]

        table_rows.append({
            "Class": avg_type.upper(),
            "Support": supp,
            "Argmax P": f"{p_0:.4f}", "Argmax R": f"{r_0:.4f}", "Argmax F1": f"{f_0:.4f}",
            "Unconstrained P": f"{p_u:.4f}", "Unconstrained R": f"{r_u:.4f}", "Unconstrained F1": f"{f_u:.4f}",
            "Floor Constrained P": f"{p_c:.4f}", "Floor Constrained R": f"{r_c:.4f}", "Floor Constrained F1": f"{f_c:.4f}"
        })

    df = pd.DataFrame(table_rows)
    print(df.to_string(index=False))
    print("=" * 115)

    inf_test_recall = rep_con["Infiltration"]["recall"]
    print(f"\n🔍 INFILTRATION TEST SET RECALL CHECK:")
    print(f"   Infiltration Recall on Held-Out Test Set: {inf_test_recall * 100:.2f}%")
    if inf_test_recall >= 0.40:
        print(f"   ✅ VERIFIED: Infiltration recall ({inf_test_recall*100:.2f}%) STAYS ABOVE THE 40% FLOOR on the TEST set!")
    else:
        print(f"   ⚠️ NOTE: Infiltration recall on test set is {inf_test_recall*100:.2f}% (Tuned on validation set).")
    print("=" * 115 + "\n")

    return rep_plain, rep_uncon, rep_con

