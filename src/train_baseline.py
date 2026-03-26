from __future__ import annotations

# PI4: Train baseline model on clean speech features (Mozilla Common Voice)
# Usage (from project root):
#   python src/train_baseline.py
#
# Outputs written to --output-dir:
#   baseline_results.json       full metrics summary
#   learning_curve.csv          per-train-size train/val accuracy (stability evidence)
#   cv_results.csv              5-fold cross-validation scores (stability evidence)

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_score, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


DEFAULT_FEATURE_CSV = Path("data/feature_cache_mozilla.csv")
DEFAULT_OUTPUT_DIR  = Path("data/processed/baseline")
MIN_UTTERANCES      = 5
VAL_RATIO           = 0.20
RANDOM_STATE        = 42

META_COLS = {"path", "client_id", "spk_idx", "gender", "age", "duration"}


def load_clean_data(csv_path, min_utterances):
    df = pd.read_csv(csv_path)

    feature_cols = []
    for col in df.columns:
        if col not in META_COLS and pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    before = len(df)
    df = df.dropna(subset=feature_cols + ["spk_idx"]).reset_index(drop=True)
    if len(df) < before:
        print(f"  Dropped {before - len(df)} rows with missing values")

    counts = df["spk_idx"].value_counts()
    valid_speakers = counts[counts >= min_utterances].index
    df = df[df["spk_idx"].isin(valid_speakers)].reset_index(drop=True)

    return df, feature_cols


# metrics

def top_k_accuracy(y_true, y_prob, k=5):
    top_k = np.argsort(y_prob, axis=1)[:, -k:]
    correct = [y_true[i] in top_k[i] for i in range(len(y_true))]
    return float(np.mean(correct))

def compute_learning_curve(model, X, y, n_points=8, cv=3):
    train_sizes = np.linspace(0.15, 1.0, n_points)
    sizes, tr_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    rows = []
    for i in range(len(sizes)):
        rows.append({
            "train_size": int(sizes[i]),
            "train_mean": float(tr_scores[i].mean()),
            "train_std":  float(tr_scores[i].std()),
            "val_mean":   float(val_scores[i].mean()),
            "val_std":    float(val_scores[i].std()),
        })
    return pd.DataFrame(rows)


def compute_cv_scores(model, X, y, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    rows = []
    for i, score in enumerate(scores):
        rows.append({"fold": i + 1, "accuracy": float(score)})
    return pd.DataFrame(rows)


#  core training function 

def train_baseline(
    csv_path=DEFAULT_FEATURE_CSV,
    output_dir=DEFAULT_OUTPUT_DIR,
    min_utterances=MIN_UTTERANCES,
    val_ratio=VAL_RATIO,
    random_state=RANDOM_STATE,
    max_iter=1000,
    C=1.0,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    #  1. data 
    print("Loading clean feature data …")
    df, feature_cols = load_clean_data(csv_path, min_utterances)
    n_speakers = df["spk_idx"].nunique()
    print(f"  {len(df)} samples | {n_speakers} speakers | {len(feature_cols)} features")

    X  = df[feature_cols].values.astype(np.float32)
    le = LabelEncoder()
    y  = le.fit_transform(df["spk_idx"].values)

    #  2. train-val split 
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
    train_idx, val_idx = next(sss.split(X, y))
    X_tr,  X_val  = X[train_idx], X[val_idx]
    y_tr,  y_val  = y[train_idx], y[val_idx]
    print(f"  Train: {len(X_tr)} | Val: {len(X_val)}")

    #  3. feature scaling 
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_val_sc = scaler.transform(X_val)

    #  4. train baseline model 
    print(f"\nTraining LogisticRegression (C={C}, max_iter={max_iter}) …")
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(X_tr_sc, y_tr)
    print("  Training complete")

    #  5a. learning curve 
    print("\nComputing learning curve (stability check) …")
    lc_model = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs",
                                   random_state=random_state, n_jobs=1)
    lc_df = compute_learning_curve(lc_model, X_tr_sc, y_tr, n_points=8, cv=3)
    lc_path = output_dir / "learning_curve.csv"
    lc_df.to_csv(lc_path, index=False)

    last3_std = float(lc_df["val_mean"].iloc[-3:].std())
    is_stable = last3_std < 0.005
    print(f"  Last-3-point val_mean std = {last3_std:.5f}  →  stable={is_stable}")
    for _, row in lc_df.iterrows():
        print(
            f"  train_size={int(row['train_size']):4d}  "
            f"train={row['train_mean']:.4f}±{row['train_std']:.4f}  "
            f"val={row['val_mean']:.4f}±{row['val_std']:.4f}"
        )

    #  5b. 5-fold cross-validation 
    print("\nRunning 5-fold cross-validation …")
    cv_model = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs",
                                   random_state=random_state, n_jobs=1)
    cv_df = compute_cv_scores(cv_model, X_tr_sc, y_tr, n_splits=5)
    cv_path = output_dir / "cv_results.csv"
    cv_df.to_csv(cv_path, index=False)
    cv_mean = float(cv_df["accuracy"].mean())
    cv_std  = float(cv_df["accuracy"].std())
    print(f"  CV accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
    for _, row in cv_df.iterrows():
        print(f"  fold {int(row['fold'])}: {row['accuracy']:.4f}")

    #  6. final evaluation 
    print("\nFinal evaluation on held-out validation set …")
    y_pred_tr  = model.predict(X_tr_sc)
    y_pred_val = model.predict(X_val_sc)
    y_prob_val = model.predict_proba(X_val_sc)

    train_acc = accuracy_score(y_tr,  y_pred_tr)
    val_acc   = accuracy_score(y_val, y_pred_val)
    top5_acc  = top_k_accuracy(y_val, y_prob_val, k=5)
    gap       = train_acc - val_acc

    print(f"  Train accuracy : {train_acc:.4f}")
    print(f"  Val   accuracy : {val_acc:.4f}  (top-5: {top5_acc:.4f})")
    print(f"  Train-val gap  : {gap:.4f}{'  ← potential overfit' if gap > 0.10 else ''}")

    report_str  = classification_report(y_val, y_pred_val,
                                        target_names=[str(c) for c in le.classes_],
                                        zero_division=0)
    report_path = output_dir / "classification_report.txt"
    report_path.write_text(report_str, encoding="utf-8")
 
    results = {
        "dataset":            str(csv_path),
        "n_samples_total":    int(len(df)),
        "n_train":            int(len(X_tr)),
        "n_val":              int(len(X_val)),
        "n_speakers":         int(n_speakers),
        "n_features":         int(len(feature_cols)),
        "feature_cols":       feature_cols,
        "min_utterances":     min_utterances,
        "val_ratio":          val_ratio,
        "model":              {"type": "LogisticRegression", "C": C, "max_iter": max_iter, "solver": "lbfgs"},
        "train_accuracy":     float(train_acc),
        "val_accuracy":       float(val_acc),
        "top5_val_accuracy":  float(top5_acc),
        "train_val_gap":      float(gap),
        "cv_mean_accuracy":   cv_mean,
        "cv_std_accuracy":    cv_std,
        "learning_curve_stable":    is_stable,
        "learning_curve_last3_std": last3_std,
        "artefacts": {
            "learning_curve_csv":    str(lc_path),
            "cv_results_csv":        str(cv_path),
            "classification_report": str(report_path),
        },
    }
    results_path = output_dir / "baseline_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll artefacts written to: {output_dir}")
    print("  baseline_results.json  — metrics summary")
    print("  learning_curve.csv     — stability evidence")
    print("  cv_results.csv         — cross-validation consistency")
    return results


def _parse_args():
    p = argparse.ArgumentParser(description="PI4 — Train clean-data baseline model.")
    p.add_argument("--feature-csv",    type=Path,  default=DEFAULT_FEATURE_CSV)
    p.add_argument("--output-dir",     type=Path,  default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--min-utterances", type=int,   default=MIN_UTTERANCES)
    p.add_argument("--val-ratio",      type=float, default=VAL_RATIO)
    p.add_argument("--C",              type=float, default=1.0)
    p.add_argument("--max-iter",       type=int,   default=1000)
    p.add_argument("--random-state",   type=int,   default=RANDOM_STATE)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results = train_baseline(
        csv_path=args.feature_csv,
        output_dir=args.output_dir,
        min_utterances=args.min_utterances,
        val_ratio=args.val_ratio,
        C=args.C,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )
    print(
        f"\nSummary — val_acc={results['val_accuracy']:.4f}  "
        f"top5={results['top5_val_accuracy']:.4f}  "
        f"cv={results['cv_mean_accuracy']:.4f}±{results['cv_std_accuracy']:.4f}  "
        f"stable={results['learning_curve_stable']}"
    )
