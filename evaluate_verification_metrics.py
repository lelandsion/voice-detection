
# #21: Define evaluation metrics (FAR, FRR, EER, ROC)
# Use this script to compute verification metrics from trial scores.
# 1. Prepare an input CSV with label and score columns.
# 2. Run: python evaluate_verification_metrics.py --input-csv <path_to_scores_csv>
# 3. Outputs:
# - data/processed/evaluation_metrics.json (FAR, FRR, EER, EER threshold, AUC)
# - data/processed/roc_curve.csv (threshold-wise ROC points: threshold, far, frr, tar)
# Notes:
# - Higher score means more likely genuine.
# - Default genuine label is "1" and all other labels are treated as impostor.
# - If --threshold is omitted, FAR/FRR are reported at the EER threshold.

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_METRICS = Path("data/processed/evaluation_metrics.json")
DEFAULT_OUTPUT_ROC = Path("data/processed/roc_curve.csv")


def _to_binary_labels(labels: pd.Series, genuine_label: str) -> np.ndarray:
    normalized = labels.astype(str).str.strip().str.lower()
    target = genuine_label.strip().lower()
    return (normalized == target).astype(np.int32).to_numpy()


def far_frr_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[float, float]:
    accepts = scores >= threshold

    impostor_mask = y_true == 0
    genuine_mask = y_true == 1

    impostor_total = int(np.sum(impostor_mask))
    genuine_total = int(np.sum(genuine_mask))

    if impostor_total == 0 or genuine_total == 0:
        raise ValueError("Both genuine and impostor samples are required to compute FAR/FRR.")

    false_accepts = int(np.sum(accepts & impostor_mask))
    false_rejects = int(np.sum((~accepts) & genuine_mask))

    far = false_accepts / impostor_total
    frr = false_rejects / genuine_total
    return float(far), float(frr)


def roc_points(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    thresholds = np.unique(scores)
    thresholds = np.concatenate(([np.inf], thresholds[::-1], [-np.inf]))

    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        far, frr = far_frr_at_threshold(y_true, scores, float(threshold))
        tar = 1.0 - frr
        rows.append(
            {
                "threshold": float(threshold),
                "far": far,
                "frr": frr,
                "tar": tar,
            }
        )

    roc_df = pd.DataFrame(rows)
    roc_df = roc_df.sort_values("far").reset_index(drop=True)
    return roc_df


def eer_from_roc(roc_df: pd.DataFrame) -> tuple[float, float]:
    far_vals = roc_df["far"].astype(float).to_numpy()
    frr_vals = roc_df["frr"].astype(float).to_numpy()
    thr_vals = roc_df["threshold"].astype(float).to_numpy()

    diffs = np.abs(far_vals - frr_vals)
    idx = int(np.argmin(diffs))

    far = float(far_vals[idx])
    frr = float(frr_vals[idx])
    threshold = float(thr_vals[idx])
    eer = (far + frr) / 2.0

    return eer, threshold


def auc_from_roc(roc_df: pd.DataFrame) -> float:
    x = roc_df["far"].to_numpy()
    y = roc_df["tar"].to_numpy()
    return float(np.trapz(y, x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute verification metrics (FAR, FRR, EER, ROC, AUC) from score CSV."
    )
    parser.add_argument("--input-csv", type=Path, required=True, help="Input CSV with label and score columns.")
    parser.add_argument("--label-col", type=str, default="label", help="Column containing class labels.")
    parser.add_argument("--score-col", type=str, default="score", help="Column containing verification scores.")
    parser.add_argument(
        "--genuine-label",
        type=str,
        default="1",
        help="Label value for genuine samples. All other labels are treated as impostor.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional fixed acceptance threshold. If omitted, EER threshold is used.",
    )
    parser.add_argument("--output-metrics", type=Path, default=DEFAULT_OUTPUT_METRICS)
    parser.add_argument("--output-roc", type=Path, default=DEFAULT_OUTPUT_ROC)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input_csv)
    required_cols = {args.label_col, args.score_col}
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    subset = df[[args.label_col, args.score_col]].dropna()
    labels = subset[args.label_col]
    scores = subset[args.score_col].astype(float).to_numpy()
    y_true = _to_binary_labels(labels, args.genuine_label)

    roc_df = roc_points(y_true, scores)
    eer, eer_threshold = eer_from_roc(roc_df)
    auc = auc_from_roc(roc_df)

    threshold = float(args.threshold) if args.threshold is not None else eer_threshold
    far, frr = far_frr_at_threshold(y_true, scores, threshold)

    metrics = {
        "n_samples": int(len(subset)),
        "n_genuine": int(np.sum(y_true == 1)),
        "n_impostor": int(np.sum(y_true == 0)),
        "threshold_used": threshold,
        "far": far,
        "frr": frr,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "auc": auc,
    }

    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output_roc.parent.mkdir(parents=True, exist_ok=True)

    with args.output_metrics.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    roc_df.to_csv(args.output_roc, index=False)

    print(f"Computed metrics for {metrics['n_samples']} samples")
    print(f"FAR={far:.4f}, FRR={frr:.4f}, EER={eer:.4f}, AUC={auc:.4f}")
    print(f"Metrics JSON: {args.output_metrics}")
    print(f"ROC CSV: {args.output_roc}")


if __name__ == "__main__":
    main()
