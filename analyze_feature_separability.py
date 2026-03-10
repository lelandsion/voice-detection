
# #23: Analyze feature/embedding separability (same vs different speaker)
# Use this script to measure how well extracted features separate speakers.
# 1. Prepare a feature CSV with a speaker identity column and numeric feature columns.
# 2. Run: python analyze_feature_separability.py --input-csv <path_to_feature_csv>
# 3. Outputs:
# - data/processed/separability_summary.json (same/different similarity stats, AUC, EER)
# - data/processed/separability_pairs.csv (sampled pair scores for plotting/inspection)
# Notes:
# - Pair score is cosine similarity between two feature vectors.
# - Higher similarity should correspond to same-speaker pairs.
# - Sampling caps keep runtime practical for large datasets.
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_SUMMARY = Path("data/processed/separability_summary.json")
DEFAULT_OUTPUT_PAIRS = Path("data/processed/separability_pairs.csv")


def _cosine_similarity_matrix_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_safe = a / np.maximum(a_norm, 1e-12)
    b_safe = b / np.maximum(b_norm, 1e-12)
    return np.sum(a_safe * b_safe, axis=1)


def _build_feature_matrix(
    df: pd.DataFrame,
    speaker_col: str,
    feature_cols: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if speaker_col not in df.columns:
        raise ValueError(f"Speaker column '{speaker_col}' not found in input CSV.")

    if feature_cols:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")
        used_feature_cols = feature_cols
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        used_feature_cols = [c for c in numeric_cols if c not in {speaker_col}]

    if not used_feature_cols:
        raise ValueError("No numeric feature columns available after filtering.")

    subset = df[[speaker_col, *used_feature_cols]].dropna()
    speakers = subset[speaker_col].astype(str).to_numpy()
    x = subset[used_feature_cols].astype(np.float32).to_numpy()

    if len(x) < 2:
        raise ValueError("Need at least 2 samples after dropping missing values.")

    return x, speakers, used_feature_cols


def _sample_pairs(
    speakers: np.ndarray,
    max_same_pairs: int,
    max_diff_pairs: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(speakers)

    same_pairs: set[tuple[int, int]] = set()
    diff_pairs: set[tuple[int, int]] = set()

    max_attempts = max(5000, 30 * (max_same_pairs + max_diff_pairs))
    attempts = 0

    while attempts < max_attempts and (len(same_pairs) < max_same_pairs or len(diff_pairs) < max_diff_pairs):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n - 1))
        if j >= i:
            j += 1
        a, b = (i, j) if i < j else (j, i)

        if speakers[a] == speakers[b]:
            if len(same_pairs) < max_same_pairs:
                same_pairs.add((a, b))
        else:
            if len(diff_pairs) < max_diff_pairs:
                diff_pairs.add((a, b))

        attempts += 1

    rows: list[dict[str, int]] = []
    for a, b in same_pairs:
        rows.append({"idx_a": a, "idx_b": b, "is_same": 1})
    for a, b in diff_pairs:
        rows.append({"idx_a": a, "idx_b": b, "is_same": 0})

    if not rows:
        raise ValueError("Could not sample any valid pairs. Check speaker labels and dataset size.")

    return pd.DataFrame(rows)


def _roc_df_from_scores(labels: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    thresholds = np.unique(scores)
    thresholds = np.concatenate(([np.inf], thresholds[::-1], [-np.inf]))

    pos = labels == 1
    neg = labels == 0
    pos_n = int(np.sum(pos))
    neg_n = int(np.sum(neg))
    if pos_n == 0 or neg_n == 0:
        raise ValueError("Need both same-speaker and different-speaker pairs.")

    rows: list[dict[str, float]] = []
    for t in thresholds:
        accept = scores >= float(t)
        far = float(np.sum(accept & neg) / neg_n)
        frr = float(np.sum((~accept) & pos) / pos_n)
        rows.append({"threshold": float(t), "far": far, "frr": frr, "tar": 1.0 - frr})

    return pd.DataFrame(rows).sort_values("far").reset_index(drop=True)


def _auc_from_roc(roc_df: pd.DataFrame) -> float:
    return float(np.trapz(roc_df["tar"].to_numpy(), roc_df["far"].to_numpy()))


def _eer_from_roc(roc_df: pd.DataFrame) -> tuple[float, float]:
    far_vals = roc_df["far"].astype(float).to_numpy()
    frr_vals = roc_df["frr"].astype(float).to_numpy()
    thr_vals = roc_df["threshold"].astype(float).to_numpy()
    idx = int(np.argmin(np.abs(far_vals - frr_vals)))
    eer = float((far_vals[idx] + frr_vals[idx]) / 2.0)
    return eer, float(thr_vals[idx])


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled = ((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2)
    if pooled <= 1e-12:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze feature separability between same-speaker and different-speaker pairs."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--speaker-col", type=str, default="spk_idx")
    parser.add_argument(
        "--feature-cols",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit feature columns. If omitted, all numeric columns except speaker-col are used.",
    )
    parser.add_argument("--max-same-pairs", type=int, default=20000)
    parser.add_argument("--max-diff-pairs", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_OUTPUT_SUMMARY)
    parser.add_argument("--output-pairs", type=Path, default=DEFAULT_OUTPUT_PAIRS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input_csv)
    x, speakers, used_feature_cols = _build_feature_matrix(
        df,
        speaker_col=args.speaker_col,
        feature_cols=args.feature_cols,
    )

    pair_df = _sample_pairs(
        speakers=speakers,
        max_same_pairs=args.max_same_pairs,
        max_diff_pairs=args.max_diff_pairs,
        seed=args.seed,
    )

    a = x[pair_df["idx_a"].to_numpy()]
    b = x[pair_df["idx_b"].to_numpy()]
    sims = _cosine_similarity_matrix_rows(a, b)

    pair_df["similarity"] = sims.astype(float)
    pair_df["speaker_a"] = speakers[pair_df["idx_a"].to_numpy()]
    pair_df["speaker_b"] = speakers[pair_df["idx_b"].to_numpy()]

    labels = pair_df["is_same"].to_numpy(dtype=np.int32)
    scores = pair_df["similarity"].to_numpy(dtype=np.float32)
    roc_df = _roc_df_from_scores(labels, scores)

    auc = _auc_from_roc(roc_df)
    eer, eer_threshold = _eer_from_roc(roc_df)

    same_scores = scores[labels == 1]
    diff_scores = scores[labels == 0]

    summary = {
        "n_rows_used": int(len(x)),
        "n_unique_speakers": int(len(np.unique(speakers))),
        "n_features": int(len(used_feature_cols)),
        "feature_columns_used": used_feature_cols,
        "n_same_pairs": int(np.sum(labels == 1)),
        "n_diff_pairs": int(np.sum(labels == 0)),
        "same_similarity_mean": float(np.mean(same_scores)) if len(same_scores) else float("nan"),
        "same_similarity_std": float(np.std(same_scores)) if len(same_scores) else float("nan"),
        "diff_similarity_mean": float(np.mean(diff_scores)) if len(diff_scores) else float("nan"),
        "diff_similarity_std": float(np.std(diff_scores)) if len(diff_scores) else float("nan"),
        "mean_gap_same_minus_diff": float(np.mean(same_scores) - np.mean(diff_scores))
        if len(same_scores) and len(diff_scores)
        else float("nan"),
        "cohens_d": _cohens_d(same_scores, diff_scores),
        "auc": auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
    }

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_pairs.parent.mkdir(parents=True, exist_ok=True)

    with args.output_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pair_df.to_csv(args.output_pairs, index=False)

    print(f"Rows used: {summary['n_rows_used']} | speakers: {summary['n_unique_speakers']}")
    print(f"Pairs sampled: same={summary['n_same_pairs']}, diff={summary['n_diff_pairs']}")
    print(
        f"same_mean={summary['same_similarity_mean']:.4f}, "
        f"diff_mean={summary['diff_similarity_mean']:.4f}, "
        f"gap={summary['mean_gap_same_minus_diff']:.4f}"
    )
    print(f"AUC={summary['auc']:.4f}, EER={summary['eer']:.4f}")
    print(f"Summary JSON: {args.output_summary}")
    print(f"Pairs CSV: {args.output_pairs}")


if __name__ == "__main__":
    main()
