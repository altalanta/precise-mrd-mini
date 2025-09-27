"""Metric helpers for evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

def roc_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    labels = labels[order]
    scores = scores[order]
    positives = labels.sum()
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    tpr = np.cumsum(labels) / positives
    fpr = np.cumsum(1 - labels) / negatives
    return float(np.trapz(tpr, fpr))

def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    labels = labels[order]
    scores = scores[order]
    positives = labels.sum()
    if positives == 0:
        return float("nan")
    cum_tp = np.cumsum(labels)
    cum_fp = np.cumsum(1 - labels)
    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / positives
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.trapz(precision, recall))

def brier_score(labels: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))

def calibration_curve(labels: np.ndarray, probs: np.ndarray, bins: int) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.digitize(probs, edges, right=True) - 1
    records = []
    for idx in range(bins):
        mask = indices == idx
        if not np.any(mask):
            continue
        labels_bin = labels[mask]
        probs_bin = probs[mask]
        records.append({
            "bin": idx,
            "lower": float(edges[idx]),
            "upper": float(edges[idx + 1]),
            "count": int(mask.sum()),
            "event_rate": float(labels_bin.mean()),
            "confidence": float(probs_bin.mean()),
        })
    return pd.DataFrame(records)

def bootstrap_metric(labels: np.ndarray, scores: np.ndarray, func, samples: int, ci_level: float, rng: np.random.Generator) -> dict[str, float]:
    stats = []
    n = len(labels)
    for _ in range(samples):
        idx = rng.integers(0, n, size=n)
        stats.append(func(labels[idx], scores[idx]))
    stats_arr = np.array(stats)
    lower_q = (1 - ci_level) / 2
    upper_q = 1 - lower_q
    return {
        "mean": float(stats_arr.mean()),
        "lower": float(np.quantile(stats_arr, lower_q)),
        "upper": float(np.quantile(stats_arr, upper_q)),
        "std": float(stats_arr.std(ddof=1)),
    }
