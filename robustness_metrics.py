from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    from scipy import stats as scipy_stats
    from scipy.spatial.distance import jensenshannon
except Exception:  # pragma: no cover
    scipy_stats = None
    jensenshannon = None

try:
    from sklearn.metrics import (
        roc_auc_score,
        f1_score,
        precision_score,
        recall_score,
        average_precision_score,
    )
except Exception:  # pragma: no cover
    roc_auc_score = None

from shift_generator import ShiftedDataset


# ---------------------------------------------------------------------------
# Distribution drift metrics  (operate on 1-D arrays)
# ---------------------------------------------------------------------------

def psi(baseline: np.ndarray, shifted: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index.
    Interpretation: <0.10 stable, 0.10–0.25 moderate shift, >0.25 significant shift.
    """
    eps = 1e-8
    lo, hi = float(baseline.min()), float(baseline.max())
    if lo == hi:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    base_cnt, _ = np.histogram(baseline, bins=bins)
    shift_cnt, _ = np.histogram(shifted, bins=bins)
    base_pct = (base_cnt + eps) / (len(baseline) + eps)
    shift_pct = (shift_cnt + eps) / (len(shifted) + eps)
    return float(np.sum((shift_pct - base_pct) * np.log(shift_pct / base_pct)))


def ks_statistic(baseline: np.ndarray, shifted: np.ndarray) -> Tuple[float, float]:
    """Kolmogorov-Smirnov two-sample test. Returns (statistic, p_value)."""
    if scipy_stats is None:
        raise ImportError("scipy is required for ks_statistic")
    result = scipy_stats.ks_2samp(baseline.astype(float), shifted.astype(float))
    return float(result.statistic), float(result.pvalue)


def wasserstein_distance(baseline: np.ndarray, shifted: np.ndarray) -> float:
    """Wasserstein (Earth Mover's) distance between two 1-D distributions."""
    if scipy_stats is None:
        raise ImportError("scipy is required for wasserstein_distance")
    return float(scipy_stats.wasserstein_distance(baseline.astype(float), shifted.astype(float)))


def js_divergence(baseline: np.ndarray, shifted: np.ndarray, n_bins: int = 50) -> float:
    """Jensen-Shannon divergence (histogram-based, symmetrised KL). Returns value in [0, 1]."""
    if jensenshannon is None:
        raise ImportError("scipy is required for js_divergence")
    lo = min(float(baseline.min()), float(shifted.min()))
    hi = max(float(baseline.max()), float(shifted.max()))
    if lo == hi:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    eps = 1e-8
    p, _ = np.histogram(baseline, bins=bins, density=True)
    q, _ = np.histogram(shifted, bins=bins, density=True)
    return float(jensenshannon(p + eps, q + eps))


def compute_distribution_metrics(
    X_baseline: np.ndarray,
    X_shifted: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_bins: int = 10,
) -> "pd.DataFrame":
    """
    Compute per-feature distribution drift.
    Returns DataFrame with columns: feature, psi, ks_stat, ks_pvalue, wasserstein, js_divergence.
    """
    if pd is None:
        raise ImportError("pandas is required for compute_distribution_metrics")

    n_features = X_baseline.shape[1]
    names = feature_names or [f"feature_{i}" for i in range(n_features)]

    rows = []
    for i, name in enumerate(names):
        base_col = X_baseline[:, i].astype(float)
        shift_col = X_shifted[:, i].astype(float)
        ks_s, ks_p = ks_statistic(base_col, shift_col)
        rows.append({
            "feature": name,
            "psi": psi(base_col, shift_col, n_bins=n_bins),
            "ks_stat": ks_s,
            "ks_pvalue": ks_p,
            "wasserstein": wasserstein_distance(base_col, shift_col),
            "js_divergence": js_divergence(base_col, shift_col),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model performance metrics
# ---------------------------------------------------------------------------

def compute_performance_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Returns auc_roc, f1, precision, recall, avg_precision.
    y_prob: predicted probabilities for the positive class.
    """
    if roc_auc_score is None:
        raise ImportError("scikit-learn is required for performance metrics")

    y_pred = (y_prob >= threshold).astype(int)
    metrics: Dict[str, float] = {}

    if len(np.unique(y_true)) < 2:
        metrics["auc_roc"] = float("nan")
        metrics["avg_precision"] = float("nan")
    else:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        metrics["avg_precision"] = float(average_precision_score(y_true, y_prob))

    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    return metrics


def relative_degradation(
    baseline: Dict[str, float],
    shifted: Dict[str, float],
) -> Dict[str, float]:
    """
    Relative change (%) per metric.
    Negative values indicate degradation: (shifted - baseline) / |baseline| * 100.
    """
    eps = 1e-8
    result: Dict[str, float] = {}
    for key in baseline:
        if key not in shifted:
            continue
        b, s = baseline[key], shifted[key]
        if np.isnan(b) or np.isnan(s):
            result[key] = float("nan")
        else:
            result[key] = (s - b) / (abs(b) + eps) * 100.0
    return result


# ---------------------------------------------------------------------------
# Core evaluation classes
# ---------------------------------------------------------------------------

@dataclass
class RobustnessResult:
    shift_type: str
    intensity: float
    mode: str
    performance: Dict[str, float]
    degradation: Dict[str, float]
    distribution_drift: Any            # pd.DataFrame or None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RobustnessEvaluator:
    """
    Evaluates a model's robustness against distribution shifts.

    Example::

        evaluator = RobustnessEvaluator(model, predict_fn, X_train, y_train)
        results = evaluator.evaluate_all(shifts)   # shifts from ShiftPipeline
        report = RobustnessReport(results)
        report.to_csv("robustness_report.csv")
    """

    def __init__(
        self,
        model: Any,
        predict_fn: Optional[Callable[[np.ndarray], np.ndarray]],
        X_baseline: np.ndarray,
        y_baseline: np.ndarray,
        feature_names: Optional[List[str]] = None,
        threshold: float = 0.5,
        max_drift_samples: int = 5000,
    ):
        self.model = model
        self.predict_fn = predict_fn or self._default_predict
        self.X_baseline = X_baseline
        self.y_baseline = y_baseline
        self.feature_names = feature_names
        self.threshold = threshold
        self.max_drift_samples = max_drift_samples
        self._baseline_metrics: Optional[Dict[str, float]] = None

    def _default_predict(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X).astype(float)

    @property
    def baseline_metrics(self) -> Dict[str, float]:
        if self._baseline_metrics is None:
            y_prob = self.predict_fn(self.X_baseline)
            self._baseline_metrics = compute_performance_metrics(
                self.y_baseline, y_prob, self.threshold
            )
        return self._baseline_metrics

    def evaluate(self, shifted: ShiftedDataset) -> RobustnessResult:
        y_prob = self.predict_fn(shifted.X)
        perf = compute_performance_metrics(shifted.y, y_prob, self.threshold)
        degrad = relative_degradation(self.baseline_metrics, perf)

        # Distribution drift — subsample to keep it fast
        try:
            n = min(self.max_drift_samples, len(self.X_baseline), len(shifted.X))
            drift_df = compute_distribution_metrics(
                self.X_baseline[:n],
                shifted.X[:n],
                feature_names=self.feature_names,
            )
        except Exception:
            drift_df = None

        return RobustnessResult(
            shift_type=shifted.config.shift_type,
            intensity=shifted.config.intensity,
            mode=shifted.config.mode,
            performance=perf,
            degradation=degrad,
            distribution_drift=drift_df,
            metadata=shifted.metadata,
        )

    def evaluate_all(
        self,
        shifts: Dict[str, List[ShiftedDataset]],
    ) -> List[RobustnessResult]:
        results: List[RobustnessResult] = []
        for datasets in shifts.values():
            for shifted in datasets:
                results.append(self.evaluate(shifted))
        return results


class RobustnessReport:
    """
    Aggregates RobustnessResult objects and provides tabular / CSV output.

    Columns in summary_table():
        shift_type, mode, intensity,
        perf_auc_roc, perf_f1, perf_precision, perf_recall, perf_avg_precision,
        deg_auc_roc, deg_f1, deg_precision, deg_recall, deg_avg_precision
    """

    def __init__(self, results: List[RobustnessResult]):
        self.results = results

    def summary_table(self) -> "pd.DataFrame":
        if pd is None:
            raise ImportError("pandas is required for summary_table")
        rows = []
        for r in self.results:
            row: Dict[str, Any] = {
                "shift_type": r.shift_type,
                "mode": r.mode,
                "intensity": r.intensity,
            }
            row.update({f"perf_{k}": v for k, v in r.performance.items()})
            row.update({f"deg_{k}": v for k, v in r.degradation.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def to_csv(self, path: str) -> None:
        self.summary_table().to_csv(path, index=False)

    def flag_critical(self, metric: str, threshold: float) -> List[RobustnessResult]:
        """
        Return results where the relative degradation of `metric` exceeds `threshold` percent.
        E.g. flag_critical("auc_roc", 5.0) returns cases where AUC dropped >5%.
        """
        flagged = []
        for r in self.results:
            deg = r.degradation.get(metric, 0.0)
            if not np.isnan(deg) and deg < -abs(threshold):
                flagged.append(r)
        return flagged
