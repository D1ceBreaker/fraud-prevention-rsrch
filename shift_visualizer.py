from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
except Exception:  # pragma: no cover
    plt = None

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from robustness_metrics import RobustnessReport, RobustnessResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHIFT_COLORS: Dict[str, str] = {
    "covariate":   "#2196F3",
    "label":       "#4CAF50",
    "concept":     "#FF9800",
    "adversarial": "#F44336",
}

METRIC_LABELS: Dict[str, str] = {
    "auc_roc":       "AUC-ROC",
    "f1":            "F1",
    "precision":     "Precision",
    "recall":        "Recall",
    "avg_precision": "Avg Precision",
}

# PSI severity thresholds
PSI_THRESHOLDS = {"stable": 0.10, "moderate": 0.25}


# ---------------------------------------------------------------------------
# ShiftVisualizer
# ---------------------------------------------------------------------------

class ShiftVisualizer:
    """
    Generates plots from a RobustnessReport.

    Usage::

        viz = ShiftVisualizer(report, output_dir="shift_plots")
        viz.plot_performance_curves()
        viz.plot_degradation_bar()
        viz.save_all()
    """

    def __init__(self, report: RobustnessReport, output_dir: str = "shift_plots"):
        if plt is None:
            raise ImportError("matplotlib is required for ShiftVisualizer")
        self.report = report
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._figures: Dict[str, plt.Figure] = {}

    # ------------------------------------------------------------------
    # 1. Performance degradation curves
    # ------------------------------------------------------------------

    def plot_performance_curves(
        self,
        metrics: Optional[List[str]] = None,
    ) -> plt.Figure:
        """
        Line charts: shift intensity → metric value, one line per shift type.
        One subplot per metric.
        """
        metrics = metrics or ["auc_roc", "f1"]
        df = self.report.summary_table()
        shift_types = sorted(df["shift_type"].unique())

        n = len(metrics)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
        fig.suptitle("Model Performance Under Distribution Shifts",
                     fontsize=13, fontweight="bold", y=1.02)

        for ax, metric in zip(axes[0], metrics):
            col = f"perf_{metric}"
            if col not in df.columns:
                ax.set_visible(False)
                continue
            for st in shift_types:
                grp = df[df["shift_type"] == st].sort_values("intensity")
                color = SHIFT_COLORS.get(st, "#888888")
                ax.plot(grp["intensity"], grp[col],
                        marker="o", linewidth=2, color=color, label=st.capitalize())
            ax.set_xlabel("Shift Intensity", fontsize=11)
            ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=11)
            ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.02, 1.05)

        plt.tight_layout()
        self._figures["performance_curves"] = fig
        return fig

    # ------------------------------------------------------------------
    # 2. Degradation bar chart
    # ------------------------------------------------------------------

    def plot_degradation_bar(self, metric: str = "auc_roc") -> plt.Figure:
        """
        Horizontal bar chart: % degradation per shift type at maximum intensity.
        """
        df = self.report.summary_table()
        col = f"deg_{metric}"
        if col not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in report")

        idx_max = df.groupby("shift_type")["intensity"].idxmax()
        summary = df.loc[idx_max].copy()

        fig, ax = plt.subplots(figsize=(8, max(3, len(summary) * 1.2)))
        colors = [SHIFT_COLORS.get(st, "#888888") for st in summary["shift_type"]]
        bars = ax.barh(
            summary["shift_type"].str.capitalize(),
            summary[col],
            color=colors,
            edgecolor="white",
            height=0.5,
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(
            f"Relative Degradation of {METRIC_LABELS.get(metric, metric)} (%)",
            fontsize=11,
        )
        ax.set_title("Performance Degradation at Maximum Shift Intensity",
                     fontsize=12, fontweight="bold")

        for bar, val in zip(bars, summary[col]):
            x = bar.get_width()
            offset = 0.4 if x >= 0 else -0.4
            ha = "left" if x >= 0 else "right"
            if not np.isnan(val):
                ax.text(x + offset, bar.get_y() + bar.get_height() / 2,
                        f"{val:+.1f}%", va="center", ha=ha, fontsize=9)

        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        self._figures["degradation_bar"] = fig
        return fig

    # ------------------------------------------------------------------
    # 3. Feature distribution histograms (before vs after)
    # ------------------------------------------------------------------

    def plot_distribution_shifts(
        self,
        X_baseline: np.ndarray,
        X_shifted: np.ndarray,
        feature_names: Optional[List[str]] = None,
        top_n: int = 10,
        drift_df: Optional[Any] = None,
    ) -> plt.Figure:
        """
        Overlaid histograms for the top-N features ranked by PSI.
        `drift_df` is the output of compute_distribution_metrics(); if None,
        features are selected by index.
        """
        if pd is None:
            raise ImportError("pandas is required")

        n_features = X_baseline.shape[1]
        names = feature_names or [f"feature_{i}" for i in range(n_features)]

        if drift_df is not None and "psi" in drift_df.columns:
            top_feats = drift_df.nlargest(top_n, "psi")["feature"].tolist()
            feat_indices = [names.index(f) for f in top_feats if f in names]
        else:
            feat_indices = list(range(min(top_n, n_features)))
            top_feats = [names[i] for i in feat_indices]

        n_show = len(feat_indices)
        ncols = min(5, n_show)
        nrows = (n_show + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4 * ncols, 3.5 * nrows),
                                 squeeze=False)
        fig.suptitle(
            f"Feature Distributions: Baseline vs Shifted  (Top {n_show} by PSI)",
            fontsize=12, fontweight="bold",
        )
        axes_flat = axes.flatten()

        for ax, feat_idx, feat_name in zip(axes_flat, feat_indices, top_feats):
            base_v = X_baseline[:, feat_idx].astype(float)
            shift_v = X_shifted[:, feat_idx].astype(float)
            lo = min(base_v.min(), shift_v.min())
            hi = max(base_v.max(), shift_v.max())
            bins = np.linspace(lo, hi, 30) if lo < hi else 10
            ax.hist(base_v, bins=bins, alpha=0.6, color="#2196F3",
                    label="Baseline", density=True)
            ax.hist(shift_v, bins=bins, alpha=0.6, color="#F44336",
                    label="Shifted", density=True)
            ax.set_title(feat_name[:22], fontsize=8)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)

        for ax in axes_flat[len(feat_indices):]:
            ax.set_visible(False)

        plt.tight_layout()
        self._figures["distribution_shifts"] = fig
        return fig

    # ------------------------------------------------------------------
    # 4. PSI heatmap  (features × shift types)
    # ------------------------------------------------------------------

    def plot_psi_heatmap(
        self,
        drift_results: Dict[str, Any],
        top_n: int = 30,
    ) -> plt.Figure:
        """
        Heatmap of PSI values.
        `drift_results`: dict mapping shift_type → pd.DataFrame from compute_distribution_metrics.
        """
        if pd is None:
            raise ImportError("pandas is required")

        shift_types = list(drift_results.keys())
        if not shift_types:
            raise ValueError("drift_results is empty")

        feature_names = drift_results[shift_types[0]]["feature"].tolist()
        n_feats = len(feature_names)
        matrix = np.zeros((n_feats, len(shift_types)))

        for j, st in enumerate(shift_types):
            df_st = drift_results[st]
            psi_map = dict(zip(df_st["feature"], df_st["psi"]))
            for i, feat in enumerate(feature_names):
                matrix[i, j] = psi_map.get(feat, 0.0)

        # Keep top features by max PSI across shift types
        top_n = min(top_n, n_feats)
        top_idx = np.argsort(-matrix.max(axis=1))[:top_n]
        matrix_top = matrix[top_idx]
        feat_labels = [feature_names[i] for i in top_idx]

        fig_h = max(6, top_n * 0.38)
        fig_w = max(5, len(shift_types) * 1.8)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(matrix_top, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.5)
        ax.set_xticks(range(len(shift_types)))
        ax.set_xticklabels([s.capitalize() for s in shift_types], fontsize=10)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([f[:28] for f in feat_labels], fontsize=7)
        plt.colorbar(im, ax=ax, label="PSI", fraction=0.03)
        ax.set_title("Feature PSI Heatmap by Shift Type",
                     fontsize=12, fontweight="bold")

        for i in range(top_n):
            for j in range(len(shift_types)):
                val = matrix_top[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if val > 0.3 else "black")

        # Dashed lines for PSI thresholds (on colorbar scale)
        plt.tight_layout()
        self._figures["psi_heatmap"] = fig
        return fig

    # ------------------------------------------------------------------
    # 5. Multi-model comparison heatmap  (stub — for task 3)
    # ------------------------------------------------------------------

    def plot_model_comparison_heatmap(
        self,
        models_results: Dict[str, RobustnessReport],
        metric: str = "auc_roc",
    ) -> plt.Figure:
        """
        Heatmap: models × shift types, color = mean % degradation of `metric`.
        Intended for task 3 (comparing multiple fraud detection models).
        """
        if pd is None:
            raise ImportError("pandas is required")

        model_names = list(models_results.keys())
        all_shift_types: List[str] = []
        for report in models_results.values():
            for st in report.summary_table()["shift_type"].unique():
                if st not in all_shift_types:
                    all_shift_types.append(st)

        deg_col = f"deg_{metric}"
        matrix = np.full((len(model_names), len(all_shift_types)), float("nan"))
        for i, name in enumerate(model_names):
            df = models_results[name].summary_table()
            for j, st in enumerate(all_shift_types):
                sub = df[df["shift_type"] == st]
                if not sub.empty and deg_col in sub.columns:
                    matrix[i, j] = float(sub[deg_col].mean())

        fig, ax = plt.subplots(
            figsize=(max(5, len(all_shift_types) * 2), max(3, len(model_names) * 1.2))
        )
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-50, vmax=10)
        ax.set_xticks(range(len(all_shift_types)))
        ax.set_xticklabels([s.capitalize() for s in all_shift_types], fontsize=10)
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names, fontsize=10)
        plt.colorbar(im, ax=ax,
                     label=f"Mean Degradation of {METRIC_LABELS.get(metric, metric)} (%)")
        ax.set_title(
            f"Model Robustness Comparison  —  {METRIC_LABELS.get(metric, metric)}",
            fontsize=12, fontweight="bold",
        )
        for i in range(len(model_names)):
            for j in range(len(all_shift_types)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                            fontsize=8, color="black")

        plt.tight_layout()
        self._figures["model_comparison_heatmap"] = fig
        return fig

    # ------------------------------------------------------------------
    # 6. Radar / spider chart  (stub — for task 3)
    # ------------------------------------------------------------------

    def plot_radar_chart(
        self,
        models_results: Dict[str, RobustnessReport],
        metric: str = "auc_roc",
        intensity: float = 0.5,
    ) -> plt.Figure:
        """
        Spider chart: axes = shift types, values = model metric at given intensity.
        Intended for task 3.
        """
        if pd is None:
            raise ImportError("pandas is required")

        model_names = list(models_results.keys())
        all_shift_types: List[str] = []
        for report in models_results.values():
            for st in report.summary_table()["shift_type"].unique():
                if st not in all_shift_types:
                    all_shift_types.append(st)

        if len(all_shift_types) < 3:
            raise ValueError("Need at least 3 shift types for a radar chart")

        n_axes = len(all_shift_types)
        angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
        angles_closed = angles + angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles)
        ax.set_xticklabels([s.capitalize() for s in all_shift_types], fontsize=10)

        perf_col = f"perf_{metric}"
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))  # type: ignore

        for model_name, color in zip(model_names, colors):
            df = models_results[model_name].summary_table()
            values = []
            for st in all_shift_types:
                sub = df[
                    (df["shift_type"] == st)
                    & (np.abs(df["intensity"] - intensity) < 0.06)
                ]
                if not sub.empty and perf_col in sub.columns:
                    values.append(float(sub[perf_col].mean()))
                else:
                    values.append(0.0)
            values_closed = values + values[:1]
            ax.plot(angles_closed, values_closed, "o-", linewidth=2,
                    color=color, label=model_name)
            ax.fill(angles_closed, values_closed, alpha=0.12, color=color)

        ax.set_title(
            f"Model Robustness — {METRIC_LABELS.get(metric, metric)}  "
            f"(intensity={intensity})",
            fontsize=11, fontweight="bold", pad=20,
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

        plt.tight_layout()
        self._figures["radar_chart"] = fig
        return fig

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_all(self, formats: Optional[List[str]] = None) -> List[str]:
        """Save all generated figures. Returns list of saved file paths."""
        formats = formats or ["png"]
        saved: List[str] = []
        for name, fig in self._figures.items():
            for fmt in formats:
                path = os.path.join(self.output_dir, f"{name}.{fmt}")
                fig.savefig(path, dpi=150, bbox_inches="tight")
                saved.append(path)
        return saved
