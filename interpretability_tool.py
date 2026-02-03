from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import csv

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LimeTabularExplainer = None

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None


@dataclass
class Explanation:
    method: str
    local_importance: Optional[np.ndarray] = None
    global_importance: Optional[np.ndarray] = None
    extra: Optional[Dict[str, Any]] = None


@dataclass
class MetricResult:
    name: str
    value: float
    details: Optional[Dict[str, Any]] = None


class DataManager:
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = X
        self.y = y

    def sample(self, n: int, seed: int = 42) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.X), size=min(n, len(self.X)), replace=False)
        Xs = self.X[idx]
        ys = self.y[idx] if self.y is not None else None
        return Xs, ys


def select_false_positive_indices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    max_examples: int = 5,
) -> List[int]:
    """
    Select indices of false positives ordered by highest predicted score.

    A false positive is defined as y_true == 0 with y_pred >= threshold.

    Example:
        y_true = np.array([0, 0, 1, 0])
        y_pred = np.array([0.2, 0.9, 0.8, 0.6])
        select_false_positive_indices(y_true, y_pred, threshold=0.5, max_examples=2)
        # -> [1, 3]
    """
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if max_examples <= 0:
        return []

    fp_mask = (y_true == 0) & (y_pred >= threshold)
    fp_indices = np.where(fp_mask)[0]
    if fp_indices.size == 0:
        return []

    scores = y_pred[fp_indices]
    order = np.argsort(-scores)
    selected = fp_indices[order][:max_examples]
    return selected.tolist()


class ExplainEngine:
    def __init__(self, model: Any, predict_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.model = model
        self.predict_fn = predict_fn or self._default_predict

    def _default_predict(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    def explain(
        self,
        X: np.ndarray,
        method: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Explanation:
        params = params or {}
        method = method.lower()

        if method == "permutation":
            return self._permutation_importance(X, params)
        if method == "pdp":
            return self._pdp_importance(X, params)
        if method == "lime":
            return self._lime(X, params)
        if method == "shap":
            return self._shap(X, params)
        if method == "ig":
            return self._integrated_gradients(X, params)

        raise ValueError(f"Unknown method: {method}")

    def _permutation_importance(self, X: np.ndarray, params: Dict[str, Any]) -> Explanation:
        metric = params.get("metric", self._default_metric)
        rng = np.random.default_rng(params.get("seed", 42))
        base = metric(self.predict_fn(X), params.get("y_true"))
        importances = []
        for j in range(X.shape[1]):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            score = metric(self.predict_fn(Xp), params.get("y_true"))
            importances.append(base - score)
        return Explanation(method="permutation", global_importance=np.array(importances))

    def _pdp_importance(self, X: np.ndarray, params: Dict[str, Any]) -> Explanation:
        feature = params.get("feature")
        if feature is None:
            raise ValueError("PDP requires 'feature' in params")
        grid = params.get("grid")
        if grid is None:
            grid = np.quantile(X[:, feature], np.linspace(0.05, 0.95, 9))
        pdp = []
        for val in grid:
            Xp = X.copy()
            Xp[:, feature] = val
            pdp.append(np.mean(self.predict_fn(Xp)))
        return Explanation(method="pdp", extra={"feature": feature, "grid": grid, "pdp": np.array(pdp)})

    def _lime(self, X: np.ndarray, params: Dict[str, Any]) -> Explanation:
        if LimeTabularExplainer is None:
            raise ImportError("lime is not installed. Install with: pip install lime")

        idx = params.get("index")
        indices = params.get("indices")
        feature_names = params.get("feature_names")
        class_names = params.get("class_names")
        discretize_continuous = params.get("discretize_continuous", True)
        y_true = params.get("y_true")
        fp_threshold = params.get("fp_threshold", 0.5)
        fp_max_examples = params.get("fp_max_examples", 5)

        explainer = LimeTabularExplainer(
            training_data=X,
            feature_names=feature_names,
            class_names=class_names,
            discretize_continuous=discretize_continuous,
        )
        if indices is None:
            if idx is not None:
                indices = [idx]
            else:
                if y_true is None:
                    raise ValueError("LIME requires 'index' or 'indices', or y_true for false-positive selection")
                y_pred = self.predict_fn(X)
                indices = select_false_positive_indices(
                    y_true=y_true,
                    y_pred=y_pred,
                    threshold=fp_threshold,
                    max_examples=fp_max_examples,
                )

        if not indices:
            raise ValueError("No LIME indices available (no false positives found)")

        per_index = []
        weights_list = []
        for sel_idx in indices:
            exp = explainer.explain_instance(
                data_row=X[sel_idx],
                predict_fn=lambda x: np.column_stack([1 - self.predict_fn(x), self.predict_fn(x)]),
                num_features=params.get("num_features", X.shape[1]),
                num_samples=params.get("num_samples", 5000),
            )
            weights = np.zeros(X.shape[1])
            for feature_idx, weight in exp.as_map()[1]:
                weights[feature_idx] = weight
            weights_list.append(weights)
            per_index.append({"index": int(sel_idx), "weights": weights, "raw": exp})

        if len(weights_list) == 1:
            local_importance = weights_list[0]
        else:
            local_importance = np.mean(np.stack(weights_list, axis=0), axis=0)

        return Explanation(
            method="lime",
            local_importance=local_importance,
            extra={"indices": indices, "per_index": per_index},
        )

    def _shap(self, X: np.ndarray, params: Dict[str, Any]) -> Explanation:
        if shap is None:
            raise ImportError("shap is not installed. Install with: pip install shap")

        idx = params.get("index", 0)
        background = params.get("background", X[: min(200, len(X))])
        explainer = shap.Explainer(self.predict_fn, background)
        shap_values = explainer(X)
        local = np.array(shap_values[idx].values)
        global_imp = np.mean(np.abs(shap_values.values), axis=0)
        return Explanation(
            method="shap",
            local_importance=local,
            global_importance=global_imp,
            extra={"expected_value": shap_values.base_values},
        )

    def _integrated_gradients(self, X: np.ndarray, params: Dict[str, Any]) -> Explanation:
        if torch is None:
            raise ImportError("torch is not installed. Install with: pip install torch")
        if not hasattr(self.model, "__call__"):
            raise ValueError("Integrated Gradients requires a callable torch model")

        idx = params.get("index", 0)
        baseline = params.get("baseline", np.zeros(X.shape[1], dtype=np.float32))
        steps = params.get("steps", 50)
        output_index = params.get("output_index", None)
        apply_sigmoid = params.get("apply_sigmoid", False)

        x0 = X[idx].astype(np.float32)
        baseline = baseline.astype(np.float32)

        x0_t = torch.tensor(x0, requires_grad=False)
        base_t = torch.tensor(baseline, requires_grad=False)
        alphas = torch.linspace(0.0, 1.0, steps)
        grads = []

        for a in alphas:
            x = base_t + a * (x0_t - base_t)
            x = x.clone().detach().requires_grad_(True)
            out = self.model(x)
            if apply_sigmoid:
                out = torch.sigmoid(out)
            if out.ndim > 0 and output_index is not None:
                out = out[output_index]
            out = out if out.ndim == 0 else out.squeeze()
            out.backward()
            grads.append(x.grad.detach().cpu().numpy())

        avg_grads = np.mean(np.stack(grads, axis=0), axis=0)
        ig = (x0 - baseline) * avg_grads
        return Explanation(method="ig", local_importance=ig)

    @staticmethod
    def _default_metric(y_pred: np.ndarray, y_true: Optional[np.ndarray]) -> float:
        if y_true is None:
            return float(np.mean(y_pred))
        return float(np.mean((y_pred - y_true) ** 2))


class MetricEngine:
    def fidelity(self, model_preds: np.ndarray, surrogate_preds: np.ndarray) -> MetricResult:
        num = np.sum((model_preds - surrogate_preds) ** 2)
        denom = np.sum((model_preds - np.mean(model_preds)) ** 2) + 1e-12
        r2 = 1 - num / denom
        return MetricResult(name="fidelity_r2", value=float(r2))

    def stability(
        self,
        importance_a: np.ndarray,
        importance_b: np.ndarray,
        top_k: int = 5,
    ) -> MetricResult:
        a = np.argsort(-np.abs(importance_a))[:top_k]
        b = np.argsort(-np.abs(importance_b))[:top_k]
        overlap = len(set(a) & set(b)) / max(1, top_k)
        return MetricResult(name="stability_topk_overlap", value=float(overlap))

    def sparsity(self, importance: np.ndarray, threshold: float = 0.8) -> MetricResult:
        abs_imp = np.abs(importance)
        if abs_imp.sum() == 0:
            return MetricResult(name="sparsity_k", value=float(len(importance)))
        order = np.argsort(-abs_imp)
        cum = np.cumsum(abs_imp[order]) / abs_imp.sum()
        k = int(np.searchsorted(cum, threshold) + 1)
        return MetricResult(name="sparsity_k", value=float(k))

    def consistency(
        self,
        importance: np.ndarray,
        rule_features: Iterable[int],
        top_k: int = 5,
    ) -> MetricResult:
        top = set(np.argsort(-np.abs(importance))[:top_k])
        rules = set(rule_features)
        if not rules:
            return MetricResult(name="consistency_rules", value=0.0, details={"note": "no rules"})
        hit = len(top & rules) / len(rules)
        return MetricResult(name="consistency_rules", value=float(hit))


class ReportBuilder:
    def summary_table(self, results: List[Dict[str, MetricResult]]) -> List[Dict[str, float]]:
        table = []
        for item in results:
            row = {"method": item["method"]}
            for k, v in item.items():
                if k == "method":
                    continue
                row[k] = v.value
            table.append(row)
        return table

    def to_csv(self, rows: List[Dict[str, float]], path: str) -> None:
        if not rows:
            return
        fieldnames: List[str] = []
        for key in ("model", "method"):
            if any(key in row for row in rows):
                fieldnames.append(key)
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def run_experiment(
    model: Any,
    X: np.ndarray,
    y: Optional[np.ndarray],
    methods: List[str],
    rule_features: Optional[List[int]] = None,
    method_params: Optional[Dict[str, Dict[str, Any]]] = None,
    predict_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> List[Dict[str, MetricResult]]:
    rule_features = rule_features or []
    method_params = method_params or {}
    explainer = ExplainEngine(model, predict_fn=predict_fn)
    metrics = MetricEngine()
    results = []

    for method in methods:
        params = dict(method_params.get(method, {}))
        params["y_true"] = y
        explanation = explainer.explain(X, method, params=params)
        item: Dict[str, MetricResult] = {"method": method}
        if explanation.local_importance is not None:
            item["sparsity"] = metrics.sparsity(explanation.local_importance)
            item["consistency"] = metrics.consistency(explanation.local_importance, rule_features)
        if explanation.global_importance is not None:
            item["global_sparsity"] = metrics.sparsity(explanation.global_importance)
        results.append(item)

    return results


def run_interpretability_suite(
    model: Any,
    X: np.ndarray,
    y: Optional[np.ndarray],
    methods: List[str],
    method_params: Optional[Dict[str, Dict[str, Any]]] = None,
    predict_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    lime_fp_threshold: float = 0.5,
    lime_max_examples: int = 5,
) -> Dict[str, Any]:
    """
    Unified runner for interpretability methods (LIME/SHAP/etc.).

    If LIME is selected and no explicit indices are provided, the runner will
    auto-select false positives based on y_true and the provided threshold.

    Example:
        results = run_interpretability_suite(
            model=model,
            X=X_eval,
            y=y_eval,
            methods=["lime", "shap"],
            method_params={"lime": {"num_samples": 3000}},
            lime_fp_threshold=0.7,
            lime_max_examples=3,
        )
    """
    method_params = method_params or {}
    explainer = ExplainEngine(model, predict_fn=predict_fn)
    metrics = MetricEngine()
    outputs: Dict[str, Any] = {"explanations": {}, "metrics": []}

    for method in methods:
        params = dict(method_params.get(method, {}))
        params["y_true"] = y
        if method.lower() == "lime":
            params.setdefault("fp_threshold", lime_fp_threshold)
            params.setdefault("fp_max_examples", lime_max_examples)
            if y is None and "index" not in params and "indices" not in params:
                raise ValueError("LIME requires y_true or explicit 'index'/'indices'")

        explanation = explainer.explain(X, method, params=params)
        outputs["explanations"][method] = explanation

        item: Dict[str, MetricResult] = {"method": method}
        if explanation.local_importance is not None:
            item["sparsity"] = metrics.sparsity(explanation.local_importance)
        if explanation.global_importance is not None:
            item["global_sparsity"] = metrics.sparsity(explanation.global_importance)
        outputs["metrics"].append(item)

    return outputs
