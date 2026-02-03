from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from interpretability_tool import ReportBuilder, run_experiment, run_interpretability_suite


@dataclass
class ModelOutputs:
    y_true: np.ndarray
    preds: Dict[str, np.ndarray]


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _train_torch_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dim: int,
    epochs: int = 30,
    lr: float = 1e-3,
) -> MLP:
    model = MLP(X_train.shape[1], hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    X_v = torch.tensor(X_val, dtype=torch.float32)
    y_v = torch.tensor(y_val, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        opt.step()

    # quick eval to ensure forward works
    model.eval()
    with torch.no_grad():
        _ = model(X_v)
    return model


def _predict_proba_torch(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        probs = torch.sigmoid(logits)
    return probs.cpu().numpy()


def build_models_and_predict(
    seed: int = 42,
) -> Tuple[ModelOutputs, np.ndarray, Dict[str, Tuple[object, Callable[[np.ndarray], np.ndarray] | None, np.ndarray]]]:
    X, y = make_classification(
        n_samples=3000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=seed,
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(random_state=seed)

    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    nn_small = _train_torch_model(X_train_s, y_train, X_test_s, y_test, hidden_dim=16)
    nn_large = _train_torch_model(X_train_s, y_train, X_test_s, y_test, hidden_dim=64)

    preds = {
        "random_forest": rf.predict_proba(X_test)[:, 1],
        "gradient_boosting": gb.predict_proba(X_test)[:, 1],
        "nn_small": _predict_proba_torch(nn_small, X_test_s),
        "nn_large": _predict_proba_torch(nn_large, X_test_s),
    }

    models = {
        "random_forest": (rf, None, X_test),
        "gradient_boosting": (gb, None, X_test),
        "nn_small": (nn_small, lambda x: _predict_proba_torch(nn_small, x), X_test_s),
        "nn_large": (nn_large, lambda x: _predict_proba_torch(nn_large, x), X_test_s),
    }

    return ModelOutputs(y_true=y_test, preds=preds), X_test, models


if __name__ == "__main__":
    outputs, X_test, models = build_models_and_predict()
    print("y_true shape:", outputs.y_true.shape)
    for name, preds in outputs.preds.items():
        print(name, "preds shape:", preds.shape, "mean:", float(np.mean(preds)))

    methods = ["permutation", "shap", "lime"]
    method_params = {
        "pdp": {"feature": 0},
        # LIME will auto-select false positives when index/indices are omitted
        "lime": {"num_features": 10, "num_samples": 3000, "fp_threshold": 0.6, "fp_max_examples": 3},
        "shap": {"index": 0},
    }

    report = ReportBuilder()
    all_rows = []
    for name, (model, predict_fn, X_eval) in models.items():
        results = run_experiment(
            model=model,
            X=X_eval,
            y=outputs.y_true,
            methods=methods,
            method_params=method_params,
            predict_fn=predict_fn,
        )
        suite = run_interpretability_suite(
            model=model,
            X=X_eval,
            y=outputs.y_true,
            methods=["lime", "shap"],
            method_params=method_params,
            predict_fn=predict_fn,
            lime_fp_threshold=0.6,
            lime_max_examples=3,
        )
        table = report.summary_table(results)
        print("\n=== Interpretability metrics for", name, "===")
        for row in table:
            print(row)
            row_with_model = dict(row)
            row_with_model["model"] = name
            all_rows.append(row_with_model)
        if "lime" in suite["explanations"]:
            lime_info = suite["explanations"]["lime"].extra or {}
            print("LIME indices (false positives):", lime_info.get("indices"))

    report.to_csv(all_rows, "interpretability_report.csv")
    print("\nCSV report saved to interpretability_report.csv")
