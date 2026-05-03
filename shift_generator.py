from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


@dataclass
class ShiftConfig:
    shift_type: str   # "covariate" | "label" | "concept" | "adversarial"
    intensity: float  # 0.0..1.0
    mode: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShiftedDataset:
    X: np.ndarray
    y: np.ndarray
    config: ShiftConfig
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseShiftGenerator(ABC):
    def _validate_intensity(self, intensity: float) -> None:
        if not 0.0 <= intensity <= 1.0:
            raise ValueError(f"intensity must be in [0, 1], got {intensity}")

    @abstractmethod
    def apply(self, X: np.ndarray, y: np.ndarray, intensity: float, **kwargs) -> ShiftedDataset:
        ...


# ---------------------------------------------------------------------------
# Covariate Shift  —  P(X) changes, P(Y|X) unchanged
# ---------------------------------------------------------------------------

class CovariateShiftGenerator(BaseShiftGenerator):
    """
    Three modes:
      gaussian_noise : add N(0, intensity*std) per numerical feature
      scaling        : multiply each feature by 1 + intensity*U(-1,1)
      correlation    : rotate feature space via a random orthogonal perturbation
    """

    MODES = ("gaussian_noise", "scaling", "correlation")

    def apply(
        self,
        X: np.ndarray,
        y: np.ndarray,
        intensity: float,
        mode: str = "gaussian_noise",
        numerical_mask: Optional[np.ndarray] = None,
        seed: int = 42,
        **kwargs,
    ) -> ShiftedDataset:
        self._validate_intensity(intensity)
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode '{mode}'. Choose from {self.MODES}")

        rng = np.random.default_rng(seed)
        X_out = X.astype(float).copy()

        if numerical_mask is None:
            numerical_mask = np.ones(X.shape[1], dtype=bool)

        if mode == "gaussian_noise":
            stds = X_out[:, numerical_mask].std(axis=0) + 1e-8
            noise = rng.normal(0, intensity * stds, size=(X_out.shape[0], numerical_mask.sum()))
            X_out[:, numerical_mask] += noise
            meta: Dict[str, Any] = {"mode": mode, "noise_std_mean": float((intensity * stds).mean())}

        elif mode == "scaling":
            scale_factors = 1.0 + intensity * rng.uniform(-1.0, 1.0, size=numerical_mask.sum())
            X_out[:, numerical_mask] *= scale_factors
            meta = {"mode": mode, "scale_factor_mean": float(scale_factors.mean())}

        elif mode == "correlation":
            n_feats = int(numerical_mask.sum())
            X_sub = X_out[:, numerical_mask]
            # Random skew-symmetric matrix → first-order matrix-exponential → QR orthonormalize
            A = rng.standard_normal((n_feats, n_feats)) * intensity
            A = (A - A.T) / 2
            Q, _ = np.linalg.qr(np.eye(n_feats) + A)
            # Blend: original * (1-t) + rotated * t
            X_out[:, numerical_mask] = (1.0 - intensity) * X_sub + intensity * (X_sub @ Q.T)
            meta = {"mode": mode, "rotation_norm": float(np.linalg.norm(A))}

        config = ShiftConfig(shift_type="covariate", intensity=intensity, mode=mode)
        return ShiftedDataset(X=X_out, y=y.copy(), config=config, metadata=meta)


# ---------------------------------------------------------------------------
# Label Shift  —  P(Y) changes, P(X|Y) unchanged
# ---------------------------------------------------------------------------

class LabelShiftGenerator(BaseShiftGenerator):
    """
    Resamples the dataset so that the fraud rate shifts toward `target_fraud_rate`.
    intensity=0 → baseline rate, intensity=1 → 30% fraud.
    """

    def apply(
        self,
        X: np.ndarray,
        y: np.ndarray,
        intensity: float,
        target_fraud_rate: Optional[float] = None,
        seed: int = 42,
        **kwargs,
    ) -> ShiftedDataset:
        self._validate_intensity(intensity)
        rng = np.random.default_rng(seed)

        baseline_rate = float(y.mean())
        if target_fraud_rate is None:
            target_fraud_rate = baseline_rate + intensity * (0.30 - baseline_rate)

        fraud_idx = np.where(y == 1)[0]
        nonfr_idx = np.where(y == 0)[0]

        n_total = len(X)
        n_fraud = max(1, int(round(target_fraud_rate * n_total)))
        n_nonfr = n_total - n_fraud

        sel_fraud = rng.choice(fraud_idx, size=n_fraud, replace=(n_fraud > len(fraud_idx)))
        sel_nonfr = rng.choice(nonfr_idx, size=n_nonfr, replace=(n_nonfr > len(nonfr_idx)))
        sel = np.concatenate([sel_fraud, sel_nonfr])
        rng.shuffle(sel)

        config = ShiftConfig(shift_type="label", intensity=intensity)
        meta = {
            "baseline_fraud_rate": baseline_rate,
            "target_fraud_rate": target_fraud_rate,
            "actual_fraud_rate": float(y[sel].mean()),
        }
        return ShiftedDataset(X=X[sel], y=y[sel], config=config, metadata=meta)


# ---------------------------------------------------------------------------
# Concept Drift  —  P(Y|X) changes via conditional label flipping
# ---------------------------------------------------------------------------

class ConceptDriftGenerator(BaseShiftGenerator):
    """
    Two modes:
      random_flip   : flip intensity*100% of labels uniformly at random
      boundary_flip : flip samples whose predicted confidence is closest to 0.5
    """

    MODES = ("random_flip", "boundary_flip")

    def apply(
        self,
        X: np.ndarray,
        y: np.ndarray,
        intensity: float,
        mode: str = "random_flip",
        predict_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        seed: int = 42,
        **kwargs,
    ) -> ShiftedDataset:
        self._validate_intensity(intensity)
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode '{mode}'. Choose from {self.MODES}")
        if mode == "boundary_flip" and predict_fn is None:
            raise ValueError("'boundary_flip' requires predict_fn")

        rng = np.random.default_rng(seed)
        y_out = y.copy()
        n_flip = max(0, int(round(intensity * len(y))))

        if mode == "random_flip":
            flip_idx = rng.choice(len(y), size=n_flip, replace=False)
        else:  # boundary_flip
            probs = predict_fn(X)
            flip_idx = np.argsort(np.abs(probs - 0.5))[:n_flip]

        y_out[flip_idx] ^= 1  # 0 ↔ 1

        config = ShiftConfig(shift_type="concept", intensity=intensity, mode=mode)
        meta = {
            "n_flipped": int(n_flip),
            "baseline_fraud_rate": float(y.mean()),
            "shifted_fraud_rate": float(y_out.mean()),
        }
        return ShiftedDataset(X=X.copy(), y=y_out, config=config, metadata=meta)


# ---------------------------------------------------------------------------
# Adversarial Shift  —  FGSM for PyTorch, statistical fallback for sklearn
# ---------------------------------------------------------------------------

class AdversarialShiftGenerator(BaseShiftGenerator):
    """
    PyTorch model  : FGSM — X_adv = X + ε*sign(∇_X L(f(X), y))
                     ε = intensity * feature_std per numerical feature
    sklearn model  : shift each sample toward opposite-class mean
    Categorical / one-hot columns are left unchanged (controlled by numerical_mask).
    """

    def apply(
        self,
        X: np.ndarray,
        y: np.ndarray,
        intensity: float,
        model: Any = None,
        predict_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        numerical_mask: Optional[np.ndarray] = None,
        seed: int = 42,
        **kwargs,
    ) -> ShiftedDataset:
        self._validate_intensity(intensity)

        if numerical_mask is None:
            numerical_mask = np.ones(X.shape[1], dtype=bool)

        use_fgsm = (
            torch is not None
            and model is not None
            and hasattr(model, "parameters")
        )

        if use_fgsm:
            X_adv = self._fgsm_attack(X, y, model, intensity, numerical_mask)
            method = "fgsm"
        else:
            rng = np.random.default_rng(seed)
            X_adv = self._statistical_attack(X, y, intensity, numerical_mask)
            method = "statistical"

        config = ShiftConfig(shift_type="adversarial", intensity=intensity, mode=method)
        meta = {
            "method": method,
            "perturbation_l2": float(
                np.linalg.norm(X_adv[:, numerical_mask] - X[:, numerical_mask])
            ),
        }
        return ShiftedDataset(X=X_adv, y=y.copy(), config=config, metadata=meta)

    def _fgsm_attack(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Any,
        intensity: float,
        numerical_mask: np.ndarray,
    ) -> np.ndarray:
        stds = X[:, numerical_mask].std(axis=0) + 1e-8
        eps = intensity * stds

        X_t = torch.tensor(X.astype(np.float32), requires_grad=True)
        y_t = torch.tensor(y.astype(np.float32))

        model.eval()
        out = model(X_t)
        # Normalise output shape
        if out.ndim > 1:
            out = out[:, 0] if out.shape[1] == 1 else out[:, 1]
        out = out.squeeze()

        # Choose loss: if output exceeds [0,1] treat as logits
        if float(out.max().item()) > 1.0 or float(out.min().item()) < 0.0:
            loss = nn.BCEWithLogitsLoss()(out, y_t)
        else:
            loss = nn.BCELoss()(out.clamp(1e-7, 1.0 - 1e-7), y_t)

        loss.backward()

        grad_sign = X_t.grad.detach().numpy()
        X_adv = X.astype(float).copy()
        X_adv[:, numerical_mask] += eps * np.sign(grad_sign[:, numerical_mask])
        return X_adv

    def _statistical_attack(
        self,
        X: np.ndarray,
        y: np.ndarray,
        intensity: float,
        numerical_mask: np.ndarray,
    ) -> np.ndarray:
        """Shift each sample toward the opposite-class feature mean."""
        X_adv = X.astype(float).copy()

        class_means: Dict[int, np.ndarray] = {}
        for cls in np.unique(y):
            class_means[int(cls)] = X[y == cls][:, numerical_mask].mean(axis=0)

        for i, label in enumerate(y):
            opposite = 1 - int(label)
            if opposite in class_means:
                direction = class_means[opposite] - X[i, numerical_mask]
                X_adv[i, numerical_mask] += intensity * direction

        return X_adv


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ShiftPipeline:
    """
    Applies multiple shift generators across a sweep of intensities.

    Example::

        pipeline = ShiftPipeline()
        shifts = pipeline.apply_sweep(X, y)
        # shifts["covariate"] → list of ShiftedDataset at each intensity
    """

    DEFAULT_INTENSITIES: List[float] = [0.1, 0.3, 0.5, 0.7, 1.0]
    DEFAULT_TYPES: List[str] = ["covariate", "label", "concept", "adversarial"]

    def __init__(
        self,
        shift_types: Optional[List[str]] = None,
        generators: Optional[Dict[str, BaseShiftGenerator]] = None,
    ):
        self.shift_types = shift_types or self.DEFAULT_TYPES
        self.generators = generators or {t: create_generator(t) for t in self.shift_types}

    def apply_sweep(
        self,
        X: np.ndarray,
        y: np.ndarray,
        intensities: Optional[List[float]] = None,
        generator_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, List[ShiftedDataset]]:
        import warnings

        intensities = intensities or self.DEFAULT_INTENSITIES
        generator_kwargs = generator_kwargs or {}
        results: Dict[str, List[ShiftedDataset]] = {}

        for shift_type, generator in self.generators.items():
            kwargs = generator_kwargs.get(shift_type, {})
            datasets: List[ShiftedDataset] = []
            for intensity in intensities:
                try:
                    shifted = generator.apply(X, y, intensity, **kwargs)
                    datasets.append(shifted)
                except Exception as exc:
                    warnings.warn(
                        f"Shift '{shift_type}' at intensity={intensity} skipped: {exc}"
                    )
            results[shift_type] = datasets

        return results


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_generator(shift_type: str) -> BaseShiftGenerator:
    mapping: Dict[str, type] = {
        "covariate": CovariateShiftGenerator,
        "label": LabelShiftGenerator,
        "concept": ConceptDriftGenerator,
        "adversarial": AdversarialShiftGenerator,
    }
    if shift_type not in mapping:
        raise ValueError(f"Unknown shift_type '{shift_type}'. Choose from {list(mapping)}")
    return mapping[shift_type]()
