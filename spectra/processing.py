"""Signal processing operations for spectra."""
from __future__ import annotations

from typing import Iterable
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .io import Spectrum


def baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """Asymmetric least squares baseline correction."""
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L - 2, L))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.diags(w, 0, shape=(L, L))
        Z = W + lam * D.T @ D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def correct_baseline(spec: Spectrum, **kwargs) -> Spectrum:
    """Return a new spectrum with baseline subtracted."""
    base = baseline_als(spec.y, **kwargs)
    return Spectrum(x=spec.x.copy(), y=spec.y - base, label=spec.label)


def average_spectra(spectra: Iterable[Spectrum]) -> Spectrum:
    """Average spectra after interpolating to common x-axis."""
    spectra = list(spectra)
    if not spectra:
        raise ValueError("No spectra provided")
    x_common = spectra[0].x
    ys = [np.interp(x_common, s.x, s.y) for s in spectra]
    mean_y = np.mean(ys, axis=0)
    return Spectrum(x=x_common, y=mean_y, label="average")


def subtract_spectra(spec_a: Spectrum, spec_b: Spectrum) -> Spectrum:
    """Subtract ``spec_b`` from ``spec_a`` after interpolation."""
    x_common = spec_a.x
    y_b = np.interp(x_common, spec_b.x, spec_b.y)
    return Spectrum(x=x_common, y=spec_a.y - y_b, label=f"{spec_a.label}-{spec_b.label}")


__all__ = [
    "baseline_als",
    "correct_baseline",
    "average_spectra",
    "subtract_spectra",
]
