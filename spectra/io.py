"""Input/output utilities for IR spectra."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import numpy as np

try:
    from wdfreader import WdfReader  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    WdfReader = None  # type: ignore


@dataclass
class Spectrum:
    x: np.ndarray
    y: np.ndarray
    label: str = ""


def read_txt(path: str | Path) -> Spectrum:
    """Read a simple two-column TXT spectrum file."""
    data = np.loadtxt(path)
    return Spectrum(x=data[:, 0], y=data[:, 1], label=Path(path).stem)


def read_wdf(path: str | Path) -> Spectrum:
    """Read a Renishaw WDF file if ``wdfreader`` is available."""
    if WdfReader is None:
        raise ImportError("wdfreader is required to read WDF files")
    reader = WdfReader(str(path))
    x = reader.xdata
    y = reader.ydata
    return Spectrum(x=x, y=y, label=Path(path).stem)


def load_spectra(paths: List[str | Path]) -> List[Spectrum]:
    """Load spectra from given file paths guessing format by suffix."""
    spectra: List[Spectrum] = []
    for p in paths:
        p = Path(p)
        if p.suffix.lower() == ".txt":
            spectra.append(read_txt(p))
        elif p.suffix.lower() == ".wdf":
            spectra.append(read_wdf(p))
        else:
            raise ValueError(f"Unsupported spectrum format: {p.suffix}")
    return spectra
