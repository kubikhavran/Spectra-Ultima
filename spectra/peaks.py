"""Peak detection utilities."""
from __future__ import annotations

from typing import List, Tuple
import numpy as np
from scipy.signal import find_peaks

from .io import Spectrum


def detect_peaks(spec: Spectrum, height: float | None = None, prominence: float | None = None) -> List[Tuple[float, float]]:
    """Detect peaks returning list of (x, y) tuples."""
    idx, props = find_peaks(spec.y, height=height, prominence=prominence)
    return list(zip(spec.x[idx], spec.y[idx]))


__all__ = ["detect_peaks"]
