"""IR spectra processing utilities.

This package exposes helpers for reading spectrum files, applying
baseline correction, averaging or subtracting spectra and detecting peaks.
"""

from .io import Spectrum, load_spectra, read_txt, read_wdf
from .peaks import detect_peaks
from .processing import (
    average_spectra,
    baseline_als,
    correct_baseline,
    subtract_spectra,
)

__all__ = [
    "Spectrum",
    "read_txt",
    "read_wdf",
    "load_spectra",
    "baseline_als",
    "correct_baseline",
    "average_spectra",
    "subtract_spectra",
    "detect_peaks",
]

