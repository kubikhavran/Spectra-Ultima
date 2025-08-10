"""Simple command line interface for spectra operations."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from spectra.io import load_spectra, Spectrum
from spectra.processing import correct_baseline, average_spectra, subtract_spectra
from spectra.peaks import detect_peaks


def save_spectrum(spec: Spectrum, path: str | Path) -> None:
    data = np.column_stack([spec.x, spec.y])
    np.savetxt(path, data)


def cmd_baseline(args: argparse.Namespace) -> None:
    spec = load_spectra([args.input])[0]
    corrected = correct_baseline(spec)
    save_spectrum(corrected, args.output)


def cmd_average(args: argparse.Namespace) -> None:
    specs = load_spectra(args.inputs)
    avg = average_spectra(specs)
    save_spectrum(avg, args.output)


def cmd_subtract(args: argparse.Namespace) -> None:
    spec_a, spec_b = load_spectra([args.a, args.b])
    result = subtract_spectra(spec_a, spec_b)
    save_spectrum(result, args.output)


def cmd_peaks(args: argparse.Namespace) -> None:
    spec = load_spectra([args.input])[0]
    peaks = detect_peaks(spec, height=args.height, prominence=args.prominence)
    for x, y in peaks:
        print(f"{x}\t{y}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IR spectra processing toolkit")
    sub = parser.add_subparsers()

    p_base = sub.add_parser("baseline", help="Baseline correction of a spectrum")
    p_base.add_argument("input")
    p_base.add_argument("output")
    p_base.set_defaults(func=cmd_baseline)

    p_avg = sub.add_parser("average", help="Average multiple spectra")
    p_avg.add_argument("inputs", nargs="+")
    p_avg.add_argument("output")
    p_avg.set_defaults(func=cmd_average)

    p_sub = sub.add_parser("subtract", help="Subtract spectrum B from A")
    p_sub.add_argument("a")
    p_sub.add_argument("b")
    p_sub.add_argument("output")
    p_sub.set_defaults(func=cmd_subtract)

    p_peaks = sub.add_parser("peaks", help="Detect peaks in a spectrum")
    p_peaks.add_argument("input")
    p_peaks.add_argument("--height", type=float, default=None)
    p_peaks.add_argument("--prominence", type=float, default=None)
    p_peaks.set_defaults(func=cmd_peaks)

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
