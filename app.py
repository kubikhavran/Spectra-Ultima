"""
Spectral Analysis Web App
~~~~~~~~~~~~~~~~~~~~~~~~~

This Streamlit application provides an interactive interface for loading,
visualising and processing spectral data.  It is designed as a modern,
web‑based alternative to traditional desktop spectroscopy tools.  Core
functionality includes bulk file import, baseline correction via
`pybaselines`, peak detection using SciPy, averaging and subtraction of
spectra, smoothing, normalisation and export of processed data.

The app is deliberately modular: new processing functions can be added
simply by defining additional helper routines and inserting them into
the sidebar.  Spectra are stored in an in‑memory dictionary keyed by
user‑readable labels.  Each entry contains a Pandas DataFrame with
`x` (abscissa) and `y` (ordinate) columns.

Note
----
This script depends on the following third‑party libraries: streamlit,
numpy, pandas, plotly, scipy, pybaselines and optionally ramanspy for
Renishaw `.wdf` support.  Install them via `pip install -r
requirements.txt`.
"""

from __future__ import annotations

import io
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import find_peaks, savgol_filter

# Import baseline correction tools.  pybaselines provides a unified API
# containing dozens of algorithms【330007600347264†L30-L41】.  We use the
# `Baseline` class to access AsLS (asymmetric least squares) and
# related methods.
try:
    from pybaselines import Baseline
    _has_baseline = True
except Exception:
    _has_baseline = False

# Try to import RamanSPy for Renishaw .wdf files.  If unavailable the
# loader will skip those files gracefully【405946903641446†L134-L149】.
try:
    import ramanspy
    _has_ramanspy = True
except Exception:
    _has_ramanspy = False


def read_text_spectrum(file: io.BytesIO) -> pd.DataFrame:
    """Read a plain text or CSV spectrum.

    The function attempts to auto‑detect the delimiter and assume the
    first two numeric columns correspond to the x and y axes.  Lines
    containing fewer than two numeric values are ignored.

    Parameters
    ----------
    file : BytesIO
        Uploaded file object from Streamlit.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns `x` and `y`.
    """
    # Reset pointer to start
    file.seek(0)
    content = file.read().decode('utf-8', errors='ignore').splitlines()
    rows: List[Tuple[float, float]] = []
    for line in content:
        # Replace commas with spaces to handle CSV files
        tokens = line.replace(',', ' ').split()
        if len(tokens) < 2:
            continue
        try:
            x_val = float(tokens[0])
            y_val = float(tokens[1])
            rows.append((x_val, y_val))
        except ValueError:
            # Skip header lines or malformed lines
            continue
    if not rows:
        raise ValueError("No numeric data found in file")
    data = pd.DataFrame(rows, columns=['x', 'y'])
    return data


def read_wdf_spectrum(file_path: str) -> Dict[str, pd.DataFrame]:
    """Load spectra from a Renishaw .wdf file using RamanSPy.

    A Renishaw `.wdf` file may contain one or many spectra.  RamanSPy
    returns a container that exposes `.x` (wavenumbers) and `.y`
    (intensities) arrays.  Each spectrum is returned as its own
    DataFrame in the output dictionary.

    Parameters
    ----------
    file_path : str
        Temporary path to the uploaded .wdf file on disk.

    Returns
    -------
    Dict[str, pandas.DataFrame]
        Mapping from spectrum label to DataFrame with columns `x` and
        `y`.
    """
    spectra: Dict[str, pd.DataFrame] = {}
    if not _has_ramanspy:
        raise ImportError(
            "ramanspy is required for reading Renishaw .wdf files. "
            "Please install ramanspy or upload another format."
        )
    # Use RamanSPy's loader.  According to its documentation, the
    # load.renishaw method accepts a file path and returns a RamanData
    # container with .spectra attribute.  Each entry contains .x and .y
    # arrays.  If the API changes, catch exceptions gracefully.
    try:
        raman_data = ramanspy.load.renishaw(file_path)
        # raman_data.spectra is an iterable of (x, y) pairs
        for idx, spec in enumerate(raman_data.spectra):
            # spec.x and spec.y are numpy arrays
            x = np.asarray(spec.x)
            y = np.asarray(spec.y)
            spectra[f"{file_path.name}_spectrum_{idx}"] = pd.DataFrame({
                'x': x,
                'y': y
            })
    except Exception as exc:
        # If load.renishaw returned a single spectrum or fails, fall back
        # to reading .txt in plain text format
        raise ValueError(f"Failed to read .wdf file: {exc}")
    return spectra


def interpolate_to_common_axis(spectra: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate a list of spectra to a common x axis and compute the mean.

    This helper finds the overlapping x range among all spectra and
    interpolates each y array to evenly spaced points within that range.

    Parameters
    ----------
    spectra : list of pandas.DataFrame
        Each DataFrame must have columns `x` and `y`.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        Common x axis and averaged y values.
    """
    # Determine the common x range
    x_min = max(s['x'].min() for s in spectra)
    x_max = min(s['x'].max() for s in spectra)
    if x_min >= x_max:
        raise ValueError("Spectra have no overlapping x‑range for averaging")
    # Use 1000 points by default; adjust if resolution too high/low
    num_points = 1000
    common_x = np.linspace(x_min, x_max, num_points)
    ys: List[np.ndarray] = []
    for df in spectra:
        ys.append(np.interp(common_x, df['x'], df['y']))
    mean_y = np.mean(np.vstack(ys), axis=0)
    return common_x, mean_y


def subtract_spectra(spec_a: pd.DataFrame, spec_b: pd.DataFrame) -> pd.DataFrame:
    """Subtract one spectrum from another, interpolating if necessary.

    The output uses the x axis of the first spectrum.  The second
    spectrum is interpolated onto the first x axis prior to subtraction.
    """
    x = spec_a['x'].values
    y_a = spec_a['y'].values
    y_b = np.interp(x, spec_b['x'], spec_b['y'])
    diff_y = y_a - y_b
    return pd.DataFrame({'x': x, 'y': diff_y})


def baseline_correct(y: np.ndarray, method: str = 'asls', lam: float = 1e5, p: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """Apply baseline correction to a 1D signal.

    This wrapper uses the `pybaselines` Baseline class to compute the
    baseline and return the corrected signal.  If pybaselines is not
    installed, the original signal is returned unchanged.
    """
    if not _has_baseline:
        return np.zeros_like(y), y
    base = Baseline()
    # Use AsLS (asymmetric least squares) by default
    if method == 'asls':
        baseline_y, params = base.asls(y, lam=lam, p=p)
    elif method == 'arpls':
        baseline_y, params = base.arpls(y, lam=lam)
    elif method == 'airpls':
        baseline_y, params = base.airpls(y, lam=lam)
    elif method == 'drpls':
        baseline_y, params = base.drpls(y, lam=lam)
    elif method == 'modpoly':
        baseline_y, params = base.modpoly(y, poly_order=3)
    else:
        baseline_y, params = base.asls(y, lam=lam, p=p)
    corrected = y - baseline_y
    return baseline_y, corrected


def smooth_signal(y: np.ndarray, window_length: int = 11, polyorder: int = 2) -> np.ndarray:
    """Apply Savitzky–Golay smoothing to a 1D signal.

    The window length must be odd and greater than the polynomial order.
    """
    # Ensure window_length is odd and not larger than the signal
    wl = max(3, window_length)
    if wl % 2 == 0:
        wl += 1
    wl = min(wl, len(y) - (len(y) % 2 == 0))
    return savgol_filter(y, wl, max(1, polyorder))


def normalize_vector(y: np.ndarray) -> np.ndarray:
    """Normalize a 1D signal so that its vector norm equals one."""
    norm = np.linalg.norm(y)
    return y / norm if norm > 0 else y


def plot_spectra(selected: List[str], spectra: Dict[str, pd.DataFrame], peaks: Optional[Dict[str, np.ndarray]] = None) -> None:
    """Plot selected spectra and optionally overlay peak positions.

    Parameters
    ----------
    selected : list of str
        Keys of spectra to plot.
    spectra : dict
        Dictionary mapping names to DataFrames.
    peaks : dict, optional
        Mapping from spectrum name to array of peak indices in the
        corresponding DataFrame.
    """
    fig = go.Figure()
    for name in selected:
        df = spectra[name]
        fig.add_trace(
            go.Scatter(x=df['x'], y=df['y'], mode='lines', name=name)
        )
        if peaks and name in peaks:
            # Mark peaks on the line
            pk_inds = peaks[name]
            x_vals = df['x'].iloc[pk_inds]
            y_vals = df['y'].iloc[pk_inds]
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='markers',
                    marker=dict(symbol='x', size=8, color='red'),
                    name=f"{name} peaks"
                )
            )
    fig.update_layout(
        xaxis_title='Wavenumber / Frequency / X',
        yaxis_title='Intensity / Absorbance / Y',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Spectral Analysis", layout="wide")
    st.title("Spectral Analysis Web App")
    st.markdown(
        "Upload one or more spectra and apply baseline correction, peak "
        "detection, averaging, subtraction, smoothing and normalisation."
    )

    # Session state to hold spectra across reruns
    if 'spectra' not in st.session_state:
        st.session_state.spectra: Dict[str, pd.DataFrame] = {}

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload spectra", type=['txt', 'csv', 'wdf'], accept_multiple_files=True
    )
    if uploaded_files:
        for file in uploaded_files:
            name = file.name
            # Save .wdf files temporarily because ramanspy expects a path
            if name.lower().endswith('.wdf'):
                # Use NamedTemporaryFile to write to disk
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wdf') as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name
                try:
                    wdf_spectra = read_wdf_spectrum(tmp_path)
                    for key, df in wdf_spectra.items():
                        # Guarantee unique keys in the session state
                        label = key
                        counter = 1
                        while label in st.session_state.spectra:
                            label = f"{key}_{counter}"
                            counter += 1
                        st.session_state.spectra[label] = df
                except Exception as exc:
                    st.error(f"Failed to load {name}: {exc}")
                finally:
                    import os
                    os.unlink(tmp_path)
            else:
                try:
                    df = read_text_spectrum(file)
                    label = name
                    counter = 1
                    while label in st.session_state.spectra:
                        label = f"{name}_{counter}"
                        counter += 1
                    st.session_state.spectra[label] = df
                except Exception as exc:
                    st.error(f"Failed to read {name}: {exc}")

    spectra = st.session_state.spectra
    if not spectra:
        st.info("No spectra loaded yet.  Upload files to get started.")
        return

    st.sidebar.header("Spectra selection")
    spectrum_names = list(spectra.keys())
    selected = st.sidebar.multiselect(
        "Choose spectra to plot", spectrum_names, default=spectrum_names[:1]
    )

    peaks_dict: Dict[str, np.ndarray] = {}
    # Plot the selected spectra
    if selected:
        plot_spectra(selected, spectra)

        # Options: smoothing and normalisation
        with st.expander("Pre‑processing: smoothing & normalisation"):
            smooth_enabled = st.checkbox("Apply Savitzky–Golay smoothing")
            if smooth_enabled:
                win_length = st.slider("Window length", min_value=3, max_value=101, value=11, step=2)
                polyorder = st.slider("Polynomial order", min_value=1, max_value=5, value=2)
                if st.button("Smooth selected spectra"):
                    for name in selected:
                        df = spectra[name]
                        smoothed_y = smooth_signal(df['y'].values, window_length=win_length, polyorder=polyorder)
                        new_df = pd.DataFrame({'x': df['x'], 'y': smoothed_y})
                        new_name = f"{name}_smoothed"
                        counter = 1
                        while new_name in spectra:
                            new_name = f"{name}_smoothed_{counter}"
                            counter += 1
                        spectra[new_name] = new_df
                    st.experimental_rerun()
            norm_enabled = st.checkbox("Normalize selected spectra (vector norm = 1)")
            if norm_enabled:
                if st.button("Normalize"):
                    for name in selected:
                        df = spectra[name]
                        norm_y = normalize_vector(df['y'].values)
                        new_df = pd.DataFrame({'x': df['x'], 'y': norm_y})
                        new_name = f"{name}_norm"
                        counter = 1
                        while new_name in spectra:
                            new_name = f"{name}_norm_{counter}"
                            counter += 1
                        spectra[new_name] = new_df
                    st.experimental_rerun()

        # Baseline correction options
        with st.expander("Baseline correction"):
            if not _has_baseline:
                st.warning("pybaselines not installed – baseline correction disabled.")
            else:
                method = st.selectbox(
                    "Algorithm",
                    options=['asls', 'arpls', 'airpls', 'drpls', 'modpoly'],
                    index=0
                )
                lam_exp = st.slider(
                    "log10(lambda)", min_value=2.0, max_value=8.0, value=5.0, step=0.5,
                    help="Regularisation parameter controlling smoothness"
                )
                lam = 10 ** lam_exp
                p = st.slider(
                    "p (asymmetry parameter, only for AsLS)", min_value=0.001, max_value=0.1, value=0.01, step=0.001
                )
                if st.button("Apply baseline correction"):
                    for name in selected:
                        df = spectra[name]
                        baseline_y, corrected = baseline_correct(df['y'].values, method=method, lam=lam, p=p)
                        new_df = pd.DataFrame({'x': df['x'], 'y': corrected})
                        new_name = f"{name}_blcorr"
                        counter = 1
                        while new_name in spectra:
                            new_name = f"{name}_blcorr_{counter}"
                            counter += 1
                        spectra[new_name] = new_df
                    st.experimental_rerun()

        # Peak detection options
        with st.expander("Peak detection"):
            height = st.number_input("Minimum peak height", value=0.0, format="%.4f")
            prominence = st.number_input("Minimum prominence", value=0.0, format="%.4f")
            distance = st.number_input("Minimum distance between peaks", value=0.0, format="%.1f")
            if st.button("Detect peaks"):
                for name in selected:
                    df = spectra[name]
                    y_vals = df['y'].values
                    # Determine dynamic defaults for height if not set
                    kwargs = {}
                    if height > 0:
                        kwargs['height'] = height
                    if prominence > 0:
                        kwargs['prominence'] = prominence
                    if distance > 0:
                        kwargs['distance'] = distance
                    indices, properties = find_peaks(y_vals, **kwargs)
                    peaks_dict[name] = indices
                    # Display a table of peak positions/intensities
                    peak_positions = df['x'].iloc[indices].values
                    peak_heights = df['y'].iloc[indices].values
                    table = pd.DataFrame({
                        'Position (x)': peak_positions,
                        'Height (y)': peak_heights
                    })
                    st.write(f"Peaks for **{name}**:")
                    st.dataframe(table)
                # After detection, update plot with markers
                plot_spectra(selected, spectra, peaks=peaks_dict)

        # Averaging and subtraction
        with st.expander("Combine spectra: averaging & subtraction"):
            st.write("Select two or more spectra for averaging, or exactly two for subtraction.")
            if st.button("Average selected"):
                if len(selected) < 2:
                    st.warning("Select at least two spectra to average.")
                else:
                    dfs = [spectra[name] for name in selected]
                    try:
                        avg_x, avg_y = interpolate_to_common_axis(dfs)
                        new_df = pd.DataFrame({'x': avg_x, 'y': avg_y})
                        new_name = "avg(" + ",".join(selected) + ")"
                        counter = 1
                        while new_name in spectra:
                            new_name = f"avg({'_'.join(selected)})_{counter}"
                            counter += 1
                        spectra[new_name] = new_df
                        st.success(f"Added averaged spectrum: {new_name}")
                        st.experimental_rerun()
                    except Exception as exc:
                        st.error(f"Failed to average spectra: {exc}")
            spec_a = st.selectbox("Spectrum A (minuend)", spectrum_names, index=0, key='sub_a')
            spec_b = st.selectbox("Spectrum B (subtrahend)", spectrum_names, index=1 if len(spectrum_names) > 1 else 0, key='sub_b')
            if st.button("Subtract A − B"):
                if spec_a == spec_b:
                    st.warning("Please choose two different spectra for subtraction.")
                else:
                    result = subtract_spectra(spectra[spec_a], spectra[spec_b])
                    new_name = f"{spec_a}_minus_{spec_b}"
                    counter = 1
                    while new_name in spectra:
                        new_name = f"{new_name}_{counter}"
                        counter += 1
                    spectra[new_name] = result
                    st.success(f"Added difference spectrum: {new_name}")
                    st.experimental_rerun()

        # Export selected spectra
        with st.expander("Export"):
            export_format = st.selectbox("Format", options=['csv'])
            if st.button("Download selected spectra"):
                for name in selected:
                    df = spectra[name]
                    csv_bytes = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"Download {name}.{export_format}",
                        data=csv_bytes,
                        file_name=f"{name}.{export_format}",
                        mime='text/csv'
                    )


if __name__ == '__main__':
    main()