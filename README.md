# Spectral Analysis Web Application

This repository contains a web‑based tool for processing and analysing spectroscopic data.  It was designed as a modern, user‑friendly alternative to legacy programs such as **Omnic**.  The application is built with **Python** and **Streamlit**, providing an accessible interface for common spectroscopic tasks, including:

* **Bulk file loading** – upload multiple spectra at once (e.g. `.txt`, `.csv`, `.wdf`) and switch between them using a sidebar selector.
* **Interactive plotting** – visualise one or more spectra simultaneously using interactive Plotly line graphs.
* **Baseline correction** – remove low‑frequency drifts using a variety of algorithms from the open‑source `pybaselines` library.  `pybaselines` implements more than fifty baseline correction algorithms—including AsLS, airPLS and ModPoly—designed for Raman, FTIR, NMR, XRD and related techniques【330007600347264†L30-L41】.
* **Peak detection** – automatically detect local maxima using SciPy’s `find_peaks` function.  Peaks are returned as both a table of positions/intensities and markers on the plot.  Parameter controls allow you to set minimum peak height, prominence and spacing【146054626248010†L1274-L1281】【146054626248010†L1415-L1439】.
* **Averaging and subtraction** – average an arbitrary number of selected spectra or subtract one spectrum from another.  Spectra with differing abscissa values are interpolated to a common axis before averaging or subtraction.
* **Normalization and smoothing** – optional Savitzky–Golay filtering for noise reduction and vector normalization to compare spectra on the same scale.
* **Export** – download processed spectra (CSV) directly from the interface.

In addition, the application supports Renishaw’s `.wdf` format via the `ramanspy` library.  `RamanSPy` provides a `load.renishaw()` helper that parses `.wdf` files into a spectral data container【405946903641446†L134-L149】.  If `ramanspy` is installed, `.wdf` files will be loaded automatically.

## Getting started

1. Clone this repository locally or deploy it on a server:

```bash
git clone https://github.com/YOUR_USERNAME/spectral-analysis-app.git
cd spectral-analysis-app
pip install -r requirements.txt
streamlit run app.py
```

2. Open the app in your browser (Streamlit will print a local URL).  Use the file uploader to import spectra and explore the processing options in the sidebar.

## Repository creation via the GitHub API

The code in this repository can be automatically pushed to your GitHub account using the REST API.  GitHub’s `Create a repository for the authenticated user` endpoint (`POST /user/repos`) accepts a JSON body with at least a `name` parameter and optional settings such as a description, visibility flags, and whether to initialise the repository with a README【504460264174486†L5940-L5990】.  A personal access token with the `repo` scope is required for authentication.  The script `push_to_github.py` (see below) demonstrates how to call this endpoint programmatically.

## License

This project is released under the MIT License.