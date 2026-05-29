# LSD Reconfigures Cortical Dynamics Through Faster Brain Rhythms and Increased Fractal Dimension

Analysis code for the MEG study examining how LSD alters cortical spectral and complexity dynamics across multiple sensory contexts : https://doi.org/10.64898/2026.01.28.702361

---

## Overview

This repository contains the full analysis pipeline, from raw MEG preprocessing through source localisation, spectral/complexity feature extraction, statistical testing, and machine learning decoding.

**Drug conditions**: LSD vs. Placebo (PLA). Single-blind crossover design.
**Tasks/contexts**: Rest eyes-closed with Music, Rest eyes closed WithoutMusic
**Subjects**: 18 participants (IDs: 001–020, with gaps)  
**Brain parcellation**: 360-region Glasser HCP-MMP1 atlas

---

## Pipeline

```
Raw MEG (BIDS .fif)
        │
        ▼
  1. Preprocessing
     Notch filter (50 Hz harmonics) → bandpass (1–125 Hz)
     → manual epoch selection → AutoReject → ICA (picard)
     → AutoReject (post-ICA)
        │
        ▼
  2. Source Localisation
     FreeSurfer BEM (single-shell) → forward model (ico5)
     → dSPM inverse → morph to fsaverage → HCP-MMP1 parcellation
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
  3a. Source-space PSD (1–45 Hz)     3b. Source time series (stc)
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
  4. Feature Extraction
     FOOOF (aperiodic exponent/offset, band peaks & power)
     Higuchi Fractal Dimension (kmax=18)
     Lempel-Ziv Complexity (broadband + band-specific)
                       │
                       ▼
  5. Statistics & ML
     Mixed models │ Random Forest + permutation testing
```

---

## Repository Structure

| Directory | Contents |
|-----------|----------|
| `preprocessing/` | MEG cleaning pipeline (notch, bandpass, ICA, AutoReject) |
| `source_localization/` | BEM surfaces, forward model, dSPM inverse, parcellation |
| `spectral_analysis/` | Source PSD computation; FOOOF, HFD, LZC feature extraction |
| `stats/` | Statistical analyses (mixed models, t-tests) |
| `ML/` | Random Forest decoding with permutation tests and brain maps |
| `visualization/` | Figure generation notebooks |
| `Revision/` | Sensor-level sensitivity analyses and band-specific LZC (added at revision) |
| `phenomenology_regression/` | Regression of neural features onto subjective experience ratings |

---

## Usage

All CLI scripts share the same argument structure:

```bash
# -s: subject IDs   -t: task   -d: drug condition
python preprocessing/preprocessing.py -s 001 002 003 -t Music -d LSD
python spectral_analysis/source_PSD_compute.py -s 001 002 -t Music -d LSD
python source_localization/forward_and_inverse_solver.py -s 001 002 -t Music -d LSD
python spectral_analysis/extract_spectral_temporal_features.py   # loops subjects internally
```

Notebooks (`ML/ML.ipynb`, `stats/statistics.ipynb`, `visualization/*.ipynb`) are run interactively.

> **Note**: Raw data and derivatives are stored externally at `/Brain/private/v20subra/LSD_project/` and are not included in this repository.

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `mne==1.8.0`, `fooof==1.1.0`, `antropy==0.1.9`, `lempel-ziv-complexity==0.2.2`, `autoreject==0.4.3`, `scikit-learn==1.5.2`, `pingouin==0.5.5`.

For revision-only analyses:
```bash
pip install -r Revision/requirements_revision.txt
```

---

## Citation

> *LSD Reconfigures Cortical Dynamics Through Faster Brain Rhythms and Increased Fractal Dimension*  
> (citation to be added upon publication)
