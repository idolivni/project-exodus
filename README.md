# EXODUS — Multi-Channel Technosignature Search Engine

A systematic pipeline for detecting anomalous stellar signatures across multiple independent observation channels, designed for large-scale SETI surveys.

## What It Does

EXODUS cross-correlates data from 11 independent detection channels to identify stars exhibiting statistically unlikely combinations of anomalies that resist conventional astrophysical explanation:

- **Infrared excess** (WISE W3/W4) — anomalous mid-IR emission beyond photospheric models
- **Proper motion anomaly** (Gaia–CatWISE) — astrometric discrepancies suggesting unseen companions or structures
- **UV anomaly** (GALEX NUV/FUV) — unexpected ultraviolet deficits or excesses
- **HR diagram anomaly** — positions inconsistent with standard stellar evolution
- **Abundance anomaly** — chemical peculiarities from spectroscopic surveys
- **Transit anomaly** — irregular or aperiodic dimming events
- **Radio emission** — unexpected radio detections (VLASS/NVSS cross-match)
- **IR variability** — mid-IR flux changes across NEOWISE epochs
- **Gaia photometric anomaly** — excess astrometric noise or photometric scatter
- **Habitable zone prior** — known exoplanet host weighting

The pipeline applies Stouffer's method for p-value combination, FDR correction (Benjamini-Hochberg), binary/known-object template matching, and multi-messenger cross-correlation (Fermi-LAT, IceCube, FRB, NANOGrav).

## Architecture

```
src/
├── core/           # Statistics, evidence tracking, provenance
├── detection/      # 11 channel detectors
├── ingestion/      # Catalog loaders (Gaia, WISE, 2MASS, GALEX, etc.)
├── processing/     # Signal processors
├── scoring/        # EXODUS score (geometric mean + convergence)
└── vetting/        # Cross-match, SED fitting, artifact rejection

scripts/            # Campaign runners, target builders, analysis tools
explorer/           # Interactive sky map (React + FastAPI + Aladin Lite)
tests/              # 94 tests (regression, stress, channel, speed validation)
config/             # Pipeline configuration
```

## Galaxy Explorer

Web-based visualization tool for exploring results on an interactive sky map:

- **Aladin Lite** sky background with multi-layer overlays
- Channel-specific heatmaps (IR, PM, UV, radio, etc.)
- Multi-messenger catalog overlays (Fermi, IceCube, FRB, pulsars)
- Convergence zone detection engine
- Target detail panels with radar fingerprint charts
- Side-by-side target comparison
- Annotation system for marking regions of interest

## Key Methodological Features

- **Channel independence**: Each detector runs independently; scores combine via geometric mean to reward multi-channel convergence
- **Binary template matching**: Expanded template library catches M-dwarf binary signatures (PM+HR+UV pattern) that dominate multi-channel anomalies
- **FDR correction**: Benjamini-Hochberg procedure controls false discovery rate across full target populations
- **Unexplainability scoring**: Residual score after subtracting best-matching known-object template
- **Two-component SED fitting**: 6-band (H, Ks, W1-W4) photospheric subtraction with hot+warm dust model (ndof=2)
- **Calibrated p-values**: Per-channel empirical null distributions from control populations

## Requirements

- Python 3.11+ with scientific stack (numpy, scipy, astropy, astroquery)
- Node.js 18+ for the Galaxy Explorer frontend
- See `requirements.txt` for full Python dependencies

## Status

Active research project. Pipeline operational across multiple stellar populations. Results forthcoming in a methods paper.

## License

All rights reserved. Code published for priority documentation and review purposes.
