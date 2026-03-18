# EXODUS — Multi-Channel Technosignature Search Engine

A systematic pipeline for identifying stars with statistically unlikely combinations of anomalies across multiple independent observation channels. Designed for large-scale surveys where false positive control is the primary challenge.

## Overview

EXODUS cross-matches data from major astronomical surveys (Gaia DR3, CatWISE2020, AllWISE, 2MASS, GALEX, VLASS, NVSS) and scores each target across independent detection channels. Candidates are filtered through astrophysical template matching, matched-control calibration, and empirical false positive rate estimation.

**Design:** Tested across multiple stellar populations with diverse selection functions. Calibrated on known astrophysical objects (binaries, debris disks, YSOs, giants) with zero false detections.

## Detection Channels

| Channel | Data Source | Method |
|---------|------------|--------|
| IR excess | WISE W3/W4 | Photospheric model subtraction, σ-excess |
| Proper motion anomaly | Gaia–CatWISE | Multi-epoch astrometric offset |
| UV anomaly | GALEX NUV/FUV | Predicted vs observed UV flux |
| HR diagram anomaly | Gaia | Distance from main sequence locus |
| Radio emission | VLASS/NVSS | Cross-match + flux density |
| IR variability | NEOWISE multi-epoch | Secular trend + chi-squared |
| Abundance anomaly | Spectroscopic surveys | Chemical peculiarity flags |
| Gaia photometric anomaly | Gaia epoch photometry | Excess noise / scatter |

Plus multi-messenger cross-correlation (Fermi-LAT, IceCube, FRB, NANOGrav) and a habitable zone prior for known exoplanet hosts.

## Statistical Framework

- **Scoring**: Geometric mean across active channels with convergence bonus (4^(n-1) for n active channels) and coverage penalty
- **P-value combination**: Empirical Brown's Method (EBM; Poole et al. 2016) using empirical covariance from matched control stars. Also computes Fisher and Stouffer combinations.
- **FDR correction**: Benjamini-Hochberg at α = 0.05
- **Calibrated p-values**: Per-channel empirical null distributions from matched control populations
- **Channel independence**: Verified empirically — max inter-channel correlation |r| = 0.198

## False Positive Control

- **9 astrophysical templates**: Binary (3 subtypes), debris disk, YSO, active flare star, RS CVn, background contamination, instrumental systematic — each with channel-specific conflict checks
- **Unexplainability scoring**: Residual anomaly after subtracting best-matching template
- **14 red-team checks**: Artifact rejection (WISE confusion, crowded fields, diffraction spikes, etc.)
- **PM–IR correlation penalty**: Detects cases where WISE photocenter shift creates correlated IR excess and PM anomaly
- **Two-component SED fitting**: 6-band (H, Ks, W1–W4) photospheric subtraction with hot+warm dust model
- **Calibration chain**: 4 populations of known objects (binaries, disks, YSOs, giants) — all return 0 FDR

## Architecture

```
src/
├── core/           # EBM statistics, evidence tracking, provenance
├── detection/      # Channel detectors (one module per channel)
├── ingestion/      # Catalog loaders (Gaia, WISE, 2MASS, GALEX, VLASS, etc.)
├── processing/     # Signal processors
├── scoring/        # EXODUS score (geometric mean + convergence priority)
└── vetting/        # Template matching, SED fitting, artifact rejection

scripts/            # Campaign runners, target builders, calibration tools
explorer/           # Interactive sky map (React + FastAPI + Aladin Lite)
tests/              # 114 tests (regression, stress, channel, EBM, speed)
config/             # Pipeline configuration
```

~80K lines of Python across 139 files.

## Galaxy Explorer

Web-based visualization tool for exploring results on an interactive sky map:

- **Aladin Lite** sky background with multi-layer overlays
- Channel-specific heatmaps (IR, PM, UV, radio, etc.)
- Multi-messenger catalog overlays (Fermi, IceCube, FRB, pulsars)
- Convergence zone detection engine
- Target detail panels with radar fingerprint charts
- Side-by-side target comparison
- Annotation system for marking regions of interest

## Requirements

- Python 3.11+ with scientific stack (numpy, scipy, astropy, astroquery)
- Node.js 18+ for the Galaxy Explorer frontend
- See `requirements.txt` for full Python dependencies

## Status

Active research project. Pipeline operational across multiple stellar populations. Results forthcoming in a methods paper.

## License

MIT License. See [LICENSE](LICENSE) for details.
