"""
Breakthrough Listen (BL) radio observation data ingestion module.

Accesses HDF5/filterbank files from the BL Open Data Archive,
parses observation metadata, and provides spectrogram arrays
(frequency x time) for downstream technosignature analysis.

The BL archive exposes a REST API at http://seti.berkeley.edu/opendata.
Actual data files are large HDF5/filterbank format, readable with blimpy
or h5py.  This module provides a cache-first approach: metadata is always
cached, and data files are cached locally when small enough.

When the BL API is unreachable, a simulated mode generates realistic
synthetic radio observation data for development and testing.
"""

from __future__ import annotations

import json
import os
import sys
import time
import hashlib
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project utilities
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import (
    get_logger,
    get_config,
    cache_key,
    load_cache,
    save_cache,
    PROJECT_ROOT,
)

logger = get_logger("ingestion.breakthrough_listen")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BL_API_BASE = "http://seti.berkeley.edu/opendata"
BL_DATA_URL_TEMPLATE = "http://blpd0.ssl.berkeley.edu/dl/{file_path}"
CACHE_SUBFOLDER = "breakthrough_listen"
# Maximum individual file size we are willing to cache locally (50 MB)
MAX_CACHE_FILE_BYTES = 50 * 1024 * 1024
# HTTP request timeout (seconds)
REQUEST_TIMEOUT = 30

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ObservationMeta:
    """Metadata for a single Breakthrough Listen observation."""
    target: str
    ra: Optional[float] = None          # Right ascension (degrees)
    dec: Optional[float] = None         # Declination (degrees)
    freq_start_mhz: Optional[float] = None
    freq_end_mhz: Optional[float] = None
    obs_time_utc: Optional[str] = None  # ISO-8601 string
    duration_sec: Optional[float] = None
    telescope: Optional[str] = None     # e.g. "GBT", "Parkes", "APF"
    file_url: Optional[str] = None
    file_size_bytes: Optional[int] = None
    file_type: Optional[str] = None     # "hdf5" | "filterbank" | "fits"
    source: str = "bl_archive"          # "bl_archive" | "simulated"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ObservationMeta":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Observation:
    """Full observation: metadata + optional spectrogram payload."""
    meta: ObservationMeta
    spectrogram: Optional[np.ndarray] = field(default=None, repr=False)
    frequencies_mhz: Optional[np.ndarray] = field(default=None, repr=False)
    timestamps_sec: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def shape(self) -> Optional[Tuple[int, int]]:
        if self.spectrogram is not None:
            return self.spectrogram.shape
        return None


# ---------------------------------------------------------------------------
# Optional heavy imports (blimpy, h5py, requests)
# ---------------------------------------------------------------------------

def _try_import_blimpy():
    """Return blimpy.Waterfall class or None."""
    try:
        from blimpy import Waterfall
        return Waterfall
    except ImportError:
        return None


def _try_import_h5py():
    try:
        import h5py
        return h5py
    except ImportError:
        return None


def _try_import_requests():
    try:
        import requests
        return requests
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# BL Archive API client
# ---------------------------------------------------------------------------

class BLArchiveClient:
    """Thin wrapper around the Breakthrough Listen Open Data REST API."""

    def __init__(self, base_url: str = BL_API_BASE, timeout: int = REQUEST_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._requests = _try_import_requests()

    # -- connectivity check --------------------------------------------------

    def is_reachable(self) -> bool:
        """Return True if the BL API responds to a basic request."""
        if self._requests is None:
            logger.warning("requests library not installed; BL API unreachable")
            return False
        try:
            resp = self._requests.get(
                self.base_url,
                timeout=self.timeout,
            )
            return resp.status_code < 500
        except Exception as exc:
            logger.debug("BL API unreachable: %s", exc)
            return False

    # -- query helpers -------------------------------------------------------

    def query_targets(self) -> List[Dict[str, Any]]:
        """
        Query the BL archive for available observation targets.

        Returns a list of dicts, each representing a target with fields
        such as 'target', 'ra', 'dec', 'telescope', etc.
        """
        if self._requests is None:
            raise RuntimeError("requests library not available")

        # The BL open data archive serves a JSON index; try known endpoints.
        # Primary: /api/targets  (newer)
        # Fallback: /data  (older static index)
        for endpoint in ("/api/targets", "/api/observations", "/data"):
            url = f"{self.base_url}{endpoint}"
            try:
                resp = self._requests.get(url, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    # Normalise: the response may be a list or a dict
                    # wrapping a list.
                    if isinstance(data, list):
                        return data
                    if isinstance(data, dict):
                        for key in ("targets", "observations", "data", "results"):
                            if key in data and isinstance(data[key], list):
                                return data[key]
                        # If nothing matched, return the dict in a list
                        return [data]
            except Exception as exc:
                logger.debug("Endpoint %s failed: %s", url, exc)
                continue

        raise RuntimeError("All BL API endpoints failed")

    def download_file(self, file_url: str, dest_path: Path) -> Path:
        """Download a BL data file to *dest_path*."""
        if self._requests is None:
            raise RuntimeError("requests library not available")

        logger.info("Downloading BL file: %s -> %s", file_url, dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        resp = self._requests.get(file_url, stream=True, timeout=self.timeout)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
        logger.info("Download complete: %s (%d bytes)", dest_path, dest_path.stat().st_size)
        return dest_path


# ---------------------------------------------------------------------------
# HDF5 / filterbank file reader
# ---------------------------------------------------------------------------

class BLFileReader:
    """
    Reads Breakthrough Listen HDF5 / filterbank files.

    Prefers blimpy (Waterfall) when available; falls back to raw h5py.
    """

    def __init__(self):
        self._Waterfall = _try_import_blimpy()
        self._h5py = _try_import_h5py()
        if self._Waterfall:
            logger.debug("Using blimpy Waterfall for BL file reading")
        elif self._h5py:
            logger.debug("blimpy not found; using h5py fallback")
        else:
            logger.warning("Neither blimpy nor h5py available; file reading disabled")

    @property
    def can_read(self) -> bool:
        return self._Waterfall is not None or self._h5py is not None

    def read_metadata(self, filepath: Path) -> ObservationMeta:
        """Extract observation metadata from a BL data file."""
        filepath = Path(filepath)
        if self._Waterfall:
            return self._meta_blimpy(filepath)
        if self._h5py:
            return self._meta_h5py(filepath)
        raise RuntimeError("No HDF5 reader available")

    def read_spectrogram(self, filepath: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Read the spectrogram from a BL data file.

        Returns
        -------
        spectrogram : np.ndarray  (n_freq, n_time)
        frequencies_mhz : np.ndarray
        timestamps_sec : np.ndarray
        """
        filepath = Path(filepath)
        if self._Waterfall:
            return self._spectrogram_blimpy(filepath)
        if self._h5py:
            return self._spectrogram_h5py(filepath)
        raise RuntimeError("No HDF5 reader available")

    # -- blimpy path ---------------------------------------------------------

    def _meta_blimpy(self, filepath: Path) -> ObservationMeta:
        wf = self._Waterfall(str(filepath), load_data=False)
        header = wf.header
        fch1 = header.get("fch1", 0.0)
        foff = header.get("foff", 0.0)
        nchans = header.get("nchans", 0)
        freq_end = fch1
        freq_start = fch1 + foff * nchans
        if freq_start > freq_end:
            freq_start, freq_end = freq_end, freq_start
        return ObservationMeta(
            target=header.get("source_name", filepath.stem),
            ra=header.get("src_raj"),
            dec=header.get("src_dej"),
            freq_start_mhz=freq_start,
            freq_end_mhz=freq_end,
            obs_time_utc=header.get("tstart"),
            duration_sec=header.get("tsamp", 0) * header.get("nifs", 1) * nchans,
            telescope=header.get("telescope_id"),
            file_url=None,
            file_size_bytes=filepath.stat().st_size,
            file_type="hdf5" if filepath.suffix in (".h5", ".hdf5") else "filterbank",
            source="bl_archive",
        )

    def _spectrogram_blimpy(self, filepath: Path):
        wf = self._Waterfall(str(filepath))
        data = wf.data  # shape: (n_ifs, n_time, n_freq) typically
        # Squeeze IF axis if present and shape is 3-d
        if data.ndim == 3:
            data = data[0]  # first IF
        # Transpose to (n_freq, n_time) convention
        if data.shape[0] < data.shape[1]:
            spectrogram = data
        else:
            spectrogram = data.T
        freqs = np.linspace(
            wf.header.get("fch1", 0),
            wf.header.get("fch1", 0) + wf.header.get("foff", 0) * wf.header.get("nchans", spectrogram.shape[0]),
            spectrogram.shape[0],
        )
        times = np.arange(spectrogram.shape[1]) * wf.header.get("tsamp", 1.0)
        return spectrogram.astype(np.float32), freqs, times

    # -- h5py fallback -------------------------------------------------------

    def _meta_h5py(self, filepath: Path) -> ObservationMeta:
        with self._h5py.File(str(filepath), "r") as hf:
            # BL HDF5 files typically store data under "data" group
            # and header attributes at root or under "data".
            attrs = dict(hf.attrs) if hf.attrs else {}
            if "data" in hf:
                attrs.update(dict(hf["data"].attrs))
            fch1 = float(attrs.get("fch1", 0.0))
            foff = float(attrs.get("foff", 0.0))
            nchans = int(attrs.get("nchans", 0))
            freq_end = fch1
            freq_start = fch1 + foff * nchans
            if freq_start > freq_end:
                freq_start, freq_end = freq_end, freq_start
            return ObservationMeta(
                target=str(attrs.get("source_name", filepath.stem)),
                ra=float(attrs["src_raj"]) if "src_raj" in attrs else None,
                dec=float(attrs["src_dej"]) if "src_dej" in attrs else None,
                freq_start_mhz=freq_start,
                freq_end_mhz=freq_end,
                obs_time_utc=str(attrs.get("tstart", "")),
                duration_sec=float(attrs.get("tsamp", 0)) * nchans,
                telescope=str(attrs.get("telescope_id", "unknown")),
                file_url=None,
                file_size_bytes=filepath.stat().st_size,
                file_type="hdf5",
                source="bl_archive",
            )

    def _spectrogram_h5py(self, filepath: Path):
        with self._h5py.File(str(filepath), "r") as hf:
            # Locate the primary data array
            data = None
            for key in ("data", "Data", "filterbank"):
                if key in hf:
                    node = hf[key]
                    if hasattr(node, "shape"):
                        data = node[:]
                        break
                    # Nested dataset
                    for subkey in node:
                        if hasattr(node[subkey], "shape"):
                            data = node[subkey][:]
                            break
                    if data is not None:
                        break
            if data is None:
                # Last resort: first dataset in the file
                def _find_first_dataset(group):
                    for k in group:
                        item = group[k]
                        if hasattr(item, "shape") and len(item.shape) >= 2:
                            return item[:]
                        if hasattr(item, "keys"):
                            result = _find_first_dataset(item)
                            if result is not None:
                                return result
                    return None
                data = _find_first_dataset(hf)

            if data is None:
                raise ValueError(f"No suitable data array found in {filepath}")

            # Squeeze to 2-D
            while data.ndim > 2:
                data = data[0]

            attrs = dict(hf.attrs)
            if "data" in hf:
                attrs.update(dict(hf["data"].attrs))

            fch1 = float(attrs.get("fch1", 0.0))
            foff = float(attrs.get("foff", 1.0))
            tsamp = float(attrs.get("tsamp", 1.0))

            # Ensure (n_freq, n_time)
            if data.shape[0] < data.shape[1]:
                spectrogram = data
            else:
                spectrogram = data.T

            n_freq, n_time = spectrogram.shape
            freqs = np.linspace(fch1, fch1 + foff * n_freq, n_freq)
            times = np.arange(n_time) * tsamp

            return spectrogram.astype(np.float32), freqs, times


# ---------------------------------------------------------------------------
# Simulated / mock data generator
# ---------------------------------------------------------------------------

# Well-known BL targets for realistic simulation
_SIMULATED_TARGETS = [
    {
        "target": "Kepler-160",
        "ra": 291.406,
        "dec": 42.470,
        "telescope": "GBT",
        "freq_start_mhz": 1000.0,
        "freq_end_mhz": 2000.0,
        "duration_sec": 300.0,
        "obs_time_utc": "2019-05-14T08:12:00Z",
    },
    {
        "target": "Tabby's Star",
        "ra": 301.564,
        "dec": 44.457,
        "telescope": "GBT",
        "freq_start_mhz": 1000.0,
        "freq_end_mhz": 11000.0,
        "duration_sec": 300.0,
        "obs_time_utc": "2017-10-26T04:30:00Z",
    },
    {
        "target": "Proxima Centauri",
        "ra": 217.429,
        "dec": -62.680,
        "telescope": "Parkes",
        "freq_start_mhz": 700.0,
        "freq_end_mhz": 4000.0,
        "duration_sec": 1800.0,
        "obs_time_utc": "2019-04-29T12:00:00Z",
    },
    {
        "target": "TRAPPIST-1",
        "ra": 346.622,
        "dec": -5.043,
        "telescope": "GBT",
        "freq_start_mhz": 1000.0,
        "freq_end_mhz": 12000.0,
        "duration_sec": 300.0,
        "obs_time_utc": "2018-02-12T10:45:00Z",
    },
    {
        "target": "Ross 128",
        "ra": 176.937,
        "dec": 0.800,
        "telescope": "Arecibo",
        "freq_start_mhz": 1300.0,
        "freq_end_mhz": 1700.0,
        "duration_sec": 600.0,
        "obs_time_utc": "2017-07-16T03:20:00Z",
    },
    {
        "target": "GJ 273",
        "ra": 109.859,
        "dec": 5.228,
        "telescope": "GBT",
        "freq_start_mhz": 1000.0,
        "freq_end_mhz": 2000.0,
        "duration_sec": 300.0,
        "obs_time_utc": "2018-03-20T09:15:00Z",
    },
    {
        "target": "Kepler-442",
        "ra": 291.812,
        "dec": 39.241,
        "telescope": "GBT",
        "freq_start_mhz": 1000.0,
        "freq_end_mhz": 2000.0,
        "duration_sec": 300.0,
        "obs_time_utc": "2019-06-10T06:00:00Z",
    },
    {
        "target": "HD 164922",
        "ra": 271.360,
        "dec": 26.307,
        "telescope": "APF",
        "freq_start_mhz": 3750.0,
        "freq_end_mhz": 9500.0,
        "duration_sec": 1200.0,
        "obs_time_utc": "2018-09-02T11:30:00Z",
    },
    {
        "target": "Teegarden's Star",
        "ra": 43.267,
        "dec": 16.883,
        "telescope": "GBT",
        "freq_start_mhz": 1000.0,
        "freq_end_mhz": 11000.0,
        "duration_sec": 300.0,
        "obs_time_utc": "2020-01-15T07:00:00Z",
    },
    {
        "target": "Alpha Centauri",
        "ra": 219.902,
        "dec": -60.834,
        "telescope": "Parkes",
        "freq_start_mhz": 700.0,
        "freq_end_mhz": 4000.0,
        "duration_sec": 1800.0,
        "obs_time_utc": "2019-11-22T14:00:00Z",
    },
]


def _generate_simulated_spectrogram(
    meta: ObservationMeta,
    n_freq: int = 512,
    n_time: int = 256,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a realistic-looking synthetic radio spectrogram.

    The output contains Gaussian noise floor, broadband RFI spikes,
    a faint narrowband drifting signal (simulated technosignature
    candidate), and bandpass roll-off.
    """
    if seed is None:
        # Deterministic per target name for reproducibility
        seed = int(hashlib.md5(meta.target.encode()).hexdigest()[:8], 16) % (2**31)
    rng = np.random.RandomState(seed)

    freq_start = meta.freq_start_mhz or 1000.0
    freq_end = meta.freq_end_mhz or 2000.0
    duration = meta.duration_sec or 300.0

    freqs = np.linspace(freq_start, freq_end, n_freq)
    times = np.linspace(0, duration, n_time)

    # ---- noise floor (radiometer equation approximation) ----
    # Use 1/f-ish noise + white noise to mimic real receiver
    white = rng.normal(0, 1.0, (n_freq, n_time)).astype(np.float32)
    pink = np.cumsum(rng.normal(0, 0.02, (n_freq, n_time)), axis=1).astype(np.float32)
    spectrogram = white + pink

    # ---- bandpass shape (cosine roll-off at edges) ----
    bandpass = np.ones(n_freq, dtype=np.float32)
    edge = max(1, n_freq // 20)
    bandpass[:edge] = 0.5 * (1 - np.cos(np.linspace(0, np.pi, edge)))
    bandpass[-edge:] = 0.5 * (1 - np.cos(np.linspace(np.pi, 0, edge)))
    spectrogram *= bandpass[:, np.newaxis]

    # ---- broadband RFI (random time-domain spikes) ----
    n_rfi = rng.randint(2, 8)
    for _ in range(n_rfi):
        t_idx = rng.randint(0, n_time)
        amplitude = rng.uniform(3.0, 8.0)
        width = rng.randint(1, max(2, n_time // 50))
        t_start = max(0, t_idx - width // 2)
        t_end = min(n_time, t_idx + width // 2 + 1)
        spectrogram[:, t_start:t_end] += amplitude

    # ---- narrowband drifting signal (candidate technosignature) ----
    drift_rate_hz_per_sec = rng.uniform(-2.0, 2.0)  # Hz/s
    signal_freq_idx = rng.randint(n_freq // 4, 3 * n_freq // 4)
    signal_snr = rng.uniform(5.0, 15.0)
    for t_idx in range(n_time):
        f_shift = int(drift_rate_hz_per_sec * times[t_idx] / ((freq_end - freq_start) / n_freq))
        f_idx = signal_freq_idx + f_shift
        if 0 <= f_idx < n_freq:
            spectrogram[f_idx, t_idx] += signal_snr
            # Spread to adjacent channels
            if f_idx > 0:
                spectrogram[f_idx - 1, t_idx] += signal_snr * 0.3
            if f_idx < n_freq - 1:
                spectrogram[f_idx + 1, t_idx] += signal_snr * 0.3

    # ---- scale to positive power values ----
    spectrogram = spectrogram - spectrogram.min() + 0.1

    return spectrogram, freqs, times


# ---------------------------------------------------------------------------
# Main public API class
# ---------------------------------------------------------------------------

class BreakthroughListenIngest:
    """
    Primary interface for ingesting Breakthrough Listen observation data.

    Provides three main methods:
        - list_available_targets()  -> list of target name strings
        - get_observation(target)   -> Observation with metadata
        - get_spectrogram(target)   -> (spectrogram, freqs, times) arrays

    Behaviour:
        1. Attempts to use the BL Open Data Archive REST API.
        2. Caches metadata and (small) data files locally.
        3. Falls back to simulated data when the archive is unreachable
           or when blimpy/h5py are not installed.
    """

    def __init__(self, force_simulated: bool = False):
        """
        Parameters
        ----------
        force_simulated : bool
            If True, skip all network access and use simulated data.
        """
        self._cfg = get_config()
        self._bl_url = self._cfg.get("catalogs", {}).get(
            "breakthrough_listen", {}
        ).get("url", BL_API_BASE)
        self._client = BLArchiveClient(base_url=self._bl_url)
        self._reader = BLFileReader()
        self._force_simulated = force_simulated
        self._simulated_mode = force_simulated

        self._data_dir = PROJECT_ROOT / self._cfg["project"]["data_dir"] / "breakthrough_listen"
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Eagerly check API reachability (only if not forced simulated)
        if not self._force_simulated:
            if self._client.is_reachable():
                logger.info("BL Open Data Archive is reachable at %s", self._bl_url)
                self._simulated_mode = False
            else:
                logger.warning(
                    "BL Open Data Archive unreachable; operating in simulated mode"
                )
                self._simulated_mode = True

        if self._simulated_mode:
            logger.info("Simulated mode active — synthetic data will be generated")

    # -- public API ----------------------------------------------------------

    def list_available_targets(self) -> List[str]:
        """
        Return a list of available target names.

        Tries cache first, then the BL API, then falls back to simulation.
        """
        ck = cache_key("bl", "targets_list")
        cached = load_cache(ck, subfolder=CACHE_SUBFOLDER)
        if cached is not None:
            logger.debug("Returning cached target list (%d targets)", len(cached))
            return cached

        targets = self._fetch_target_list()
        save_cache(ck, targets, subfolder=CACHE_SUBFOLDER)
        return targets

    def get_observation(self, target_name: str) -> Observation:
        """
        Return an Observation (metadata + optional data) for a given target.

        Parameters
        ----------
        target_name : str
            The name of the target to look up (case-insensitive matching).
        """
        target_name_lower = target_name.strip().lower()
        ck = cache_key("bl", "obs", target_name_lower)

        # Try cached metadata first
        cached_meta = load_cache(ck, subfolder=CACHE_SUBFOLDER)
        if cached_meta is not None:
            logger.debug("Loaded cached metadata for %s", target_name)
            meta = ObservationMeta.from_dict(cached_meta)
        else:
            meta = self._fetch_observation_meta(target_name)
            save_cache(ck, meta.to_dict(), subfolder=CACHE_SUBFOLDER)

        # Try to load actual data if we have a file URL and a reader
        spectrogram, freqs, times = None, None, None
        local_path = self._local_data_path(meta)

        if local_path is not None and local_path.exists() and self._reader.can_read:
            try:
                spectrogram, freqs, times = self._reader.read_spectrogram(local_path)
                logger.info("Loaded spectrogram from local file: %s", local_path)
            except Exception as exc:
                logger.warning("Failed to read local data file: %s", exc)

        return Observation(
            meta=meta,
            spectrogram=spectrogram,
            frequencies_mhz=freqs,
            timestamps_sec=times,
        )

    def get_spectrogram(
        self,
        target_name: str,
        n_freq: int = 512,
        n_time: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Return spectrogram arrays for a target.

        If actual BL data is available locally and readable, it is used.
        Otherwise synthetic spectrogram data is generated.

        Parameters
        ----------
        target_name : str
            Target name (case-insensitive).
        n_freq : int
            Number of frequency channels for simulated data (ignored when
            reading real files).
        n_time : int
            Number of time bins for simulated data (ignored when reading
            real files).

        Returns
        -------
        spectrogram : np.ndarray of shape (n_freq, n_time)
        frequencies_mhz : np.ndarray of shape (n_freq,)
        timestamps_sec : np.ndarray of shape (n_time,)
        source : str
            ``"bl_archive"`` if real BL data was used,
            ``"simulated"`` if synthetic data was generated.
        """
        # First attempt: read from a real observation file
        obs = self.get_observation(target_name)
        if obs.spectrogram is not None:
            return obs.spectrogram, obs.frequencies_mhz, obs.timestamps_sec, "bl_archive"

        # Second attempt: download file if URL is known, size is manageable,
        # and we have a reader.
        if (
            not self._simulated_mode
            and obs.meta.file_url
            and self._reader.can_read
            and (obs.meta.file_size_bytes or 0) <= MAX_CACHE_FILE_BYTES
        ):
            try:
                local_path = self._download_data_file(obs.meta)
                spectrogram, freqs, times = self._reader.read_spectrogram(local_path)
                logger.info("Read spectrogram from downloaded file for %s", target_name)
                return spectrogram, freqs, times, "bl_archive"
            except Exception as exc:
                logger.warning(
                    "Failed to download/read data for %s: %s — falling back to simulation",
                    target_name,
                    exc,
                )

        # Fallback: generate synthetic spectrogram
        logger.info("Generating simulated spectrogram for %s", target_name)
        spectrogram, freqs, times = _generate_simulated_spectrogram(
            obs.meta, n_freq=n_freq, n_time=n_time
        )
        return spectrogram, freqs, times, "simulated"

    def download_observation(self, target_name: str) -> Optional[Path]:
        """
        Download the BL data file for a target (if available).

        Returns the local file path, or None if download is not possible.
        """
        obs = self.get_observation(target_name)
        if not obs.meta.file_url:
            logger.warning("No file URL available for target %s", target_name)
            return None
        if self._simulated_mode:
            logger.warning("Cannot download in simulated mode")
            return None
        try:
            return self._download_data_file(obs.meta)
        except Exception as exc:
            logger.error("Download failed for %s: %s", target_name, exc)
            return None

    # -- internal helpers ----------------------------------------------------

    def _fetch_target_list(self) -> List[str]:
        """Fetch target names from API or return simulated list."""
        if not self._simulated_mode:
            try:
                raw = self._client.query_targets()
                targets = []
                for item in raw:
                    name = (
                        item.get("target")
                        or item.get("target_name")
                        or item.get("source_name")
                        or item.get("name")
                    )
                    if name and name not in targets:
                        targets.append(name)
                if targets:
                    logger.info("Fetched %d targets from BL archive", len(targets))
                    return targets
            except Exception as exc:
                logger.warning("Failed to fetch targets from API: %s — using simulated list", exc)
                self._simulated_mode = True

        # Simulated
        return [t["target"] for t in _SIMULATED_TARGETS]

    def _fetch_observation_meta(self, target_name: str) -> ObservationMeta:
        """Fetch or synthesise metadata for one target."""
        target_lower = target_name.strip().lower()

        # Try API
        if not self._simulated_mode:
            try:
                raw = self._client.query_targets()
                for item in raw:
                    name = (
                        item.get("target")
                        or item.get("target_name")
                        or item.get("source_name")
                        or item.get("name", "")
                    )
                    if name.lower() == target_lower:
                        return ObservationMeta(
                            target=name,
                            ra=item.get("ra"),
                            dec=item.get("dec"),
                            freq_start_mhz=item.get("freq_start") or item.get("freq_start_mhz"),
                            freq_end_mhz=item.get("freq_end") or item.get("freq_end_mhz"),
                            obs_time_utc=item.get("obs_time") or item.get("utc_start"),
                            duration_sec=item.get("duration") or item.get("duration_sec"),
                            telescope=item.get("telescope") or item.get("telescope_id"),
                            file_url=item.get("url") or item.get("file_url"),
                            file_size_bytes=item.get("size") or item.get("file_size"),
                            file_type=item.get("file_type", "hdf5"),
                            source="bl_archive",
                        )
            except Exception as exc:
                logger.warning("API lookup for %s failed: %s", target_name, exc)

        # Simulated / fallback
        for t in _SIMULATED_TARGETS:
            if t["target"].lower() == target_lower:
                return ObservationMeta(
                    target=t["target"],
                    ra=t.get("ra"),
                    dec=t.get("dec"),
                    freq_start_mhz=t.get("freq_start_mhz"),
                    freq_end_mhz=t.get("freq_end_mhz"),
                    obs_time_utc=t.get("obs_time_utc"),
                    duration_sec=t.get("duration_sec"),
                    telescope=t.get("telescope"),
                    file_url=None,
                    file_size_bytes=None,
                    file_type=None,
                    source="simulated",
                )

        # Target not in simulated list: create generic metadata
        logger.info("Target '%s' not found; generating generic simulated metadata", target_name)
        rng = np.random.RandomState(
            int(hashlib.md5(target_lower.encode()).hexdigest()[:8], 16) % (2**31)
        )
        return ObservationMeta(
            target=target_name,
            ra=rng.uniform(0, 360),
            dec=rng.uniform(-90, 90),
            freq_start_mhz=1000.0,
            freq_end_mhz=2000.0,
            obs_time_utc=datetime.now(timezone.utc).isoformat(),
            duration_sec=300.0,
            telescope="GBT",
            file_url=None,
            file_size_bytes=None,
            file_type=None,
            source="simulated",
        )

    def _local_data_path(self, meta: ObservationMeta) -> Optional[Path]:
        """Determine the expected local path for a data file."""
        if not meta.file_url:
            return None
        filename = Path(meta.file_url).name
        if not filename:
            filename = cache_key("bl_file", meta.target) + ".h5"
        return self._data_dir / filename

    def _download_data_file(self, meta: ObservationMeta) -> Path:
        """Download a data file, using cache if already present."""
        local_path = self._local_data_path(meta)
        if local_path is None:
            raise ValueError("No file URL in observation metadata")
        if local_path.exists():
            logger.debug("Data file already cached: %s", local_path)
            return local_path
        return self._client.download_file(meta.file_url, local_path)


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_default_ingest: Optional[BreakthroughListenIngest] = None


def _get_ingest() -> BreakthroughListenIngest:
    """Lazily initialise and return the default ingest instance."""
    global _default_ingest
    if _default_ingest is None:
        _default_ingest = BreakthroughListenIngest()
    return _default_ingest


def list_available_targets() -> List[str]:
    """List available targets in the Breakthrough Listen archive."""
    return _get_ingest().list_available_targets()


def get_observation(target_name: str) -> Observation:
    """Return observation data for a target."""
    return _get_ingest().get_observation(target_name)


def get_spectrogram(
    target_name: str,
    n_freq: int = 512,
    n_time: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return spectrogram arrays (frequency x time) for a target.

    Returns (spectrogram, frequencies_mhz, timestamps_sec, source)
    where source is ``"bl_archive"`` or ``"simulated"``.
    """
    return _get_ingest().get_spectrogram(target_name, n_freq=n_freq, n_time=n_time)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import textwrap

    print("=" * 72)
    print("  Project EXODUS — Breakthrough Listen Data Ingestion")
    print("=" * 72)
    print()

    ingest = BreakthroughListenIngest()

    # 1. List targets
    print("[1] Available targets:")
    print("-" * 40)
    targets = ingest.list_available_targets()
    for i, t in enumerate(targets, 1):
        print(f"  {i:>3}. {t}")
    print(f"\n  Total: {len(targets)} targets")
    print()

    # 2. Show metadata for a selection of targets
    print("[2] Observation metadata (first 5 targets):")
    print("-" * 40)
    for t in targets[:5]:
        obs = ingest.get_observation(t)
        meta = obs.meta
        print(f"  Target:      {meta.target}")
        print(f"  RA / Dec:    {meta.ra}, {meta.dec}")
        print(f"  Freq range:  {meta.freq_start_mhz} - {meta.freq_end_mhz} MHz")
        print(f"  Telescope:   {meta.telescope}")
        print(f"  Obs time:    {meta.obs_time_utc}")
        print(f"  Duration:    {meta.duration_sec} s")
        print(f"  Source:      {meta.source}")
        print()

    # 3. Generate a spectrogram for one target
    demo_target = targets[0]
    print(f"[3] Spectrogram for '{demo_target}':")
    print("-" * 40)
    spec, freqs, times = ingest.get_spectrogram(demo_target)
    print(f"  Shape:         {spec.shape}  (freq x time)")
    print(f"  Freq range:    {freqs[0]:.2f} - {freqs[-1]:.2f} MHz")
    print(f"  Time range:    {times[0]:.2f} - {times[-1]:.2f} s")
    print(f"  Value range:   {spec.min():.4f} - {spec.max():.4f}")
    print(f"  Mean power:    {spec.mean():.4f}")
    print(f"  Std dev:       {spec.std():.4f}")
    print()

    print("Done.")
