"""Characterization analysis routines extracted from spectrometer_charactarization.py.

The goal is to mirror the analysis flow of the standalone script while providing
callable utilities that return matplotlib Figure objects for embedding in the GUI.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


@dataclass
class CharacterizationConfig:
    """Configuration constants that mirror the standalone script."""

    laser_sequence: Sequence[str] = ("532", "445", "405", "377", "Hg_Ar")
    laser_reference_map: Dict[str, float] = field(
        default_factory=lambda: {
            "377": 375.0,
            "405": 403.46,
            "445": 445.0,
            "517": 517.0,
            "532": 532.0,
            "640": 640.0,
        }
    )
    known_lines_nm: Sequence[float] = (
        289.36,
        296.73,
        302.15,
        313.16,
        334.19,
        365.01,
        404.66,
        407.78,
        435.84,
        507.30,
        546.08,
    )
    ib_region_size: int = 20
    sat_threshold: float = 65400.0
    win_hg: int = 30
    # Integration cycles taken from script
    n_sig_640: int = 10
    n_dark_640: int = 10


@dataclass
class AnalysisArtifact:
    name: str
    figure: Figure
    path: str


@dataclass
class CharacterizationResult:
    artifacts: List[AnalysisArtifact]
    summary_lines: List[str]

    @property
    def summary_text(self) -> str:
        return "\n".join(self.summary_lines)


# ---------------------------------------------------------------------------
# Data helpers – largely lifted from the script but rewritten into functions
# ---------------------------------------------------------------------------

def _pixel_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if str(c).startswith("Pixel_")]


def get_normalized_lsf(
    df: pd.DataFrame,
    wavelength: str,
    sat_thresh: float,
    use_latest: bool = True,
) -> Optional[np.ndarray]:
    pixel_cols = _pixel_columns(df)
    if not pixel_cols:
        return None

    sig_rows = df[df["Wavelength"] == wavelength]
    dark_rows = df[df["Wavelength"] == f"{wavelength}_dark"]
    if sig_rows.empty or dark_rows.empty:
        return None

    sig_row = sig_rows.iloc[-1] if use_latest else sig_rows.iloc[0]
    dark_row = dark_rows.iloc[-1] if use_latest else dark_rows.iloc[0]

    try:
        sig = sig_row[pixel_cols].astype(float).to_numpy()
        dark = dark_row[pixel_cols].astype(float).to_numpy()
    except Exception:
        return None

    if sig.shape != dark.shape or sig.size == 0:
        return None
    if not np.all(np.isfinite(sig)) or not np.all(np.isfinite(dark)):
        return None
    if np.any(sig >= sat_thresh):
        return None

    corrected = sig - dark
    corrected -= np.nanmin(corrected)
    denom = float(np.nanmax(corrected))
    if not np.isfinite(denom) or denom <= 0:
        return None

    normed = corrected / denom
    if not np.all(np.isfinite(normed)):
        return None
    return normed


def get_corrected_signal(df: pd.DataFrame, base: str) -> Optional[np.ndarray]:
    pixel_cols = _pixel_columns(df)
    if not pixel_cols:
        return None
    sig_rows = df[df["Wavelength"] == base]
    dark_rows = df[df["Wavelength"] == f"{base}_dark"]
    if sig_rows.empty or dark_rows.empty:
        return None
    sig = sig_rows.iloc[-1][pixel_cols].astype(float).to_numpy()
    dark = dark_rows.iloc[-1][pixel_cols].astype(float).to_numpy()
    corrected = sig - dark
    corrected = np.clip(corrected, 1e-5, None)
    if not np.all(np.isfinite(corrected)):
        return None
    return corrected


def best_ordered_linear_match(
    peaks_pix: Sequence[int],
    candidate_wls: Sequence[float],
    min_points: int = 5,
) -> Optional[Tuple[float, float, float, np.ndarray, np.ndarray]]:
    peaks_pix = np.asarray(peaks_pix, dtype=float)
    candidate_wls = np.asarray(candidate_wls, dtype=float)
    P, L = len(peaks_pix), len(candidate_wls)
    best: Optional[Tuple[float, float, float, np.ndarray, np.ndarray]] = None

    def score(pix_sel: np.ndarray, wl_sel: np.ndarray) -> Tuple[float, float, float]:
        A = np.vstack([pix_sel, np.ones_like(pix_sel)]).T
        a, b = np.linalg.lstsq(A, wl_sel, rcond=None)[0]
        pred = a * pix_sel + b
        rmse = float(np.sqrt(np.mean((wl_sel - pred) ** 2)))
        return rmse, float(a), float(b)

    if P >= L:
        for i in range(P - L + 1):
            pix_sel = peaks_pix[i : i + L]
            wl_sel = candidate_wls.copy()
            rmse, a, b = score(pix_sel, wl_sel)
            if best is None or rmse < best[0]:
                best = (rmse, a, b, pix_sel.copy(), wl_sel.copy())
    else:
        for j in range(L - P + 1):
            pix_sel = peaks_pix.copy()
            wl_sel = candidate_wls[j : j + P]
            rmse, a, b = score(pix_sel, wl_sel)
            if best is None or rmse < best[0]:
                best = (rmse, a, b, pix_sel.copy(), wl_sel.copy())

    if best and len(best[3]) >= min_points:
        return best
    return None


def normalize_lsf_stray_light(lsf: np.ndarray, pixel_number: int, ib_size: int) -> np.ndarray:
    ib_start = max(0, pixel_number - ib_size // 2)
    ib_end = min(len(lsf), pixel_number + ib_size // 2 + 1)
    ib_region = np.arange(ib_start, ib_end)
    ib_sum = float(np.sum(lsf[ib_region]))
    lsf = lsf.copy()
    lsf[ib_region] = 0.0
    if not np.isfinite(ib_sum) or ib_sum <= 0:
        return np.zeros_like(lsf)
    return lsf / ib_sum


def slit_func(x: np.ndarray, A2: float, A3: float, C1: float) -> np.ndarray:
    return np.exp(-np.abs(x / A2) ** A3) + C1


def compute_fwhm(x: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size == 0:
        return 0.0
    y = y - np.min(y)
    if np.max(y) <= 0:
        return 0.0
    y = y / np.max(y)
    half = 0.5
    above = np.where(y >= half)[0]
    if len(above) < 2:
        return 0.0
    left, right = above[0], above[-1]

    def interp(idx1: int, idx2: int) -> float:
        if idx1 < 0 or idx2 >= len(x):
            return float(x[min(max(idx1, 0), len(x) - 1)])
        x1, x2 = x[idx1], x[idx2]
        y1, y2 = y[idx1], y[idx2]
        if y2 == y1:
            return float(x1)
        return float(x1 + (x2 - x1) * (half - y1) / (y2 - y1))

    x_left = interp(left - 1, left)
    x_right = interp(right, right + 1)
    return abs(x_right - x_left)


def compute_width_at_percent_max(x: np.ndarray, y: np.ndarray, percent: float = 0.2) -> float:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size == 0:
        return 0.0
    y = y - np.min(y)
    if np.max(y) <= 0:
        return 0.0
    y = y / np.max(y)
    thresh = percent
    above = np.where(y >= thresh)[0]
    if len(above) < 2:
        return 0.0
    left, right = above[0], above[-1]

    def interp(idx1: int, idx2: int) -> float:
        if idx1 < 0 or idx2 >= len(x):
            return float(x[min(max(idx1, 0), len(x) - 1)])
        x1, x2 = x[idx1], x[idx2]
        y1, y2 = y[idx1], y[idx2]
        if y2 == y1:
            return float(x1)
        return float(x1 + (x2 - x1) * (thresh - y1) / (y2 - y1))

    x_left = interp(left - 1, left)
    x_right = interp(right, right + 1)
    return abs(x_right - x_left)


def generate_adaptive_x(A2: float, spacing: float = 0.01) -> np.ndarray:
    half_width = 3 * A2
    num_points = int(2 * half_width / spacing) + 1
    return np.linspace(-half_width, half_width, max(num_points, 3))


def _safe_polyfit(x: np.ndarray, y: np.ndarray, deg: int) -> np.ndarray:
    if len(x) < deg + 1:
        deg = len(x) - 1
    if deg < 0:
        return np.array([0.0])
    return np.polyfit(x, y, deg=deg)


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------

def perform_characterization(
    df: pd.DataFrame,
    sn: str,
    folder: str,
    timestamp: Optional[str] = None,
    config: Optional[CharacterizationConfig] = None,
) -> CharacterizationResult:
    if config is None:
        config = CharacterizationConfig()
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    artifacts: List[AnalysisArtifact] = []
    summary_lines: List[str] = []

    df = df.copy()
    df["Wavelength"] = df["Wavelength"].astype(str)
    pixel_cols = _pixel_columns(df)
    if not pixel_cols:
        summary_lines.append("No pixel columns detected in dataframe.")
        return CharacterizationResult(artifacts, summary_lines)
    npix = len(pixel_cols)

    # --- Laser LSFs ------------------------------------------------------
    lsf_list: List[np.ndarray] = []
    pixel_locations: List[int] = []
    laser_wavelengths: List[float] = []

    for tag in config.laser_sequence:
        lsf = get_normalized_lsf(df, tag, config.sat_threshold)
        if lsf is None:
            summary_lines.append(f"⚠️ Missing/invalid LSF for {tag} nm; skipped.")
            continue
        lsf_list.append(lsf)
        pixel_locations.append(int(np.nanargmax(lsf)))
        laser_wavelengths.append(config.laser_reference_map.get(tag, float(tag)))

    if not lsf_list:
        summary_lines.append("No valid LSFs were computed; skipping plots.")
        return CharacterizationResult(artifacts, summary_lines)

    lsfs = np.array(lsf_list, dtype=object)
    pixel_locations_arr = np.array(pixel_locations, dtype=int)
    laser_wavelengths_arr = np.array(laser_wavelengths, dtype=float)

    fig_norm = Figure(figsize=(12, 6))
    ax_norm = fig_norm.add_subplot(111)
    ax_norm.set_yscale("log")
    ax_norm.set_xticks(np.arange(0, npix, 100))
    ax_norm.set_ylim(1e-5, 1.4)
    for lsf, wl in zip(lsfs, laser_wavelengths_arr):
        ax_norm.plot(lsf, label=f"{wl:.1f} nm")
    ax_norm.set_title(f"Spectrometer= {sn}: Normalized LSFs")
    ax_norm.set_xlabel("Pixel Index")
    ax_norm.set_ylabel("Normalized Intensity")
    ax_norm.grid(True)
    ax_norm.legend()
    path1 = os.path.join(folder, f"Normalized_Laser_Plot_{sn}_{timestamp}.png")
    fig_norm.savefig(path1, dpi=300, bbox_inches="tight")
    artifacts.append(AnalysisArtifact("Normalized LSFs", fig_norm, path1))

    # --- 640 nm Dark-corrected spectra ----------------------------------
    fig_640 = Figure(figsize=(12, 6))
    ax_640 = fig_640.add_subplot(111)
    ax_640.set_xticks(np.arange(0, npix, 100))
    sig_entries = df[df["Wavelength"].str.startswith("640") & ~df["Wavelength"].str.contains("dark")]
    if sig_entries.empty:
        summary_lines.append("⚠️ No 640 nm signal entries found.")
    else:
        for _, row in sig_entries.iterrows():
            tag = row["Wavelength"]
            dark_tag = f"{tag}_dark"
            dark_rows = df[df["Wavelength"] == dark_tag]
            if dark_rows.empty:
                continue
            signal = row[pixel_cols].astype(float).to_numpy()
            dark = dark_rows.iloc[0][pixel_cols].astype(float).to_numpy()
            corrected = np.clip(signal - dark, 1e-5, None)
            it_ms = float(row["IntegrationTime"]) if "IntegrationTime" in row else float(row.iloc[2])
            ax_640.plot(corrected, label=f"{tag} @ {it_ms:.1f} ms")
        ax_640.set_title(f"Spectrometer= {sn}: Dark-Corrected 640 nm Measurements")
        ax_640.set_xlabel("Pixel Index")
        ax_640.set_ylabel("Corrected Intensity")
        ax_640.grid(True)
        ax_640.legend()
    path2 = os.path.join(folder, f"OOR_640nm_Plot_{sn}_{timestamp}.png")
    fig_640.savefig(path2, dpi=300, bbox_inches="tight")
    artifacts.append(AnalysisArtifact("640 nm Dark-Corrected", fig_640, path2))

    # --- Hg-Ar peak detection -------------------------------------------
    signal_corr = get_corrected_signal(df, "Hg_Ar")
    fig_hg = Figure(figsize=(14, 6))
    ax_hg = fig_hg.add_subplot(111)
    if signal_corr is None:
        summary_lines.append("⚠️ Unable to compute Hg-Ar corrected signal.")
        peaks = np.array([], dtype=int)
        props = {}
        matched_pixels = np.array([], dtype=float)
        matched_wavelengths = np.array([], dtype=float)
        rmse = math.nan
        a_lin = b_lin = math.nan
    else:
        peaks, props = find_peaks(
            signal_corr,
            prominence=0.014 * np.max(signal_corr) if np.max(signal_corr) > 0 else 0,
            distance=20,
        )
        peaks = np.sort(peaks)
        candidates = [config.known_lines_nm, config.known_lines_nm[:-1]]
        solutions = [best_ordered_linear_match(peaks, cand) for cand in candidates]
        solutions = [s for s in solutions if s is not None]
        if not solutions:
            summary_lines.append("❌ No valid match between Hg-Ar peaks and known lines.")
            matched_pixels = np.array([], dtype=float)
            matched_wavelengths = np.array([], dtype=float)
            rmse = math.nan
            a_lin = b_lin = math.nan
        else:
            solutions.sort(key=lambda t: t[0])
            rmse, a_lin, b_lin, matched_pixels, matched_wavelengths = solutions[0]
            matched_pixels = np.array(matched_pixels)
            matched_wavelengths = np.array(matched_wavelengths)
            summary_lines.append(f"✅ Matched {len(matched_pixels)} Hg-Ar lines (RMSE={rmse:.2f} nm)")
        pixels = np.arange(len(signal_corr))
        ax_hg.set_yscale("log")
        ax_hg.plot(pixels, signal_corr, label="Dark-Corrected Hg-Ar", color="blue")
        if peaks.size:
            ax_hg.plot(peaks, signal_corr[peaks], "ro", label="Detected Peaks")
        if len(matched_pixels):
            for pix, wl in zip(matched_pixels, matched_wavelengths):
                ax_hg.text(
                    pix,
                    signal_corr[int(pix)] + 2500,
                    f"{wl:.1f} nm",
                    rotation=0,
                    color="brown",
                    fontsize=10,
                    ha="center",
                    va="bottom",
                )
        ax_hg.set_xlabel("Pixel")
        ax_hg.set_ylabel("Signal (Counts)")
        ax_hg.set_title(f"Spectrometer= {sn}: Hg-Ar Lamp Spectrum with Detected Peaks")
        ax_hg.legend()
        ax_hg.grid(True)
    path3 = os.path.join(folder, f"HgAr_Peaks_Plot_{sn}_{timestamp}.png")
    fig_hg.savefig(path3, dpi=300, bbox_inches="tight")
    artifacts.append(AnalysisArtifact("Hg-Ar Peaks", fig_hg, path3))

    # --- Hg-Ar LSF extraction -------------------------------------------
    lsf_list_lamp: List[np.ndarray] = []
    pixel_loc: List[int] = []
    if signal_corr is not None and len(peaks) > 0:
        for pix in matched_pixels:
            start = max(int(pix - config.win_hg), 0)
            end = min(int(pix + config.win_hg + 1), npix)
            seg = signal_corr[start:end]
            seg = seg - seg.min()
            denom = max(1e-12, seg.max())
            seg = seg / denom
            if np.all(np.isfinite(seg)):
                lsf_list_lamp.append(seg)
                pixel_loc.append(int(pix))
    lsf_list_lamp = np.array(lsf_list_lamp, dtype=object)
    matched_pixels_arr = np.array(pixel_loc, dtype=int)
    matched_wavelengths_arr = np.array(matched_wavelengths[: len(pixel_loc)], dtype=float)

    # --- Stray light matrix and SDF plots --------------------------------
    total_pixels = npix
    SDF_matrix = np.zeros((total_pixels, total_pixels))
    for lsf, pix in zip(lsfs, pixel_locations_arr):
        norm = normalize_lsf_stray_light(np.asarray(lsf, dtype=float), int(pix), config.ib_region_size)
        SDF_matrix[:, int(pix)] = norm

    # shift operations replicating the standalone script
    for i in range(len(pixel_locations_arr) - 1, 0, -1):
        current_pixel = int(pixel_locations_arr[i])
        previous_pixel = int(pixel_locations_arr[i - 1])
        for col in range(current_pixel - 1, previous_pixel, -1):
            shift_amount = current_pixel - col
            SDF_matrix[:-shift_amount, col] = SDF_matrix[shift_amount:, current_pixel]
            SDF_matrix[-shift_amount:, col] = 0
    first_pixel = int(pixel_locations_arr[0])
    for col in range(first_pixel - 1, -1, -1):
        shift_amount = first_pixel - col
        SDF_matrix[:-shift_amount, col] = SDF_matrix[shift_amount:, first_pixel]
        SDF_matrix[-shift_amount:, col] = 0
    last_lsf_pixel = int(pixel_locations_arr[-1])
    for col in range(last_lsf_pixel + 1, total_pixels):
        shift_amount = col - last_lsf_pixel
        SDF_matrix[shift_amount:, col] = SDF_matrix[:-shift_amount, last_lsf_pixel]
        SDF_matrix[:shift_amount, col] = 0
    for i in range(len(pixel_locations_arr) - 1, -1, -1):
        current_pixel = int(pixel_locations_arr[i])
        stop_col = int(pixel_locations_arr[i - 1]) + 1 if i > 0 else 0
        last_value = SDF_matrix[-1, current_pixel]
        for col in range(current_pixel - 1, stop_col - 1, -1):
            ib_start = max(0, col - config.ib_region_size // 2)
            ib_end = min(total_pixels, col + config.ib_region_size // 2 + 1)
            for row in range(ib_end, total_pixels):
                if SDF_matrix[row, col] == 0:
                    SDF_matrix[row, col] = last_value
    first_value = SDF_matrix[0, last_lsf_pixel]
    for col in range(last_lsf_pixel + 1, total_pixels):
        ib_start = max(0, col - config.ib_region_size // 2)
        for row in range(0, ib_start):
            if SDF_matrix[row, col] == 0:
                SDF_matrix[row, col] = first_value

    fig_sdf = Figure(figsize=(12, 6))
    ax_sdf = fig_sdf.add_subplot(111)
    ax_sdf.set_xlim(0, total_pixels)
    ax_sdf.set_xticks(np.arange(0, total_pixels, 100))
    for col in pixel_locations_arr:
        ax_sdf.plot(SDF_matrix[:, int(col)], label=f"{int(col)} pixel")
    ax_sdf.set_xlabel("Pixels")
    ax_sdf.set_ylabel("SDF Value")
    ax_sdf.set_title(f"Spectrometer= {sn}: Spectral Distribution Function (SDF)")
    ax_sdf.grid(True)
    ax_sdf.legend()
    path4 = os.path.join(folder, f"SDF_Plot_{sn}_{timestamp}.png")
    fig_sdf.savefig(path4, dpi=300, bbox_inches="tight")
    artifacts.append(AnalysisArtifact("SDF", fig_sdf, path4))

    fig_sdf_heat = Figure(figsize=(10, 6))
    ax_heat = fig_sdf_heat.add_subplot(111)
    im = ax_heat.imshow(SDF_matrix, aspect="auto", cmap="coolwarm", origin="lower")
    fig_sdf_heat.colorbar(im, ax=ax_heat, label="SDF Value")
    ax_heat.set_xlabel("Pixels")
    ax_heat.set_ylabel("Spectral Pixel Index")
    ax_heat.set_title(f"Spectrometer= {sn}: SDF Matrix Heatmap")
    path5 = os.path.join(folder, f"SDF_Heatmap_{sn}_{timestamp}.png")
    fig_sdf_heat.savefig(path5, dpi=300, bbox_inches="tight")
    artifacts.append(AnalysisArtifact("SDF Heatmap", fig_sdf_heat, path5))

    # --- A2/A3 polynomial fit from slit function parameters --------------
    def fit_slit_parameters(lsf: np.ndarray, peak_pixel: int, dispersion_nm_per_pixel: float) -> Optional[Tuple[float, float, float]]:
        center = len(lsf) // 2
        x = (np.arange(len(lsf)) - center) * dispersion_nm_per_pixel
        y = (lsf - np.min(lsf))
        if np.max(y) <= 0:
            return None
        y /= np.max(y)
        try:
            popt, _ = curve_fit(slit_func, x, y, p0=(0.5, 2.0, 0.0), maxfev=2000)
            return tuple(map(float, popt))
        except Exception:
            return None

    # dispersion derivative approximated by polynomial fit using Hg-Ar match + lasers
    comb_peak_pixels = np.concatenate((pixel_locations_arr, matched_pixels_arr)) if len(matched_pixels_arr) else pixel_locations_arr
    comb_wavelengths = np.concatenate((laser_wavelengths_arr, matched_wavelengths_arr)) if len(matched_wavelengths_arr) else laser_wavelengths_arr
    order = np.argsort(comb_peak_pixels)
    comb_peak_pixels_sorted = comb_peak_pixels[order]
    comb_wavelengths_sorted = comb_wavelengths[order]
    degree = 2 if len(comb_peak_pixels_sorted) >= 3 else 1
    disp_coeffs = _safe_polyfit(comb_peak_pixels_sorted, comb_wavelengths_sorted, deg=degree)
    terms = []
    for i, coeff in enumerate(disp_coeffs):
        power = degree - i
        if power == 0:
            terms.append(f"{coeff:.6e}")
        elif power == 1:
            terms.append(f"{coeff:.6e}·p")
        else:
            terms.append(f"{coeff:.6e}·p^{power}")
    summary_lines.append("Dispersion Polynomial: λ(p) = " + " + ".join(terms))
    dispersion_poly = np.poly1d(disp_coeffs)
    dispersion_deriv = dispersion_poly.deriv()

    A2_list: List[Tuple[float, float]] = []
    A3_list: List[Tuple[float, float]] = []
    C1_list: List[Tuple[float, float]] = []

    all_lsfs = []
    all_peak_pixels = []
    all_wavelengths = []
    win = 25
    for lsf, peak_pix, wl in zip(lsfs, pixel_locations_arr, laser_wavelengths_arr):
        start = max(0, int(peak_pix) - win)
        end = min(len(lsf), int(peak_pix) + win + 1)
        cropped = np.array(lsf[start:end], dtype=float)
        if len(cropped) < (2 * win + 1):
            pad_left = max(0, win - int(peak_pix))
            pad_right = max(0, win - (len(lsf) - int(peak_pix) - 1))
            cropped = np.pad(cropped, (pad_left, pad_right), mode="constant")
        all_lsfs.append(cropped)
        all_peak_pixels.append(int(peak_pix))
        all_wavelengths.append(float(wl))
    for lsf, pix, wl in zip(lsf_list_lamp, matched_pixels_arr, matched_wavelengths_arr):
        center = len(lsf) // 2
        start = max(0, center - win)
        end = min(len(lsf), center + win + 1)
        cropped = np.array(lsf[start:end], dtype=float)
        if len(cropped) < (2 * win + 1):
            pad_left = max(0, win - center)
            pad_right = max(0, win - (len(lsf) - center - 1))
            cropped = np.pad(cropped, (pad_left, pad_right), mode="constant")
        all_lsfs.append(cropped)
        all_peak_pixels.append(int(pix))
        all_wavelengths.append(float(wl))

    all_lsfs = np.array(all_lsfs, dtype=object)
    all_peak_pixels = np.array(all_peak_pixels, dtype=int)
    all_wavelengths = np.array(all_wavelengths, dtype=float)
    order = np.argsort(all_peak_pixels)
    all_lsfs = all_lsfs[order]
    all_peak_pixels = all_peak_pixels[order]
    all_wavelengths = all_wavelengths[order]

    for lsf, peak_pix, wl in zip(all_lsfs, all_peak_pixels, all_wavelengths):
        disp_nm_per_pixel = float(dispersion_deriv(peak_pix)) if dispersion_deriv.order >= 0 else 0.0
        params = fit_slit_parameters(np.asarray(lsf, dtype=float), peak_pix, disp_nm_per_pixel if disp_nm_per_pixel else 1.0)
        if params is None:
            continue
        A2, A3, C1 = params
        wl_um = wl / 1000.0
        A2_list.append((wl_um, A2))
        A3_list.append((wl_um, A3))
        C1_list.append((wl_um, C1))

    fig_A2A3 = Figure(figsize=(10, 6))
    ax_A = fig_A2A3.add_subplot(111)
    if A2_list:
        wl_um, vals = zip(*A2_list)
        ax_A.plot(np.array(wl_um) * 1000, vals, "o", label="A2")
    if A3_list:
        wl_um3, vals3 = zip(*A3_list)
        ax_A.plot(np.array(wl_um3) * 1000, vals3, "s", label="A3")
    ax_A.set_xlabel("Wavelength (nm)")
    ax_A.set_ylabel("Parameter Value")
    ax_A.set_title(f"Spectrometer= {sn}: Slit Parameters A2/A3")
    ax_A.grid(True)
    ax_A.legend()
    path6 = os.path.join(folder, f"Slit_Params_{sn}_{timestamp}.png")
    fig_A2A3.savefig(path6, dpi=300, bbox_inches="tight")
    artifacts.append(AnalysisArtifact("Slit Parameters", fig_A2A3, path6))

    # --- Spectral resolution --------------------------------------------
    wv_range_nm = np.linspace(min(all_wavelengths, default=300), max(all_wavelengths, default=800), 200)
    A2_poly = _safe_polyfit(np.array([wl for wl, _ in A2_list]), np.array([v for _, v in A2_list]), deg=2) if A2_list else np.array([0.0])
    A3_poly = _safe_polyfit(np.array([wl for wl, _ in A3_list]), np.array([v for _, v in A3_list]), deg=2) if A3_list else np.array([0.0])
    A2_vals = np.polyval(A2_poly, wv_range_nm / 1000.0)
    A3_vals = np.polyval(A3_poly, wv_range_nm / 1000.0)
    fwhm_vals = 2 * A2_vals * (np.log(2)) ** (1 / np.maximum(A3_vals, 1e-6))
    fig_resolution = Figure(figsize=(10, 6))
    ax_res = fig_resolution.add_subplot(111)
    ax_res.plot(wv_range_nm, fwhm_vals, label=f"Spectrometer = {sn}")
    ax_res.set_xlabel("Wavelength (nm)")
    ax_res.set_ylabel("FWHM (nm)")
    ax_res.set_title(f"Spectrometer= {sn}: Spectral Resolution vs Wavelength")
    ax_res.grid(True)
    ax_res.legend()
    path7 = os.path.join(folder, f"Spectral_Resolution_with_wavelength_{sn}_{timestamp}.png")
    fig_resolution.savefig(path7, dpi=300, bbox_inches="tight")
    artifacts.append(AnalysisArtifact("Spectral Resolution", fig_resolution, path7))

    # --- Slit function examples -----------------------------------------
    fig_slit = Figure(figsize=(10, 6))
    ax_slit = fig_slit.add_subplot(111)
    for center_nm in (350, 400, 480):
        lam_um = center_nm / 1000.0
        A2 = np.clip(np.polyval(A2_poly, lam_um), 0.2, 5.0)
        A3 = np.polyval(A3_poly, lam_um)
        C1 = C1_list[0][1] if C1_list else 0.0
        x_vals = generate_adaptive_x(A2)
        S = slit_func(x_vals, A2, A3, C1)
        fwhm = compute_fwhm(x_vals, S)
        ax_slit.plot(x_vals, S, label=f"λ₀ = {center_nm} nm, FWHM = {fwhm:.3f} nm")
    ax_slit.set_title(f"Spectrometer= {sn}: Slit Function with FWHM")
    ax_slit.set_xlabel("Wavelength Offset from Center (nm)")
    ax_slit.set_ylabel("Normalized Intensity")
    ax_slit.grid(True)
    ax_slit.legend()
    path8 = os.path.join(folder, f"Slit_Functions_{sn}_{timestamp}.png")
    fig_slit.savefig(path8, dpi=300, bbox_inches="tight")
    artifacts.append(AnalysisArtifact("Slit Functions", fig_slit, path8))

    # --- Overlay of normalized LSFs -------------------------------------
    fig_overlay = Figure(figsize=(9, 10))
    ax1 = fig_overlay.add_subplot(211)
    ax2 = fig_overlay.add_subplot(212, sharex=ax1)
    for lsf, peak_pixel, λ0 in zip(lsfs, pixel_locations_arr, laser_wavelengths_arr):
        disp_nm_per_pixel = float(dispersion_deriv(peak_pixel)) if dispersion_deriv.order >= 0 else 0.0
        center = int(np.nanargmax(lsf))
        x = (np.arange(len(lsf)) - center) * (disp_nm_per_pixel if disp_nm_per_pixel else 1.0)
        lsf_norm = (lsf - np.min(lsf)) / max(1e-12, np.max(lsf) - np.min(lsf))
        fwhm = compute_fwhm(x, lsf_norm)
        fw20 = compute_width_at_percent_max(x, lsf_norm, percent=0.2)
        ax1.plot(x, lsf_norm, label=f"{λ0:.0f} nm, FWHM={fwhm:.2f} nm, FW_20%={fw20:.2f} nm")
    ax1.set_yscale("log")
    ax1.set_title(f"Spectrometer = {sn}: Normalized LSFs of Lasers")
    ax1.set_ylabel("Normalized Intensity")
    ax1.set_xlim(-7, 7)
    ax1.set_ylim(1e-4, 1.5)
    ax1.grid(True)
    ax1.legend(fontsize=9)
    for lsf, pix, wl in zip(lsf_list_lamp, matched_pixels_arr, matched_wavelengths_arr):
        disp_nm_per_pixel = float(dispersion_deriv(pix)) if dispersion_deriv.order >= 0 else 0.0
        center = len(lsf) // 2
        x = (np.arange(len(lsf)) - center) * (disp_nm_per_pixel if disp_nm_per_pixel else 1.0)
        lsf_norm = (lsf - np.min(lsf)) / max(1e-12, np.max(lsf) - np.min(lsf))
        fwhm = compute_fwhm(x, lsf_norm)
        fw20 = compute_width_at_percent_max(x, lsf_norm, percent=0.2)
        ax2.plot(x, lsf_norm, label=f"{wl:.1f} nm, FWHM={fwhm:.2f} nm, FW_20%={fw20:.2f} nm")
    ax2.set_yscale("log")
    ax2.set_title(f"Spectrometer = {sn}: Normalized LSFs of Hg-Ar Lamp")
    ax2.set_xlabel("Wavelength Offset from Peak (nm)")
    ax2.set_ylabel("Normalized Intensity")
    ax2.set_xlim(-7, 7)
    ax2.set_ylim(1e-4, 1.5)
    ax2.grid(True)
    ax2.legend(fontsize=9)
    fig_overlay.tight_layout()
    path9 = os.path.join(folder, f"Overlapped_LSF_Lasers_HgAr_{sn}_{timestamp}.png")
    fig_overlay.savefig(path9, dpi=300, bbox_inches="tight")
    artifacts.append(AnalysisArtifact("Overlay LSFs", fig_overlay, path9))

    return CharacterizationResult(artifacts, summary_lines)
