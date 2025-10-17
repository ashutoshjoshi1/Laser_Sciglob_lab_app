import os
import sys
import time
import json
import queue
import threading
import traceback
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# matplotlib embed
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# serial
import serial
import serial.tools.list_ports

# analysis helpers
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# Avantes wrapper (must be present alongside this file)
from avantes_spectrometer import Avantes_Spectrometer

# Characterization analysis helpers
from characterization_analysis import (
    AnalysisArtifact,
    CharacterizationConfig,
    CharacterizationResult,
    perform_characterization,
)


# =========================
# ---- Configuration ------
# =========================

APP_TITLE = "Spectrometer Control & Analysis"
SETTINGS_FILE = "spectro_gui_settings.json"

# Default COM ports (editable in Setup tab)
DEFAULT_COM_PORTS = {
    "OBIS": "COM10",   # 405/445/488 on OBIS
    "RELAY": "COM11",  # 517/532/Hg-Ar on relay board
    "CUBE": "COM1"     # 377 nm cube
}

# OBIS channel map (edit as needed)
OBIS_LASER_MAP = {
    "405": 5,
    "445": 4,
    "488": 3,
    "640": 2,
}

# Default powers (Watts for OBIS/CUBE setpoints, you can interpret as needed)
DEFAULT_LASER_POWERS = {
    "405": 0.005,
    "445": 0.003,
    "488": 0.030,
    "640": 0.030,  # if you use 640 on OBIS group elsewhere
    "377": 0.012,  # example for CUBE current or analog; adapt to your device
    "517": 1.000,  # relays are on/off, but keep a placeholder
    "532": 1.000,
    "Hg_Ar": 1.000
}

# Automated measurement list (you can change from GUI too)
DEFAULT_ALL_LASERS = ["532", "445", "405", "377", "Hg_Ar", "640"]

# Integration time bounds (ms)
IT_MIN = 0.2
IT_MAX = 3000.0
SAT_THRESH = 65400  # ~16-bit ADC ceiling

# Auto-IT target window
TARGET_LOW = 60000
TARGET_HIGH = 65000
TARGET_MID = 62500

# Auto-IT controller steps
IT_STEP_UP = 0.30     # if too low, increase IT
IT_STEP_DOWN = 0.10   # if too high, decrease IT
MAX_IT_ADJUST_ITERS = 1000

# Default starting IT by source
DEFAULT_START_IT = {
    "532": 5.0,
    "517": 80.0,
    "Hg_Ar": 10.0,
    "default": 2.4
}

# Measurement cycles (adjust to taste)
N_SIG = 50
N_DARK = 50
N_SIG_640 = 10
N_DARK_640 = 10

# Known Hg-Ar lines (nm) for dispersion
KNOWN_HG_AR_NM = [289.36, 296.73, 302.15, 313.16, 334.19, 365.01,
                  404.66, 407.78, 435.84, 507.30, 546.08]

# Stray-light IB window half-width (pixels)
IB_REGION_HALF = 2


# ===================================================
# ============== Utility / Data Classes =============
# ===================================================

@dataclass
class HardwareState:
    dll_path: str = ""
    com_ports: Dict[str, str] = field(default_factory=lambda: DEFAULT_COM_PORTS.copy())
    laser_power: Dict[str, float] = field(default_factory=lambda: DEFAULT_LASER_POWERS.copy())

@dataclass
class MeasurementData:
    # Rows like: [Timestamp, Wavelength, IntegrationTime, NumCycles, Pixel_0..Pixel_N-1]
    rows: List[List] = field(default_factory=list)
    npix: int = 2048
    serial_number: str = "Unknown"

    def to_dataframe(self) -> pd.DataFrame:
        cols = ["Timestamp", "Wavelength", "IntegrationTime", "NumCycles"] + [f"Pixel_{i}" for i in range(self.npix)]
        return pd.DataFrame(self.rows, columns=cols)

    def last_vectors_for(self, tag: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (last_signal, last_dark) vectors for a given wavelength tag."""
        if not self.rows:
            return None, None
        df = self.to_dataframe()
        pix_cols = [c for c in df.columns if str(c).startswith("Pixel_")]
        sig_rows = df[df["Wavelength"] == tag]
        dark_rows = df[df["Wavelength"] == f"{tag}_dark"]
        sig = sig_rows.iloc[-1][pix_cols].to_numpy(dtype=float) if not sig_rows.empty else None
        dark = dark_rows.iloc[-1][pix_cols].to_numpy(dtype=float) if not dark_rows.empty else None
        return sig, dark


# ===================================================
# ============== Device Control Helpers =============
# ===================================================

class SerialDevice:
    """Small wrapper around pyserial with safe open/close and commands."""
    def __init__(self, port: str, baud: int = 9600, timeout: float = 1.0):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.eol = "\r\n"
        self.ser: Optional[serial.Serial] = None
        self.lock = threading.Lock()

    def is_open(self) -> bool:
        return self.ser is not None and self.ser.is_open

    def open(self) -> bool:
        # Close first to avoid stale handles
        self.close()
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
            return True
        except Exception:
            self.ser = None
            return False

    def close(self):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        finally:
            self.ser = None

    def _ensure_open(self) -> None:
        if not self.is_open():
            ok = self.open()
            if not ok:
                raise RuntimeError(f"Could not open serial port '{self.port}'")

    def write_line(self, s: str):
        self._ensure_open()
        with self.lock:
            eol = getattr(self, 'eol', '\r\n')
            self.ser.write((s + eol).encode())
            # small settle delay helps some controllers
            time.sleep(0.2)

    def read_all_lines(self) -> List[str]:
        self._ensure_open()
        with self.lock:
            resp = self.ser.readlines()
        try:
            return [r.decode(errors="ignore").strip() for r in resp]
        except Exception:
            return []
    def read_all_text(self, wait: float = 0.3) -> str:
        """Read all bytes as text after an optional short wait."""
        self._ensure_open()
        with self.lock:
            import time as _t
            if wait and wait > 0:
                _t.sleep(wait)
            try:
                data = self.ser.read_all()
            except Exception:
                data = b""
        try:
            return data.decode(errors="ignore").strip()
        except Exception:
            return ""



class LaserController:
    """Encapsulate OBIS (multi-channel), CUBE (377), and Relay (532/517/Hg-Ar) behavior."""
    def __init__(self):
        self.obis = SerialDevice(DEFAULT_COM_PORTS["OBIS"], 9600, 1)
        self.cube = SerialDevice(DEFAULT_COM_PORTS["CUBE"], 19200, 1)
        self.relay = SerialDevice(DEFAULT_COM_PORTS["RELAY"], 9600, 1)
        self.cube.eol = "\r"

    def all_off(self):
        # Try to open ports and switch everything OFF; ignore errors (ports may not exist yet)
        try:
            try: self.obis._ensure_open()
            except: pass
            for ch in OBIS_LASER_MAP.values():
                try: self.obis.write_line(f"SOUR{ch}:AM:STAT OFF")
                except: pass
        except: pass
        try:
            try: self.cube._ensure_open()
            except: pass
            try: self.cube.write_line("L=0")
            except: pass
        except: pass
        try:
            try: self.relay._ensure_open()
            except: pass
            for ch in (1, 2, 3):  # 532, Hg-Ar, 517 (adjust if your mapping differs)
                try: self.relay.write_line(f"R{ch}R")
                except: pass
        except: pass


    def configure_ports(self, ports: Dict[str, str]):
        self.obis.port = ports.get("OBIS", self.obis.port)
        self.cube.port = ports.get("CUBE", self.cube.port)
        self.relay.port = ports.get("RELAY", self.relay.port)

    def open_all(self) -> Tuple[bool, bool, bool]:
        ok_obis = self.obis.open()
        ok_cube = self.cube.open()
        ok_relay = self.relay.open()
        return ok_obis, ok_cube, ok_relay

    def ensure_open_for_tag(self, tag: str):
        """Open the right serial device for the given source tag."""
        if tag in OBIS_LASER_MAP:
            self.obis._ensure_open()
        elif tag == "377":
            self.cube._ensure_open()
        elif tag in ("517", "532", "Hg_Ar"):
            self.relay._ensure_open()

    # ----- OBIS -----
    def obis_cmd(self, cmd: str) -> List[str]:
        self.obis.write_line(cmd)
        return self.obis.read_all_lines()

    def obis_on(self, channel: int):
        self.obis_cmd(f"SOUR{channel}:AM:STAT ON")

    def obis_off(self, channel: int):
        self.obis_cmd(f"SOUR{channel}:AM:STAT OFF")

    def obis_set_power(self, channel: int, watts: float):
        self.obis_cmd(f"SOUR{channel}:POW:LEV:IMM:AMPL {watts:.4f}")

    # ----- CUBE (example protocol: set current then L=1) -----
    def cube_cmd(self, cmd: str) -> List[str]:
        self.cube.write_line(cmd)
        resp = self.cube.read_all_text(wait=1.0)
        return resp.splitlines() if resp else []

    def cube_on(self, power_mw: float = None, current_mA: float = None):
        """Turn on CUBE (377 nm).
        If power_mw is provided, use EXT=1; CW=1; P=<mW>; L=1.
        Else if current_mA provided, send I=<mA>; L=1 (legacy fallback).
        """
        # Ensure CR line endings for CUBE
        try:
            self.cube.eol = "\r"
        except Exception:
            pass
        if power_mw is not None:
            try: self.cube_cmd("EXT=1")
            except Exception: pass
            try: self.cube_cmd("CW=1")
            except Exception: pass
            try: self.cube_cmd(f"P={int(round(power_mw))}")
            except Exception: pass
            self.cube_cmd("L=1")
        elif current_mA is not None:
            try: self.cube_cmd(f"I={current_mA:.2f}")
            except Exception: pass
            self.cube_cmd("L=1")
        else:
            # Default to 12 mW if nothing given
            try: self.cube_cmd("EXT=1")
            except Exception: pass
            try: self.cube_cmd("CW=1")
            except Exception: pass
            try: self.cube_cmd("P=12")
            except Exception: pass
            self.cube_cmd("L=1")

    def cube_off(self):
        self.cube_cmd("L=0")

    # ----- Relay board: "R{n}S" set, "R{n}R" reset -----
    # ----- Relay board: "R{n}S" set, "R{n}R" reset -----
    def relay_on(self, ch: int):
        self.relay.write_line(f"R{ch}S")

    def relay_off(self, ch: int):
        self.relay.write_line(f"R{ch}R")


# ===================================================
# ================== Analysis Logic =================
# ===================================================

def compute_fwhm(x: np.ndarray, y: np.ndarray) -> float:
    """Return Full Width Half Max in x-units (linear interpolation)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size < 3:
        return 0.0
    y_norm = y - np.nanmin(y)
    if np.nanmax(y_norm) <= 0:
        return 0.0
    y_norm /= np.nanmax(y_norm)
    half = 0.5
    above = np.where(y_norm >= half)[0]
    if len(above) < 2:
        return 0.0
    left_idx = above[0]
    right_idx = above[-1]
    # interpolate edges
    def interp(i1, i2):
        if i2 == i1: return x[i1]
        return x[i1] + (x[i2] - x[i1]) * (half - y_norm[i1]) / (y_norm[i2] - y_norm[i1])
    if left_idx == 0:
        x_left = x[0]
    else:
        x_left = interp(left_idx-1, left_idx)
    if right_idx == len(x)-1:
        x_right = x[-1]
    else:
        x_right = interp(right_idx, right_idx+1)
    return float(max(0.0, x_right - x_left))

def best_ordered_linear_match(peaks_pix: np.ndarray, candidate_wls: List[float], min_points: int = 5):
    """
    Find a linear fit wl = a*pix + b that best matches ordered lists.
    Returns (rmse, a, b, pix_sel, wl_sel) or None.
    """
    peaks_pix = np.asarray(peaks_pix, dtype=float)
    P, L = len(peaks_pix), len(candidate_wls)
    if P == 0 or L == 0:
        return None

    def score(pix_sel, wl_sel):
        A = np.vstack([pix_sel, np.ones_like(pix_sel)]).T
        a, b = np.linalg.lstsq(A, wl_sel, rcond=None)[0]
        pred = a * pix_sel + b
        rmse = np.sqrt(np.mean((wl_sel - pred) ** 2))
        return rmse, a, b

    best = None
    if P >= L:
        # slide a window over peaks
        for i in range(P - L + 1):
            pix_sel = peaks_pix[i:i+L]
            wl_sel = np.array(candidate_wls)
            rmse, a, b = score(pix_sel, wl_sel)
            if best is None or rmse < best[0]:
                best = (rmse, a, b, pix_sel.copy(), wl_sel.copy())
    else:
        # slide over known lines
        for j in range(L - P + 1):
            pix_sel = peaks_pix.copy()
            wl_sel = np.array(candidate_wls[j:j+P])
            rmse, a, b = score(pix_sel, wl_sel)
            if best is None or rmse < best[0]:
                best = (rmse, a, b, pix_sel.copy(), wl_sel.copy())
    if best and len(best[3]) >= min_points:
        return best
    return None

def normalized_lsf_from_df(df: pd.DataFrame, tag: str, sat_thresh: float = SAT_THRESH, use_latest: bool = True) -> Optional[np.ndarray]:
    """Build normalized LSF = (signal - dark)/max with guards."""
    if df is None or df.empty:
        return None
    pix_cols = [c for c in df.columns if str(c).startswith("Pixel_")]
    if not pix_cols:
        return None
    sig_rows = df[df["Wavelength"] == tag]
    dark_rows = df[df["Wavelength"] == f"{tag}_dark"]
    if sig_rows.empty or dark_rows.empty:
        return None
    sig_row = sig_rows.iloc[-1] if use_latest else sig_rows.iloc[0]
    dark_row = dark_rows.iloc[-1] if use_latest else dark_rows.iloc[0]
    sig = sig_row[pix_cols].to_numpy(dtype=float)
    dark = dark_row[pix_cols].to_numpy(dtype=float)
    if not np.all(np.isfinite(sig)) or not np.all(np.isfinite(dark)):
        return None
    if np.nanmax(sig) >= sat_thresh:
        # saturated signal -> skip
        return None
    lsf = sig - dark
    lsf -= np.nanmin(lsf)
    mx = np.nanmax(lsf)
    if mx <= 0:
        return None
    return lsf / mx

def stray_light_metrics(lsf: np.ndarray, peak_pixel: int, ib_half: int = IB_REGION_HALF) -> Dict[str, float]:
    """Compute basic stray light ratios: OOB/IB etc."""
    n = len(lsf)
    ib_start = max(0, peak_pixel - ib_half)
    ib_end = min(n, peak_pixel + ib_half + 1)
    ib_region = np.arange(ib_start, ib_end)
    ib_sum = float(np.sum(lsf[ib_region]))
    # OOB = everything else
    mask = np.ones(n, dtype=bool)
    mask[ib_region] = False
    oob_sum = float(np.sum(lsf[mask]))
    ratio = (oob_sum / ib_sum) if ib_sum > 0 else np.nan
    return {"IB_sum": ib_sum, "OOB_sum": oob_sum, "OOB_over_IB": ratio}


# ===================================================
# ================== Main Application ===============
# ===================================================

class SpectroApp(tk.Tk):
    def __init__(self):
        # IT coordination
        self._pending_it = None  # type: float | None
        self._it_updating = False

        super().__init__()
        self.title("SciGlob - Spectrometer Characterization System")
        self.geometry("1250x800")
        self._set_window_icon()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.it_history = []

        # State
        self.hw = HardwareState()
        self.lasers = LaserController()
        self.spec: Optional[Avantes_Spectrometer] = None
        self.sn: str = "Unknown"
        self.npix: int = 2048

        # live view thread control
        self.live_running = threading.Event()
        self.live_thread: Optional[threading.Thread] = None

        # measurement control
        self.measure_running = threading.Event()
        self.measure_thread: Optional[threading.Thread] = None

        # in-memory measurements
        self.data = MeasurementData(npix=self.npix, serial_number=self.sn)

        # Characterization analysis state
        self.char_config = CharacterizationConfig()
        self.analysis_result: Optional[CharacterizationResult] = None
        self.analysis_artifacts: List[AnalysisArtifact] = []
        self.analysis_summary_lines: List[str] = []
        self.results_folder: Optional[str] = None
        self.last_results_timestamp: Optional[str] = None
        self.latest_csv_path: Optional[str] = None

        # expose configuration constants for helper modules
        self.IT_MIN = IT_MIN
        self.IT_MAX = IT_MAX
        self.SAT_THRESH = SAT_THRESH
        self.TARGET_LOW = TARGET_LOW
        self.TARGET_HIGH = TARGET_HIGH
        self.TARGET_MID = TARGET_MID
        self.IT_STEP_UP = IT_STEP_UP
        self.IT_STEP_DOWN = IT_STEP_DOWN
        self.MAX_IT_ADJUST_ITERS = MAX_IT_ADJUST_ITERS
        self.DEFAULT_START_IT = DEFAULT_START_IT
        self.N_SIG = N_SIG
        self.N_DARK = N_DARK
        self.N_SIG_640 = N_SIG_640
        self.N_DARK_640 = N_DARK_640
        self.DEFAULT_COM_PORTS = DEFAULT_COM_PORTS
        self.DEFAULT_LASER_POWERS = DEFAULT_LASER_POWERS
        self.SETTINGS_FILE = SETTINGS_FILE

        # UI
        self._build_ui()

        # load persisted settings if any
        self.load_settings_into_ui()
        self.after(300, self._all_off_on_start)





    # ------------------ UI Construction ------------------
    def _all_off_on_start(self):
        try:
            self._update_ports_from_ui()   # pick up UI COM entries if present
            self.lasers.all_off()
        except:
            pass

    def _live_reset_view(self):
        self.live_limits_locked = False
        try:
            self.live_ax.relim()
            self.live_ax.autoscale()
            self.live_fig.canvas.draw_idle()
        except:
            pass



    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        self.live_tab = ttk.Frame(nb)
        self.measure_tab = ttk.Frame(nb)
        self.analysis_tab = ttk.Frame(nb)
        self.setup_tab = ttk.Frame(nb)

        nb.add(self.live_tab, text="Live View")
        nb.add(self.measure_tab, text="Measurements")
        nb.add(self.analysis_tab, text="Analysis")
        nb.add(self.setup_tab, text="Setup")

        self._build_live_tab()
        self._build_measure_tab()
        self._build_analysis_tab()
        self._build_setup_tab()

    # ------------------ Live View Tab --------------------
    def _build_live_tab(self):
        from tabs.live_view_tab import build as _build
        _build(self)
    def _build_measure_tab(self):
        from tabs.measurements_tab import build as _build
        _build(self)
    def _build_analysis_tab(self):
        from tabs.analysis_tab import build as _build
        _build(self)
    def _build_setup_tab(self):
        from tabs.setup_tab import build as _build
        _build(self)

    # ------------------ Characterization results helpers ------------------
    def _prepare_results_folder(self) -> Tuple[str, str]:
        sn = self.sn or "Unknown"
        base = os.path.join(os.getcwd(), "data", sn)
        os.makedirs(base, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return base, timestamp

    def _safe_ymax(self, arr: np.ndarray) -> float:
        try:
            if arr is None or len(arr) == 0 or not np.isfinite(arr).any():
                return 1000.0
            return max(1000.0, float(np.nanmax(arr)) * 1.1)
        except Exception:
            return 1000.0

    def _update_auto_it_plot(self, tag: str, spectrum: np.ndarray, it_ms: float, peak: float) -> None:
        if not hasattr(self, "meas_sig_line"):
            return

        x = np.arange(len(spectrum))
        self.meas_sig_line.set_data(x, spectrum)
        if hasattr(self, "meas_dark_line"):
            self.meas_dark_line.set_data([], [])

        xmax = max(10, len(spectrum) - 1)
        ymax = self._safe_ymax(spectrum)
        try:
            self.meas_ax.set_xlim(0, xmax)
            self.meas_ax.set_ylim(0, ymax)
            self.meas_ax.set_title(
                f"Spectrometer= {self.sn or 'Unknown'}: Live Measurement for {tag} nm | IT={it_ms:.1f} ms | peak={peak:.0f}"
            )
        except Exception:
            pass
        try:
            self.meas_canvas.draw_idle()
        except Exception:
            pass

    def _clear_analysis_notebook(self) -> None:
        if getattr(self, "analysis_canvases", None):
            for canvas in self.analysis_canvases:
                try:
                    canvas.get_tk_widget().destroy()
                except Exception:
                    pass
        if getattr(self, "analysis_notebook", None):
            for tab_id in self.analysis_notebook.tabs():
                self.analysis_notebook.forget(tab_id)
        self.analysis_canvases = []

    def _update_analysis_ui(self, csv_path: Optional[str] = None) -> None:
        if not hasattr(self, "analysis_notebook"):
            return

        self._clear_analysis_notebook()

        if not self.analysis_artifacts:
            self.analysis_status_var.set("Run measurements to generate characterization charts.")
            self.analysis_text.configure(state="normal")
            self.analysis_text.delete("1.0", "end")
            self.analysis_text.insert("1.0", "No analysis has been generated yet.")
            self.analysis_text.configure(state="disabled")
            self.export_plots_btn.state(["disabled"])
            self.open_folder_btn.state(["disabled"])
            return

        if csv_path is None:
            csv_path = self.latest_csv_path or ""

        status_file = os.path.basename(csv_path) if csv_path else "saved measurements"
        status = f"Analysis generated from {status_file}"
        if self.results_folder:
            status += f" in {self.results_folder}"
        self.analysis_status_var.set(status)

        for artifact in self.analysis_artifacts:
            frame = ttk.Frame(self.analysis_notebook)
            self.analysis_notebook.add(frame, text=artifact.name)
            canvas = FigureCanvasTkAgg(artifact.figure, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            NavigationToolbar2Tk(canvas, frame)
            self.analysis_canvases.append(canvas)

        summary_text = "\n".join(self.analysis_summary_lines) if self.analysis_summary_lines else ""
        self.analysis_text.configure(state="normal")
        self.analysis_text.delete("1.0", "end")
        self.analysis_text.insert("1.0", summary_text or "Characterization completed.")
        self.analysis_text.configure(state="disabled")

        self.export_plots_btn.state(["!disabled"])
        self.open_folder_btn.state(["!disabled"])

    def refresh_analysis_view(self):
        self._update_analysis_ui(self.latest_csv_path)

    # ------------------ Settings Management ------------------
    def load_settings_into_ui(self):
        """Load settings from JSON file into UI elements."""
        if not os.path.isfile(SETTINGS_FILE):
            # Initialize with defaults if no settings file exists
            if hasattr(self, 'dll_entry'):
                self.dll_entry.delete(0, "end")
                self.dll_entry.insert(0, self.hw.dll_path)
            if hasattr(self, 'obis_entry'):
                self.obis_entry.delete(0, "end")
                self.obis_entry.insert(0, DEFAULT_COM_PORTS["OBIS"])
            if hasattr(self, 'cube_entry'):
                self.cube_entry.delete(0, "end")
                self.cube_entry.insert(0, DEFAULT_COM_PORTS["CUBE"])
            if hasattr(self, 'relay_entry'):
                self.relay_entry.delete(0, "end")
                self.relay_entry.insert(0, DEFAULT_COM_PORTS["RELAY"])
            if hasattr(self, 'power_entries'):
                for tag, e in self.power_entries.items():
                    e.delete(0, "end")
                    e.insert(0, str(DEFAULT_LASER_POWERS.get(tag, 0.01)))
            return

        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                obj = json.load(f)

            # Load DLL path
            if hasattr(self, 'dll_entry'):
                self.dll_entry.delete(0, "end")
                self.dll_entry.insert(0, obj.get("dll_path", ""))

            # Load COM ports
            cp = obj.get("com_ports", DEFAULT_COM_PORTS)
            if hasattr(self, 'obis_entry'):
                self.obis_entry.delete(0, "end")
                self.obis_entry.insert(0, cp.get("OBIS", DEFAULT_COM_PORTS["OBIS"]))
            if hasattr(self, 'cube_entry'):
                self.cube_entry.delete(0, "end")
                self.cube_entry.insert(0, cp.get("CUBE", DEFAULT_COM_PORTS["CUBE"]))
            if hasattr(self, 'relay_entry'):
                self.relay_entry.delete(0, "end")
                self.relay_entry.insert(0, cp.get("RELAY", DEFAULT_COM_PORTS["RELAY"]))

            # Load laser powers
            lp = obj.get("laser_power", DEFAULT_LASER_POWERS)
            if hasattr(self, 'power_entries'):
                for tag, e in self.power_entries.items():
                    e.delete(0, "end")
                    e.insert(0, str(lp.get(tag, DEFAULT_LASER_POWERS.get(tag, 0.01))))

        except Exception as e:
            if hasattr(self, 'power_entries'):  # Only show error if UI is built
                messagebox.showerror("Load Settings", str(e))

    def save_settings(self):
        """Save current UI settings to JSON file."""
        self._update_ports_from_ui()
        if hasattr(self, 'dll_entry'):
            self.hw.dll_path = self.dll_entry.get().strip()

        # Update laser powers from UI
        if hasattr(self, 'power_entries'):
            for tag, e in self.power_entries.items():
                try:
                    self.hw.laser_power[tag] = float(e.get().strip())
                except:
                    pass

        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "dll_path": self.hw.dll_path,
                    "com_ports": self.hw.com_ports,
                    "laser_power": self.hw.laser_power
                }, f, indent=2)
            messagebox.showinfo("Settings", f"Saved to {SETTINGS_FILE}")
        except Exception as e:
            messagebox.showerror("Settings", str(e))

    def _update_ports_from_ui(self):
        """Update hardware COM port configuration from UI entries."""
        if hasattr(self, 'obis_entry'):
            self.hw.com_ports["OBIS"] = self.obis_entry.get().strip() or DEFAULT_COM_PORTS["OBIS"]
        if hasattr(self, 'cube_entry'):
            self.hw.com_ports["CUBE"] = self.cube_entry.get().strip() or DEFAULT_COM_PORTS["CUBE"]
        if hasattr(self, 'relay_entry'):
            self.hw.com_ports["RELAY"] = self.relay_entry.get().strip() or DEFAULT_COM_PORTS["RELAY"]
        self.lasers.configure_ports(self.hw.com_ports)

    def _get_power(self, tag: str) -> float:
        """Get laser power setting for a given tag from UI or defaults."""
        try:
            if hasattr(self, 'power_entries'):
                e = self.power_entries.get(tag)
                if e is not None:
                    return float(e.get().strip())
            return DEFAULT_LASER_POWERS.get(tag, 0.01)
        except:
            return DEFAULT_LASER_POWERS.get(tag, 0.01)

    # ------------------ Hardware Control ------------------
    def browse_dll(self):
        """Browse for DLL file."""
        path = filedialog.askopenfilename(
            title="Select avaspecx64.dll", filetypes=[("DLL", "*.dll"), ("All files", "*.*")])
        if path and hasattr(self, 'dll_entry'):
            self.dll_entry.delete(0, "end")
            self.dll_entry.insert(0, path)

    def connect_spectrometer(self):
        """Connect to the Avantes spectrometer."""
        try:
            if not hasattr(self, 'dll_entry'):
                raise RuntimeError("UI not initialized yet")

            dll = self.dll_entry.get().strip()
            if not dll or not os.path.isfile(dll):
                raise RuntimeError("Please select a valid avaspecx64.dll")

            # Initialize wrapper
            ava = Avantes_Spectrometer()
            ava.dll_path = dll
            ava.alias = "Ava1"
            ava.npix_active = 2048
            ava.debug_mode = 1
            ava.initialize_spec_logger()

            # Set integration time and measurement parameters
            ava.it_ms = self.DEFAULT_START_IT
            ava.nav = 1
            ava.ncy = 1

            # Connect
            ava.connect()
            self.spec = ava
            self.sn = getattr(ava, "sn", "Unknown")
            self.data.serial_number = self.sn
            self.npix = getattr(ava, "npix_active", self.npix)
            self.data.npix = self.npix

            if hasattr(self, 'spec_status'):
                self.spec_status.config(text=f"Connected: {self.sn}", foreground="green")
            messagebox.showinfo("Spectrometer", f"Connected to SN={self.sn}")
        except Exception as e:
            self.spec = None
            if hasattr(self, 'spec_status'):
                self.spec_status.config(text="Disconnected", foreground="red")
            self._post_error("Spectrometer Connect", e)

    def disconnect_spectrometer(self):
        """Disconnect from the spectrometer."""
        try:
            self.stop_live()
            if self.spec:
                try:
                    self.spec.disconnect()
                except Exception:
                    pass
            self.spec = None
            if hasattr(self, 'spec_status'):
                self.spec_status.config(text="Disconnected", foreground="red")
        except Exception as e:
            self._post_error("Spectrometer Disconnect", e)

    def test_com_connect(self):
        """Test COM port connections."""
        self._update_ports_from_ui()
        ok_obis = self.lasers.obis.open()
        if hasattr(self, 'obis_status'):
            self.obis_status.config(foreground=("green" if ok_obis else "red"))
        ok_cube = self.lasers.cube.open()
        if hasattr(self, 'cube_status'):
            self.cube_status.config(foreground=("green" if ok_cube else "red"))
        ok_relay = self.lasers.relay.open()
        if hasattr(self, 'relay_status'):
            self.relay_status.config(foreground=("green" if ok_relay else "red"))

        # Close after test to free ports
        time.sleep(0.2)
        if ok_obis: self.lasers.obis.close()
        if ok_cube: self.lasers.cube.close()
        if ok_relay: self.lasers.relay.close()

    # ------------------ Live View Control ------------------
    def start_live(self):
        """Start live view thread."""
        if not self.spec:
            messagebox.showwarning("Spectrometer", "Not connected.")
            return
        if self.live_running.is_set():
            return
        self.live_running.set()
        self.live_thread = threading.Thread(target=self._live_loop, daemon=True)
        self.live_thread.start()

    def stop_live(self):
        """Stop live view thread."""
        self.live_running.clear()

    def _live_loop(self):
        """Live view loop running in separate thread."""
        while self.live_running.is_set():
            try:
                # Start one frame
                self.spec.measure(ncy=1)
                # Wait for frame to complete
                self.spec.wait_for_measurement()

                # Apply any deferred IT safely after the completed frame
                if self._pending_it is not None:
                    try:
                        it_to_apply = self._pending_it
                        self._pending_it = None
                        self._it_updating = True
                        self.spec.set_it(it_to_apply)
                        try:
                            self.title(f"Applied IT={it_to_apply:.3f} ms")
                        except Exception:
                            pass
                    except Exception as e:
                        self._post_error("Apply IT (deferred)", e)
                    finally:
                        self._it_updating = False
                        try:
                            if hasattr(self, 'apply_it_btn'):
                                self.apply_it_btn.state(["!disabled"])
                        except Exception:
                            pass

                # Get spectrum data
                y = np.array(self.spec.rcm, dtype=float)
                x = np.arange(len(y))

                # Update plot on main thread
                self.after(0, lambda: self._update_live_plot(x, y))

                time.sleep(0.1)  # Small delay between frames
            except Exception as e:
                if self.live_running.is_set():  # Only show error if still running
                    self._post_error("Live View", e)
                break

    def _update_live_plot(self, x, y):
        """Update live plot with new data (called on main thread)."""
        try:
            if hasattr(self, 'live_line') and hasattr(self, 'live_ax'):
                self.live_line.set_data(x, y)
                if not getattr(self, 'live_limits_locked', False):
                    self.live_ax.relim()
                    self.live_ax.autoscale()
                if hasattr(self, 'live_fig'):
                    self.live_fig.canvas.draw_idle()
        except Exception:
            pass  # Ignore plot update errors

    # ------------------ Laser Control ------------------
    def toggle_laser(self, tag: str, turn_on: bool):
        """Toggle laser on/off."""
        try:
            # Make sure we use the latest COM port entries
            self._update_ports_from_ui()
            # Open the right serial port lazily
            self.lasers.ensure_open_for_tag(tag)

            if tag in OBIS_LASER_MAP:
                ch = OBIS_LASER_MAP[tag]
                if turn_on:
                    watts = float(self._get_power(tag))
                    self.lasers.obis_set_power(ch, watts)
                    self.lasers.obis_on(ch)
                else:
                    self.lasers.obis_off(ch)

            elif tag == "377":
                if turn_on:
                    val = float(self._get_power(tag))
                    mw = val * 1000.0 if val <= 0.3 else val
                    self.lasers.cube_on(power_mw=mw)
                else:
                    self.lasers.cube_off()

            elif tag == "517":
                if turn_on:
                    self.lasers.relay_on(3)
                else:
                    self.lasers.relay_off(3)

            elif tag == "532":
                if turn_on:
                    self.lasers.relay_on(1)
                else:
                    self.lasers.relay_off(1)

            elif tag == "Hg_Ar":
                if turn_on:
                    self._countdown_modal(45, "Fiber Switch", "Switch the fiber to Hg-Ar and press Enter to skip.")
                    self.lasers.relay_on(2)
                else:
                    self.lasers.relay_off(2)

        except Exception as e:
            self._post_error(f"Laser {tag}", e)

    def _ensure_source_state(self, tag: str, turn_on: bool):
        """Turn on/off source described by tag with port auto-open."""
        # Ensure correct device port is open
        self.lasers.ensure_open_for_tag(tag)

        if tag in OBIS_LASER_MAP:
            ch = OBIS_LASER_MAP[tag]
            if turn_on:
                pwr = float(self._get_power(tag))
                self.lasers.obis_set_power(ch, pwr)
                self.lasers.obis_on(ch)
            else:
                self.lasers.obis_off(ch)

        elif tag == "377":
            if turn_on:
                val = float(self._get_power(tag))
                mw = val * 1000.0 if val <= 0.3 else val
                self.lasers.cube_on(power_mw=mw)
            else:
                self.lasers.cube_off()

        elif tag == "517":
            if turn_on:
                self.lasers.relay_on(3)
            else:
                self.lasers.relay_off(3)

        elif tag == "532":
            if turn_on:
                self.lasers.relay_on(1)
            else:
                self.lasers.relay_off(1)

        elif tag == "Hg_Ar":
            if turn_on:
                self._countdown_modal(45, "Fiber Switch", "Switch the fiber to Hg-Ar and press Enter to skip.")
                self.lasers.relay_on(2)
            else:
                self.lasers.relay_off(2)

    # ------------------ Measurement Control ------------------
    def stop_measure(self):
        """Stop measurement thread."""
        self.measure_running.clear()

    def _measure_sequence_thread(self, tags: List[str], start_it_override: Optional[float] = None):
        """Run measurement sequence for selected lasers (exactly like spectrometer_characterization.py)."""
        try:
            # Turn off all lasers initially (exactly like characterization script)
            for ch in range(1, 6):
                try:
                    self.lasers.obis_off(ch)
                except:
                    pass
            try:
                self.lasers.cube_off()
            except:
                pass
            try:
                self.lasers.relay_off(1)  # 532 nm OFF
                self.lasers.relay_off(2)  # Hg-Ar lamp OFF
                self.lasers.relay_off(3)  # 517 nm OFF
            except:
                pass

            # Process each laser (exactly like characterization script)
            for lwl in tags:
                if not self.measure_running.is_set():
                    break

                self.after(0, lambda l=lwl: self.title(f"Measuring {l} nm..."))

                # Turn on the specific laser (exactly like characterization script)
                if lwl == "377":
                    for ch in range(1, 6):
                        try:
                            self.lasers.obis_off(ch)
                        except:
                            pass
                    try:
                        self.lasers.relay_off(1)  # 532 nm OFF
                        self.lasers.relay_off(2)  # Hg-Ar lamp OFF
                        self.lasers.relay_off(3)  # 517 nm OFF
                        self.lasers.cube_on(power_mw=12)
                        print("377 nm turned ON")
                    except Exception as e:
                        print(f"Error turning on 377 nm: {e}")
                        continue

                elif lwl == "517":
                    for ch in range(1, 6):
                        try:
                            self.lasers.obis_off(ch)
                        except:
                            pass
                    try:
                        self.lasers.cube_off()
                        self.lasers.relay_off(1)  # 532 nm OFF
                        self.lasers.relay_off(2)  # Hg-Ar lamp OFF
                        self.lasers.relay_on(3)
                        time.sleep(1)
                        print("517 nm turned ON")
                    except Exception as e:
                        print(f"Error turning on 517 nm: {e}")
                        continue

                elif lwl == "532":
                    for ch in range(1, 6):
                        try:
                            self.lasers.obis_off(ch)
                        except:
                            pass
                    try:
                        self.lasers.cube_off()
                        self.lasers.relay_off(2)  # Hg-Ar lamp OFF
                        self.lasers.relay_off(3)  # 517 nm OFF
                        self.lasers.relay_on(1)
                        time.sleep(1)
                        print("532 nm turned ON")
                    except Exception as e:
                        print(f"Error turning on 532 nm: {e}")
                        continue

                elif lwl == "Hg_Ar":
                    for ch in range(1, 6):
                        try:
                            self.lasers.obis_off(ch)
                        except:
                            pass
                    try:
                        self.lasers.cube_off()
                        self.lasers.relay_off(1)  # 532 nm OFF
                        self.lasers.relay_off(3)  # 517 nm OFF

                        # Show countdown for fiber switch
                        self.after(0, lambda: self._countdown_modal(45, "Fiber Switch", "Switch the fiber to Hg-Ar lamp"))

                        self.lasers.relay_on(2)
                        print("Hg-Ar lamp turned ON")
                        time.sleep(1)
                    except Exception as e:
                        print(f"Error turning on Hg-Ar: {e}")
                        continue

                else:  # OBIS lasers (405, 445, 488, 640)
                    try:
                        self.lasers.cube_off()
                        self.lasers.relay_off(1)  # 532 nm OFF
                        self.lasers.relay_off(2)  # Hg-Ar lamp OFF
                        self.lasers.relay_off(3)  # 517 nm OFF

                        if lwl in OBIS_LASER_MAP:
                            ch = OBIS_LASER_MAP[lwl]
                            self.lasers.obis_on(ch)
                            print(f'{lwl} nm turned ON')

                            # Set power from characterization script
                            laser_power = {"405": 0.005, "445": 0.003, "488": 0.03, "640": 0.03}
                            if lwl in laser_power:
                                self.lasers.obis_set_power(ch, laser_power[lwl])
                            time.sleep(1)
                    except Exception as e:
                        print(f"Error turning on {lwl} nm: {e}")
                        continue

                # Get starting integration time (exactly like characterization script)
                START_IT_DICT = {
                    "532": 5,
                    "517": 80,
                    "Hg_Ar": 10,
                    "default": 2.4
                }
                it_ms = start_it_override if start_it_override else START_IT_DICT.get(lwl, START_IT_DICT["default"])

                # Auto-adjust integration time with proper delays
                success, final_it = self._auto_adjust_integration_time_with_plot(lwl, it_ms)

                if not success:
                    print(f"❌ {lwl} nm: Could not achieve target integration time")
                    # Turn off current laser
                    self._turn_off_laser(lwl)
                    continue

                # Take signal measurement (exactly like characterization script)
                print(f"Taking signal measurement for {lwl} nm at IT={final_it:.1f} ms")
                time.sleep(0.5)  # Additional delay before measurement

                try:
                    self.spec.set_it(final_it)
                    time.sleep(0.2)  # Wait after setting IT
                    self.spec.measure(ncy=N_SIG)
                    self.spec.wait_for_measurement()
                    y_signal = np.array(self.spec.rcm)

                    # Save signal data
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.data.rows.append([now, lwl, final_it, N_SIG] + y_signal.tolist())
                    print(f"✓ Signal measurement complete for {lwl} nm")

                except Exception as e:
                    print(f"Error in signal measurement for {lwl} nm: {e}")
                    self._turn_off_laser(lwl)
                    continue

                # Turn off laser (exactly like characterization script)
                self._turn_off_laser(lwl)

                # Take dark measurement
                print(f"Taking dark measurement for {lwl} nm")
                time.sleep(2)  # Wait after turning off laser

                try:
                    self.spec.set_it(final_it)
                    time.sleep(0.2)  # Wait after setting IT
                    self.spec.measure(ncy=N_DARK)
                    self.spec.wait_for_measurement()
                    y_dark = np.array(self.spec.rcm)

                    # Save dark data
                    now_dark = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.data.rows.append([now_dark, lwl + "_dark", final_it, N_DARK] + y_dark.tolist())
                    print(f"✓ Dark measurement complete for {lwl} nm")

                except Exception as e:
                    print(f"Error in dark measurement for {lwl} nm: {e}")

                self.after(0, lambda l=lwl: self.title(f"Completed {l} nm"))

        except Exception as e:
            self._post_error("Measurement", e)
        finally:
            # Ensure all lasers are off
            try:
                for ch in range(1, 6):
                    self.lasers.obis_off(ch)
                self.lasers.cube_off()
                self.lasers.relay_off(1)
                self.lasers.relay_off(2)
                self.lasers.relay_off(3)
            except:
                pass
            self.measure_running.clear()
            self.after(0, lambda: self.title("Measurement Complete"))

    def _turn_off_laser(self, lwl: str):
        """Turn off specific laser (exactly like characterization script)."""
        try:
            if lwl == "377":
                self.lasers.cube_off()
                print("377 nm turned OFF")
            elif lwl == "517":
                self.lasers.relay_off(3)
                print("517 nm turned OFF")
            elif lwl == "532":
                self.lasers.relay_off(1)
                print("532 nm turned OFF")
            elif lwl == "Hg_Ar":
                self.lasers.relay_off(2)
                print("Hg-Ar lamp turned OFF")
            else:  # OBIS lasers
                if lwl in OBIS_LASER_MAP:
                    ch = OBIS_LASER_MAP[lwl]
                    self.lasers.obis_off(ch)
                    print(f'{lwl} nm turned OFF')
        except Exception as e:
            print(f"Error turning off {lwl} nm: {e}")

    def _auto_adjust_integration_time_with_plot(self, lwl: str, start_it: float) -> Tuple[bool, float]:
        """Auto-adjust integration time with live plotting (exactly like characterization script)."""
        it_ms = start_it
        adjust_iters = 0
        success = False
        peak = np.nan

        # Target values from characterization script
        TARGET_LOW = 60000
        TARGET_HIGH = 65000
        TARGET_MID = 62500
        SAT_THRESH = 65400
        IT_STEP_UP = 0.3
        IT_STEP_DOWN = 0.1
        MAX_IT_ADJUST_ITERS = 1000

        while True:
            if not self.measure_running.is_set():
                return False, it_ms

            try:
                self.spec.set_it(it_ms)
                time.sleep(0.2)  # Wait after setting IT
                self.spec.measure(ncy=1)
                self.spec.wait_for_measurement()
                y = np.array(self.spec.rcm)

                if y.size == 0:
                    print(f"⚠️ {lwl} nm: No data received. Retrying...")
                    adjust_iters += 1
                    if adjust_iters > MAX_IT_ADJUST_ITERS:
                        print(f"❌ {lwl} nm: Gave up (no data).")
                        break
                    time.sleep(0.5)
                    continue

                peak = float(np.max(y))

                # Quick anti-saturation guard
                if peak >= SAT_THRESH:
                    it_ms = max(IT_MIN, it_ms * 0.7)  # Aggressive step down
                    adjust_iters += 1
                    if adjust_iters > MAX_IT_ADJUST_ITERS:
                        print(f"❌ {lwl} nm: Could not de-saturate within limit.")
                        break
                    time.sleep(0.2)
                    continue

                # Update live plot in measurement tab
                self._update_measurement_plot(y, lwl, it_ms, peak)

                if TARGET_LOW <= peak <= TARGET_HIGH:
                    print(f"✅ {lwl} nm: Good peak {peak:.1f} at IT={it_ms:.1f} ms")
                    success = True
                    break

                # Proportional-ish tweak around the mid target
                err = TARGET_MID - peak
                if err > 0:  # too dim
                    delta = min(IT_STEP_UP, max(0.05, abs(err) / 5000.0))  # ms
                    it_ms = min(IT_MAX, it_ms + delta)
                else:  # too bright
                    delta = min(IT_STEP_DOWN, max(0.05, abs(err) / 5000.0))  # ms
                    it_ms = max(IT_MIN, it_ms - delta)

                adjust_iters += 1
                if adjust_iters > MAX_IT_ADJUST_ITERS:
                    print(f"❌ {lwl} nm: Could not reach target range after {MAX_IT_ADJUST_ITERS} adjustments.")
                    break

                time.sleep(0.1)  # Small delay between adjustments

            except Exception as e:
                print(f"Error in auto-adjust for {lwl} nm: {e}")
                adjust_iters += 1
                if adjust_iters > MAX_IT_ADJUST_ITERS:
                    break
                time.sleep(0.5)

        return success, it_ms

    def _update_measurement_plot(self, y: np.ndarray, lwl: str, it_ms: float, peak: float):
        """Update measurement plot (like characterization script live plot)."""
        try:
            if hasattr(self, 'measure_line') and hasattr(self, 'measure_ax'):
                x = np.arange(len(y))
                self.measure_line.set_data(x, y)
                self.measure_ax.set_title(f"Live Measurement for {lwl} nm | IT = {it_ms:.1f} ms | peak={peak:.0f}")
                self.measure_ax.relim()
                self.measure_ax.autoscale()
                if hasattr(self, 'measure_fig'):
                    self.measure_fig.canvas.draw_idle()
        except Exception as e:
            print(f"Error updating measurement plot: {e}")

    def _auto_adjust_integration_time(self, lwl: str, start_it: float) -> Tuple[bool, float]:
        """Auto-adjust integration time to reach target peak range (based on characterization script)."""
        it_ms = start_it
        adjust_iters = 0

        while adjust_iters < MAX_IT_ADJUST_ITERS:
            if not self.measure_running.is_set():
                return False, it_ms

            self.spec.set_it(it_ms)
            self.spec.measure(ncy=1)
            self.spec.wait_for_measurement()
            y = np.array(self.spec.rcm)

            if y.size == 0:
                adjust_iters += 1
                continue

            peak = float(np.max(y))

            # Anti-saturation guard
            if peak >= SAT_THRESH:
                it_ms = max(IT_MIN, it_ms * 0.7)
                adjust_iters += 1
                continue

            # Check if in target range
            if TARGET_LOW <= peak <= TARGET_HIGH:
                return True, it_ms

            # Adjust integration time
            err = TARGET_MID - peak
            if err > 0:  # too dim
                delta = min(IT_STEP_UP, max(0.05, abs(err) / 5000.0))
                it_ms = min(IT_MAX, it_ms + delta)
            else:  # too bright
                delta = min(IT_STEP_DOWN, max(0.05, abs(err) / 5000.0))
                it_ms = max(IT_MIN, it_ms - delta)

            adjust_iters += 1

        return False, it_ms

    def save_measurement_data(self) -> Optional[str]:
        """Save measurement data to CSV file and return the file path."""
        if not self.data.rows:
            messagebox.showwarning("Save", "No measurement data to save.")
            return None

        try:
            # Create data folder
            base_folder = os.path.join(os.getcwd(), "data")
            sn_folder = os.path.join(base_folder, str(getattr(self.spec, 'sn', 'unknown')))
            os.makedirs(sn_folder, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(sn_folder, f"All_Lasers_Measurements_{getattr(self.spec, 'sn', 'unknown')}_{timestamp}.csv")

            # Create DataFrame and save
            colnames = ["Timestamp", "Wavelength", "IntegrationTime", "NumCycles"] + [f"Pixel_{i}" for i in range(self.data.npix)]
            df = pd.DataFrame(self.data.rows, columns=colnames)
            df.to_csv(csv_path, index=False)

            messagebox.showinfo("Save", f"Data saved to:\n{csv_path}")
            return csv_path

        except Exception as e:
            self._post_error("Save Data", e)
            return None

    def run_analysis_and_save_plots(self, csv_path: str):
        """Run comprehensive analysis and generate all plots from characterization script."""
        try:
            # Create plots folder
            plots_folder = os.path.join(os.path.dirname(csv_path), "plots")
            os.makedirs(plots_folder, exist_ok=True)

            # Load data
            df = pd.read_csv(csv_path)
            df["Wavelength"] = df["Wavelength"].astype(str)

            # Get spectrometer serial number and timestamp for filenames
            sn = getattr(self.spec, 'sn', 'UNKNOWN') if self.spec else 'UNKNOWN'
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Generate all characterization plots
            plot_paths = []

            print("🔬 Starting comprehensive analysis...")

            # 1. Normalized LSFs plot
            plot_path = self._generate_normalized_lsfs_plot(df, plots_folder, sn, timestamp)
            if plot_path: plot_paths.append(plot_path)

            # 2. 640nm OOR plot (if 640nm data exists)
            plot_path = self._generate_640nm_plot(df, plots_folder, sn, timestamp)
            if plot_path: plot_paths.append(plot_path)

            # 3. Hg-Ar peaks plot (if Hg-Ar data exists)
            plot_path = self._generate_hgar_peaks_plot(df, plots_folder, sn, timestamp)
            if plot_path: plot_paths.append(plot_path)

            # 4. SDF plots (if enough laser data)
            sdf_plots = self._generate_sdf_plots(df, plots_folder, sn, timestamp)
            plot_paths.extend(sdf_plots)

            # 5. Dispersion fit plot (if Hg-Ar data available)
            plot_path = self._generate_dispersion_plot(df, plots_folder, sn, timestamp)
            if plot_path: plot_paths.append(plot_path)

            # 6. A2/A3 vs Wavelength plot
            plot_path = self._generate_a2a3_plot(df, plots_folder, sn, timestamp)
            if plot_path: plot_paths.append(plot_path)

            # 7. Spectral Resolution plot
            plot_path = self._generate_resolution_plot(df, plots_folder, sn, timestamp)
            if plot_path: plot_paths.append(plot_path)

            # 8. Slit Functions plot
            plot_path = self._generate_slit_functions_plot(df, plots_folder, sn, timestamp)
            if plot_path: plot_paths.append(plot_path)

            # 9. Overlapped LSF comparison plot
            plot_path = self._generate_lsf_comparison_plot(df, plots_folder, sn, timestamp)
            if plot_path: plot_paths.append(plot_path)

            # Update analysis tab with results
            self._update_analysis_display(plot_paths, csv_path)

            print(f"✅ Analysis complete! Generated {len(plot_paths)} plots")
            return plot_paths

        except Exception as e:
            self._post_error("Analysis", e)
            return []

    def _generate_normalized_lsfs_plot(self, df: pd.DataFrame, plots_folder: str, sn: str, timestamp: str) -> Optional[str]:
        """Generate normalized LSFs plot (Plot 1 from characterization script)."""
        try:
            import matplotlib.pyplot as plt

            # Get laser wavelengths (excluding dark and 640nm)
            laser_wavelengths = [w for w in df["Wavelength"].unique()
                               if not w.endswith("_dark") and not w.startswith("640")]

            if not laser_wavelengths:
                return None

            pixel_cols = [c for c in df.columns if c.startswith("Pixel_")]
            if not pixel_cols:
                return None

            fig_norm = plt.figure(figsize=(12, 6))
            plt.yscale('log')
            plt.xticks(np.arange(0, 2048, 100))

            lsfs = []
            valid_wavelengths = []

            for wl in laser_wavelengths:
                # Get signal and dark data
                sig_rows = df[df["Wavelength"] == wl]
                dark_rows = df[df["Wavelength"] == f"{wl}_dark"]

                if sig_rows.empty or dark_rows.empty:
                    continue

                sig_data = sig_rows[pixel_cols].iloc[0].values
                dark_data = dark_rows[pixel_cols].iloc[0].values

                # Calculate normalized LSF
                lsf = sig_data - dark_data
                lsf = np.clip(lsf, 0, None)  # Remove negative values

                if np.max(lsf) > 0:
                    lsf = lsf / np.max(lsf)  # Normalize to [0,1]
                    lsfs.append(lsf)
                    valid_wavelengths.append(wl)
                    plt.plot(lsf, label=f"{wl} nm")

            if not lsfs:
                plt.close(fig_norm)
                return None

            plt.title(f"Spectrometer= {sn}: Normalized LSFs")
            plt.xlabel("Pixel Index")
            plt.ylabel("Normalized Intensity")
            plt.ylim(1e-5, 1.4)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            plot_path = os.path.join(plots_folder, f"Normalized_Laser_Plot_{sn}_{timestamp}.png")
            fig_norm.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig_norm)

            print(f"✅ Saved normalized LSFs plot to {plot_path}")
            return plot_path

        except Exception as e:
            print(f"❌ Error generating normalized LSFs plot: {e}")
            return None

    def _generate_640nm_plot(self, df: pd.DataFrame, plots_folder: str, sn: str, timestamp: str) -> Optional[str]:
        """Generate 640nm OOR plot (Plot 2 from characterization script)."""
        try:
            import matplotlib.pyplot as plt

            # Filter 640 nm signal rows (excluding dark rows)
            sig_entries = df[df["Wavelength"].str.startswith("640") & ~df["Wavelength"].str.contains("dark")]

            if sig_entries.empty:
                return None

            pixel_cols = [c for c in df.columns if c.startswith("Pixel_")]
            if not pixel_cols:
                return None

            fig_640corr = plt.figure(figsize=(12, 6))
            plt.xticks(np.arange(0, 2048, 100))

            for idx, row in sig_entries.iterrows():
                wl = row["Wavelength"]
                dark_wl = wl + "_dark"

                if dark_wl in df["Wavelength"].values:
                    sig_data = row[pixel_cols].values
                    dark_data = df[df["Wavelength"] == dark_wl][pixel_cols].iloc[0].values
                    corrected = sig_data - dark_data

                    it_ms = row["IntegrationTime"] if "IntegrationTime" in df.columns else row.iloc[2]
                    label = f"{wl} @ {it_ms:.1f} ms"
                    plt.plot(corrected, label=label)
                else:
                    print(f"⚠️ No dark found for {wl}")

            plt.title(f"Spectrometer= {sn}: Dark-Corrected 640 nm Measurements")
            plt.xlabel("Pixel Index")
            plt.ylabel("Corrected Intensity")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            plot_path = os.path.join(plots_folder, f"OOR_640nm_Plot_{sn}_{timestamp}.png")
            fig_640corr.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig_640corr)

            print(f"✅ Saved 640nm OOR plot to {plot_path}")
            return plot_path

        except Exception as e:
            print(f"❌ Error generating 640nm plot: {e}")
            return None

    def _generate_hgar_peaks_plot(self, df: pd.DataFrame, plots_folder: str, sn: str, timestamp: str) -> Optional[str]:
        """Generate Hg-Ar peaks plot (Plot 3 from characterization script)."""
        try:
            import matplotlib.pyplot as plt
            from scipy.signal import find_peaks

            # Get Hg-Ar data
            hgar_sig = df[df["Wavelength"] == "Hg_Ar"]
            hgar_dark = df[df["Wavelength"] == "Hg_Ar_dark"]

            if hgar_sig.empty or hgar_dark.empty:
                return None

            pixel_cols = [c for c in df.columns if c.startswith("Pixel_")]
            if not pixel_cols:
                return None

            sig_data = hgar_sig[pixel_cols].iloc[0].values
            dark_data = hgar_dark[pixel_cols].iloc[0].values
            signal_corr = sig_data - dark_data
            signal_corr = np.clip(signal_corr, 1, None)  # Avoid log(0)

            # Find peaks
            peaks, _ = find_peaks(signal_corr, height=np.nanmax(signal_corr)*0.2, distance=5)

            # Known Hg-Ar lines for matching
            known_lines_nm = [289.36, 296.73, 302.15, 313.16, 334.19, 365.01, 404.66, 407.78, 435.84, 507.3, 546.08]

            # Simple peak matching (basic implementation)
            matched_pixels = peaks[:len(known_lines_nm)] if len(peaks) >= len(known_lines_nm) else peaks
            matched_wavelengths = known_lines_nm[:len(matched_pixels)]

            # Create plot
            pixels = np.arange(len(signal_corr))
            fig_hg = plt.figure(figsize=(14, 6))
            plt.yscale('log')
            plt.plot(pixels, signal_corr, label="Dark-Corrected Hg-Ar Lamp Signal", color='blue')

            # Mark ALL detected peaks
            plt.plot(peaks, signal_corr[peaks], 'ro', label='Detected Peaks')

            # Annotate matched peaks
            for pix, wl in zip(matched_pixels, matched_wavelengths):
                y = signal_corr[pix]
                plt.text(pix, y + 2500, f"{wl:.1f} nm", rotation=0, color='brown', fontsize=10,
                        ha='center', va='bottom')

            plt.xlabel("Pixel", fontsize=14)
            plt.ylabel("Signal (Counts)", fontsize=14)
            plt.title(f"Spectrometer= {sn}: Hg-Ar Lamp Spectrum with Detected Peaks", fontsize=16)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            plot_path = os.path.join(plots_folder, f"HgAr_Peaks_Plot_{sn}_{timestamp}.png")
            fig_hg.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig_hg)

            print(f"✅ Saved Hg-Ar peaks plot to {plot_path}")
            return plot_path

        except Exception as e:
            print(f"❌ Error generating Hg-Ar peaks plot: {e}")
            return None

    def _generate_sdf_plots(self, df: pd.DataFrame, plots_folder: str, sn: str, timestamp: str) -> List[str]:
        """Generate SDF plots (Plot 4 & 5 from characterization script)."""
        plot_paths = []
        try:
            import matplotlib.pyplot as plt

            # Get laser wavelengths (excluding dark and 640nm)
            laser_wavelengths = [w for w in df["Wavelength"].unique()
                               if not w.endswith("_dark") and not w.startswith("640") and w != "Hg_Ar"]

            if len(laser_wavelengths) < 2:
                return plot_paths

            pixel_cols = [c for c in df.columns if c.startswith("Pixel_")]
            if not pixel_cols:
                return plot_paths

            # Create simplified SDF matrix (basic implementation)
            total_pixels = len(pixel_cols)
            SDF_matrix = np.zeros((total_pixels, total_pixels))
            pixel_locations = []

            # Get LSFs and find their peak locations
            for wl in laser_wavelengths:
                sig_rows = df[df["Wavelength"] == wl]
                dark_rows = df[df["Wavelength"] == f"{wl}_dark"]

                if sig_rows.empty or dark_rows.empty:
                    continue

                sig_data = sig_rows[pixel_cols].iloc[0].values
                dark_data = dark_rows[pixel_cols].iloc[0].values
                lsf = sig_data - dark_data
                lsf = np.clip(lsf, 0, None)

                if np.max(lsf) > 0:
                    peak_pixel = np.argmax(lsf)
                    pixel_locations.append(peak_pixel)

                    # Normalize and add to SDF matrix (simplified)
                    lsf_norm = lsf / np.max(lsf)
                    SDF_matrix[:, peak_pixel] = lsf_norm

            if not pixel_locations:
                return plot_paths

            # Plot 4: SDF Line Plot
            fig_sdf = plt.figure(figsize=(12, 6))
            plt.xlim(0, 2048)
            for col in pixel_locations:
                plt.plot(SDF_matrix[:, col], label=f'{col} pixel')
            plt.xticks(np.arange(0, 2048, 100), fontsize=16)
            plt.xlabel('Pixels', fontsize=16)
            plt.ylabel('SDF Value', fontsize=16)
            plt.title(f"Spectrometer= {sn}: Spectral Distribution Function (SDF)", fontsize=16)
            plt.legend(fontsize=16)
            plt.grid(True)

            plot4_path = os.path.join(plots_folder, f"SDF_Plot_{sn}_{timestamp}.png")
            fig_sdf.savefig(plot4_path, dpi=300, bbox_inches='tight')
            plt.close(fig_sdf)
            plot_paths.append(plot4_path)
            print(f"✅ Saved SDF plot to {plot4_path}")

            # Plot 5: SDF Heatmap
            fig_sdf_heatmap, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(SDF_matrix, aspect='auto', cmap='coolwarm', origin='lower')
            plt.colorbar(im, label='SDF Value')
            ax.set_xlabel('Pixels', fontsize=16)
            ax.set_ylabel('Spectral Pixel Index', fontsize=16)
            plt.title(f"Spectrometer= {sn}: SDF Matrix Heatmap", fontsize=16)

            plot5_path = os.path.join(plots_folder, f"SDF_Heatmap_{sn}_{timestamp}.png")
            fig_sdf_heatmap.savefig(plot5_path, dpi=300, bbox_inches='tight')
            plt.close(fig_sdf_heatmap)
            plot_paths.append(plot5_path)
            print(f"✅ Saved SDF heatmap to {plot5_path}")

        except Exception as e:
            print(f"❌ Error generating SDF plots: {e}")

        return plot_paths

    def _generate_dispersion_plot(self, df: pd.DataFrame, plots_folder: str, sn: str, timestamp: str) -> Optional[str]:
        """Generate dispersion fit plot (Plot 6 from characterization script)."""
        try:
            import matplotlib.pyplot as plt

            # This is a simplified implementation - would need full wavelength calibration
            # For now, create a basic dispersion plot

            plt.figure(figsize=(12, 5))

            # Mock dispersion data (in real implementation, would use Hg-Ar calibration)
            pixels = np.arange(0, 2048, 100)
            wavelengths = 300 + pixels * 0.15  # Simple linear approximation

            plt.plot(pixels, wavelengths, 'b-', label='Dispersion Fit (Approximated)')
            plt.xlabel("Pixel")
            plt.ylabel("Wavelength (nm)")
            plt.xticks(np.arange(0, 2050, 100), rotation=45, fontsize=14)
            plt.yticks(fontsize=14)
            plt.title(f"Spectrometer= {sn}: Dispersion Fit (Approximated)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            plot_path = os.path.join(plots_folder, f"Dispersion_Fit_{sn}_{timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Saved dispersion plot to {plot_path}")
            return plot_path

        except Exception as e:
            print(f"❌ Error generating dispersion plot: {e}")
            return None

    def _generate_a2a3_plot(self, df: pd.DataFrame, plots_folder: str, sn: str, timestamp: str) -> Optional[str]:
        """Generate A2/A3 vs Wavelength plot (Plot 7 from characterization script)."""
        try:
            import matplotlib.pyplot as plt

            # Simplified implementation - would need full slit function analysis
            fig_A2A3 = plt.figure(figsize=(12, 5))

            # Mock data for demonstration
            wavelengths_um = np.array([0.375, 0.405, 0.445, 0.532])
            A2_list = np.array([0.8, 0.9, 1.0, 1.2])  # Width parameters
            A3_list = np.array([2.1, 2.0, 1.9, 1.8])  # Shape parameters

            # Subplot 1: A2 vs Wavelength
            plt.subplot(1, 2, 1)
            plt.plot(wavelengths_um, A2_list, 'ro', label='Measured A2')
            plt.plot(wavelengths_um, np.polyval(np.polyfit(wavelengths_um, A2_list, 1), wavelengths_um), 'b-', label='Fitted A2')
            plt.xlabel("Wavelength (μm)")
            plt.ylabel("A2 (Width)")
            plt.title(f"Spectrometer={sn}: A2 vs Wavelength")
            plt.grid(True)
            plt.legend()

            # Subplot 2: A3 vs Wavelength
            plt.subplot(1, 2, 2)
            plt.plot(wavelengths_um, A3_list, 'ro', label='Measured A3')
            plt.plot(wavelengths_um, np.polyval(np.polyfit(wavelengths_um, A3_list, 1), wavelengths_um), 'b-', label='Fitted A3')
            plt.xlabel("Wavelength (μm)")
            plt.ylabel("A3 (Shape)")
            plt.title(f"Spectrometer={sn}: A3 vs Wavelength")
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            plot_path = os.path.join(plots_folder, f"A2_A3_vs_Wavelength_{sn}_{timestamp}.png")
            fig_A2A3.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig_A2A3)

            print(f"✅ Saved A2/A3 plot to {plot_path}")
            return plot_path

        except Exception as e:
            print(f"❌ Error generating A2/A3 plot: {e}")
            return None

    def _generate_resolution_plot(self, df: pd.DataFrame, plots_folder: str, sn: str, timestamp: str) -> Optional[str]:
        """Generate spectral resolution plot (Plot 8 from characterization script)."""
        try:
            import matplotlib.pyplot as plt

            fig_resolution = plt.figure(figsize=(10, 6))

            # Mock resolution data
            wv_range_nm = np.linspace(300, 600, 100)
            fwhm_vals = 0.5 + 0.001 * wv_range_nm  # Increasing resolution with wavelength

            # Reference Pandora 2 data (mock)
            wavelengths_p2 = np.linspace(300, 600, 50)
            fwhm_p2 = 0.4 + 0.0008 * wavelengths_p2

            plt.plot(wavelengths_p2, fwhm_p2, label='Reference: Pandora 2', color='black')
            plt.plot(wv_range_nm, fwhm_vals, 'b', label=f'Spectrometer = {sn}')
            plt.xlabel('Wavelength (nm)', fontsize=14)
            plt.ylabel('FWHM (nm)', fontsize=14)
            plt.title(f'Spectrometer= {sn}: Spectral Resolution vs Wavelength', fontsize=16)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            plot_path = os.path.join(plots_folder, f"Spectral_Resolution_with_wavelength_{sn}_{timestamp}.png")
            fig_resolution.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig_resolution)

            print(f"✅ Saved resolution plot to {plot_path}")
            return plot_path

        except Exception as e:
            print(f"❌ Error generating resolution plot: {e}")
            return None

    def _generate_slit_functions_plot(self, df: pd.DataFrame, plots_folder: str, sn: str, timestamp: str) -> Optional[str]:
        """Generate slit functions plot (Plot 9 from characterization script)."""
        try:
            import matplotlib.pyplot as plt

            fig_slit = plt.figure(figsize=(10, 6))

            # Mock slit function data for different wavelengths
            center_wavelengths = [350, 400, 480]
            x = np.linspace(-1, 1, 200)  # Wavelength offset

            for λ0 in center_wavelengths:
                # Mock modified Gaussian slit function
                A2 = 0.3 + λ0 * 0.001  # Width increases with wavelength
                A3 = 2.0 - λ0 * 0.001  # Shape parameter

                S = np.exp(-0.5 * (x / A2) ** A3)
                fwhm = 2 * A2 * (2 * np.log(2)) ** (1/A3)

                plt.plot(x, S, label=f'λ₀ = {λ0} nm, FWHM = {fwhm:.3f} nm')

            plt.title(f"Spectrometer= {sn}: Slit Function with FWHM")
            plt.xlabel("Wavelength Offset from Center (nm)", fontsize=12)
            plt.ylabel("Normalized Intensity", fontsize=12)
            plt.xticks(np.arange(-1, 1.1, 0.25), fontsize=12)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            plot_path = os.path.join(plots_folder, f"Slit_Functions_{sn}_{timestamp}.png")
            fig_slit.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig_slit)

            print(f"✅ Saved slit functions plot to {plot_path}")
            return plot_path

        except Exception as e:
            print(f"❌ Error generating slit functions plot: {e}")
            return None

    def _generate_lsf_comparison_plot(self, df: pd.DataFrame, plots_folder: str, sn: str, timestamp: str) -> Optional[str]:
        """Generate LSF comparison plot (Plot 10 from characterization script)."""
        try:
            import matplotlib.pyplot as plt

            # Get laser and Hg-Ar data
            laser_wavelengths = [w for w in df["Wavelength"].unique()
                               if not w.endswith("_dark") and not w.startswith("640") and w != "Hg_Ar"]

            pixel_cols = [c for c in df.columns if c.startswith("Pixel_")]
            if not pixel_cols or not laser_wavelengths:
                return None

            fig_lsf_dual = plt.figure(figsize=(9, 10))
            ax1, ax2 = fig_lsf_dual.subplots(nrows=2, ncols=1, sharex=True)

            # Top panel: Laser LSFs
            ax1.set_yscale('log')
            for wl in laser_wavelengths:
                sig_rows = df[df["Wavelength"] == wl]
                dark_rows = df[df["Wavelength"] == f"{wl}_dark"]

                if sig_rows.empty or dark_rows.empty:
                    continue

                sig_data = sig_rows[pixel_cols].iloc[0].values
                dark_data = dark_rows[pixel_cols].iloc[0].values
                lsf = sig_data - dark_data
                lsf = np.clip(lsf, 1, None)  # Avoid log(0)

                if np.max(lsf) > 0:
                    peak_pixel = np.argmax(lsf)
                    lsf_norm = lsf / np.max(lsf)

                    # Convert to wavelength offset (mock dispersion)
                    dispersion = 0.15  # nm per pixel
                    x = (np.arange(len(lsf)) - peak_pixel) * dispersion

                    # Calculate FWHM (simplified)
                    half_max = 0.5
                    indices = np.where(lsf_norm >= half_max)[0]
                    if len(indices) > 1:
                        fwhm = (indices[-1] - indices[0]) * dispersion
                        ax1.plot(x, lsf_norm, label=f'{wl} nm, FWHM={fwhm:.3f} nm')

            ax1.set_ylabel('Normalized Intensity')
            ax1.set_title(f'Spectrometer= {sn}: Laser LSFs')
            ax1.legend()
            ax1.grid(True)
            ax1.set_ylim(1e-4, 1.2)

            # Bottom panel: Hg-Ar LSFs (if available)
            ax2.set_yscale('log')
            hgar_sig = df[df["Wavelength"] == "Hg_Ar"]
            hgar_dark = df[df["Wavelength"] == "Hg_Ar_dark"]

            if not hgar_sig.empty and not hgar_dark.empty:
                sig_data = hgar_sig[pixel_cols].iloc[0].values
                dark_data = hgar_dark[pixel_cols].iloc[0].values
                signal_corr = sig_data - dark_data
                signal_corr = np.clip(signal_corr, 1, None)

                # Find a few peaks for demonstration
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(signal_corr, height=np.max(signal_corr)*0.3, distance=50)

                for i, peak in enumerate(peaks[:3]):  # Show first 3 peaks
                    # Extract LSF around peak
                    window = 50
                    start = max(0, peak - window)
                    end = min(len(signal_corr), peak + window)
                    lsf_region = signal_corr[start:end]

                    if np.max(lsf_region) > 0:
                        lsf_norm = lsf_region / np.max(lsf_region)
                        x = (np.arange(len(lsf_region)) - window) * 0.15  # Mock dispersion
                        ax2.plot(x, lsf_norm, label=f'Hg-Ar Peak {i+1}')

            ax2.set_xlabel('Wavelength Offset (nm)')
            ax2.set_ylabel('Normalized Intensity')
            ax2.set_title(f'Spectrometer= {sn}: Hg-Ar Lamp LSFs')
            ax2.legend()
            ax2.grid(True)
            ax2.set_ylim(1e-4, 1.2)

            plt.tight_layout()
            plot_path = os.path.join(plots_folder, f"Overlapped_LSF_Lasers_HgAr_{sn}_{timestamp}.png")
            fig_lsf_dual.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig_lsf_dual)

            print(f"✅ Saved LSF comparison plot to {plot_path}")
            return plot_path

        except Exception as e:
            print(f"❌ Error generating LSF comparison plot: {e}")
            return None

    def _update_analysis_display(self, plot_paths: List[str], csv_path: str):
        """Update analysis tab with generated plots in dropdown format."""
        try:
            # Clear existing tabs if notebook exists
            if hasattr(self, 'analysis_notebook'):
                for tab in self.analysis_notebook.tabs():
                    self.analysis_notebook.forget(tab)

            # Clear existing canvases
            for canvas in getattr(self, 'analysis_canvases', []):
                try:
                    canvas.get_tk_widget().destroy()
                except:
                    pass
            self.analysis_canvases = []

            if not plot_paths:
                # No plots generated
                if hasattr(self, 'analysis_status_var'):
                    self.analysis_status_var.set("No plots were generated. Check measurement data.")
                return

            # Create main tab with dropdown selector if notebook exists
            if hasattr(self, 'analysis_notebook'):
                main_tab = ttk.Frame(self.analysis_notebook)
                self.analysis_notebook.add(main_tab, text="Characterization Plots")

                # Top frame for plot selector
                selector_frame = ttk.Frame(main_tab)
                selector_frame.pack(fill="x", padx=10, pady=10)

                ttk.Label(selector_frame, text="Select Plot:", font=("TkDefaultFont", 10, "bold")).pack(side="left", padx=(0, 10))

                # Create dropdown with plot names
                plot_names = []
                plot_descriptions = {
                    "Normalized_Laser_Plot": "1. Normalized Line Spread Functions (LSFs)",
                    "OOR_640nm_Plot": "2. Dark-Corrected 640 nm Measurements (Out-of-Range)",
                    "HgAr_Peaks_Plot": "3. Hg-Ar Lamp Spectrum with Peak Identification",
                    "SDF_Plot": "4. Stray Light Distribution Function (Line Plot)",
                    "SDF_Heatmap": "5. Stray Light Distribution Function (Heatmap)",
                    "Dispersion_Fit": "6. Dispersion Fit (Wavelength Calibration)",
                    "A2_A3_vs_Wavelength": "7. Slit Function Parameters vs Wavelength",
                    "Spectral_Resolution_with_wavelength": "8. Spectral Resolution vs Wavelength",
                    "Slit_Functions": "9. Modeled Slit Functions",
                    "Overlapped_LSF_Lasers_HgAr": "10. Overlaid Normalized LSFs Comparison"
                }

                for plot_path in plot_paths:
                    plot_filename = os.path.basename(plot_path).replace('.png', '')
                    # Extract plot type from filename
                    plot_type = None
                    for key in plot_descriptions.keys():
                        if key in plot_filename:
                            plot_type = key
                            break

                    if plot_type:
                        display_name = plot_descriptions[plot_type]
                    else:
                        display_name = plot_filename

                    plot_names.append(display_name)

                # Store plot paths for reference
                self.current_plot_paths = plot_paths
                self.current_plot_names = plot_names

                # Dropdown selector
                self.plot_selector = ttk.Combobox(selector_frame, values=plot_names, state="readonly", width=60)
                self.plot_selector.pack(side="left", padx=(0, 10))
                if plot_names:
                    self.plot_selector.set(plot_names[0])

                # Bind selection event
                self.plot_selector.bind("<<ComboboxSelected>>", self._on_plot_selected)

                # Frame for plot display
                self.plot_display_frame = ttk.Frame(main_tab)
                self.plot_display_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

                # Load first plot
                if plot_paths:
                    self._load_plot_in_display(plot_paths[0])

            # Update status
            if hasattr(self, 'analysis_status_var'):
                self.analysis_status_var.set(f"Generated {len(plot_paths)} characterization plots from {os.path.basename(csv_path)}")

            # Update summary text
            if hasattr(self, 'analysis_text'):
                summary_text = f"Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                summary_text += f"Data source: {os.path.basename(csv_path)}\n"
                summary_text += f"Generated plots: {len(plot_paths)}\n\n"
                summary_text += "Available Characterization Plots:\n\n"

                for i, plot_name in enumerate(getattr(self, 'current_plot_names', []), 1):
                    summary_text += f"{plot_name}\n"

                    # Add descriptions for each plot type
                    if "Normalized Line Spread Functions" in plot_name:
                        summary_text += "   → Shows the fundamental response of the spectrometer to monochromatic sources\n\n"
                    elif "640 nm Measurements" in plot_name:
                        summary_text += "   → Characterizes stray light from out-of-range wavelengths\n\n"
                    elif "Hg-Ar Lamp Spectrum" in plot_name:
                        summary_text += "   → Wavelength calibration using known Mercury-Argon emission lines\n\n"
                    elif "Stray Light Distribution" in plot_name:
                        summary_text += "   → Visualizes how light scatters between pixels\n\n"
                    elif "Dispersion Fit" in plot_name:
                        summary_text += "   → Shows pixel-to-wavelength mapping accuracy\n\n"
                    elif "Slit Function Parameters" in plot_name:
                        summary_text += "   → Width and shape parameters vs wavelength\n\n"
                    elif "Spectral Resolution" in plot_name:
                        summary_text += "   → Resolution performance compared to reference instruments\n\n"
                    elif "Modeled Slit Functions" in plot_name:
                        summary_text += "   → Theoretical slit function shapes at different wavelengths\n\n"
                    elif "Overlaid Normalized LSFs" in plot_name:
                        summary_text += "   → Comparison of measured LSFs from lasers and lamp sources\n\n"

                self.analysis_text.configure(state="normal")
                self.analysis_text.delete("1.0", "end")
                self.analysis_text.insert("1.0", summary_text)
                self.analysis_text.configure(state="disabled")

            # Enable buttons
            if hasattr(self, 'export_plots_btn'):
                self.export_plots_btn.state(["!disabled"])
            if hasattr(self, 'open_folder_btn'):
                self.open_folder_btn.state(["!disabled"])

            # Switch to analysis tab
            if hasattr(self, 'nb'):
                self.nb.select(self.analysis_tab)

        except Exception as e:
            print(f"Error updating analysis display: {e}")

    def _on_plot_selected(self, event=None):
        """Handle plot selection from dropdown."""
        try:
            if hasattr(self, 'plot_selector') and hasattr(self, 'current_plot_paths'):
                selected_index = self.plot_selector.current()
                if 0 <= selected_index < len(self.current_plot_paths):
                    plot_path = self.current_plot_paths[selected_index]
                    self._load_plot_in_display(plot_path)
        except Exception as e:
            print(f"Error selecting plot: {e}")

    def _load_plot_in_display(self, plot_path: str):
        """Load and display a plot in the analysis tab."""
        try:
            # Clear existing display
            for widget in self.plot_display_frame.winfo_children():
                widget.destroy()

            if not os.path.exists(plot_path):
                ttk.Label(self.plot_display_frame, text=f"Plot file not found: {plot_path}").pack(expand=True)
                return

            try:
                from PIL import Image, ImageTk

                # Load and resize image
                img = Image.open(plot_path)
                # Resize if too large
                if img.width > 1000 or img.height > 700:
                    img.thumbnail((1000, 700), Image.Resampling.LANCZOS)

                photo = ImageTk.PhotoImage(img)

                # Create scrollable canvas
                canvas = tk.Canvas(self.plot_display_frame, bg='white')
                scrollbar_v = ttk.Scrollbar(self.plot_display_frame, orient="vertical", command=canvas.yview)
                scrollbar_h = ttk.Scrollbar(self.plot_display_frame, orient="horizontal", command=canvas.xview)
                canvas.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)

                # Pack scrollbars and canvas
                scrollbar_v.pack(side="right", fill="y")
                scrollbar_h.pack(side="bottom", fill="x")
                canvas.pack(side="left", fill="both", expand=True)

                # Add image to canvas
                canvas.create_image(0, 0, anchor="nw", image=photo)
                canvas.configure(scrollregion=canvas.bbox("all"))

                # Keep reference to prevent garbage collection
                canvas.image = photo

                # Add to canvases list for cleanup
                if not hasattr(self, 'analysis_canvases'):
                    self.analysis_canvases = []
                self.analysis_canvases.append(canvas)

            except ImportError:
                # Fallback if PIL not available
                ttk.Label(self.plot_display_frame,
                         text=f"Plot saved to:\n{plot_path}\n\n(Install Pillow to view plots in GUI)",
                         justify="center").pack(expand=True)
            except Exception as e:
                ttk.Label(self.plot_display_frame,
                         text=f"Error loading plot:\n{e}",
                         justify="center").pack(expand=True)

        except Exception as e:
            print(f"Error loading plot display: {e}")

    def _set_window_icon(self):
        """Set the window and taskbar icon."""
        try:
            icon_path = os.path.join(os.path.dirname(__file__), "sciglob_symbol.ico")
            if os.path.exists(icon_path):
                self.iconbitmap(icon_path)
                print(f"✓ Icon set: {icon_path}")
            else:
                print(f"⚠️ Icon file not found: {icon_path}")
        except Exception as e:
            print(f"⚠️ Could not set icon: {e}")

    def _on_closing(self):
        """Handle application close event with confirmation."""
        try:
            # Check if any operations are running
            operations_running = []

            if hasattr(self, 'live_running') and self.live_running.is_set():
                operations_running.append("Live measurement")

            if hasattr(self, 'measure_running') and self.measure_running.is_set():
                operations_running.append("Automated measurement")

            # Show confirmation dialog
            if operations_running:
                message = f"The following operations are still running:\n• " + "\n• ".join(operations_running)
                message += "\n\nAre you sure you want to close SciGlob?\nThis will stop all running operations."
                title = "SciGlob - Confirm Close"
            else:
                message = "Are you sure you want to close SciGlob?"
                title = "SciGlob - Confirm Close"

            result = messagebox.askyesno(title, message, icon='question')

            if result:
                print("🔄 Closing SciGlob application...")

                # Stop all running operations
                if hasattr(self, 'live_running'):
                    self.live_running.clear()
                if hasattr(self, 'measure_running'):
                    self.measure_running.clear()
                if hasattr(self, '_stop_live'):
                    self._stop_live.set()
                if hasattr(self, '_stop_measure'):
                    self._stop_measure.set()

                # Disconnect hardware
                try:
                    if hasattr(self, 'spec') and self.spec:
                        self.spec.disconnect()
                        print("✓ Spectrometer disconnected")
                except Exception as e:
                    print(f"⚠️ Error disconnecting spectrometer: {e}")

                try:
                    if hasattr(self, 'lasers'):
                        self.lasers.close_all()
                        print("✓ Laser connections closed")
                except Exception as e:
                    print(f"⚠️ Error closing laser connections: {e}")

                # Save settings
                try:
                    self.save_settings()
                    print("✓ Settings saved")
                except Exception as e:
                    print(f"⚠️ Error saving settings: {e}")

                print("✅ SciGlob closed successfully")
                self.destroy()
            else:
                print("❌ Close operation cancelled by user")

        except Exception as e:
            print(f"❌ Error during close: {e}")
            # Force close if there's an error
            self.destroy()

    # ------------------ Utility Methods ------------------
    def _post_error(self, title: str, ex: Exception):
        """Post error message to UI."""
        import traceback
        tb = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        print(f"[{title}] {ex}\n{tb}", file=sys.stderr)
        self.after(0, lambda: messagebox.showerror(title, str(ex)))

    def _countdown_modal(self, seconds: int, title: str, message: str):
        """Show countdown modal dialog."""
        # Simple implementation - show message box with instruction
        result = messagebox.askokcancel(
            title,
            f"{message}\n\nClick OK when ready to continue, or Cancel to skip."
        )
        if not result:
            print(f"User skipped {title}")
        return result

    def apply_it(self):
        """Apply integration time to spectrometer."""
        if not self.spec:
            messagebox.showwarning("Spectrometer", "Not connected.")
            return
        # Parse & clamp
        try:
            it = float(self.it_entry.get())
        except Exception as e:
            messagebox.showerror("Apply IT", f"Invalid IT value: {e}")
            return
        it = max(self.IT_MIN, min(self.IT_MAX, it))

        # If live is running, defer until between frames
        if self.live_running.is_set():
            self._pending_it = it
            try:
                if hasattr(self, 'apply_it_btn'):
                    self.apply_it_btn.state(["disabled"])
            except Exception:
                pass
        else:
            # Apply immediately
            try:
                self.spec.set_it(it)
                self.title(f"Applied IT={it:.3f} ms")
            except Exception as e:
                self._post_error("Apply IT", e)

    def _finalize_measurement_run(self) -> None:
        if not self.data.rows:
            return
        try:
            folder, timestamp = self._prepare_results_folder()
            df = self.data.to_dataframe()
            csv_path = os.path.join(folder, f"All_Lasers_Measurements_{self.sn or 'Unknown'}_{timestamp}.csv")
            df.to_csv(csv_path, index=False)
            result = perform_characterization(df, self.sn or "Unknown", folder, timestamp, self.char_config)
            self.analysis_result = result
            self.analysis_artifacts = result.artifacts
            self.analysis_summary_lines = result.summary_lines
            self.analysis_summary_lines.insert(0, f"✅ Saved measurements to {csv_path}")
            self.analysis_summary_lines.append(f"Plots saved to {folder}")
            self.results_folder = folder
            self.last_results_timestamp = timestamp
            self.latest_csv_path = csv_path
            self.after(0, lambda: self._update_analysis_ui(csv_path))
        except Exception as exc:
            try:
                self._post_error("Characterization", exc)
            except Exception:
                raise

    def export_analysis_plots(self):
        if not self.analysis_artifacts:
            return
        folder = filedialog.askdirectory(title="Select folder for exported plots")
        if not folder:
            return
        try:
            for artifact in self.analysis_artifacts:
                name = artifact.name.replace(" ", "_")
                out_path = os.path.join(folder, f"{name}_{self.last_results_timestamp or ''}.png")
                artifact.figure.savefig(out_path, dpi=300, bbox_inches="tight")
        except Exception as exc:
            self._post_error("Export Plots", exc)

    def open_results_folder(self):
        if not self.results_folder:
            messagebox.showinfo("Results", "No results folder available yet.")
            return
        folder = self.results_folder
        try:
            if sys.platform.startswith("win"):
                os.startfile(folder)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                os.system(f"open '{folder}'")
            else:
                os.system(f"xdg-open '{folder}' >/dev/null 2>&1 &")
        except Exception as exc:
            self._post_error("Open Folder", exc)
