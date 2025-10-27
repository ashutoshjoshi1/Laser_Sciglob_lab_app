#!/usr/bin/env python3
import os
import sys
import time
import json
import queue
import threading
import traceback
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

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
# ============== Utility / Data Classes ============
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
# ============== Device Control Helpers ============
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
            # previously small settle delay removed to comply with user's request

    def read_all_lines(self) -> List[str]:
        self._ensure_open()
        with self.lock:
            resp = self.ser.readlines()
        try:
            return [r.decode(errors="ignore").strip() for r in resp]
        except Exception:
            return []
    def read_all_text(self, wait: float = 0.0) -> str:
        """Read all bytes as text after an optional short wait. wait parameter retained for compatibility."""
        self._ensure_open()
        with self.lock:
            # Removed blocking wait to eliminate delays
            try:
                data = self.ser.read_all()
            except Exception:
                data = b""
        try:
            return data.decode(errors="ignore").strip()
        except Exception:
            return ""


    def clear_input(self):
        self._ensure_open()
        with self.lock:
            try:
                self.ser.reset_input_buffer()
            except Exception:
                pass

    def clear_output(self):
        self._ensure_open()
        with self.lock:
            try:
                self.ser.reset_output_buffer()
            except Exception:
                pass

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
        """Robust CUBE command: enforce CR line endings, drain input, pace, and retry read."""
        try:
            # CUBE expects carriage return (CR) as line ending
            self.cube.eol = "\r"
        except Exception:
            pass

        # Drain any stale buffers so this command's response is clean
        try:
            self.cube.clear_input()
            self.cube.clear_output()
        except Exception:
            pass

        # Write the command
        try:
            self.cube.write_line(cmd)
        except Exception:
            return []

        # Brief processing time, then non-blocking read with a short retry
        import time as _t
        _t.sleep(0.05)
        try:
            resp = self.cube.read_all_text(wait=0.0)
        except Exception:
            resp = ""

        if not resp:
            _t.sleep(0.05)
            try:
                resp = self.cube.read_all_text(wait=0.0)
            except Exception:
                resp = ""

        return [line.strip() for line in resp.splitlines()] if resp else []


    def cube_on(self, power_mw: float = None, current_mA: float = None):
        """Turn on CUBE (377 nm) with small inter-command pacing.
        If power_mw is provided: EXT=1; CW=1; P=<mW>; L=1.
        Else if current_mA provided: EXT=1; CW=1; I=<mA>; L=1.
        """
        try:
            self.cube.eol = "\r"
        except Exception:
            pass

        import time as _t

        def _safe(cmd: str, delay: float = 0.03):
            try:
                self.cube_cmd(cmd)
            except Exception:
                pass
            _t.sleep(delay)

        _safe("EXT=1")
        _safe("CW=1")

        if power_mw is not None:
            try:
                self.cube_cmd(f"P={int(round(power_mw))}")
            except Exception:
                pass
            _t.sleep(0.03)
            self.cube_cmd("L=1")
        elif current_mA is not None:
            try:
                self.cube_cmd(f"I={current_mA:.2f}")
            except Exception:
                pass
            _t.sleep(0.03)
            self.cube_cmd("L=1")
        else:
            _safe("P=12")
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
        # Global thread exception hook to suppress AvaSpec watchdog crashes
        try:
            import threading, traceback
            def _thread_excepthook(args):
                tb = ''.join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
                if 'avantes_spectrometer.py' in tb and 'data_handling_watchdog' in tb:
                    print('[AvaSpec Watchdog] Suppressed thread error:', args.exc_value)
                    # We already handle device stop/disconnect via _handle_avaspec_error
                    return
                # Otherwise, use default handler
                sys.__excepthook__(args.exc_type, args.exc_value, args.exc_traceback)
            threading.excepthook = _thread_excepthook
        except Exception as _eh:
            print('[Init] Could not set threading.excepthook:', _eh)
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
        # Removed scheduled all-off delay; call immediately
        self.after(0, self._all_off_on_start)


     # ---------------- Safety: Avantes -5 handling ----------------
    def _handle_avaspec_error(self, err: Any, context: str = ""):
        """
        Central handler for Avantes spectrometer critical errors.

        The Avantes wrapper sometimes returns integer error codes (e.g. -5).
        If we detect this, we proactively stop live/measure threads and disconnect
        the spectrometer to avoid the app raising the -5 error repeatedly.

        err: either an Exception or an int code.
        context: optional text describing where the error occurred.
        """
        try:
            # Format message
            if isinstance(err, int) and err == -5:
                msg = "Avantes spectrometer returned error code -5 (driver/communication). Stopping operations and disconnecting the spectrometer to prevent further errors."
            elif isinstance(err, Exception):
                msg = f"Avantes spectrometer error in {context}: {err}"
            else:
                msg = f"Avantes spectrometer error in {context}: {err}"

            print(f"[AvaSpec Error] {msg}")

            # Notify user on main thread
            try:
                self.after(0, lambda: messagebox.showerror("Spectrometer Error", msg))
            except Exception:
                pass

            # Stop threaded operations
            try:
                self.live_running.clear()
            except Exception:
                pass
            try:
                self.measure_running.clear()
            except Exception:
                pass

            # Try graceful disconnect
            try:
                if self.spec:
                    try:
                        # If the wrapper has a 'stop' or 'cancel' call try it
                        if hasattr(self.spec, 'stop'):
                            try:
                                self.spec.stop()
                            except Exception:
                                pass
                        # Attempt disconnect
                        try:
                            self.spec.disconnect()
                        except Exception:
                            pass
                    finally:
                        self.spec = None
                if hasattr(self, 'spec_status'):
                    try:
                        self.spec_status.config(text="Disconnected", foreground="red")
                    except Exception:
                        pass
            except Exception as e2:
                print(f"[AvaSpec Error] Error while disconnecting: {e2}")

            # Ensure live UI buttons reflect stopped state
            try:
                if hasattr(self, 'start_live_btn'):
                    self.start_live_btn.state(["!disabled"])
                if hasattr(self, 'stop_live_btn'):
                    self.stop_live_btn.state(["disabled"])
            except Exception:
                pass

        except Exception as final_e:
            print(f"[AvaSpec Error] Unhandled during error handling: {final_e}")

    def _safe_spec_call(self, call_fn, *args, **kwargs):
        """
        Execute a call against self.spec safely. If the wrapper returns an integer
        error code (e.g. -5) or raises an Exception, handle it and return None.

        call_fn: a callable that performs the operation (using self.spec).
        Returns the callable's return value on success, or None on failure.
        """
        if not self.spec:
            return None
        try:
            ret = call_fn(*args, **kwargs)
            # Some wrappers return integer error codes directly OR embed them in strings.
            # Treat Avantes pending-op error (-5) as critical and stop the spectrometer.
            if (isinstance(ret, int) and ret == -5) or (isinstance(ret, str) and "error code -5" in ret.lower()):
                self._handle_avaspec_error(-5, context=getattr(call_fn, "__name__", "spec_call"))
                return None
            return ret
        except Exception as e:
            # Catch exceptions from the wrapper and handle
            # Also detect stringified -5 in the exception text, if present.
            try:
                if hasattr(e, "args") and any(isinstance(a, str) and "error code -5" in a.lower() for a in e.args):
                    self._handle_avaspec_error(-5, context=getattr(call_fn, "__name__", "spec_call"))
                    return None
            except Exception:
                pass
            self._handle_avaspec_error(e, context=getattr(call_fn, "__name__", "spec_call"))
            return None
        
    def call_fn(self, fn_name: str, *args, **kwargs):
        """
        Convenience wrapper to call a spectrometer method by name through
        the app's safe executor. Usage:
            self.call_fn('set_it', 5.0)
            self.call_fn('measure', ncy=1)
            self.call_fn('wait_for_measurement')
        Returns the callable's return value on success, or None on failure.
        """
        if not self.spec or not hasattr(self.spec, fn_name):
            return None
        def _op():
            return getattr(self.spec, fn_name)(*args, **kwargs)
        return self._safe_spec_call(_op)

    # Optional: typed convenience shims (useful in other modules/UI)
    def spec_set_it(self, it_ms: float):
        return self.call_fn('set_it', it_ms)

    def spec_measure(self, **kwargs):
        return self.call_fn('measure', **kwargs)

    def spec_wait(self):
        return self.call_fn('wait_for_measurement')

    def spec_abort(self):
        # best-effort abort; ignore outcome
        try:
            return self.call_fn('abort')
        except Exception:
            return None

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

        self.nb = nb  # keep reference used elsewhere

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
        # The analysis_tab builder should create:
        # - self.analysis_notebook (ttk.Notebook) or we create one later
        # - self.analysis_text (tk.Text)
        # - self.analysis_status_var (tk.StringVar)
        # - self.export_plots_btn, self.open_folder_btn
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
            self.meas_ax.set_ylim(0, 65000)
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
        # Remove previous tab contents and canvases
        if getattr(self, "analysis_canvases", None):
            for entry in getattr(self, "analysis_canvases", []):
                try:
                    # entry might be a FigureCanvasTkAgg or tk.Canvas widget
                    if hasattr(entry, 'get_tk_widget'):
                        entry.get_tk_widget().destroy()
                    elif isinstance(entry, tk.Widget):
                        entry.destroy()
                except Exception:
                    pass
        if getattr(self, "analysis_notebook", None):
            for tab_id in self.analysis_notebook.tabs():
                try:
                    self.analysis_notebook.forget(tab_id)
                except Exception:
                    pass
        self.analysis_canvases = []

    def _update_analysis_ui(self, csv_path: Optional[str] = None) -> None:
        if not hasattr(self, "analysis_notebook"):
            # create one if tab builder didn't
            try:
                self.analysis_notebook = ttk.Notebook(self.analysis_tab)
                self.analysis_notebook.pack(fill="both", expand=True)
            except Exception:
                pass

        self._clear_analysis_notebook()

        if not self.analysis_artifacts:
            if hasattr(self, 'analysis_status_var'):
                self.analysis_status_var.set("Run measurements to generate characterization charts.")
            if hasattr(self, 'analysis_text'):
                self.analysis_text.configure(state="normal")
                self.analysis_text.delete("1.0", "end")
                self.analysis_text.insert("1.0", "No analysis has been generated yet.")
                self.analysis_text.configure(state="disabled")
            if hasattr(self, 'export_plots_btn'):
                self.export_plots_btn.state(["disabled"])
            if hasattr(self, 'open_folder_btn'):
                self.open_folder_btn.state(["disabled"])
            return

        if csv_path is None:
            csv_path = self.latest_csv_path or ""

        status_file = os.path.basename(csv_path) if csv_path else "saved measurements"
        status = f"Analysis generated from {status_file}"
        if self.results_folder:
            status += f" in {self.results_folder}"
        if hasattr(self, 'analysis_status_var'):
            self.analysis_status_var.set(status)

        # If artifacts (Figure objects) are available, show in tabs
        if self.analysis_artifacts:
            for artifact in self.analysis_artifacts:
                frame = ttk.Frame(self.analysis_notebook)
                self.analysis_notebook.add(frame, text=artifact.name)
                canvas = FigureCanvasTkAgg(artifact.figure, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)
                try:
                    NavigationToolbar2Tk(canvas, frame)
                except Exception:
                    pass
                self.analysis_canvases.append(canvas)

        summary_text = "\n".join(self.analysis_summary_lines) if self.analysis_summary_lines else ""
        if hasattr(self, 'analysis_text'):
            self.analysis_text.configure(state="normal")
            self.analysis_text.delete("1.0", "end")
            self.analysis_text.insert("1.0", summary_text or "Characterization completed.")
            self.analysis_text.configure(state="disabled")

        if hasattr(self, 'export_plots_btn'):
            self.export_plots_btn.state(["!disabled"])
        if hasattr(self, 'open_folder_btn'):
            self.open_folder_btn.state(["!disabled"])

    def refresh_analysis_view(self):
        self._update_analysis_ui(self.latest_csv_path)

    # ------------------ Settings Management ------------------
    def load_settings_into_ui(self):
        try:
            if not os.path.isfile(SETTINGS_FILE):
                # Initialize with defaults
                default_settings = {
                    'dll_path': getattr(self.hw, 'dll_path', ''),
                    'com_ports': DEFAULT_COM_PORTS,
                    'power_settings': DEFAULT_POWER_SETTINGS
                }
                
                for key, widget in {
                    'dll': self.dll_entry,
                    'obis': self.obis_entry,
                    'cube': self.cube_entry,
                    'relay': self.relay_entry
                }.items():
                    if hasattr(self, f'{key}_entry'):
                        widget.delete(0, "end")
                        if key == 'dll':
                            widget.insert(0, default_settings['dll_path'])
                        else:
                            widget.insert(0, default_settings['com_ports'][key.upper()])
                            
                if hasattr(self, 'power_entries'):
                    for tag, entry in self.power_entries.items():
                        entry.delete(0, "end")
                        entry.insert(0, str(default_settings['power_settings'].get(tag, 1.0)))
        except Exception as e:
            print(f"Error loading settings: {e}")

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
            
            # ==========================================================
            # ==== MODIFICATION: Disable driver-level saturation abort ===
            ava.abort_on_saturation = False
            # ==========================================================
            
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
            ret = self._safe_spec_call(ava.connect)
            if ret is None and ava is None:
                raise RuntimeError("Failed to connect to spectrometer")

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

        # No delays here per user request

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
                # Start one frame safely
                ret = self._safe_spec_call(lambda: self.spec.measure(ncy=1) if self.spec else None)
                if ret is None and not self.spec:
                    break  # _safe_spec_call already handled error and disconnected

                # Wait for frame to complete
                ret2 = self._safe_spec_call(lambda: self.spec.wait_for_measurement() if self.spec else None)
                if ret2 is None and not self.spec:
                    break

                # Apply any deferred IT safely after the completed frame
                if self._pending_it is not None:
                    try:
                        it_to_apply = self._pending_it
                        self._pending_it = None
                        self._it_updating = True
                        ret3 = self._safe_spec_call(lambda: self.spec.set_it(it_to_apply) if self.spec else None)
                        if ret3 is None and not self.spec:
                            break
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
                try:
                    if not self.spec:
                        break
                    y_raw = getattr(self.spec, 'rcm', None)
                    # Some wrappers may expose rcm as property or method
                    if callable(y_raw):
                        y = np.array(y_raw(), dtype=float)
                    else:
                        y = np.array(self.spec.rcm, dtype=float)
                except Exception as e:
                    self._handle_avaspec_error(e, context="reading rcm in live loop")
                    break

                # Update plot on main thread
                self.after(0, lambda: self._update_live_plot(np.arange(len(y)), y))

                # Removed thread sleep to eliminate delays

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
                    # Replaced countdown blocking modal by immediate prompt (no delay)
                    self._countdown_modal(0, "Fiber Switch", "Switch the fiber to Hg-Ar and press OK to continue.")
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
                self._countdown_modal(0, "Fiber Switch", "Switch the fiber to Hg-Ar and press OK to continue.")
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

                        # Non-blocking prompt (no long countdown)
                        self.after(0, lambda: self._countdown_modal(0, "Fiber Switch", "Switch the fiber to Hg-Ar lamp (press OK when ready)"))

                        self.lasers.relay_on(2)
                        print("Hg-Ar lamp turned ON")
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

                # Auto-adjust integration time with proper delays (non-blocking internal calls)
                success, final_it = self._auto_adjust_integration_time_with_plot(lwl, it_ms)

                if not success:
                    print(f"❌ {lwl} nm: Could not achieve target integration time")
                    # Turn off current laser
                    self._turn_off_laser(lwl)
                    continue

                # Take signal measurement (exactly like characterization script)
                print(f"Taking signal measurement for {lwl} nm at IT={final_it:.1f} ms")

                try:
                    # set IT safely
                    ret_setit = self._safe_spec_call(lambda: self.spec.set_it(final_it) if self.spec else None)
                    if ret_setit is None and not self.spec:
                        break

                    ret_meas = self._safe_spec_call(lambda: self.spec.measure(ncy=N_SIG) if self.spec else None)
                    if ret_meas is None and not self.spec:
                        break

                    ret_wait = self._safe_spec_call(lambda: self.spec.wait_for_measurement() if self.spec else None)
                    if ret_wait is None and not self.spec:
                        break

                    y_signal_raw = getattr(self.spec, 'rcm', None)
                    if callable(y_signal_raw):
                        y_signal = np.array(y_signal_raw(), dtype=float)
                    else:
                        y_signal = np.array(self.spec.rcm, dtype=float)

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

                try:
                    ret_setit2 = self._safe_spec_call(lambda: self.spec.set_it(final_it) if self.spec else None)
                    if ret_setit2 is None and not self.spec:
                        break

                    ret_meas2 = self._safe_spec_call(lambda: self.spec.measure(ncy=N_DARK) if self.spec else None)
                    if ret_meas2 is None and not self.spec:
                        break

                    ret_wait2 = self._safe_spec_call(lambda: self.spec.wait_for_measurement() if self.spec else None)
                    if ret_wait2 is None and not self.spec:
                        break

                    y_dark_raw = getattr(self.spec, 'rcm', None)
                    if callable(y_dark_raw):
                        y_dark = np.array(y_dark_raw(), dtype=float)
                    else:
                        y_dark = np.array(self.spec.rcm, dtype=float)

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
                # set IT safely
                ret_set = self._safe_spec_call(lambda: self.spec.set_it(it_ms) if self.spec else None)
                if ret_set is None and not self.spec:
                    return False, it_ms

                ret_meas = self._safe_spec_call(lambda: self.spec.measure(ncy=1) if self.spec else None)
                if ret_meas is None and not self.spec:
                    return False, it_ms

                ret_wait = self._safe_spec_call(lambda: self.spec.wait_for_measurement() if self.spec else None)
                if ret_wait is None and not self.spec:
                    return False, it_ms

                # retrieve spectrum
                try:
                    y_raw = getattr(self.spec, 'rcm', None)
                    if callable(y_raw):
                        y = np.array(y_raw(), dtype=float)
                    else:
                        y = np.array(self.spec.rcm, dtype=float)
                except Exception as e:
                    self._handle_avaspec_error(e, context="_auto_adjust_integration_time_with_plot: reading rcm")
                    return False, it_ms

                if y.size == 0:
                    print(f"⚠️ {lwl} nm: No data received. Retrying...")
                    adjust_iters += 1
                    if adjust_iters > MAX_IT_ADJUST_ITERS:
                        print(f"❌ {lwl} nm: Gave up (no data).")
                        break
                    continue

                peak = float(np.max(y))
                
                # Update live plot in measurement tab
                self._update_measurement_plot(y, lwl, it_ms, peak)

                # ==========================================================
                # ==== MODIFICATION: Accept saturated peaks as success =====
                
                # Check if peak is high enough (in range OR saturated)
                if peak >= TARGET_LOW:
                    if peak >= SAT_THRESH:
                        print(f"⚠️ {lwl} nm: Saturated peak {peak:.1f} at IT={it_ms:.1f} ms. Accepting.")
                    else:
                        print(f"✅ {lwl} nm: Good peak {peak:.1f} at IT={it_ms:.1f} ms")
                    success = True
                    break # Success, exit loop
                
                # If we are here, the peak is too dim (peak < TARGET_LOW)
                err = TARGET_MID - peak # Error will be positive
                delta = min(IT_STEP_UP, max(0.05, abs(err) / 5000.0))  # ms
                it_ms = min(IT_MAX, it_ms + delta)
                
                # ==========================================================

                adjust_iters += 1
                if adjust_iters > MAX_IT_ADJUST_ITERS:
                    print(f"❌ {lwl} nm: Could not reach target range after {MAX_IT_ADJUST_ITERS} adjustments.")
                    break

            except Exception as e:
                print(f"Error in auto-adjust for {lwl} nm: {e}")
                adjust_iters += 1
                if adjust_iters > MAX_IT_ADJUST_ITERS:
                    break

        return success, it_ms

    def _update_measurement_plot(self, y: np.ndarray, lwl: str, it_ms: float, peak: float):
        """Update measurement plot (like characterization script live plot)."""
        try:
            if hasattr(self, 'measure_line') and hasattr(self, 'measure_ax'):
                x = np.arange(len(y))
                self.measure_line.set_data(x, y)
                self.measure_ax.set_title(f"Live Measurement for {lwl} nm | IT = {it_ms:.1f} ms | peak={peak:.0f}")
                self.measure_ax.relim()
                self.measure_ax.set_ylim(0, 65000)
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

            ret_set = self._safe_spec_call(lambda: self.spec.set_it(it_ms) if self.spec else None)
            if ret_set is None and not self.spec:
                return False, it_ms

            ret_meas = self._safe_spec_call(lambda: self.spec.measure(ncy=1) if self.spec else None)
            if ret_meas is None and not self.spec:
                return False, it_ms

            ret_wait = self._safe_spec_call(lambda: self.spec.wait_for_measurement() if self.spec else None)
            if ret_wait is None and not self.spec:
                return False, it_ms

            try:
                y_raw = getattr(self.spec, 'rcm', None)
                if callable(y_raw):
                    y = np.array(y_raw(), dtype=float)
                else:
                    y = np.array(self.spec.rcm, dtype=float)
            except Exception:
                adjust_iters += 1
                continue

            if y.size == 0:
                adjust_iters += 1
                continue

            peak = float(np.max(y))

            # ==========================================================
            # ==== MODIFICATION: Accept saturated peaks as success =====
            
            # Check if peak is high enough (in range OR saturated)
            if peak >= TARGET_LOW:
                return True, it_ms

            # If we are here, the peak is too dim (peak < TARGET_LOW)
            err = TARGET_MID - peak # Error will be positive
            delta = min(IT_STEP_UP, max(0.05, abs(err) / 5000.0))
            it_ms = min(IT_MAX, it_ms + delta)
            
            # ==========================================================

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
            """Run comprehensive analysis and generate all plots using characterization_analysis.perform_characterization."""
            try:
                # Prepare output folder
                plots_folder = os.path.join(os.path.dirname(csv_path), "plots")
                os.makedirs(plots_folder, exist_ok=True)

                # Load data
                df = pd.read_csv(csv_path)
                if "Wavelength" in df.columns:
                    df["Wavelength"] = df["Wavelength"].astype(str)

                # Identify spectrometer
                sn = getattr(self.spec, 'sn', 'UNKNOWN') if self.spec else 'UNKNOWN'
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                print("🔬 Starting comprehensive analysis via perform_characterization...")
                result = perform_characterization(df, sn, plots_folder, timestamp)

                # Collect saved figure paths
                plot_paths: List[str] = []
                if result and getattr(result, "artifacts", None):
                    for art in result.artifacts:
                        if getattr(art, "path", None) and os.path.isfile(art.path):
                            plot_paths.append(art.path)

                # Update UI with generated plots
                self._update_analysis_display_tabs(plot_paths, csv_path)

                # Write summary text if available
                if result and getattr(result, "summary_lines", None):
                    try:
                        self.analysis_text.configure(state="normal")
                        self.analysis_text.delete("1.0", "end")
                        self.analysis_text.insert("end", "\n".join(result.summary_lines))
                        self.analysis_text.configure(state="disabled")
                    except Exception:
                        pass

                print(f"✅ Analysis complete. {len(plot_paths)} plots saved to: {plots_folder}")
                return plot_paths

            except Exception as e:
                self._post_error("Analysis Error", e)
                return None

    def _update_analysis_display_tabs(self, plot_paths: List[str], csv_path: str):
        """Update analysis tab by creating a separate notebook tab for each generated plot image."""
        try:
            # Ensure we have an analysis notebook reference (created by the analysis tab builder)
            if not hasattr(self, 'analysis_notebook'):
                # If not present, create one inside self.analysis_tab
                self.analysis_notebook = ttk.Notebook(self.analysis_tab)
                self.analysis_notebook.pack(fill="both", expand=True)

            # Clear existing tabs & canvases
            self._clear_analysis_notebook()

            if not plot_paths:
                if hasattr(self, 'analysis_status_var'):
                    self.analysis_status_var.set("No plots were generated. Check measurement data.")
                return

            # Iterate and create one tab per plot image
            plot_names = []
            for p in plot_paths:
                display_name = os.path.basename(p).replace('.png', '')
                plot_names.append(display_name)

                frame = ttk.Frame(self.analysis_notebook)
                self.analysis_notebook.add(frame, text=display_name)

                # Load image into this tab's frame
                try:
                    from PIL import Image, ImageTk
                    img = Image.open(p)
                    # Resize for display while keeping original available for scroll
                    max_w, max_h = 1100, 750
                    if img.width > max_w or img.height > max_h:
                        try:
                            resample = Image.Resampling.LANCZOS
                        except AttributeError:
                            resample = Image.LANCZOS
                        img_thumb = img.copy()
                        img_thumb.thumbnail((max_w, max_h), resample)
                    else:
                        img_thumb = img

                    photo = ImageTk.PhotoImage(img_thumb)

                    # Create canvas + scrollbars so user can view large images
                    canvas = tk.Canvas(frame, bg='white')
                    vsb = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
                    hsb = ttk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
                    canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

                    vsb.pack(side="right", fill="y")
                    hsb.pack(side="bottom", fill="x")
                    canvas.pack(side="left", fill="both", expand=True)

                    # Place the image
                    canvas.create_image(0, 0, anchor="nw", image=photo)
                    canvas.image = photo  # keep reference
                    # Set the scroll region to the image size
                    try:
                        canvas.configure(scrollregion=(0, 0, img_thumb.width, img_thumb.height))
                    except Exception:
                        canvas.configure(scrollregion=canvas.bbox("all"))

                    # Save original file path reference for potential export/use
                    if not hasattr(self, 'analysis_canvases'):
                        self.analysis_canvases = []
                    self.analysis_canvases.append(canvas)

                except ImportError:
                    ttk.Label(frame, text=f"Plot saved to:\n{p}\n\n(Install Pillow to view plots in GUI)", justify="center").pack(expand=True)
                except Exception as e:
                    ttk.Label(frame, text=f"Error loading plot {p}:\n{e}", justify="center").pack(expand=True)

            # Update status
            if hasattr(self, 'analysis_status_var'):
                self.analysis_status_var.set(f"Generated {len(plot_paths)} characterization plots from {os.path.basename(csv_path)}")

            # Update summary text area (descriptions)
            if hasattr(self, 'analysis_text'):
                summary_text = f"Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                summary_text += f"Data source: {os.path.basename(csv_path)}\n"
                summary_text += f"Generated plots: {len(plot_paths)}\n\n"
                summary_text += "Available Characterization Plots:\n\n"

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
                    "Overlapped_LSF_Lasers_HgAr": "10. Overlapped Normalized LSFs Comparison",
                    "Overlapped_LSF_Comparison": "10. Overlapped Normalized LSFs Comparison"
                }

                for name in plot_names:
                    # try to match to known keys
                    desc = None
                    for key, d in plot_descriptions.items():
                        if key in name:
                            desc = d
                            break
                    summary_text += f"{desc or name}\n"
                    if desc:
                        # add short help lines
                        if "Normalized Line Spread Functions" in d:
                            summary_text += "   → Shows the fundamental response of the spectrometer to monochromatic sources\n\n"
                        elif "640 nm Measurements" in d:
                            summary_text += "   → Characterizes stray light from out-of-range wavelengths\n\n"
                        elif "Hg-Ar Lamp Spectrum" in d:
                            summary_text += "   → Wavelength calibration using known Mercury-Argon emission lines\n\n"
                        elif "Stray Light Distribution" in d:
                            summary_text += "   → Visualizes how light scatters between pixels\n\n"
                        elif "Dispersion Fit" in d:
                            summary_text += "   → Shows pixel-to-wavelength mapping accuracy\n\n"
                        elif "Slit Function Parameters" in d:
                            summary_text += "   → Width and shape parameters vs wavelength\n\n"
                        elif "Spectral Resolution" in d:
                            summary_text += "   → Resolution performance compared to reference instruments\n\n"
                        elif "Modeled Slit Functions" in d:
                            summary_text += "   → Theoretical slit function shapes at different wavelengths\n\n"
                        elif "Overlaid Normalized LSFs" in d:
                            summary_text += "   → Comparison of measured LSFs from lasers and lamp sources\n\n"

                self.analysis_text.configure(state="normal")
                self.analysis_text.delete("1.0", "end")
                self.analysis_text.insert("1.0", summary_text)
                self.analysis_text.configure(state="disabled")

            # Enable export/open buttons
            if hasattr(self, 'export_plots_btn'):
                self.export_plots_btn.state(["!disabled"])
            if hasattr(self, 'open_folder_btn'):
                self.open_folder_btn.state(["!disabled"])

            # Switch to analysis tab
            if hasattr(self, 'nb'):
                self.nb.select(self.analysis_tab)

        except Exception as e:
            print(f"Error updating analysis display (tabs): {e}")

    # Deprecated/compatibility placeholder (no dropdown in analysis tab)
    def _on_plot_selected(self, event=None):
        """No-op: dropdown removed in favor of tabbed plot display."""
        pass

    # Keep a helper to load a single plot into an arbitrary frame if needed elsewhere
    def _load_plot_into_frame(self, parent_frame: ttk.Frame, plot_path: str):
        """Load a single PNG plot into the given frame (scrollable)."""
        try:
            for w in parent_frame.winfo_children():
                try:
                    w.destroy()
                except Exception:
                    pass

            if not os.path.exists(plot_path):
                ttk.Label(parent_frame, text=f"Plot file not found: {plot_path}").pack(expand=True)
                return

            try:
                from PIL import Image, ImageTk
                img = Image.open(plot_path)
                max_w, max_h = 1100, 750
                if img.width > max_w or img.height > max_h:
                    try:
                        resample = Image.Resampling.LANCZOS
                    except AttributeError:
                        resample = Image.LANCZOS
                    img_thumb = img.copy()
                    img_thumb.thumbnail((max_w, max_h), resample)
                else:
                    img_thumb = img

                photo = ImageTk.PhotoImage(img_thumb)

                canvas = tk.Canvas(parent_frame, bg='white')
                vsb = ttk.Scrollbar(parent_frame, orient="vertical", command=canvas.yview)
                hsb = ttk.Scrollbar(parent_frame, orient="horizontal", command=canvas.xview)
                canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

                vsb.pack(side="right", fill="y")
                hsb.pack(side="bottom", fill="x")
                canvas.pack(side="left", fill="both", expand=True)

                canvas.create_image(0, 0, anchor="nw", image=photo)
                canvas.image = photo
                try:
                    canvas.configure(scrollregion=(0, 0, img_thumb.width, img_thumb.height))
                except Exception:
                    canvas.configure(scrollregion=canvas.bbox("all"))

            except ImportError:
                ttk.Label(parent_frame,
                         text=f"Plot saved to:\n{plot_path}\n\n(Install Pillow to view plots in GUI)",
                         justify="center").pack(expand=True)
            except Exception as e:
                ttk.Label(parent_frame,
                         text=f"Error loading plot:\n{e}",
                         justify="center").pack(expand=True)

        except Exception as e:
            print(f"Error loading plot into frame: {e}")

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
                    try:
                        self._stop_live.set()
                    except Exception:
                        pass
                if hasattr(self, '_stop_measure'):
                    try:
                        self._stop_measure.set()
                    except Exception:
                        pass

                # Disconnect hardware
                try:
                    if hasattr(self, 'spec') and self.spec:
                        self.spec.disconnect()
                        print("✓ Spectrometer disconnected")
                except Exception as e:
                    print(f"⚠️ Error disconnecting spectrometer: {e}")

                try:
                    if hasattr(self, 'lasers'):
                        # LaserController may not have close_all; be defensive
                        if hasattr(self.lasers, 'close_all'):
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
        # Non-blocking/simple prompt without delay
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
                ret = self._safe_spec_call(lambda: self.spec.set_it(it) if self.spec else None)
                if ret is None and not self.spec:
                    return
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
        if not self.analysis_artifacts and not getattr(self, "results_folder", None):
            # If artifacts aren't available but plots are on disk, allow user to copy them
            folder = filedialog.askdirectory(title="Select folder for exported plots")
            if not folder:
                return
            # Try copying from results_folder if available
            try:
                src_folder = self.results_folder or ""
                if not src_folder:
                    messagebox.showinfo("Export Plots", "No plots available to export.")
                    return
                for fn in os.listdir(src_folder):
                    src = os.path.join(src_folder, fn)
                    if os.path.isfile(src) and fn.lower().endswith(".png"):
                        dst = os.path.join(folder, fn)
                        with open(src, "rb") as rf, open(dst, "wb") as wf:
                            wf.write(rf.read())
                messagebox.showinfo("Export Plots", f"Exported plots to {folder}")
            except Exception as exc:
                self._post_error("Export Plots", exc)
            return

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


    def update_target_peak(self, mid: int):
        """Update the auto-IT target window (LOW/MID/HIGH) at runtime based on a user-set peak.
        Keeps the original target band width (TARGET_HIGH - TARGET_LOW) constant.
        """
        try:
            mid = int(mid)
        except Exception:
            return

        try:
            band = int(self.TARGET_HIGH - self.TARGET_LOW)
        except Exception:
            band = 5000  # default band width if unknown

        half = max(1, band // 2)
        low = max(0, mid - half)
        try:
            sat = int(self.SAT_THRESH)
        except Exception:
            sat = globals().get('SAT_THRESH', 65400)
        high = min(sat, mid + half)

        # Update globals used by functions that reference module-level constants
        g = globals()
        g['TARGET_LOW'] = low
        g['TARGET_HIGH'] = high
        g['TARGET_MID'] = mid

        # Mirror on the instance for code that uses app.TARGET_*
        self.TARGET_LOW = low
        self.TARGET_HIGH = high
        self.TARGET_MID = mid

        # Lightly refresh any helper labels if present
        try:
            if hasattr(self, 'target_band_label'):
                self.target_band_label.config(text=f'Target window: {low}–{high}')
        except Exception:
            pass
