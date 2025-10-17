# Auto-generated from gui.py by splitter
from typing import Any, List, Optional, Tuple, Dict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import os, time, json, sys, traceback
import serial
import serial.tools.list_ports
from avantes_spectrometer import Avantes_Spectrometer
from datetime import datetime
import threading
import os
from datetime import datetime

def build(app):
    # Import constants from app
    DEFAULT_ALL_LASERS = ["405", "445", "488", "517", "532", "377", "Hg_Ar"]
    OBIS_LASER_MAP = {
        "405": 5,
        "445": 4,
        "488": 3,
        "640": 2,
    }

    def _build_measure_tab():
        # Create main container with better layout
        main_frame = ttk.Frame(app.measure_tab)
        main_frame.pack(fill="both", expand=True, padx=12, pady=12)

        # Top section - Live measurement plot (like characterization script)
        plot_frame = ttk.LabelFrame(main_frame, text="Live Measurement Display", padding=10)
        plot_frame.pack(fill="both", expand=True, pady=(0, 12))

        # Add matplotlib plot (exactly like characterization script)
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        import numpy as np

        app.measure_fig = Figure(figsize=(12, 6), dpi=100)
        app.measure_ax = app.measure_fig.add_subplot(111)
        app.measure_ax.set_title("Live Measurement", fontsize=14)
        app.measure_ax.set_xlabel("Pixel Index")
        app.measure_ax.set_ylabel("Counts")
        app.measure_ax.set_xticks(np.arange(0, 2048, 100))
        app.measure_ax.set_ylim(0, 69000)
        app.measure_ax.grid(True)

        # Initialize empty plot line
        app.measure_line, = app.measure_ax.plot(np.zeros(2048), lw=1, color='tab:blue')

        # Add canvas to plot frame
        canvas = FigureCanvasTkAgg(app.measure_fig, plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # Bottom section - Controls in organized layout
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill="x", pady=(0, 0))

        # Left side - Laser selection
        laser_frame = ttk.LabelFrame(controls_frame, text="Laser Selection", padding=10)
        laser_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))

        app.measure_vars = {}
        # Use the same laser list as characterization script
        all_lasers = ["532", "445", "405", "377", "Hg_Ar"]  # From characterization script

        # Create a grid layout for laser checkboxes
        for i, tag in enumerate(all_lasers):
            v = tk.BooleanVar(value=True)  # Default all selected like characterization script
            chk = ttk.Checkbutton(laser_frame, text=f"{tag} nm", variable=v)
            chk.grid(row=i // 3, column=i % 3, padx=8, pady=4, sticky="w")
            app.measure_vars[tag] = v

        # Middle - Settings
        settings_frame = ttk.LabelFrame(controls_frame, text="Settings", padding=10)
        settings_frame.pack(side="left", fill="y", padx=6)

        ttk.Label(settings_frame, text="Auto-IT start (ms):").pack(anchor="w", pady=(0, 4))
        app.auto_it_entry = ttk.Entry(settings_frame, width=12)
        app.auto_it_entry.insert(0, "")
        app.auto_it_entry.pack(anchor="w", pady=(0, 8))

        ttk.Label(settings_frame, text="(Leave blank for defaults)",
                 font=("TkDefaultFont", 8)).pack(anchor="w")

        # Right side - Action buttons
        actions_frame = ttk.LabelFrame(controls_frame, text="Actions", padding=10)
        actions_frame.pack(side="right", fill="y", padx=(6, 0))

        # Create buttons with better styling
        button_style = {"width": 15}

        app.run_all_btn = ttk.Button(actions_frame, text="▶ Run Selected",
                                   command=app.run_all_selected, **button_style)
        app.run_all_btn.pack(pady=(0, 6))

        app.stop_all_btn = ttk.Button(actions_frame, text="⏹ Stop",
                                    command=app.stop_measure, **button_style)
        app.stop_all_btn.pack(pady=(0, 6))

        app.save_csv_btn = ttk.Button(actions_frame, text="💾 Save CSV",
                                    command=app.save_csv, **button_style)
        app.save_csv_btn.pack(pady=(0, 6))

        app.start_analysis_btn = ttk.Button(actions_frame, text="📊 Analysis",
                                          command=app.refresh_analysis_view, **button_style)
        app.start_analysis_btn.pack()



    def run_all_selected():
        if not app.spec:
            messagebox.showwarning("Spectrometer", "Not connected.")
            return
        if app.measure_running.is_set():
            return
        tags = [t for t, v in app.measure_vars.items() if v.get()]
        if not tags:
            messagebox.showwarning("Run", "No lasers selected.")
            return
        start_it_override = None
        try:
            txt = app.auto_it_entry.get().strip()
            if txt:
                start_it_override = float(txt)
        except:
            start_it_override = None

        # Clear previous data
        app.data.rows.clear()

        app.measure_running.set()
        app.measure_thread = threading.Thread(
            target=run_measurement_with_analysis, args=(tags, start_it_override), daemon=True)
        app.measure_thread.start()

    def run_measurement_with_analysis(tags, start_it_override):
        """Run measurement sequence and automatically generate analysis."""
        try:
            # Run the measurement sequence
            app._measure_sequence_thread(tags, start_it_override)

            # Save data to CSV
            csv_path = app.save_measurement_data()
            if csv_path:
                # Generate analysis plots
                plot_paths = app.run_analysis_and_save_plots(csv_path)

                # Show completion message
                app.after(0, lambda: messagebox.showinfo(
                    "Measurement Complete",
                    f"Measurement and analysis complete!\n\n"
                    f"Data saved to: {os.path.basename(csv_path)}\n"
                    f"Generated {len(plot_paths)} analysis plots.\n\n"
                    f"Check the Analysis tab for results."
                ))
        except Exception as e:
            app._post_error("Measurement", e)

    def stop_measure():
        app.measure_running.clear()

    def _measure_sequence_thread(self, laser_tags: List[str], start_it_override: Optional[float]):
        # Make sure ports reflect UI and are open for the run
        try:
            app._update_ports_from_ui()
            app.lasers.open_all()
        except Exception as e:
            app._post_error("Ports", e)
            app.measure_running.clear()
            return

        # Ensure everything OFF initially (auto-opens as needed)
        try:
            for ch in OBIS_LASER_MAP.values():
                try: app.lasers.obis_off(ch)
                except: pass
            app.lasers.cube_off()
            app.lasers.relay_off(1)  # 532
            app.lasers.relay_off(2)  # Hg-Ar
            app.lasers.relay_off(3)  # 517
        except Exception:
            pass

        main_tags = [t for t in laser_tags if t != "640"]
        do_640 = "640" in laser_tags

        for tag in main_tags:
            if not app.measure_running.is_set():
                break
            try:
                app._run_single_measurement(tag, start_it_override)
            except Exception as e:
                app._post_error(f"Measurement {tag}", e)

        if do_640 and app.measure_running.is_set():
            try:
                app._run_640_measurement()
            except Exception as e:
                app._post_error("640 nm Measurement", e)

        # Turn all off at the end
        try:
            for ch in OBIS_LASER_MAP.values():
                try: app.lasers.obis_off(ch)
                except: pass
            app.lasers.cube_off()
            app.lasers.relay_off(1)
            app.lasers.relay_off(2)
            app.lasers.relay_off(3)
        except Exception:
            pass

        try:
            app._finalize_measurement_run()
        except Exception as e:
            app._post_error("Finalize Measurements", e)

        app.measure_running.clear()

    def _auto_adjust_it(self, start_it: float, tag: str) -> Tuple[float, float]:
        it_ms = max(app.IT_MIN, min(app.IT_MAX, start_it))
        peak = np.nan
        iters = 0
        app.it_history = []

        def keep_running() -> bool:
            return not hasattr(app, "measure_running") or app.measure_running.is_set()

        while iters <= app.MAX_IT_ADJUST_ITERS and keep_running():
            app.spec.set_it(it_ms)
            app.spec.measure(ncy=1)
            app.spec.wait_for_measurement()
            y = np.array(app.spec.rcm, dtype=float)
            if y.size == 0:
                iters += 1
                continue

            peak = float(np.nanmax(y))
            app.it_history.append((it_ms, peak))
            app.after(0, lambda arr=y.copy(), it_val=it_ms, pk=peak, tg=tag: app._update_auto_it_plot(tg, arr, it_val, pk))

            if peak >= app.SAT_THRESH:
                it_ms = max(app.IT_MIN, it_ms * 0.7)
                iters += 1
                continue

            if app.TARGET_LOW <= peak <= app.TARGET_HIGH:
                return it_ms, peak

            err = app.TARGET_MID - peak
            if err > 0:
                delta = min(app.IT_STEP_UP, max(0.05, abs(err) / 5000.0))
                it_ms = min(app.IT_MAX, it_ms + delta)
            else:
                delta = min(app.IT_STEP_DOWN, max(0.05, abs(err) / 5000.0))
                it_ms = max(app.IT_MIN, it_ms - delta)

            iters += 1

        return it_ms, peak


    def _ensure_source_state(self, tag: str, turn_on: bool):
        """Turn on/off source described by tag with port auto-open."""
        # ensure correct device port is open
        app.lasers.ensure_open_for_tag(tag)

        if tag in OBIS_LASER_MAP:
            ch = OBIS_LASER_MAP[tag]
            if turn_on:
                pwr = float(app._get_power(tag))
                app.lasers.obis_set_power(ch, pwr)
                app.lasers.obis_on(ch)
            else:
                app.lasers.obis_off(ch)

        elif tag == "377":
            if turn_on:
                val = float(app._get_power(tag))
                mw = val * 1000.0 if val <= 0.3 else val
                app.lasers.cube_on(power_mw=mw)
            else:
                app.lasers.cube_off()

        elif tag == "517":
            if turn_on: app.lasers.relay_on(3)
            else:       app.lasers.relay_off(3)

        elif tag == "532":
            if turn_on: app.lasers.relay_on(1)
            else:       app.lasers.relay_off(1)

        elif tag == "Hg_Ar":
            if turn_on:
                app._countdown_modal(45, "Fiber Switch", "Switch the fiber to Hg-Ar and press Enter to skip.")
                app.lasers.relay_on(2)
            else:
                app.lasers.relay_off(2)



    def _run_single_measurement(self, tag: str, start_it_override: Optional[float]):
        # Turn on only the requested tag; others off
        for k in ["377", "517", "532", "Hg_Ar"]:
            if k != tag:
                try:
                    app._ensure_source_state(k, False)
                except:
                    pass
        for k, ch in OBIS_LASER_MAP.items():
            if k != tag:
                try: app.lasers.obis_off(ch)
                except: pass

        app._ensure_source_state(tag, True)
        time.sleep(1.0)  # allow source to stabilize

        # pick start IT
        start_it = start_it_override if start_it_override is not None else app.DEFAULT_START_IT.get(tag, app.DEFAULT_START_IT["default"])
        # Auto-IT
        it_ms, peak = app._auto_adjust_it(start_it, tag)

        if app.TARGET_LOW <= peak <= app.TARGET_HIGH:
            # Signal
            app.spec.set_it(it_ms)
            app.spec.measure(ncy=app.N_SIG)
            app.spec.wait_for_measurement()
            y_signal = np.array(app.spec.rcm, dtype=float)

            # Turn OFF tag
            app._ensure_source_state(tag, False)

            # Dark
            time.sleep(0.3)
            app.spec.set_it(it_ms)
            app.spec.measure(ncy=app.N_DARK)
            app.spec.wait_for_measurement()
            y_dark = np.array(app.spec.rcm, dtype=float)

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            app.data.rows.append([now, tag, it_ms, app.N_SIG] + y_signal.tolist())
            app.data.rows.append([now, f"{tag}_dark", it_ms, app.N_DARK] + y_dark.tolist())

            app._update_last_plots(tag)
        else:
            # could not reach target -> just turn off
            app._ensure_source_state(tag, False)

    def _run_640_measurement():
        if not app.measure_running.is_set():
            return

        integration_times = [100.0, 500.0, 1000.0]

        try:
            app._ensure_source_state("640", True)
            time.sleep(3.0)

            for it_ms in integration_times:
                if not app.measure_running.is_set():
                    break
                app.spec.set_it(it_ms)
                app.spec.measure(ncy=app.N_SIG_640)
                app.spec.wait_for_measurement()
                y = np.array(app.spec.rcm, dtype=float)
                peak = float(np.nanmax(y)) if y.size else 0.0
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                app.data.rows.append([now, "640", it_ms, app.N_SIG_640] + y.tolist())
                app.after(0, lambda arr=y.copy(), it_val=it_ms, pk=peak: app._update_auto_it_plot("640", arr, it_val, pk))

            app._ensure_source_state("640", False)
            time.sleep(0.3)

            for it_ms in integration_times:
                if not app.measure_running.is_set():
                    break
                app.spec.set_it(it_ms)
                app.spec.measure(ncy=app.N_DARK_640)
                app.spec.wait_for_measurement()
                y_dark = np.array(app.spec.rcm, dtype=float)
                now_dark = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                app.data.rows.append([now_dark, "640_dark", it_ms, app.N_DARK_640] + y_dark.tolist())
        finally:
            app._ensure_source_state("640", False)

        app._update_last_plots("640")

    def _countdown_modal(self, seconds: int, title: str, message: str):
        """Blocking modal with countdown; Enter key to skip."""
        top = tk.Toplevel(self)
        top.title(title)
        top.geometry("500x180+200+200")
        ttk.Label(top, text=message, wraplength=460).pack(pady=8)
        lbl = ttk.Label(top, text="", font=("Segoe UI", 14))
        lbl.pack(pady=10)

        skip = {"flag": False}
        def on_key(ev):
            skip["flag"] = True
        top.bind("<Return>", on_key)

        for s in range(seconds, -1, -1):
            if skip["flag"]:
                break
            lbl.config(text=f"{s} sec")
            top.update()
            time.sleep(1.0)
        top.destroy()

    def _update_last_plots(self, tag: str):
        sig, dark = app.data.last_vectors_for(tag)

        def update():
            # main overlay
            xmax = 10
            ymax = 1000.0
            if sig is not None:
                xs = np.arange(len(sig))
                app.meas_sig_line.set_data(xs, sig)
                xmax = max(xmax, len(sig)-1)
                ymax = max(ymax, float(np.nanmax(sig))*1.1)
            if dark is not None:
                xd = np.arange(len(dark))
                app.meas_dark_line.set_data(xd, dark)
                xmax = max(xmax, len(dark)-1)
                ymax = max(ymax, float(np.nanmax(dark))*1.1)

            app.meas_ax.set_xlim(0, xmax)
            app.meas_ax.set_ylim(0, ymax)

            # inset: Auto-IT step history (peaks & IT)
            steps = list(app.it_history) if hasattr(self, "it_history") else []
            if steps:
                st = np.arange(len(steps))
                peaks = [p for (_, p) in steps]
                its   = [it for (it, _) in steps]  # note order in tuple

                app.meas_inset.set_xlim(-0.5, len(st)-0.5 if len(st) else 0.5)
                # Update data
                app.inset_peak_line.set_data(st, peaks)
                app.inset_it_line.set_data(st, its)

                # Rescale both y-axes
                app.meas_inset.relim();  app.meas_inset.autoscale_view()
                app.meas_inset2.relim(); app.meas_inset2.autoscale_view()
            else:
                # clear inset
                app.inset_peak_line.set_data([], [])
                app.inset_it_line.set_data([], [])
                app.meas_inset.relim();  app.meas_inset.autoscale_view()
                app.meas_inset2.relim(); app.meas_inset2.autoscale_view()

            app.meas_canvas.draw_idle()

        app.after(0, update)


    def save_csv():
        if not app.data.rows:
            messagebox.showwarning("Save CSV", "No data collected yet.")
            return
        df = app.data.to_dataframe()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default = f"Measurements_{app.data.serial_number}_{ts}.csv"
        path = filedialog.asksaveasfilename(
            title="Save CSV", defaultextension=".csv",
            initialfile=default, filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not path:
            return
        try:
            df.to_csv(path, index=False)
            messagebox.showinfo("Save CSV", f"Saved: {path}")
        except Exception as e:
            messagebox.showerror("Save CSV", str(e))

    # ------------------ Analysis Tab --------------------

    def _build_analysis_tab():
        top = ttk.Frame(app.analysis_tab)
        top.pack(fill="x", padx=8, pady=8)

        ttk.Label(top, text="Analysis Type:").pack(side="left")
        app.analysis_type = tk.StringVar(value="LSF")
        combo = ttk.Combobox(top, textvariable=app.analysis_type, width=18,
                             values=["LSF", "Dispersion", "Stray Light", "Resolution"])
        combo.pack(side="left", padx=8)

        ttk.Label(top, text="Wavelength tag (e.g., 405, 445, Hg_Ar):").pack(side="left", padx=(16, 4))
        app.analysis_tag_entry = ttk.Entry(top, width=15)
        app.analysis_tag_entry.insert(0, "Hg_Ar")
        app.analysis_tag_entry.pack(side="left", padx=4)

        ttk.Button(top, text="Run Analysis", command=app.run_analysis).pack(side="left", padx=12)

        ttk.Button(top, text="Export Plots", command=app.export_analysis_plots).pack(side="right")
        ttk.Button(top, text="Export Summary", command=app.export_analysis_summary).pack(side="right", padx=6)

        mid = ttk.Frame(app.analysis_tab)
        mid.pack(fill="both", expand=True, padx=8, pady=8)

        app.ana_fig1 = Figure(figsize=(7, 4), dpi=100)
        app.ana_ax1 = app.ana_fig1.add_subplot(111)
        app.ana_ax1.set_title("Analysis Plot 1")
        app.ana_ax1.grid(True)
        app.ana_canvas1 = FigureCanvasTkAgg(app.ana_fig1, mid)
        app.ana_canvas1.draw()
        app.ana_canvas1.get_tk_widget().pack(side="left", fill="both", expand=True)
        NavigationToolbar2Tk(app.ana_canvas1, mid)

        app.ana_fig2 = Figure(figsize=(7, 4), dpi=100)
        app.ana_ax2 = app.ana_fig2.add_subplot(111)
        app.ana_ax2.set_title("Analysis Plot 2")
        app.ana_ax2.grid(True)
        app.ana_canvas2 = FigureCanvasTkAgg(app.ana_fig2, mid)
        app.ana_canvas2.draw()
        app.ana_canvas2.get_tk_widget().pack(side="right", fill="both", expand=True)
        NavigationToolbar2Tk(app.ana_canvas2, mid)

        bottom = ttk.Frame(app.analysis_tab)
        bottom.pack(fill="x", padx=8, pady=8)
        ttk.Label(bottom, text="Analysis Summary:").pack(anchor="w")
        app.analysis_text = tk.Text(bottom, height=8)
        app.analysis_text.pack(fill="x", expand=True)


    def start_analysis_from_measure():
        """Start a measurement run for the lasers selected in the Measurement tab
        (analysis selection if available), and paint the resulting LSF/other plots in the Analysis tab.
        """
        if not app.spec:
            messagebox.showwarning("Spectrometer", "Not connected.")
            return
        tags = [t for t, v in getattr(self, 'analysis_vars', {}).items() if v.get()] or \
               [t for t, v in app.measure_vars.items() if v.get()]
        if not tags:
            messagebox.showwarning("Analysis", "No lasers selected to analyze.")
            return
        try:
            app.ana_ax1.clear(); app.ana_ax1.grid(True)
            app.ana_ax2.clear(); app.ana_ax2.grid(True)
            app.analysis_text.delete('1.0', 'end')
        except Exception:
            pass
        app.measure_running.set()
        def _runner():
            try:
                app._measure_sequence_thread(tags, None)
            finally:
                try:
                    app.nb.select(app.analysis_tab)
                except Exception:
                    pass
        app.measure_thread = threading.Thread(target=_runner, daemon=True)
        app.measure_thread.start()
    def run_analysis():
        if not app.data.rows:
            messagebox.showwarning("Analysis", "No measurement data available.")
            return
        df = app.data.to_dataframe()
        atype = app.analysis_type.get()
        tag = app.analysis_tag_entry.get().strip() or "Hg_Ar"
        app.analysis_text.delete("1.0", "end")
        app.ana_ax1.clear(); app.ana_ax1.grid(True)
        app.ana_ax2.clear(); app.ana_ax2.grid(True)

        try:
            if atype == "LSF":
                lsf = normalized_lsf_from_df(df, tag)
                if lsf is None:
                    raise RuntimeError("Could not compute LSF (missing rows or saturated).")
                x = np.arange(len(lsf))
                app.ana_ax1.set_title(f"LSF (normalized) - {tag}")
                app.ana_ax1.plot(x, lsf)
                # center/peak
                peak_pix = int(np.nanargmax(lsf))
                app.ana_ax2.set_title("Zoom near peak")
                lo = max(0, peak_pix - 50); hi = min(len(lsf), peak_pix + 50)
                app.ana_ax2.plot(np.arange(lo, hi), lsf[lo:hi])
                app.analysis_text.insert("end", f"LSF computed for {tag}\nPeak Pixel: {peak_pix}\n")

            elif atype == "Dispersion":
                # Use Hg_Ar (or tag) to find peaks, then match to known lines
                sig, _ = app.data.last_vectors_for(tag)
                if sig is None:
                    raise RuntimeError(f"No '{tag}' signal found.")
                # find prominent peaks
                peaks, _ = find_peaks(sig, height=np.nanmax(sig)*0.2, distance=5)
                peaks = np.sort(peaks)
                # attempt match with known lines
                sol = best_ordered_linear_match(peaks, KNOWN_HG_AR_NM, min_points=4)
                if not sol:
                    raise RuntimeError("Could not fit linear dispersion to known lines.")
                rmse, a, b, pix_sel, wl_sel = sol
                wl_pred = a * np.arange(len(sig)) + b
                app.ana_ax1.set_title("Dispersion Mapping (nm vs pixel)")
                app.ana_ax1.plot(np.arange(len(sig)), wl_pred, lw=1)
                app.ana_ax2.set_title("Peak Match")
                app.ana_ax2.plot(pix_sel, wl_sel, "o")
                app.analysis_text.insert("end", f"Dispersion fit: wl = {a:.6f}*pix + {b:.3f}\nRMSE: {rmse:.3f} nm\n")
                app.analysis_text.insert("end", f"Used peaks: {pix_sel.tolist()}\nMapped to: {wl_sel.tolist()}\n")

            elif atype == "Stray Light":
                lsf = normalized_lsf_from_df(df, tag)
                if lsf is None:
                    raise RuntimeError("Could not compute LSF for stray light.")
                peak_pix = int(np.nanargmax(lsf))
                metrics = stray_light_metrics(lsf, peak_pix, ib_half=IB_REGION_HALF)
                app.ana_ax1.set_title("LSF (normalized)")
                app.ana_ax1.plot(np.arange(len(lsf)), lsf)
                app.ana_ax2.set_title("Bands")
                lo = max(0, peak_pix-50); hi = min(len(lsf), peak_pix+50)
                app.ana_ax2.plot(np.arange(lo, hi), lsf[lo:hi])
                app.analysis_text.insert("end", "Stray Light Metrics:\n")
                for k, v in metrics.items():
                    app.analysis_text.insert("end", f"  {k}: {v:.6g}\n")

            elif atype == "Resolution":
                # FWHM near a peak; if dispersion fit known, you can convert to nm
                tag_use = tag
                sig, dark = app.data.last_vectors_for(tag_use)
                if sig is None or dark is None:
                    raise RuntimeError("Missing signal/dark for resolution.")
                y = (sig - dark).astype(float)
                y -= np.nanmin(y)
                if np.nanmax(y) <= 0:
                    raise RuntimeError("Flat/invalid spectrum for resolution.")
                y /= np.nanmax(y)
                xpix = np.arange(len(y))
                fwhm_pix = compute_fwhm(xpix, y)
                # try to estimate nm per pixel via simple two-line fit if available
                peaks, _ = find_peaks(y, height=0.2, distance=5)
                nm_per_pix = np.nan
                if len(peaks) >= 4:
                    sol = best_ordered_linear_match(peaks, KNOWN_HG_AR_NM, min_points=4)
                    if sol:
                        _, a, b, _, _ = sol
                        nm_per_pix = a
                app.ana_ax1.set_title("Signal - Dark (normalized)")
                app.ana_ax1.plot(xpix, y)
                app.ana_ax2.set_title("Peak Zoom")
                if len(peaks) > 0:
                    p0 = int(peaks[np.argmax(y[peaks])])
                    lo = max(0, p0-50); hi = min(len(y), p0+50)
                    app.ana_ax2.plot(np.arange(lo, hi), y[lo:hi])
                txt = f"FWHM ≈ {fwhm_pix:.3f} pixels"
                if np.isfinite(nm_per_pix):
                    txt += f"  (~{fwhm_pix*nm_per_pix:.3f} nm with slope {nm_per_pix:.6f} nm/pixel)"
                app.analysis_text.insert("end", txt + "\n")

            else:
                raise RuntimeError(f"Unknown analysis type '{atype}'")

        except Exception as e:
            app._post_error("Analysis Error", e)

        app.ana_canvas1.draw()
        app.ana_canvas2.draw()

    def export_analysis_plots():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = filedialog.askdirectory(title="Select folder to save analysis plots")
        if not base:
            return
        try:
            p1 = os.path.join(base, f"analysis_plot1_{ts}.png")
            p2 = os.path.join(base, f"analysis_plot2_{ts}.png")
            app.ana_fig1.savefig(p1, dpi=150, bbox_inches="tight")
            app.ana_fig2.savefig(p2, dpi=150, bbox_inches="tight")
            messagebox.showinfo("Export Plots", f"Saved:\n{p1}\n{p2}")
        except Exception as e:
            messagebox.showerror("Export Plots", str(e))

    def export_analysis_summary():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default = f"analysis_summary_{ts}.txt"
        path = filedialog.asksaveasfilename(
            title="Save Summary", defaultextension=".txt",
            initialfile=default, filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if not path:
            return
        try:
            txt = app.analysis_text.get("1.0", "end").strip()
            with open(path, "w", encoding="utf-8") as f:
                f.write(txt + "\n")
            messagebox.showinfo("Export Summary", f"Saved: {path}")
        except Exception as e:
            messagebox.showerror("Export Summary", str(e))

    # ------------------ Setup Tab -----------------------

    def _build_setup_tab():
        frame = ttk.Frame(app.setup_tab)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Spectrometer block
        spec_group = ttk.LabelFrame(frame, text="Spectrometer")
        spec_group.pack(fill="x", padx=6, pady=6)

        ttk.Label(spec_group, text="DLL Path (avaspecx64.dll):").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        app.dll_entry = ttk.Entry(spec_group, width=60)
        app.dll_entry.grid(row=0, column=1, sticky="w", padx=4, pady=4)
        ttk.Button(spec_group, text="Browse", command=app.browse_dll).grid(row=0, column=2, padx=4, pady=4)

        app.spec_status = ttk.Label(spec_group, text="Disconnected", foreground="red")
        app.spec_status.grid(row=0, column=3, padx=10)

        ttk.Button(spec_group, text="Connect", command=app.connect_spectrometer).grid(row=1, column=1, padx=4, pady=4, sticky="w")
        ttk.Button(spec_group, text="Disconnect", command=app.disconnect_spectrometer).grid(row=1, column=2, padx=4, pady=4, sticky="w")

        # COM ports
        ports_group = ttk.LabelFrame(frame, text="COM Port Configuration")
        ports_group.pack(fill="x", padx=6, pady=6)

        ttk.Label(ports_group, text="OBIS:").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        ttk.Label(ports_group, text="CUBE:").grid(row=1, column=0, sticky="e", padx=4, pady=4)
        ttk.Label(ports_group, text="RELAY:").grid(row=2, column=0, sticky="e", padx=4, pady=4)

        app.obis_entry = ttk.Entry(ports_group, width=12)
        app.cube_entry = ttk.Entry(ports_group, width=12)
        app.relay_entry = ttk.Entry(ports_group, width=12)
        app.obis_entry.grid(row=0, column=1, padx=4, pady=4, sticky="w")
        app.cube_entry.grid(row=1, column=1, padx=4, pady=4, sticky="w")
        app.relay_entry.grid(row=2, column=1, padx=4, pady=4, sticky="w")

        ttk.Button(ports_group, text="Refresh Ports", command=app.refresh_ports).grid(row=0, column=2, padx=6)
        ttk.Button(ports_group, text="Test Connect", command=app.test_com_connect).grid(row=1, column=2, padx=6)

        app.obis_status = ttk.Label(ports_group, text="●", foreground="red")
        app.cube_status = ttk.Label(ports_group, text="●", foreground="red")
        app.relay_status = ttk.Label(ports_group, text="●", foreground="red")
        app.obis_status.grid(row=0, column=3, padx=4)
        app.cube_status.grid(row=1, column=3, padx=4)
        app.relay_status.grid(row=2, column=3, padx=4)

        # Laser power config
        power_group = ttk.LabelFrame(frame, text="Laser Power Configuration")
        power_group.pack(fill="x", padx=6, pady=6)

        app.power_entries: Dict[str, ttk.Entry] = {}
        row = 0
        for tag in ["405", "445", "488", "640", "377", "517", "532", "Hg_Ar"]:
            ttk.Label(power_group, text=f"{tag} nm power:").grid(row=row, column=0, sticky="e", padx=4, pady=2)
            e = ttk.Entry(power_group, width=12)
            e.insert(0, str(app.DEFAULT_LASER_POWERS.get(tag, 0.01)))
            e.grid(row=row, column=1, sticky="w", padx=4, pady=2)
            app.power_entries[tag] = e
            row += 1

        # Save/Load
        save_group = ttk.Frame(frame)
        save_group.pack(fill="x", padx=6, pady=8)
        ttk.Button(save_group, text="Save Settings", command=app.save_settings).pack(side="left")
        ttk.Button(save_group, text="Load Settings", command=app.load_settings_into_ui).pack(side="left", padx=6)

    def refresh_ports():
        ports = list(serial.tools.list_ports.comports())
        names = [p.device for p in ports]
        if names:
            app.obis_entry.delete(0, "end"); app.obis_entry.insert(0, names[0])
            if len(names) > 1:
                app.cube_entry.delete(0, "end"); app.cube_entry.insert(0, names[1])
            if len(names) > 2:
                app.relay_entry.delete(0, "end"); app.relay_entry.insert(0, names[2])
            messagebox.showinfo("Ports", "Populated with first detected ports.\nAdjust as needed.")
        else:
            messagebox.showwarning("Ports", "No serial ports detected.")

    def test_com_connect():
        app._update_ports_from_ui()
        ok_obis = app.lasers.obis.open()
        app.obis_status.config(foreground=("green" if ok_obis else "red"))
        ok_cube = app.lasers.cube.open()
        app.cube_status.config(foreground=("green" if ok_cube else "red"))
        ok_relay = app.lasers.relay.open()
        app.relay_status.config(foreground=("green" if ok_relay else "red"))
        # Close after test to free ports (or keep open if you prefer)
        time.sleep(0.2)
        if ok_obis: app.lasers.obis.close()
        if ok_cube: app.lasers.cube.close()
        if ok_relay: app.lasers.relay.close()

    def browse_dll():
        path = filedialog.askopenfilename(
            title="Select avaspecx64.dll", filetypes=[("DLL", "*.dll"), ("All files", "*.*")])
        if path:
            app.dll_entry.delete(0, "end")
            app.dll_entry.insert(0, path)

    def connect_spectrometer():
        try:
            dll = app.dll_entry.get().strip()
            if not dll or not os.path.isfile(dll):
                raise RuntimeError("Please select a valid avaspecx64.dll")
            # init wrapper
            ava = Avantes_Spectrometer()
            ava.dll_path = dll
            ava.alias = "Ava1"
            ava.npix_active = 2048
            ava.debug_mode = 1
            ava.initialize_spec_logger()

            res = ava.load_spec_dll()
            if res != "OK":
                raise RuntimeError(f"load_spec_dll returned: {res}")
            res = ava.initialize_dll()
            # enumerate
            res, ndev = ava.get_number_of_devices()
            if res != "OK" or ndev <= 0:
                raise RuntimeError("No Avantes devices detected.")
            res, infos = ava.get_all_devices_info(ndev)
            # pick first SN if available, else rely on wrapper default
            try:
                sers = []
                for i in range(ndev):
                    ident = getattr(infos, f"a{i}")
                    sn = ident.SerialNumber
                    if isinstance(sn, (bytes, bytearray)):
                        sn = sn.decode("utf-8", errors="ignore")
                    sers.append(sn)
                if sers:
                    ava.sn = sers[0]
            except Exception:
                pass

            ava.connect()
            app.spec = ava
            app.sn = getattr(ava, "sn", "Unknown")
            app.data.serial_number = app.sn
            app.npix = getattr(ava, "npix_active", app.npix)
            app.data.npix = app.npix

            app.spec_status.config(text=f"Connected: {app.sn}", foreground="green")
            messagebox.showinfo("Spectrometer", f"Connected to SN={app.sn}")
        except Exception as e:
            app.spec = None
            app.spec_status.config(text="Disconnected", foreground="red")
            app._post_error("Spectrometer Connect", e)

    def disconnect_spectrometer():
        try:
            app.stop_live()
            if app.spec:
                try:
                    app.spec.disconnect()
                except Exception:
                    pass
            app.spec = None
            app.spec_status.config(text="Disconnected", foreground="red")
        except Exception as e:
            app._post_error("Spectrometer Disconnect", e)

    def _update_ports_from_ui():
        app.hw.com_ports["OBIS"] = app.obis_entry.get().strip() or app.DEFAULT_COM_PORTS["OBIS"]
        app.hw.com_ports["CUBE"] = app.cube_entry.get().strip() or app.DEFAULT_COM_PORTS["CUBE"]
        app.hw.com_ports["RELAY"] = app.relay_entry.get().strip() or app.DEFAULT_COM_PORTS["RELAY"]
        app.lasers.configure_ports(app.hw.com_ports)

    def _get_power(self, tag: str) -> float:
        try:
            e = app.power_entries.get(tag)
            if e is None:
                return app.DEFAULT_LASER_POWERS.get(tag, 0.01)
            return float(e.get().strip())
        except:
            return app.DEFAULT_LASER_POWERS.get(tag, 0.01)

    def save_settings():
        app._update_ports_from_ui()
        app.hw.dll_path = app.dll_entry.get().strip()
        for tag, e in app.power_entries.items():
            try:
                app.hw.laser_power[tag] = float(e.get().strip())
            except:
                pass
        try:
            with open(app.SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "dll_path": app.hw.dll_path,
                    "com_ports": app.hw.com_ports,
                    "laser_power": app.hw.laser_power
                }, f, indent=2)
            messagebox.showinfo("Settings", f"Saved to {app.SETTINGS_FILE}")
        except Exception as e:
            messagebox.showerror("Settings", str(e))

    def load_settings_into_ui():
        if not os.path.isfile(app.SETTINGS_FILE):
            # init defaults
            app.dll_entry.delete(0, "end")
            app.dll_entry.insert(0, app.hw.dll_path)
            app.obis_entry.delete(0, "end"); app.obis_entry.insert(0, app.DEFAULT_COM_PORTS["OBIS"])
            app.cube_entry.delete(0, "end"); app.cube_entry.insert(0, app.DEFAULT_COM_PORTS["CUBE"])
            app.relay_entry.delete(0, "end"); app.relay_entry.insert(0, app.DEFAULT_COM_PORTS["RELAY"])
            for tag, e in app.power_entries.items():
                e.delete(0, "end"); e.insert(0, str(app.DEFAULT_LASER_POWERS.get(tag, 0.01)))
            return
        try:
            with open(app.SETTINGS_FILE, "r", encoding="utf-8") as f:
                obj = json.load(f)
            app.dll_entry.delete(0, "end")
            app.dll_entry.insert(0, obj.get("dll_path", ""))
            cp = obj.get("com_ports", app.DEFAULT_COM_PORTS)
            app.obis_entry.delete(0, "end"); app.obis_entry.insert(0, cp.get("OBIS", app.DEFAULT_COM_PORTS["OBIS"]))
            app.cube_entry.delete(0, "end"); app.cube_entry.insert(0, cp.get("CUBE", app.DEFAULT_COM_PORTS["CUBE"]))
            app.relay_entry.delete(0, "end"); app.relay_entry.insert(0, cp.get("RELAY", app.DEFAULT_COM_PORTS["RELAY"]))
            lp = obj.get("laser_power", app.DEFAULT_LASER_POWERS)
            for tag, e in app.power_entries.items():
                e.delete(0, "end"); e.insert(0, str(lp.get(tag, app.DEFAULT_LASER_POWERS.get(tag, 0.01))))
        except Exception as e:
            messagebox.showerror("Load Settings", str(e))

    # ------------------ General helpers ------------------

    def _post_error(self, title: str, ex: Exception):
        tb = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        print(f"[{title}] {ex}\n{tb}", file=sys.stderr)
        app.after(0, lambda: messagebox.showerror(title, str(ex)))

    def on_close():
        try:
            app.stop_live()
            app.stop_measure()
            if app.spec:
                try: app.spec.disconnect()
                except: pass
            for dev in [app.lasers.obis, app.lasers.cube, app.lasers.relay]:
                try: dev.close()
                except: pass
        finally:
            app.destroy()

    # Bind functions to app object
    app.run_all_selected = run_all_selected
    app.stop_measure = stop_measure
    app.save_csv = save_csv
    app.start_analysis_from_measure = start_analysis_from_measure
    app.run_analysis = run_analysis
    app.export_analysis_plots = export_analysis_plots
    app.export_analysis_summary = export_analysis_summary

    # Call the UI builder
    _build_measure_tab()

