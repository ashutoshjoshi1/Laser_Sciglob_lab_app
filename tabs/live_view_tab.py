# Auto-generated from gui.py by splitter
from typing import Any, List, Optional, Tuple, Dict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import os, time

def build(app):
    # Import constants from app
    OBIS_LASER_MAP = {
        "405": 5,
        "445": 4,
        "488": 3,
        "640": 2,
    }
    IT_MIN = 0.2
    IT_MAX = 3000.0

    def _build_live_tab():
        left = ttk.Frame(app.live_tab)
        left.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        right = ttk.Frame(app.live_tab)
        right.pack(side="right", fill="y", padx=8, pady=8)

        # Matplotlib figure
        app.live_fig = Figure(figsize=(8, 5), dpi=100)
        app.live_ax = app.live_fig.add_subplot(111)
        app.live_ax.set_title("Live Spectrum")
        app.live_ax.set_xlabel("Pixel")
        app.live_ax.set_ylabel("Counts")
        app.live_line, = app.live_ax.plot([], [], lw=1, label="Signal")
        app.live_ax.grid(True)
        app.live_ax.legend(loc="upper right")

        app.live_canvas = FigureCanvasTkAgg(app.live_fig, master=left)
        app.live_canvas.draw()
        app.live_canvas.get_tk_widget().pack(fill="both", expand=True)

        app.live_toolbar = NavigationToolbar2Tk(app.live_canvas, left)

        # track zoom/pan interactions
        app.live_limits_locked = False
        app._live_mouse_down = False

        def _on_press(event):
            app._live_mouse_down = True

        def _on_release(event):
            # When user releases after an interaction on axes, lock current limits
            if event.inaxes is not None:
                app.live_limits_locked = True
            app._live_mouse_down = False

        app.live_canvas.mpl_connect("button_press_event", _on_press)
        app.live_canvas.mpl_connect("button_release_event", _on_release)

        # Controls
        ttk.Label(right, text="Integration Time (ms):").pack(anchor="w")
        ttk.Button(right, text="Reset Zoom", command=app._live_reset_view).pack(anchor="w", pady=(6, 0))
        app.it_entry = ttk.Entry(right, width=12)
        app.it_entry.insert(0, "2.4")
        app.it_entry.pack(anchor="w", pady=(0, 10))
        app.apply_it_btn = ttk.Button(right, text="Apply IT", command=app.apply_it)
        app.apply_it_btn.pack(anchor="w", pady=(0, 10))

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=6)

        ttk.Label(right, text="Laser Controls").pack(anchor="w")
        app.laser_vars = {}
        for tag in ["405", "445", "488", "377", "517", "532", "Hg_Ar"]:
            var = tk.BooleanVar(value=False)
            btn = ttk.Checkbutton(
                right, text=f"{tag} nm", variable=var,
                command=lambda t=tag, v=var: app.toggle_laser(t, v.get()))
            btn.pack(anchor="w")
            app.laser_vars[tag] = var

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=6)

        app.live_start_btn = ttk.Button(right, text="Start Live", command=app.start_live)
        app.live_stop_btn = ttk.Button(right, text="Stop Live", command=app.stop_live)
        app.live_start_btn.pack(anchor="w", pady=2)
        app.live_stop_btn.pack(anchor="w", pady=2)

    def apply_it():
        if not app.spec:
            messagebox.showwarning("Spectrometer", "Not connected.")
            return
        # Parse & clamp
        try:
            it = float(app.it_entry.get())
        except Exception as e:
            messagebox.showerror("Apply IT", f"Invalid IT value: {e}")
            return
        it = max(IT_MIN, min(IT_MAX, it))

        # If live is running, defer until between frames
        if getattr(self, 'live_running', None) and app.live_running.is_set():
            app._pending_it = it
            try:
                app.apply_it_btn.state(["disabled"])  # if button exists
            except Exception:
                pass
            # non-blocking toast via title/status
            try:
                app.title(f"Queued IT={it:.3f} ms (will apply after current frame)")
            except Exception:
                pass
            return

        # If a measurement is in-flight, wait briefly
        try:
            if getattr(app.spec, 'measuring', False):
                t0 = time.time()
                while getattr(app.spec, 'measuring', False) and time.time() - t0 < 3.0:
                    try:
                        app.spec.wait_for_measurement()
                        break
                    except Exception:
                        time.sleep(0.05)
        except Exception:
            pass

        # Apply now
        try:
            app._it_updating = True
            app.spec.set_it(it)
            messagebox.showinfo("Integration", f"Applied IT = {it:.3f} ms")
        except Exception as e:
            messagebox.showerror("Apply IT", str(e))
        finally:
            app._it_updating = False
            try:
                app.apply_it_btn.state(["!disabled"])  # re-enable
            except Exception:
                pass


    def start_live():
        if not app.spec:
            messagebox.showwarning("Spectrometer", "Not connected.")
            return
        if app.live_running.is_set():
            return
        app.live_running.set()
        app.live_thread = threading.Thread(target=app._live_loop, daemon=True)
        app.live_thread.start()

    def stop_live():
        app.live_running.clear()

    def _live_loop():
        while app.live_running.is_set():
            try:
                # Start one frame
                app.spec.measure(ncy=1)
                # Wait for frame to complete
                app.spec.wait_for_measurement()

                # Apply any deferred IT safely after the completed frame
                if app._pending_it is not None:
                    try:
                        it_to_apply = app._pending_it
                        app._pending_it = None
                        app._it_updating = True
                        app.spec.set_it(it_to_apply)
                        try:
                            app.title(f"Applied IT={it_to_apply:.3f} ms")
                        except Exception:
                            pass
                    except Exception as e:
                        app._post_error("Apply IT (deferred)", e)
                    finally:
                        app._it_updating = False
                        try:
                            app.apply_it_btn.state(["!disabled"])  # if exists
                        except Exception:
                            pass

                # After IT changes (or none), fetch data and draw
                y = np.array(app.spec.rcm, dtype=float)
                x = np.arange(len(y))
                app.npix = len(y)
                app.data.npix = app.npix
                app._update_live_plot(x, y)

            except Exception as e:
                app._post_error("Live error", e)
                break


    def _update_live_plot(self, x, y):
        def update():
            app.live_line.set_data(x, y)

            # Only adjust limits when NOT locked
            if not app.live_limits_locked:
                app.live_ax.set_xlim(0, max(10, len(x)-1))
                ymax = np.nanmax(y) if y.size else 1.0
                app.live_ax.set_ylim(0, max(1000, ymax * 1.1))

            app.live_fig.canvas.draw_idle()
        app.after(0, update)


    def toggle_laser(self, tag: str, turn_on: bool):
        try:
            # make sure we use the latest COM port entries
            app._update_ports_from_ui()
            # open the right serial port lazily
            app.lasers.ensure_open_for_tag(tag)

            if tag in OBIS_LASER_MAP:
                ch = OBIS_LASER_MAP[tag]
                if turn_on:
                    watts = float(app._get_power(tag))
                    app.lasers.obis_set_power(ch, watts)
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
                if turn_on:
                    app.lasers.relay_on(3)
                else:
                    app.lasers.relay_off(3)

            elif tag == "532":
                if turn_on:
                    app.lasers.relay_on(1)
                else:
                    app.lasers.relay_off(1)

            elif tag == "Hg_Ar":
                if turn_on:
                    app.lasers.relay_on(2)
                else:
                    app.lasers.relay_off(2)

        except Exception as e:
            app._post_error(f"Laser {tag}", e)

    # ------------------ Measurements Tab -----------------

    def _build_measure_tab():
        top = ttk.Frame(app.measure_tab)
        top.pack(fill="x", padx=8, pady=8)

        ttk.Label(top, text="Automated Measurements").pack(side="left")

        app.run_all_btn = ttk.Button(top, text="Run Selected", command=app.run_all_selected)
        app.stop_all_btn = ttk.Button(top, text="Stop", command=app.stop_measure)
        app.save_csv_btn = ttk.Button(top, text="Save CSV", command=app.save_csv)

        app.run_all_btn.pack(side="right", padx=5)
        app.stop_all_btn.pack(side="right", padx=5)
        app.save_csv_btn.pack(side="right", padx=5)

    
        app.start_analysis_btn = ttk.Button(top, text="Start Analysis", command=app.start_analysis_from_measure)
        app.start_analysis_btn.pack(side="right", padx=5)
    # Laser selection + Auto-IT options
        mid = ttk.Frame(app.measure_tab)
        mid.pack(fill="x", padx=8, pady=8)

        ttk.Label(mid, text="Select lasers to run:").grid(row=0, column=0, sticky="w", padx=4, pady=4)

        app.measure_vars = {}
        tags = ["405", "445", "488", "517", "532", "377", "Hg_Ar"]
        for i, tag in enumerate(tags):
            v = tk.BooleanVar(value=(tag in DEFAULT_ALL_LASERS))
            chk = ttk.Checkbutton(mid, text=tag + " nm", variable=v)
            chk.grid(row=1 + i // 6, column=(i % 6), padx=4, pady=4, sticky="w")
            app.measure_vars[tag] = v

        ttk.Label(mid, text="Auto-IT start (ms, default if blank):").grid(row=3, column=0, sticky="w", padx=4, pady=4)
        app.auto_it_entry = ttk.Entry(mid, width=10)
        app.auto_it_entry.insert(0, "")
        app.auto_it_entry.grid(row=3, column=1, sticky="w", padx=4, pady=4)

        # Result plots (signal & dark)
        bot = ttk.Frame(app.measure_tab)
        bot.pack(fill="both", expand=True, padx=8, pady=8)

        app.meas_fig = Figure(figsize=(12, 6), dpi=100)
        app.meas_ax  = app.meas_fig.add_subplot(111)
        app.meas_ax.set_title("Last Measurement: Signal & Dark")
        app.meas_ax.set_xlabel("Pixel")
        app.meas_ax.set_ylabel("Counts")
        app.meas_ax.grid(True)

        (app.meas_sig_line,)  = app.meas_ax.plot([], [], lw=1.2, label="Signal (ON)")
        (app.meas_dark_line,) = app.meas_ax.plot([], [], lw=1.2, linestyle="--", label="Dark (OFF)")
        app.meas_ax.legend(loc="upper left")

        # Inset for Auto-IT steps (peaks and IT vs step index)
        app.meas_inset = app.meas_ax.inset_axes([0.58, 0.52, 0.38, 0.42])  # x, y, w, h (relative)
        app.meas_inset.set_title("Auto-IT steps")
        app.meas_inset.set_xlabel("step")
        app.meas_inset.set_ylabel("peak")

        app.meas_inset2 = app.meas_inset.twinx()
        app.meas_inset2.set_ylabel("IT (ms)")

        (app.inset_peak_line,) = app.meas_inset.plot([], [], marker="o", lw=1, label="Peak")
        (app.inset_it_line,)   = app.meas_inset2.plot([], [], marker="s", lw=1, linestyle="--", label="IT (ms)")

        app.meas_canvas = FigureCanvasTkAgg(app.meas_fig, bot)
        app.meas_canvas.draw()
        app.meas_canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(app.meas_canvas, bot)

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
        app.measure_running.set()
        app.measure_thread = threading.Thread(
            target=app._measure_sequence_thread, args=(tags, start_it_override), daemon=True)
        app.measure_thread.start()

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

        for tag in laser_tags:
            if not app.measure_running.is_set():
                break
            try:
                app._run_single_measurement(tag, start_it_override)
            except Exception as e:
                app._post_error(f"Measurement {tag}", e)

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

        app.measure_running.clear()

    def _auto_adjust_it(self, start_it: float) -> Tuple[float, float]:
        it_ms = max(IT_MIN, min(IT_MAX, start_it))
        peak = np.nan
        iters = 0
        app.it_history = []  # <-- track steps for the inset
        while iters <= MAX_IT_ADJUST_ITERS:
            app.spec.set_it(it_ms)
            app.spec.measure(ncy=1)
            app.spec.wait_for_measurement()
            y = np.array(app.spec.rcm, dtype=float)
            if y.size == 0:
                iters += 1
                continue
            peak = float(np.nanmax(y))
            app.it_history.append((it_ms, peak))  # <-- record step

            if peak >= SAT_THRESH:
                it_ms = max(IT_MIN, it_ms * 0.7)
                iters += 1
                continue
            if TARGET_LOW <= peak <= TARGET_HIGH:
                return it_ms, peak
            if peak < TARGET_LOW:
                it_ms = min(IT_MAX, it_ms + IT_STEP_UP)
            else:
                it_ms = max(IT_MIN, it_ms - IT_STEP_DOWN)
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
        start_it = start_it_override if start_it_override is not None else DEFAULT_START_IT.get(tag, DEFAULT_START_IT["default"])
        # Auto-IT
        it_ms, peak = app._auto_adjust_it(start_it)

        if TARGET_LOW <= peak <= TARGET_HIGH:
            # Signal
            app.spec.set_it(it_ms)
            app.spec.measure(ncy=N_SIG)
            app.spec.wait_for_measurement()
            y_signal = np.array(app.spec.rcm, dtype=float)

            # Turn OFF tag
            app._ensure_source_state(tag, False)

            # Dark
            time.sleep(0.3)
            app.spec.set_it(it_ms)
            app.spec.measure(ncy=N_DARK)
            app.spec.wait_for_measurement()
            y_dark = np.array(app.spec.rcm, dtype=float)

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            app.data.rows.append([now, tag, it_ms, N_SIG] + y_signal.tolist())
            app.data.rows.append([now, f"{tag}_dark", it_ms, N_DARK] + y_dark.tolist())

            app._update_last_plots(tag)
        else:
            # could not reach target -> just turn off
            app._ensure_source_state(tag, False)

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
            e.insert(0, str(DEFAULT_LASER_POWERS.get(tag, 0.01)))
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
        app.hw.com_ports["OBIS"] = app.obis_entry.get().strip() or DEFAULT_COM_PORTS["OBIS"]
        app.hw.com_ports["CUBE"] = app.cube_entry.get().strip() or DEFAULT_COM_PORTS["CUBE"]
        app.hw.com_ports["RELAY"] = app.relay_entry.get().strip() or DEFAULT_COM_PORTS["RELAY"]
        app.lasers.configure_ports(app.hw.com_ports)

    def _get_power(self, tag: str) -> float:
        try:
            e = app.power_entries.get(tag)
            if e is None:
                return DEFAULT_LASER_POWERS.get(tag, 0.01)
            return float(e.get().strip())
        except:
            return DEFAULT_LASER_POWERS.get(tag, 0.01)

    def save_settings():
        app._update_ports_from_ui()
        app.hw.dll_path = app.dll_entry.get().strip()
        for tag, e in app.power_entries.items():
            try:
                app.hw.laser_power[tag] = float(e.get().strip())
            except:
                pass
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "dll_path": app.hw.dll_path,
                    "com_ports": app.hw.com_ports,
                    "laser_power": app.hw.laser_power
                }, f, indent=2)
            messagebox.showinfo("Settings", f"Saved to {SETTINGS_FILE}")
        except Exception as e:
            messagebox.showerror("Settings", str(e))

    def load_settings_into_ui():
        if not os.path.isfile(SETTINGS_FILE):
            # init defaults
            app.dll_entry.delete(0, "end")
            app.dll_entry.insert(0, app.hw.dll_path)
            app.obis_entry.delete(0, "end"); app.obis_entry.insert(0, DEFAULT_COM_PORTS["OBIS"])
            app.cube_entry.delete(0, "end"); app.cube_entry.insert(0, DEFAULT_COM_PORTS["CUBE"])
            app.relay_entry.delete(0, "end"); app.relay_entry.insert(0, DEFAULT_COM_PORTS["RELAY"])
            for tag, e in app.power_entries.items():
                e.delete(0, "end"); e.insert(0, str(DEFAULT_LASER_POWERS.get(tag, 0.01)))
            return
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                obj = json.load(f)
            app.dll_entry.delete(0, "end")
            app.dll_entry.insert(0, obj.get("dll_path", ""))
            cp = obj.get("com_ports", DEFAULT_COM_PORTS)
            app.obis_entry.delete(0, "end"); app.obis_entry.insert(0, cp.get("OBIS", DEFAULT_COM_PORTS["OBIS"]))
            app.cube_entry.delete(0, "end"); app.cube_entry.insert(0, cp.get("CUBE", DEFAULT_COM_PORTS["CUBE"]))
            app.relay_entry.delete(0, "end"); app.relay_entry.insert(0, cp.get("RELAY", DEFAULT_COM_PORTS["RELAY"]))
            lp = obj.get("laser_power", DEFAULT_LASER_POWERS)
            for tag, e in app.power_entries.items():
                e.delete(0, "end"); e.insert(0, str(lp.get(tag, DEFAULT_LASER_POWERS.get(tag, 0.01))))
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

    # Bind functions to app object (only bind functions that don't already exist in main app)
    # Note: Some functions like start_live, stop_live, toggle_laser, apply_it are already in main app
    app._live_loop = _live_loop

    # Call the UI builder
    _build_live_tab()
