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

def build(app):
    # Import constants from app
    DEFAULT_COM_PORTS = app.DEFAULT_COM_PORTS
    DEFAULT_LASER_POWERS = app.DEFAULT_LASER_POWERS
    SETTINGS_FILE = app.SETTINGS_FILE
    OBIS_LASER_MAP = {
        "405": 5,
        "445": 4,
        "488": 3,
        "640": 2,
    }

    def _build_setup_tab():
        frame = ttk.Frame(app.setup_tab)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Spectrometer block
        spec_group = ttk.LabelFrame(frame, text="Spectrometer")
        spec_group.pack(fill="x", padx=6, pady=6)

        ttk.Label(spec_group, text="DLL Path (avaspecx64.dll):").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        app.dll_entry = ttk.Entry(spec_group, width=60)
        app.dll_entry.grid(row=0, column=1, sticky="w", padx=4, pady=4)
        ttk.Button(spec_group, text="Browse", command=browse_dll).grid(row=0, column=2, padx=4, pady=4)

        app.spec_status = ttk.Label(spec_group, text="Disconnected", foreground="red")
        app.spec_status.grid(row=0, column=3, padx=10)

        ttk.Button(spec_group, text="Connect", command=connect_spectrometer).grid(row=1, column=1, padx=4, pady=4, sticky="w")
        ttk.Button(spec_group, text="Disconnect", command=disconnect_spectrometer).grid(row=1, column=2, padx=4, pady=4, sticky="w")

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

        ttk.Button(ports_group, text="Refresh Ports", command=refresh_ports).grid(row=0, column=2, padx=6)
        ttk.Button(ports_group, text="Test Connect", command=test_com_connect).grid(row=1, column=2, padx=6)

        app.obis_status = ttk.Label(ports_group, text="●", foreground="red")
        app.cube_status = ttk.Label(ports_group, text="●", foreground="red")
        app.relay_status = ttk.Label(ports_group, text="●", foreground="red")
        app.obis_status.grid(row=0, column=3, padx=4)
        app.cube_status.grid(row=1, column=3, padx=4)
        app.relay_status.grid(row=2, column=3, padx=4)

        # Laser power config
        power_group = ttk.LabelFrame(frame, text="Laser Power Configuration")
        power_group.pack(fill="x", padx=6, pady=6)

        app.power_entries = {}
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
        ttk.Button(save_group, text="Save Settings", command=save_settings).pack(side="left")
        ttk.Button(save_group, text="Load Settings", command=load_settings_into_ui).pack(side="left", padx=6)

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

    def _get_power(tag: str) -> float:
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

    def _post_error(title: str, ex: Exception):
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
    app.refresh_ports = refresh_ports
    app.test_com_connect = test_com_connect
    app.browse_dll = browse_dll
    app.connect_spectrometer = connect_spectrometer
    app.disconnect_spectrometer = disconnect_spectrometer
    app._update_ports_from_ui = _update_ports_from_ui
    app._get_power = _get_power
    app.save_settings = save_settings
    app.load_settings_into_ui = load_settings_into_ui
    app._post_error = _post_error
    app.on_close = on_close

    # Call the UI builder
    _build_setup_tab()


