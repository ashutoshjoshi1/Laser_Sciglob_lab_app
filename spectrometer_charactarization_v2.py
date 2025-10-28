import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import time
import serial
import threading
import sys
import os
from datetime import datetime
from avantes_spectrometer import Avantes_Spectrometer
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from scipy.signal import find_peaks


########################################## USER CONFIGURATION ###############################################################

# === LASER CONFIG ===
laser_map = {"405": 5, "445": 4, "488": 3}
laser_power = {"405": 0.05, "445": 0.03, "488": 0.03}
laser_power_640nm = 0.05
# all_lasers = ["532", "517", "488", "445", "405", "377", "Hg_Ar"]
#all_lasers = ["532", "445", "405", "377", "Hg_Ar"]
all_lasers = ["445", "405", "377", "532", "Hg_Ar"]
# all_lasers = ["405", "377", "Hg_Ar"]
data_rows = []

# === Hg-Ar Spectral Lines (nm) ===
known_lines_nm = [289.36, 296.73, 302.15, 313.16, 334.19, 365.01, 404.66, 407.78, 435.84, 507.3, 546.08]

# Starting integration times (ms) for each laser
START_IT_DICT = {
    "532": 20,
    "517": 80,
    "Hg_Ar": 20,
    "default": 2.4}

# === MEASUREMENT CONSTANTS ===
N_SIG = 50        # main lasers: bright measurement cycles
N_DARK = 50       # main lasers: dark measurement cycles
N_SIG_640 = 10    # 640 nm block: bright measurement cycles
N_DARK_640 = 10   # 640 nm block: dark measurement cycles

# Target peak window (counts) for auto-IT
TARGET_LOW = 60000
TARGET_HIGH = 65000
TARGET_MID = 62500

# Integration time bounds (ms)
IT_MIN = 0.2
IT_MAX = 3000

# Auto-IT controller
IT_STEP_UP = 0.3       # when peak too low
IT_STEP_DOWN = 0.1     # when peak too high (make symmetric to avoid oscillations)
MAX_IT_ADJUST_ITERS = 1000  # safety cap to avoid infinite loops

# COM port assignments
COM_PORTS = {
    "OBIS": "COM10",     # OBIS laser serial port (405, 445, 488, 640 nm)
    "RELAY": "COM11",    # Relay control board serial port (517, 532 nm)
    "CUBE": "COM1"       # Cube laser serial port (377 nm)
}

# Stray light Correction matrix
ib_region_size = 20

# Set ADC limit as saturated
SAT_THRESH = 65400  # for ~16-bit

############################################################################################################################

# === OBIS LASER CONTROL SETUP ===
obis_ser = serial.Serial(COM_PORTS["OBIS"], baudrate=9600, timeout=1)

def send_obis_cmd(cmd):
    obis_ser.write((cmd + "\r\n").encode())
    time.sleep(0.2)
    response = obis_ser.readlines()
    return [r.decode().strip() for r in response]

def obis_laser_on(channel):  send_obis_cmd(f"SOUR{channel}:AM:STAT ON")
def obis_laser_off(channel): send_obis_cmd(f"SOUR{channel}:AM:STAT OFF")
def set_obis_power(channel, pwr): send_obis_cmd(f"SOUR{channel}:POW:LEV:IMM:AMPL {pwr:.3f}")

# === CUBE LASER CONTROL SETUP (377 nm) ===
cube_ser = serial.Serial(
    port=COM_PORTS['CUBE'],
    baudrate=19200,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1
)

def send_cube_cmd(cmd):
    full_cmd = (cmd + '\r').encode('utf-8')
    cube_ser.write(full_cmd)
    time.sleep(1)
    return cube_ser.read_all().decode('utf-8', errors='ignore').strip()

def cube_laser_on(power_mw=12):
    send_cube_cmd("EXT=1")
    send_cube_cmd("CW=1")
    send_cube_cmd(f"P={power_mw}")
    send_cube_cmd("L=1")
    time.sleep(3)

def cube_laser_off():
    send_cube_cmd("L=0")

# === NEW 8-RELAY CONTROL SETUP (for 532 nm) ===
relay_ser = serial.Serial(COM_PORTS["RELAY"], baudrate=9600, timeout=1)
def relay_on(n): relay_ser.write(f"R{n}S\r".encode())
def relay_off(n): relay_ser.write(f"R{n}R\r".encode())

def wait_with_countdown_or_enter(seconds=20, title="Fiber switch", message="Switch the fiber"):
    """Matplotlib-based, Enter-skippable countdown that preserves interactive state."""
    was_ion = plt.isinteractive()          # remember current interactive state
    plt.ion()                              # ensure interactive for the countdown
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        # Try to center window if TkAgg
        try:
            if matplotlib.get_backend().lower().startswith("tkagg"):
                win = fig.canvas.manager.window
                sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
                ww, wh = 800, 400  # approx px (depends on DPI)
                win.geometry(f"{ww}x{wh}+{(sw-ww)//2}+{(sh-wh)//2}")
            # Set window title if available
            try:
                fig.canvas.manager.set_window_title(title)
            except Exception:
                pass
        except Exception:
            pass

        ax.set_axis_off()
        msg_text  = ax.text(0.5, 0.70, message, fontsize=28, ha='center', va='center')
        timer_txt = ax.text(0.5, 0.35, "", fontsize=64, color='red', ha='center', va='center')

        skip = {'flag': False}
        def on_key(event):
            if event.key == 'enter':
                skip['flag'] = True
        cid = fig.canvas.mpl_connect('key_press_event', on_key)

        for t in range(seconds, -1, -1):
            if not plt.fignum_exists(fig.number) or skip['flag']:
                break
            timer_txt.set_text(f"{t}")
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(1)

    finally:
        if plt.fignum_exists(fig.number):
            plt.close(fig)
        # restore original interactive state
        if was_ion:
            plt.ion()
        else:
            plt.ioff()




############################### === SPECTROMETER SETUP (auto-detect SN) === ##############################

spec = None
try:
    ava = Avantes_Spectrometer()
    ava.dll_path   = r'C:\Users\Administrator\Desktop\Python Code\Spectrometer_codes\spec_ava1\Avaspec-DLL_9.14.0.9_64bits\avaspecx64.dll'
    ava.alias      = "Ava1"
    ava.npix_active= 2048
    ava.debug_mode = 1
    ava.initialize_spec_logger()

    # 1) Load the DLL (wrapper provides this)
    res = ava.load_spec_dll()
    if res != "OK":
        print(f"❌ DLL load failed: {res}")
        exit()

    # 2) Initialize the DLL (wrapper calls AVS_Init internally)
    res = ava.initialize_dll()
    if res != "OK":
        print(f"⚠️ initialize_dll returned: {res} (continuing)")

    # 3) Enumerate devices
    res, ndev = ava.get_number_of_devices()
    if res != "OK" or ndev <= 0:
        print(f"❌ No Avantes spectrometers detected. res={res}, ndev={ndev}")
        exit()

    # 4) Get info for all connected devices (to read serials)
    res, l_pData_all = ava.get_all_devices_info(ndev)
    if res != "OK":
        print(f"❌ get_all_devices_info failed: {res}")
        exit()

    # 5) Pick the first serial number found
    serials = []
    for i in range(ndev):
        ident = getattr(l_pData_all, f"a{i}")
        sn = ident.SerialNumber
        if isinstance(sn, (bytes, bytearray)):
            sn = sn.decode("utf8")
        serials.append(sn)

    if not serials:
        print("❌ Could not read any SerialNumber from devices.")
        exit()

    chosen_sn = serials[0]
    ava.sn = chosen_sn
    print(f"🔎 Using spectrometer SN: {ava.sn}")

    # 6) Connect to that spectrometer
    ava.connect()
    spec = ava

except Exception as e:
    print(f"❌ Avantes spectrometer not found/connected: {e}")
    exit()



######################################### === PLOT SETUP === ########################################
plt.ion()
fig, ax = plt.subplots(figsize=(16, 8)) 
line, = ax.plot(np.ones(spec.npix_active), lw=1, color='tab:blue', label="Signal") #signal measurement (laser ON)
line_dark, = ax.plot(np.ones(spec.npix_active), lw=1, color='black', linestyle='--', label="Dark") #dark measurement (laser OFF)

ax.set_yscale('log')
ax.set_title(f"Spectrometer= {ava.sn}: Live Measurement", fontsize=18)
ax.set_xlabel("Pixel Index", fontsize=18)
ax.set_ylabel("Counts", fontsize=18)
ax.set_xticks(np.arange(0, 2048, 100))
ax.tick_params(axis='x', labelsize=16) 
ax.tick_params(axis='y', labelsize=18) 
# ax.set_ylim(0, 69000)
ax.set_ylim(1e2, 1e5)
ax.legend( fontsize=18) 
ax.grid(True)
plt.tight_layout()

########################################## === MAIN ROUTINE === ##########################################
try:
    for ch in range(1, 6): obis_laser_off(ch)
    cube_laser_off()
    relay_off(1)  # 532 nm OFF
    relay_off(2)  # Hg-Ar lamp OFF
    relay_off(3)  # 517 nm OFF

    for lwl in all_lasers:
        if lwl == "377":
            for ch in range(1, 6): obis_laser_off(ch)
            relay_off(1)  # 532 nm OFF
            relay_off(2)  # Hg-Ar lamp OFF
            relay_off(3)  # 517 nm OFF
            cube_laser_on(12)
            print("377 nm turned ON")
        elif lwl == "517":
            for ch in range(1, 6): obis_laser_off(ch)
            cube_laser_off()
            relay_off(1)  # 532 nm OFF
            relay_off(2)  # Hg-Ar lamp OFF
            relay_on(3)
            time.sleep(1)
            print("517 nm turned ON")
        elif lwl == "532":
            for ch in range(1, 6): obis_laser_off(ch)
            cube_laser_off()
            relay_off(2)  # Hg-Ar lamp OFF
            relay_off(3)  # 517 nm OFF
            relay_on(1)
            time.sleep(1)
            print("532 nm turned ON")
        elif lwl == "Hg_Ar":
            for ch in range(1, 6): obis_laser_off(ch)
            cube_laser_off()
            relay_off(1)  # 532 nm OFF
            relay_off(3)  # 517 nm OFF

            #interruptible_countdown(30, msg="Switch the fiber to Hg-Ar lamp")
            # Show countdown and allow Enter to skip
            wait_with_countdown_or_enter(seconds=45, title="fiber switch", message="Switch the fiber to Hg-Ar lamp")
            

            relay_on(2)
            print("Hg-Ar lamp turned ON")
            time.sleep(1)
        else:
            cube_laser_off()
            relay_off(1)  # 532 nm OFF
            relay_off(2)  # Hg-Ar lamp OFF
            relay_off(3)  # 517 nm OFF
            ch = laser_map[lwl]
            obis_laser_on(ch)
            print(f'{lwl} nm turned ON')
            set_obis_power(ch, laser_power[lwl])
            time.sleep(1)

        # === Adjust integration time (robust with max iterations) ===
        it_ms = START_IT_DICT.get(lwl, START_IT_DICT["default"])
        adjust_iters = 0
        success = False
        peak = np.nan  # ensure defined

        while True:
            spec.set_it(it_ms)
            spec.measure(ncy=1)
            spec.wait_for_measurement()
            y = np.array(spec.rcm)

            if y.size == 0:
                print(f"⚠️ {lwl} nm: No data received. Retrying...")
                adjust_iters += 1
                if adjust_iters > MAX_IT_ADJUST_ITERS:
                    print(f"❌ {lwl} nm: Gave up (no data).")
                    break
                continue

            peak = float(np.max(y))

            # quick anti-saturation guard
            if peak >= SAT_THRESH:
                it_ms = max(IT_MIN, it_ms * 0.7)  # aggressive step down
                adjust_iters += 1
                if adjust_iters > MAX_IT_ADJUST_ITERS:
                    print(f"❌ {lwl} nm: Could not de-saturate within limit.")
                    break
                continue

            line.set_ydata(y)
            ax.set_title(f"Spectrometer= {ava.sn}: Live Measurement for {lwl} nm | IT = {it_ms:.1f} ms | peak={peak:.0f}", fontsize=18)
            fig.canvas.draw()
            fig.canvas.flush_events()

            if TARGET_LOW <= peak <= TARGET_HIGH:
                print(f"✅ {lwl} nm: Good peak {peak:.1f} at IT={it_ms:.1f} ms")
                success = True
                break

            # Proportional-ish tweak around the mid target; cap by step sizes
            err = TARGET_MID - peak
            if err > 0:   # too dim
                delta = min(IT_STEP_UP, max(0.05, abs(err) / 5000.0))  # ms
                it_ms = min(IT_MAX, it_ms + delta)
            else:         # too bright
                delta = min(IT_STEP_DOWN, max(0.05, abs(err) / 5000.0))  # ms
                it_ms = max(IT_MIN, it_ms - delta)

            adjust_iters += 1
            if adjust_iters > MAX_IT_ADJUST_ITERS:
                print(f"❌ {lwl} nm: Could not reach target range after {MAX_IT_ADJUST_ITERS} adjustments.")
                break

        # === If not successful, turn off the current laser and skip ===
        if not success:
            if lwl == "377":
                cube_laser_off()
                print("377 nm turned OFF (no valid measurement)")
            elif lwl == "517":
                relay_off(3)
                print("517 nm turned OFF (no valid measurement)")
            elif lwl == "532":
                relay_off(1)
                print("532 nm turned OFF (no valid measurement)")
            elif lwl == "Hg_Ar":
                relay_off(2)
                print("Hg_Ar lamp turned OFF (no valid measurement)")
            else:
                obis_laser_off(ch)
                print(f"{lwl} nm turned OFF (no valid measurement)")
            time.sleep(0.3)
            continue

        # === Save signal + dark ===
        if TARGET_LOW <= peak <= TARGET_HIGH:
            # Bright (signal)
            spec.set_it(it_ms)
            spec.measure(ncy=N_SIG)
            spec.wait_for_measurement()
            y_signal = np.array(spec.rcm)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data_rows.append([now, lwl, it_ms, N_SIG] + y_signal.tolist())

            # Turn off laser
            if lwl == "377":
                cube_laser_off()
                print("377 nm turned OFF")
            elif lwl == "517":
                relay_off(3)
                print("517 nm turned OFF")
            elif lwl == "532":
                relay_off(1)
                print("532 nm turned OFF")
            elif lwl == "Hg_Ar":
                relay_off(2)
                print("Hg_Ar lamp turned OFF")
            else:
                obis_laser_off(ch)
                print(f'{lwl} nm turned OFF')

            # Dark
            time.sleep(2)
            spec.set_it(it_ms)
            spec.measure(ncy=N_DARK)
            spec.wait_for_measurement()
            y_dark = np.array(spec.rcm)
            now_dark = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data_rows.append([now_dark, lwl + "_dark", it_ms, N_DARK] + y_dark.tolist())

            line_dark.set_ydata(np.clip(y_dark, 1e-5, None))
            ax.set_title(f"Spectrometer= {ava.sn}: {lwl} nm DARK Measurement @ IT = {it_ms:.1f} ms", fontsize=18)
            fig.canvas.draw()
            fig.canvas.flush_events()

    
    # === 640 nm OBIS Laser Measurement ===
    plt.close(fig)

    #interruptible_countdown(30, msg="Switch back the fiber to previous Sphere")
    wait_with_countdown_or_enter(seconds=45, title="fiber switch", message="Switch back the fiber to previous Sphere")
    
    plt.ion()
    ch_640 = 2
    integration_times = [100, 500, 1000]

    # Turn ON laser once
    obis_laser_on(ch_640)
    set_obis_power(ch_640, laser_power_640nm)
    time.sleep(3)

    fig640, ax640 = plt.subplots(figsize=(16, 8))
    ax640.set_title(f"Spectrometer= {ava.sn}: Live 640 nm OBIS Measurement", fontsize=18)
    ax640.set_xticks(np.arange(0, 2048, 100))
    ax640.set_xlabel("Pixel Index", fontsize=18)
    ax640.set_ylabel("Counts", fontsize=18)
    ax640.tick_params(axis='x', labelsize=16) 
    ax640.tick_params(axis='y', labelsize=18)
    ax640.grid(True)
    plt.tight_layout()
    line640, = ax640.plot(np.zeros(spec.npix_active), lw=1, color='tab:red')


    # === BRIGHT MEASUREMENTS ===
    for it_ms in integration_times:
        spec.set_it(it_ms)
        spec.measure(ncy=N_SIG_640)
        spec.wait_for_measurement()
        y640 = np.array(spec.rcm)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_rows.append([now, "640", it_ms, N_SIG_640] + y640.tolist())

        # live plot for each
        line640.set_ydata(y640)
        ax640.set_ylim(0, max(1000, np.max(y640) * 1.2))
        ax640.set_title(f"Spectrometer= {ava.sn}: 640 nm Measurement @ IT={it_ms} ms")
        fig640.canvas.draw()
        fig640.canvas.flush_events()

    # Turn OFF laser once
    obis_laser_off(ch_640)
    time.sleep(1)

    # === DARK MEASUREMENTS ===
    for it_ms in integration_times:
        spec.set_it(it_ms)
        spec.measure(ncy=N_DARK_640)
        spec.wait_for_measurement()
        y640_dark = np.array(spec.rcm)
        now_dark = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_rows.append([now_dark, "640_dark", it_ms, N_DARK_640] + y640_dark.tolist())


finally:
    for ch in range(1, 6): obis_laser_off(ch)
    cube_laser_off()
    plt.ioff()
    
    # Close only live measurement figures
    try: plt.close(fig)
    except: pass
    try: plt.close(fig640)
    except: pass

    if spec: spec.disconnect()
    try: obis_ser.close()
    except: pass
    try: cube_ser.close()
    except: pass
    try: relay_ser.close()
    except: pass


############################################### === SAVE TO CSV with timestamp === ####################################################

# === Prepare folder path ===
base_folder = r"C:\Users\Administrator\Desktop\Python Code\Spectrometer_codes\data"
sn_folder = os.path.join(base_folder, str(ava.sn))

# Create folder if it doesn't exist
os.makedirs(sn_folder, exist_ok=True)

# === Save CSV ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join(sn_folder, f"All_Lasers_Measurements_{ava.sn}_{timestamp}.csv")
colnames = ["Timestamp", "Wavelength", "IntegrationTime", "NumCycles"] + [f"Pixel_{i}" for i in range(spec.npix_active)]
df = pd.DataFrame(data_rows, columns=colnames)
df.to_csv(csv_path, index=False)
print(f"✅ Saved measurements to {csv_path}")

########################################### === Spectrometer Charactarization Plots === ###############################################

# === START LSF PROCESSING AUTOMATICALLY AFTER SAVING CSV ===
df = pd.read_csv(csv_path)
df["Wavelength"] = df["Wavelength"].astype(str)  # Ensure consistent string matching
npix = sum(col.startswith("Pixel_") for col in df.columns)

# Extract pixel columns in numeric order and derive npix from that
#pixel_cols = [c for c in df.columns if c.startswith("Pixel_")]
#pixel_cols = sorted(pixel_cols, key=lambda c: int(c.split("_")[1]))
#npix = len(pixel_cols)

# === Function to generate normalized LSF from signal and dark ===
def get_normalized_lsf(df, wavelength, sat_thresh=SAT_THRESH, use_latest=True):
    """
    Build a normalized LSF = (signal - dark), scaled to [0,1], with strict guards:
      - require both rows present
      - require all finite values (no NaN/inf)
      - reject saturated signal
      - auto-detect pixel columns
      - robust normalization
    Returns:
      np.ndarray (length = #pixels) or None on any problem.
    """
    # 1) Pick pixel columns robustly (in case you add metadata columns later)
    pixel_cols = [c for c in df.columns if str(c).startswith("Pixel_")]
    if not pixel_cols:
        print(f"❌ No pixel columns found for {wavelength} nm.")
        return None

    # 2) Find signal/dark rows
    sig_rows = df[df["Wavelength"] == wavelength]
    dark_rows = df[df["Wavelength"] == f"{wavelength}_dark"]

    if sig_rows.empty or dark_rows.empty:
        print(f"⚠️ Missing data for {wavelength} nm (signal or dark).")
        return None

    # Prefer the most recent row (last) if there are multiples
    sig_row = sig_rows.iloc[-1] if use_latest else sig_rows.iloc[0]
    dark_row = dark_rows.iloc[-1] if use_latest else dark_rows.iloc[0]

    # 3) Extract vectors and force float
    try:
        sig = sig_row[pixel_cols].astype(float).to_numpy()
        dark = dark_row[pixel_cols].astype(float).to_numpy()
    except Exception as e:
        print(f"⚠️ {wavelength} nm: failed to parse pixels -> {e}")
        return None

    # 4) Basic shape/size checks
    if sig.size == 0 or dark.size == 0 or sig.shape != dark.shape:
        print(f"❌ Signal/Dark length mismatch for {wavelength} nm.")
        return None

    # 5) Finite + saturation checks
    if not np.all(np.isfinite(sig)) or not np.all(np.isfinite(dark)):
        print(f"⚠️ Non-finite values for {wavelength} nm; skipping.")
        return None
    if np.any(sig >= sat_thresh):
        print(f"⚠️ {wavelength} nm: signal saturated (>= {sat_thresh}); skipping.")
        return None

    # 6) Correct, ensure finite, and require range
    corrected = sig - dark
    if not np.all(np.isfinite(corrected)):
        print(f"⚠️ Non-finite values after correction for {wavelength} nm; skipping.")
        return None

    # For normalization, bring min to 0 and scale by max
    corrected -= np.min(corrected)
    denom = float(np.max(corrected))
    if not np.isfinite(denom) or denom <= 0:
        print(f"⚠️ Flat/invalid corrected signal for {wavelength} nm; skipping.")
        return None

    normed = corrected / denom
    #normed = (sig - dark) / np.max(sig - dark)

    # Final sanity
    if not np.all(np.isfinite(normed)):
        print(f"⚠️ Non-finite values after normalization for {wavelength} nm; skipping.")
        return None

    return normed


# === Reference wavelengths map ===
laser_reference_map = {
    "377": 375,
    "405": 403.46,
    "445": 445,
    "517": 517,
    "532": 532
}

# === Build lists ===
lsf_list = []
pixel_locations = []
laser_wavelengths = []

for lwl in all_lasers:
    if lwl not in laser_reference_map:
        print(f"⚠️ Skipping undefined laser wavelength: {lwl}")
        continue

    lsf = get_normalized_lsf(df, lwl)
    if lsf is not None:
        lsf_list.append(lsf)
        pixel_locations.append(np.argmax(lsf))
        laser_wavelengths.append(laser_reference_map[lwl])
    else:
        print(f"⚠️ No valid LSF for {lwl} nm — will skip in SDF matrix.")

# Final cleanup before fitting
keep_idx = [i for i, lsf in enumerate(lsf_list) if np.all(np.isfinite(lsf)) and np.max(lsf) > 0]
if len(keep_idx) != len(lsf_list):
    print(f"ℹ️ Dropping {len(lsf_list) - len(keep_idx)} invalid/empty LSF(s) before fitting.")
    lsf_list = [lsf_list[i] for i in keep_idx]
    pixel_locations = [pixel_locations[i] for i in keep_idx]
    laser_wavelengths = laser_wavelengths[keep_idx]

# === Convert wavelength list to NumPy array ===
laser_wavelengths = np.array(laser_wavelengths)
# === Ensure ascending order of peak pixels (left -> right) ===
peak_pixels_unsorted = [np.argmax(lsf) for lsf in lsf_list]
order = np.argsort(peak_pixels_unsorted)
pixel_locations = [peak_pixels_unsorted[i] for i in order] ##### used in SDF matrix
lsf_list = [lsf_list[i] for i in order]
laser_wavelengths = laser_wavelengths[order]
lsfs = lsf_list  ##### lsf_list used in SDF matrix
total_pixels = 2048

# # === Plot for verification ===
fig_norm = plt.figure(figsize=(16, 8))
plt.yscale('log')
plt.xticks(np.arange(0, 2048, 100), fontsize=16)
plt.yticks(fontsize=18)
for lsf, wl in zip(lsfs, laser_wavelengths):
    plt.plot(lsf, label=f"{wl} nm")
plt.title(f"Spectrometer= {ava.sn}: Normalized LSFs", fontsize=18)
plt.xlabel("Pixel Index", fontsize=18)
plt.ylabel("Normalized Intensity", fontsize=18)
plt.ylim(1e-5, 1.4)
plt.grid(True)
plt.legend(fontsize=18)
plt.tight_layout()

plot1_path = os.path.join(sn_folder, f"Normalized_Laser_Plot_{ava.sn}_{timestamp}.png")
fig_norm.savefig(plot1_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved normalized plot to {plot1_path}")

# === Filter 640 nm signal rows (excluding dark rows) ===
sig_entries = df[df["Wavelength"].str.startswith("640") & ~df["Wavelength"].str.contains("dark")]

fig_640corr = plt.figure(figsize=(16, 8))
plt.xticks(np.arange(0, 2048, 100), fontsize=16)

for idx, row in sig_entries.iterrows():
    wl = row["Wavelength"]
    dark_wl = wl + "_dark"
    if dark_wl in df["Wavelength"].values:
        signal = row.iloc[4:].astype(float).values
        dark = df[df["Wavelength"] == dark_wl].iloc[0, 4:].astype(float).values
        corrected = np.clip(signal - dark, a_min=1e-5, a_max=None)
        it_ms = row["IntegrationTime"] if "IntegrationTime" in df.columns else row.iloc[2]
        label = f"{wl} @ {it_ms:.1f} ms"
        plt.plot(corrected, label=label)
    else:
        print(f"⚠️ No dark found for {wl}")

plt.title(f"Spectrometer= {ava.sn}: Dark-Corrected 640 nm Measurements", fontsize=18)
plt.xlabel("Pixel Index", fontsize=18)
plt.ylabel("Corrected Intensity", fontsize=18)
plt.grid(True)
plt.legend(fontsize=18)
plt.tight_layout()

plot2_path = os.path.join(sn_folder, f"OOR_640nm_Plot_{ava.sn}_{timestamp}.png")
fig_640corr.savefig(plot2_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved OOR 640 nm plot to {plot2_path}")

##################################################### === Hg-Ar Lamp === #################################################################################

# === DARK CORRECTED SIGNAL ===
def get_corrected_signal_from_df(df, base="Hg_Ar"):
    sig = df[df["Wavelength"] == base].iloc[-1, 4:].astype(float).values
    dark = df[df["Wavelength"] == base + "_dark"].iloc[-1, 4:].astype(float).values
    return np.clip(sig - dark, 1e-5, None)

signal_corr = get_corrected_signal_from_df(df, "Hg_Ar")

# === PEAK DETECTION ===
#peaks, props = find_peaks(signal_corr, height=700, distance=20, prominence=500)
peaks, props = find_peaks(signal_corr, prominence=0.008*np.max(signal_corr), distance=20)
ord_idx = np.argsort(peaks)
peaks = peaks[ord_idx]

# === MATCH PEAKS TO KNOWN LINES ===
def best_ordered_linear_match(peaks_pix, candidate_wls, min_points=5):
    P, L = len(peaks_pix), len(candidate_wls)
    best = None

    def score(pix_sel, wl_sel):
        A = np.vstack([pix_sel, np.ones_like(pix_sel)]).T
        a, b = np.linalg.lstsq(A, wl_sel, rcond=None)[0]
        pred = a * pix_sel + b
        return np.sqrt(np.mean((wl_sel - pred)**2)), a, b

    if P >= L:
        for i in range(P - L + 1):
            pix_sel = peaks_pix[i:i+L]
            wl_sel = np.array(candidate_wls)
            rmse, a, b = score(pix_sel, wl_sel)
            if best is None or rmse < best[0]:
                best = (rmse, a, b, pix_sel.copy(), wl_sel.copy())
    else:
        for j in range(L - P + 1):
            pix_sel = peaks_pix.copy()
            wl_sel = np.array(candidate_wls[j:j+P])
            rmse, a, b = score(pix_sel, wl_sel)
            if best is None or rmse < best[0]:
                best = (rmse, a, b, pix_sel.copy(), wl_sel.copy())

    return best if best and len(best[3]) >= min_points else None

#candidates = [known_lines_nm, known_lines_nm[:-1]]  # with and without 546 nm
candidates = [known_lines_nm, known_lines_nm]
solutions = [best_ordered_linear_match(peaks, cand) for cand in candidates]
solutions = [s for s in solutions if s is not None]
if not solutions:
    raise RuntimeError("❌ No valid match between Hg-Ar peaks and known lines.")
solutions.sort(key=lambda t: t[0])
rmse, a_lin, b_lin, matched_pixels, matched_wavelengths = solutions[0]
matched_pixels = np.array(matched_pixels)
matched_wavelengths = np.array(matched_wavelengths)

if rmse < 2:
    print(f"✅ Matched {len(matched_pixels)} lines, RMSE = {rmse:.2f} nm")
else:
    print(f"❌ RMSE value too high = {rmse:.2f} nm, Matched {len(matched_pixels)} lines, Check the matching Hg Lamp peaks")


# === Dark-corrected signal plot ===
pixels = np.arange(len(signal_corr))
fig_hg = plt.figure(figsize=(18, 8))
plt.yscale('log')
plt.plot(pixels, signal_corr, label="Dark-Corrected Hg-Ar Lamp Signal", color='blue')

# === Mark ALL detected peaks ===
plt.plot(peaks, signal_corr[peaks], 'ro', label='Detected Peaks')

# === Annotate only matched peaks ===
for pix, wl in zip(matched_pixels, matched_wavelengths):
    y = signal_corr[pix]
    plt.text(pix, y + 2500, f"{wl:.1f} nm", rotation=0, color='brown', fontsize=11,
             ha='center', va='bottom')

# === Labels and formatting ===
plt.xticks(np.arange(0, 2048, 100), fontsize=16)
plt.xlabel("Pixel", fontsize=18)
plt.ylabel("Signal (Counts)", fontsize=18)
plt.title(f"Spectrometer= {ava.sn}: Hg-Ar Lamp Spectrum with Detected Peaks", fontsize=18)
plt.legend(fontsize=18)
plt.grid(True)
plt.tight_layout()

# === Save figure ===
plot3_path = os.path.join(sn_folder, f"HgAr_Peaks_Plot_{ava.sn}_{timestamp}.png")
fig_hg.savefig(plot3_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved Hg-Ar peak plot to {plot3_path}")


# === Hg-Ar lamp LSF EXTRACTION ===
lsf_list_lamp = []
pixel_loc = []
win = 30

for pix in matched_pixels:
    start = max(int(pix - win), 0)
    end = min(int(pix + win + 1), npix)
    lsf = signal_corr[start:end]
    lsf = (lsf - lsf.min()) / max(1e-12, lsf.max() - lsf.min())
    if np.all(np.isfinite(lsf)):
        lsf_list_lamp.append(lsf)
        pixel_loc.append(int(pix))


# Convert to NumPy arrays for easier indexing
matched_wavelengths = np.array(matched_wavelengths)
#pixel_locations = np.array(pixel_locations)
matched_pixels = np.array(pixel_loc)
lsf_list_lamp = np.array(lsf_list_lamp, dtype=object)


##############################################################################################################################################


######################################## === Stray Light Matrix === ##########################################################################

# Initialize an empty SDF matrix (size: total_pixels x total_pixels)
SDF_matrix = np.zeros((total_pixels, total_pixels))

def normalize_lsf_stray_light_correct(lsf, pixel_number):
    # Get the in-band region (IB) around the pixel where the monochromatic source was measured
    ib_start = max(0, pixel_number - ib_region_size // 2)
    ib_end = min(len(lsf), pixel_number + ib_region_size // 2 + 1)
    ib_region = np.arange(ib_start, ib_end)

    # Calculate the sum of IB values
    ib_sum = float(np.sum(lsf[ib_region]))

    # Set IB values to zero
    lsf[ib_region] = 0

    # Normalize the remaining OOB values by the IB sum (OOB/IB); avoid divide-by-zero
    if not np.isfinite(ib_sum) or ib_sum <= 0:
        normalized_lsf = np.zeros_like(lsf)
    else:
        normalized_lsf = lsf / ib_sum

    return normalized_lsf

# Place normalized LSFs in the appropriate columns
for i, (lsf, pixel_number) in enumerate(zip(lsf_list, pixel_locations)):
    normalized_lsf = normalize_lsf_stray_light_correct(np.copy(lsf), pixel_number)  # Normalize the LSF correctly
    SDF_matrix[:, pixel_number] = normalized_lsf  # Set the normalized LSF as a column in the SDF matrix

# Shift LSFs upwards from the right-most LSF, stopping at next available LSF or pixel 0
for i in range(len(pixel_locations) - 1, 0, -1):  # Start from the right-most LSF and move left
    current_pixel = pixel_locations[i]
    previous_pixel = pixel_locations[i - 1]
    
    for col in range(current_pixel - 1, previous_pixel, -1):
        shift_amount = current_pixel - col
        SDF_matrix[:-shift_amount, col] = SDF_matrix[shift_amount:, current_pixel]
        SDF_matrix[-shift_amount:, col] = 0  # Zero out the shifted-in region

# Handle the shift for the first LSF to the left (from first LSF to pixel 0)
first_pixel = pixel_locations[0]
for col in range(first_pixel - 1, -1, -1):
    shift_amount = first_pixel - col
    SDF_matrix[:-shift_amount, col] = SDF_matrix[shift_amount:, first_pixel]
    SDF_matrix[-shift_amount:, col] = 0  # Zero out the shifted-in region

# Shift the right-most LSF downward across columns to the right
last_lsf_pixel = pixel_locations[-1]  # The right-most LSF pixel

for col in range(last_lsf_pixel + 1, total_pixels):
    shift_amount = col - last_lsf_pixel
    SDF_matrix[shift_amount:, col] = SDF_matrix[:-shift_amount, last_lsf_pixel]
    SDF_matrix[:shift_amount, col] = 0  # Zero out the shifted-in region

# Replace bottom 0 values with the last row value of the shifting LSF
for i in range(len(pixel_locations) - 1, -1, -1):
    current_pixel = pixel_locations[i]
    stop_col = pixel_locations[i - 1] + 1 if i > 0 else 0
    last_value = SDF_matrix[-1, current_pixel]
    
    for col in range(current_pixel - 1, stop_col - 1, -1):
        # Define the in-band (IB) region for this column
        ib_start = max(0, col - ib_region_size // 2)
        ib_end = min(total_pixels, col + ib_region_size // 2 + 1)
        
        # Mask to identify zeros outside the IB region
        for row in range(ib_end, total_pixels):
            if SDF_matrix[row, col] == 0:
                SDF_matrix[row, col] = last_value

# Replace top 0 values with the first row value of the shifting LSF
last_lsf_pixel = pixel_locations[-1]  # The right-most LSF pixel
first_value = SDF_matrix[0, last_lsf_pixel]

for col in range(last_lsf_pixel + 1, total_pixels):
    # Define the in-band (IB) region for this column
    ib_start = max(0, col - ib_region_size // 2)
    ib_end = min(total_pixels, col + ib_region_size // 2 + 1)
    
    # Mask to identify zeros outside the IB region
    for row in range(0, ib_start):
        if SDF_matrix[row, col] == 0:
            SDF_matrix[row, col] = first_value


# Plot the updated SDF matrix
fig_sdf = plt.figure(figsize=(16, 8))
# plt.yscale('log')
plt.xlim(0,2048)
for col in pixel_locations:
    plt.plot(SDF_matrix[:, col], label= f'{col} pixel')
plt.xticks(np.arange(0, 2048, 100),fontsize=16)
plt.xlabel('Pixels',fontsize=18)
plt.ylabel('SDF Value',fontsize=18)
plt.yticks(fontsize=18)
plt.title(f"Spectrometer= {ava.sn}: Spectral Distribution Function (SDF)",fontsize=18)
#plt.ylim(0,0.0035)
plt.legend(fontsize=18)
plt.grid(True)

# Save plot
plot4_path = os.path.join(sn_folder, f"SDF_Plot_{ava.sn}_{timestamp}.png")
fig_sdf.savefig(plot4_path, dpi=300, bbox_inches='tight')  # ✅ Save with fig_sdf
print(f"✅ Saved SDF plot to {plot4_path}")


# Create figure and axis
fig_sdf_heatmap, ax = plt.subplots(figsize=(10, 6))  # ✅ Name the figure object
im = ax.imshow(SDF_matrix, aspect='auto', cmap='coolwarm', origin='lower')
plt.colorbar(im, label='SDF Value')
ax.set_xlabel('Pixels', fontsize=16)
ax.set_ylabel('Spectral Pixel Index', fontsize=16)
plt.title(f"Spectrometer= {ava.sn}: SDF Matrix Heatmap", fontsize=16)

# Save plot
plot5_path = os.path.join(sn_folder, f"SDF_Heatmap_{ava.sn}_{timestamp}.png")
fig_sdf_heatmap.savefig(plot5_path, dpi=300, bbox_inches='tight')  # ✅ Save with fig object
print(f"✅ Saved SDF matrix heatmap to {plot5_path}")


identity_matrix = np.eye(total_pixels)
A_matrix = identity_matrix + SDF_matrix
cond_A = np.linalg.cond(A_matrix) # Report conditioning and choose a safe inverse
print(f"ℹ️ cond(I+SDF) = {cond_A:.3e}")

if np.isclose(cond_A, 1.0, atol=1e-3):
    print("κ≈1: perfectly conditioned.")
elif cond_A < 10:
    print("κ≈10: Matrix is excellent for most numerical work.")
else:
    print("Matrix is ill-conditioned, results can be unreliable without special techniques.")


if np.isfinite(cond_A) and cond_A < 1e8:
    correction_matrix = np.linalg.inv(A_matrix)
else:
    # Tikhonov regularization for stability
    eps = 1e-6
    print(f"⚠️ Ill-conditioned A; applying Tikhonov regularization (ε={eps})")
    A_reg = A_matrix + eps * identity_matrix
    correction_matrix = np.linalg.pinv(A_reg, rcond=1e-10)


## Plotting the heatmap with adjusted color range
#fig, ax = plt.subplots(figsize=(10, 6))
## Primary heatmap
#im = ax.imshow(correction_matrix, aspect='auto', cmap='coolwarm', origin='lower') #, vmin=-0.2, vmax=vmax
#plt.colorbar(im, label='Correction Matrix Value')
## Set labels for the primary x-axis
#ax.set_xlabel('Pixels',fontsize=16)
#ax.set_ylabel('Spectral Pixel Index',fontsize=16)
#plt.title(f"Spectrometer= {ava.sn}: Correction Matrix",fontsize=16)


######################################## === Dispersion Polynomial === #########################################################

comb_peak_pixels = np.concatenate((pixel_locations, matched_pixels))
comb_wavelengths = np.concatenate((laser_wavelengths, matched_wavelengths))

# Combine into pairs
comb_pairs = list(zip(comb_peak_pixels, comb_wavelengths))

# Sort by wavelength (preferred for calibration)
comb_pairs_sorted = sorted(comb_pairs, key=lambda x: x[1])

# Unzip back to arrays
comb_peak_pixels_sorted, comb_wavelengths_sorted = zip(*comb_pairs_sorted)
comb_peak_pixels_sorted = np.array(comb_peak_pixels_sorted)
comb_wavelengths_sorted = np.array(comb_wavelengths_sorted)

print("Sorted all peak pixels:", comb_peak_pixels_sorted)
print("Sorted all wavelengths:", comb_wavelengths_sorted)


# === Adaptive dispersion polynomial fit ===
num_points = len(comb_peak_pixels_sorted)

if num_points < 2:
    raise ValueError("❌ At least 2 laser peaks are required for dispersion fitting.")

degree = 2 if num_points >= 3 else 1  # use degree 2 if enough points, else linear
disp_coeffs = np.polyfit(comb_peak_pixels_sorted, comb_wavelengths_sorted, deg=degree)
print(f"✅ Dispersion polynomial fitted with degree {degree}")

# === Build polynomial string dynamically ===
terms = []
for i, coeff in enumerate(disp_coeffs):
    power = degree - i
    if power == 0:
        terms.append(f"{coeff:.6e}")
    elif power == 1:
        terms.append(f"{coeff:.6e}·p")
    else:
        terms.append(f"{coeff:.6e}·p^{power}")

# Join the terms into a polynomial expression
poly_str = " + ".join(terms)
print(f"Dispersion Polynomial: λ(p) = {poly_str}  [nm]")

pixels = np.arange(2048)
wavelengths_fitted = np.polyval(disp_coeffs, pixels)
print(f'Fitted WV: {wavelengths_fitted}')

# Plot
plt.figure(figsize=(14, 6))
plt.plot(comb_peak_pixels_sorted, comb_wavelengths_sorted, 'ro', label='Laser + Lamp Peaks')
plt.plot(pixels, wavelengths_fitted, 'b-', label='Dispersion Fit')
plt.xlabel("Pixel",fontsize=18)
plt.ylabel("Wavelength (nm)",fontsize=18)
plt.xticks(np.arange(0, 2050, 100), rotation=45, fontsize=16)
plt.yticks(fontsize=16)
plt.title(f"Spectrometer= {ava.sn}: Dispersion Fit")
plt.grid(True)
plt.legend()
plt.tight_layout()

########################################### === Slit Function === #############################################

# === Ensure ascending order of peak pixels (left -> right) ===
lsf_list_lamp = np.array(lsf_list_lamp)
order = np.argsort(matched_pixels)
matched_pixels = matched_pixels[order]
lsf_list_lamp = lsf_list_lamp[order]
matched_wavelengths = matched_wavelengths[order]

# === Remove unwanted Hg-Ar lamp wavelengths ===
remove_wv = [302.15, 313.16, 365.01, 407.78]
keep_idx = [i for i, wl in enumerate(matched_wavelengths) if wl not in remove_wv]
matched_wavelengths = matched_wavelengths[keep_idx]
matched_pixels = matched_pixels[keep_idx]
lsf_list_lamp = lsf_list_lamp[keep_idx]

# === Crop laser LSFs to ±win pixels around peak ===
cropped_lsfs = []
for lsf, peak in zip(lsfs, pixel_locations):
    start = max(0, peak - win)
    end = min(len(lsf), peak + win + 1)
    cropped = lsf[start:end]

    # Pad to 2*win+1 = 51 length if cropped at edge
    if len(cropped) < (2*win + 1):
        pad_left = max(0, win - peak)
        pad_right = max(0, win - (len(lsf) - peak - 1))
        cropped = np.pad(cropped, (pad_left, pad_right), mode='constant')

    cropped_lsfs.append(cropped)

#cropped_lsfs = np.array(cropped_lsfs)
cropped_lsfs = np.asarray(cropped_lsfs, dtype=float)


# === Combine and sort all LSFs (laser + lamp) ===
all_lsfs = np.vstack((cropped_lsfs, lsf_list_lamp))
all_peak_pixels = np.concatenate((pixel_locations, matched_pixels))
all_wavelengths = np.concatenate((laser_wavelengths, matched_wavelengths))

# Sort based on peak pixels
sort_idx = np.argsort(all_peak_pixels)
all_lsfs = all_lsfs[sort_idx]
all_peak_pixels = all_peak_pixels[sort_idx]
all_wavelengths = all_wavelengths[sort_idx]


# === symmetric modified Gaussian model ===
def slit_func(x, A2, A3, C1):
    return np.exp(-np.abs(x / A2) ** A3) + C1

# Fit slit function parameters for each LSF ===
A2_list, A3_list, C1_list = [], [], []
# dispersion_nm_per_pixel = c1  # use linear slope
dispersion_poly = np.poly1d(disp_coeffs)
dispersion_deriv = dispersion_poly.deriv()

for lsf, peak_pixel in zip(all_lsfs, all_peak_pixels):
    # x = (np.arange(len(lsf)) - peak_pixel) * dispersion_nm_per_pixel  # wavelength offset in nm for linear
    dispersion_nm_per_pixel = dispersion_deriv(peak_pixel)  # local slope
    # x = (np.arange(len(lsf)) - peak_pixel) * dispersion_nm_per_pixel

    center = len(lsf) // 2  # assuming symmetric LSF (e.g., 51 -> center = 25)
    x = (np.arange(len(lsf)) - center) * dispersion_nm_per_pixel

    
    lsf_norm = (lsf - np.min(lsf)) / (np.max(lsf) - np.min(lsf))  # normalize to [0, 1]
    
    # Fit the slit function
    popt, _ = curve_fit(
        slit_func, x, lsf_norm,
        bounds=([0.01, 1.5, 0], [5.0, 10.0, 0.1])
    )
    A2_list.append(popt[0])
    A3_list.append(popt[1])
    C1_list.append(popt[2])

# Fit polynomials to A2(λ) and A3(λ) ===
wavelengths_um = all_wavelengths / 1000  # convert nm to microns

A2_poly = np.polyfit(wavelengths_um, A2_list, deg=2)
A3_poly = np.polyfit(wavelengths_um, A3_list, deg=2)
C1_poly = [np.mean(C1_list)]  # assumed constant

# Print final polynomial coefficients ===
print("Slit function fitting method -> Symmetric modified Gaussian")
print("Slit function parameter A2 polynomial ->", ' '.join(f"{c:.6e}" for c in A2_poly))
print("Slit function parameter A3 polynomial ->", ' '.join(f"{c:.6e}" for c in A3_poly))
print("Slit function parameter C1 polynomial ->", f"{C1_poly[0]:.6e}")

# plot for A2 and A3 fits
fig_A2A3 = plt.figure(figsize=(14, 6))

# === Subplot 1: A2 vs Wavelength ===
plt.subplot(1, 2, 1)
plt.plot(wavelengths_um, A2_list, 'ro', label='Measured A2')
plt.plot(wavelengths_um, np.polyval(A2_poly, wavelengths_um), 'b-', label='Fitted A2')
plt.xlabel("Wavelength (μm)")
plt.ylabel("A2 (Width)")
plt.title(f"Spectrometer={ava.sn}: A2 vs Wavelength")
plt.grid(True)
plt.legend()

# === Subplot 2: A3 vs Wavelength ===
plt.subplot(1, 2, 2)
plt.plot(wavelengths_um, A3_list, 'ro', label='Measured A3')
plt.plot(wavelengths_um, np.polyval(A3_poly, wavelengths_um), 'b-', label='Fitted A3')
plt.xlabel("Wavelength (μm)")
plt.ylabel("A3 (Shape)")
plt.title(f"Spectrometer={ava.sn}: A3 vs Wavelength")
plt.grid(True)
plt.legend()

# === Adjust layout and save ===
plt.tight_layout()
plot6_path = os.path.join(sn_folder, f"A2_A3_vs_Wavelength_{ava.sn}_{timestamp}.png")
fig_A2A3.savefig(plot6_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved A2/A3 vs Wavelength plot to {plot6_path}")



##############################  === Spectral Resolution vs Wavelength (compare with Pandora 2) === ####################

npix_ref = 2048
pix_ref = np.arange(1, npix_ref + 1)
pixs_ref = 3.46 * ((pix_ref / npix_ref) - 0.5)  # normalized pixel scale
disp_coeffs_p2 = [7.231379e-04, 4.173348e-05, -6.463429e-03, 1.715324e-04, 2.186199e-02, -3.104087e-03,
                  -3.418786e-02, 1.189230e-02, -1.164253e-01, -2.558337e+00, 7.738903e+01, 4.177510e+02]
res_poly_p2    = [1.481378e+00, -1.059595e+00, 7.612173e-01]  # nm when λ is in microns
wavelengths_p2 = np.polyval(disp_coeffs_p2, pixs_ref)                # nm
fwhm_p2        = np.polyval(res_poly_p2, wavelengths_p2 / 1000.0)    # nm

wv_range_nm = np.linspace(wavelengths_fitted[0], wavelengths_fitted[-1], 1000)
wv_range_um = wv_range_nm/1000
A2_vals = np.polyval(A2_poly, wv_range_um) 
A3_vals = np.polyval(A3_poly, wv_range_um)
fwhm_vals = 2 * A2_vals * (np.log(2))**(1/A3_vals)
resolution_poly = np.polyfit(wv_range_um, fwhm_vals, deg=2)
print("Resolution polynomial coefficients ->", ' '.join(f"{c:.6e}" for c in resolution_poly))

# Plot Spectral Resolution
fig_resolution = plt.figure(figsize=(16, 8))

# === Plot resolution curves ===
plt.plot(wavelengths_p2, fwhm_p2, label='Reference: Pandora 2', color='black')
plt.plot(wv_range_nm, fwhm_vals, 'b', label=f'Spectrometer = {ava.sn}')
plt.xlabel('Wavelength (nm)', fontsize=18)
plt.ylabel('FWHM (nm)', fontsize=18)
plt.title(f'Spectrometer= {ava.sn}: Spectral Resolution vs Wavelength', fontsize=18)
plt.grid(True)
plt.legend(fontsize=18)
plt.tight_layout()

# === Save the plot ===
plot7_path = os.path.join(sn_folder, f"Spectral_Resolution_with_wavelength_{ava.sn}_{timestamp}.png")
fig_resolution.savefig(plot7_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved Spectral Resolution plot to {plot7_path}")


##########################################  === Slit Function with FWHM === ################################ 

center_wavelengths = [350, 400, 480]  # in nm

def generate_adaptive_x(A2, spacing=0.01):
    half_width = 3 * A2
    num_points = int(2 * half_width / spacing) + 1
    return np.linspace(-half_width, half_width, num_points)

# === Width Calculation Function ===
def compute_width_at_percent_max(x, y, percent=0.2):
    y = y - np.min(y)        # Remove baseline
    y = y / np.max(y)        # Normalize to [0, 1]

    threshold = percent
    above = np.where(y >= threshold)[0]

    if len(above) < 2:
        return 0.0, None, None  # Not enough points above threshold

    left_idx = above[0]
    right_idx = above[-1]

    def interp_edge(i1, i2):
        if i2 >= len(x): return x[-1]
        return x[i1] + (x[i2] - x[i1]) * (threshold - y[i1]) / (y[i2] - y[i1])

    x_left = interp_edge(left_idx - 1, left_idx) if left_idx > 0 else x[left_idx]
    x_right = interp_edge(right_idx, right_idx + 1) if right_idx < len(x) - 1 else x[right_idx]

    return np.abs(x_right - x_left), x_left, x_right

# === FWHM Function (for comparison) ===
def compute_fwhm(x, y):
    return compute_width_at_percent_max(x, y, percent=0.5)[0]


fig_slit = plt.figure(figsize=(16, 8))

for λ0 in center_wavelengths:
    λ0_um = λ0 / 1000.0
    A2 = np.clip(np.polyval(A2_poly, λ0_um), 0.2, 5.0)
    A3 = np.polyval(A3_poly, λ0_um)
    C1 = C1_poly[0]

    x = generate_adaptive_x(A2)
    S = np.exp(-np.abs(x / A2) ** A3) + C1
    # S /= np.sum(S)  # normalize

    fwhm = compute_fwhm(x, S)
    print(f"λ₀ = {λ0} nm → A2 = {A2:.3f}, A3 = {A3:.2f}, FWHM = {fwhm:.4f} nm")

    plt.plot(x, S, label=f'λ₀ = {λ0} nm, FWHM = {fwhm:.3f} nm')

plt.title(f"Spectrometer= {ava.sn}: Slit Function with FWHM", fontsize=18)
plt.xlabel("Wavelength Offset from Center (nm)", fontsize=18)
plt.ylabel("Normalized Intensity", fontsize=18)
plt.xticks(np.arange(-1, 1.1, 0.25), fontsize=16)
plt.legend(fontsize=18)
plt.grid(True)
plt.tight_layout()

# === Save the figure ===
plot8_path = os.path.join(sn_folder, f"Slit_Functions_{ava.sn}_{timestamp}.png")
fig_slit.savefig(plot8_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved Slit Function plot to {plot8_path}")


##########################################  === Overlay of Normalized LSFs === ################################ 

# === Set path to reference Pandora 2 LSF file ===
pandora_2_csv_path = r'C:\Users\Administrator\Desktop\Python Code\Spectrometer_codes\Pandora2_All_LSFs.csv'
pandora_2_wavelengths = [325]  # Edit as needed

pandora_334_csv_path = r'C:\Users\Administrator\Desktop\Python Code\Spectrometer_codes\Pandora334_All_LSFs.csv'
pandora_334_wavelengths = [375] 

# === Load reference data ===
df_ref = pd.read_csv(pandora_2_csv_path)
df_334 = pd.read_csv(pandora_334_csv_path)

fwhm_laser = []
fwhm_lamp = []
fwhm_ref = []
fwhm_334 = []

# === CONFIGURATION ===
percent_height = 0.01

# === Create subplots ===
fig_lsf_dual = plt.figure(figsize=(14, 10))
ax1, ax2 = fig_lsf_dual.subplots(nrows=2, ncols=1, sharex=True)

for lsf, peak_pixel, λ0 in zip(lsf_list, pixel_locations, laser_wavelengths):
    dispersion_nm_per_pixel = dispersion_deriv(peak_pixel)
    x = (np.arange(len(lsf)) - peak_pixel) * dispersion_nm_per_pixel
    lsf_norm = (lsf - np.min(lsf)) / (np.max(lsf) - np.min(lsf))
    fwhm = compute_fwhm(x, lsf_norm)
    fwhm_laser.append((λ0, fwhm))
    width_p, x_left, x_right = compute_width_at_percent_max(x, lsf_norm, percent=percent_height)

    # === Plot LSF ===
    ax1.plot(x, lsf_norm, label=f'{λ0} nm, FWHM = {fwhm:.2f} nm, FW_{int(percent_height*100)}% = {width_p:.2f} nm')

# === Overlay Pandora 2 reference LSFs ===
for λ0 in pandora_2_wavelengths:
    df_w = df_ref[df_ref["Wavelength_nm"] == λ0]
    if not df_w.empty:
        x_ref = df_w["WavelengthOffset_nm"].values
        y_ref = df_w["LSF_Normalized"].values
        fwhm = compute_fwhm(x_ref, y_ref)
        fwhm_ref.append((λ0, fwhm))
        width_p, x_left, x_right = compute_width_at_percent_max(x_ref, y_ref, percent=percent_height)

        ax1.plot(x_ref, y_ref, '--', linewidth=2,
                 label=f'Pan2, {λ0} nm, FWHM={fwhm:.2f} nm, FW_{int(percent_height*100)}% = {width_p:.2f} nm',
                 color='orange')

# === Overlay Pandora 334 reference LSFs ===
for λ0 in pandora_334_wavelengths:
    df_334 = df_334[df_334["Wavelength_nm"] == λ0]
    if not df_334.empty:
        x_334 = df_334["WavelengthOffset_nm"].values
        y_334 = df_334["LSF_Normalized"].values
        fwhm_ = compute_fwhm(x_334, y_334)
        fwhm_334.append((λ0, fwhm_))
        width_p, x_left, x_right = compute_width_at_percent_max(x_334, y_334, percent=percent_height)

        ax1.plot(x_334, y_334, '--', linewidth=2,
                 label=f'Pan334, {λ0} nm, FWHM={fwhm:.2f} nm, FW_{int(percent_height*100)}% = {width_p:.2f} nm',
                 color='black')

ax1.set_yscale('log')
ax1.set_title(f"Spectrometer = {ava.sn}: Normalized LSFs of Lasers", fontsize=18)
ax1.set_ylabel("Normalized Intensity", fontsize=18)
ax1.set_xticks(np.arange(-30, 30, 1))
ax1.tick_params(axis='x', labelsize=14)
ax1.set_xlim(-7, 7)
ax1.set_ylim(1e-4, 1.5)
ax1.grid(True)
ax1.legend(title="Laser Wavelength", fontsize=14)

# === Plot 2: Hg-Ar Lamp LSFs ===
for lsf, peak_pixel, λ0 in zip(lsf_list_lamp, matched_pixels, matched_wavelengths):
    dispersion_nm_per_pixel = dispersion_deriv(peak_pixel)
    center = len(lsf) // 2
    x = (np.arange(len(lsf)) - center) * dispersion_nm_per_pixel
    lsf_norm = (lsf - np.min(lsf)) / (np.max(lsf) - np.min(lsf))
    fwhm = compute_fwhm(x, lsf_norm)
    fwhm_lamp.append((λ0, fwhm))
    width_p, x_left, x_right = compute_width_at_percent_max(x, lsf_norm, percent=percent_height)

    # === Plot LSF ===
    ax2.plot(x, lsf_norm, label=f'{λ0} nm, FWHM = {fwhm:.2f} nm, FW_{int(percent_height*100)}% = {width_p:.2f} nm')

ax2.set_yscale('log')
ax2.set_title(f"Spectrometer = {ava.sn}: Normalized LSFs of Hg-Ar Lamp", fontsize=14)
ax2.set_xlabel("Wavelength Offset from Peak (nm)", fontsize=14)
ax2.set_ylabel("Normalized Intensity", fontsize=14)
ax2.set_xticks(np.arange(-30, 30, 1))
ax2.tick_params(axis='x', labelsize=12)
ax2.set_xlim(-7, 7)
ax2.set_ylim(1e-4, 1.5)
ax2.grid(True)
ax2.legend(title="Lamp Wavelength", fontsize=14)

fig_lsf_dual.tight_layout()

# === Save the figure ===
plot9_path = os.path.join(sn_folder, f"Overlapped_LSF_Lasers_HgAr_{ava.sn}_{timestamp}.png")
fig_lsf_dual.savefig(plot9_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved LSF comparison plot to {plot9_path}")



# === Create single plot ===
fig, ax = plt.subplots(figsize=(16, 8))

# === Plot laser LSFs from current spectrometer ===
lsf_list2 = [lsf_list[0]]
for lsf, peak_pixel, λ0 in zip(lsf_list2, pixel_locations, laser_wavelengths):
    dispersion_nm_per_pixel = dispersion_deriv(peak_pixel)
    x = (np.arange(len(lsf)) - peak_pixel) * dispersion_nm_per_pixel
    lsf_norm = (lsf - np.min(lsf)) / (np.max(lsf) - np.min(lsf))
    fwhm = compute_fwhm(x, lsf_norm)
    width_p, x_left, x_right = compute_width_at_percent_max(x, lsf_norm, percent=percent_height)
    fwhm_laser.append((λ0, fwhm))

    # === Plot LSF ===
    ax.plot(x, lsf_norm, label=f'{λ0} nm, FWHM = {fwhm:.2f} nm, FW_{int(percent_height*100)}% = {width_p:.2f} nm')
    # === Horizontal line at selected percent ===
    ax.axhline(y=percent_height, color='red', linestyle='--', linewidth=1, alpha=0.7)

    # === Vertical lines at left/right crossing ===
    if x_left is not None and x_right is not None:
        ax.axvline(x=x_left, color='red', linestyle=':', linewidth=1.5)
        ax.axvline(x=x_right, color='red', linestyle=':', linewidth=1.5)

# === Overlay Pandora 2 reference LSFs ===
for λ0 in pandora_2_wavelengths:
    df_w = df_ref[df_ref["Wavelength_nm"] == λ0]
    if not df_w.empty:
        x_ref = df_w["WavelengthOffset_nm"].values
        y_ref = df_w["LSF_Normalized"].values
        fwhm = compute_fwhm(x_ref, y_ref)
        width_p, x_left, x_right = compute_width_at_percent_max(x_ref, y_ref, percent=percent_height)
        fwhm_ref.append((λ0, fwhm))

        ax.plot(x_ref, y_ref, '--', linewidth=2,
                 label=f'Pan2, {λ0} nm, FWHM={fwhm:.2f} nm, FW_{int(percent_height*100)}% = {width_p:.2f} nm',
                 color='orange')

        # Horizontal & vertical reference lines for Pan2
        ax.axhline(y=percent_height, color='orange', linestyle='--', linewidth=1, alpha=0.5)
        if x_left is not None and x_right is not None:
            ax.axvline(x=x_left, color='orange', linestyle=':', linewidth=1.5)
            ax.axvline(x=x_right, color='orange', linestyle=':', linewidth=1.5)

# === Overlay Pandora 334 reference LSFs ===
for λ0 in pandora_334_wavelengths:
    df_334 = df_334[df_334["Wavelength_nm"] == λ0]
    if not df_334.empty:
        x_334 = df_334["WavelengthOffset_nm"].values
        y_334 = df_334["LSF_Normalized"].values
        fwhm_ = compute_fwhm(x_334, y_334)
        width_p, x_left, x_right = compute_width_at_percent_max(x_334, y_334, percent=percent_height)
        fwhm_334.append((λ0, fwhm_))

        ax.plot(x_334, y_334, '--', linewidth=2,
                 label=f'Pan334, {λ0} nm, FWHM={fwhm:.2f} nm, FW_{int(percent_height*100)}% = {width_p:.2f} nm',
                 color='black')

        # Horizontal & vertical reference lines for Pan2
        ax.axhline(y=percent_height, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        if x_left is not None and x_right is not None:
            ax.axvline(x=x_left, color='gray', linestyle=':', linewidth=1.5)
            ax.axvline(x=x_right, color='gray', linestyle=':', linewidth=1.5)

# === Final Plot Settings ===
ax.set_yscale('log')
ax.set_title(f"Spectrometer = {ava.sn}: Normalized Laser LSFs", fontsize=18)
ax.set_ylabel("Normalized Intensity", fontsize=18)
ax.set_xlabel("Wavelength Offset (nm)", fontsize=18)
ax.set_xlim(-7, 7)
ax.set_ylim(1e-4, 1.5)
ax.set_xticks(np.arange(-7, 7, 1))
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.grid(True)
ax.legend(title="Wavelength", fontsize=16)
plt.tight_layout()


# === Save the figure ===
plot10_path = os.path.join(sn_folder, f"Overlapped_375nm_Laser_{ava.sn}_{timestamp}.png")
fig.savefig(plot10_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved 375 nm LSF plot to {plot10_path}")



# === Show All Plots ===
plt.show()
