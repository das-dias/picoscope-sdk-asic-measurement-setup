"""
================================================================================
ps5000a_bode_sweep.py
PicoScope 5000 Series — Swept-frequency Bode Plot (Gain & Phase)
================================================================================
Setup
  - Generator (AWG out) → your ASIC input, looped back to Channel A
  - ASIC output+ → Channel B
  - ASIC output- → Channel C  (or noise/reference as needed)
  - (B - C) represents the differential output of the ASIC

What the script does
  1. Iterates over log-spaced frequencies from 10 kHz to 50 MHz
  2. At each frequency the built-in sine generator is programmed via
     ps5000aSetSigGenBuiltInV2 (software step-and-settle, no hardware sweep)
  3. Three channels (A, B, C) are captured in block mode
  4. FFT is computed for A and (B-C)
  5. At the fundamental bin: gain [dB] and phase difference [°] are extracted
  6. All per-step results are accumulated in a pandas DataFrame and exported
     to a timestamped CSV after the sweep completes
  7. A Bode plot (gain + phase vs log frequency) is saved as a PNG

Dependencies:  pip install picosdk numpy matplotlib scipy pandas
================================================================================
"""

import ctypes
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless-safe; change to "TkAgg" for a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal.windows import blackman

from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import adc2mV, assert_pico_ok

# ──────────────────────────────────────────────────────────────────────────────
# USER CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
FREQ_START_HZ   = 10_000        # 10 kHz
FREQ_STOP_HZ    = 50_000_000    # 50 MHz
N_FREQ_POINTS   = 60            # log-spaced frequency steps
N_SAMPLES       = 8192          # samples per block capture (power-of-2 for FFT)
SETTLE_TIME_S   = 0.05          # seconds to wait after changing generator freq

# Voltage range index for all three analogue channels
# PS5000A ranges: 1=20mV 2=50mV 3=100mV 4=200mV 5=500mV
#                 6=1V   7=2V   8=5V   9=10V  10=20V
CH_RANGE        = 7             # ±2 V

# Generator
GEN_OFFSET_UV   = 0             # DC offset in µV
GEN_PK2PK_UV    = 2_000_000     # 2 Vpp
RESOLUTION      = ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_12BIT"]

# Output directory (created automatically if it does not exist)
OUTPUT_DIR      = Path("outputs")

# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORY
# ──────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Timestamp used as a common suffix for all output files from this run
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ──────────────────────────────────────────────────────────────────────────────
# OPEN DEVICE
# ──────────────────────────────────────────────────────────────────────────────
status  = {}
chandle = ctypes.c_int16()

status["openUnit"] = ps.ps5000aOpenUnit(ctypes.byref(chandle), None, RESOLUTION)
try:
    assert_pico_ok(status["openUnit"])
except Exception:
    pwr = status["openUnit"]
    if pwr in (282, 286):
        status["changePower"] = ps.ps5000aChangePowerSource(chandle, pwr)
        assert_pico_ok(status["changePower"])
    else:
        raise

print("Device opened.")

# ──────────────────────────────────────────────────────────────────────────────
# CHANNEL SETUP  (A, B, C on; D off to maximise timebase range)
# ──────────────────────────────────────────────────────────────────────────────
CH_A = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
CH_B = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_B"]
CH_C = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_C"]
CH_D = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_D"]
DC   = ps.PS5000A_COUPLING["PS5000A_DC"]

for ch, en in [(CH_A, 1), (CH_B, 1), (CH_C, 1), (CH_D, 0)]:
    status[f"setCh{ch}"] = ps.ps5000aSetChannel(
        chandle, ch, en, DC, CH_RANGE, ctypes.c_float(0.0)
    )
    assert_pico_ok(status[f"setCh{ch}"])

# Maximum ADC value for this resolution (used by adc2mV)
maxADC = ctypes.c_int16()
status["maxADC"] = ps.ps5000aMaximumValue(chandle, ctypes.byref(maxADC))
assert_pico_ok(status["maxADC"])

# ──────────────────────────────────────────────────────────────────────────────
# TIMEBASE HELPER
# ──────────────────────────────────────────────────────────────────────────────
def find_timebase(target_fs_hz: float, n_samples: int) -> tuple[int, float]:
    """
    Walk ps5000aGetTimebase2 indices until the sample interval yields at
    least 20 full cycles of *target_fs_hz* within *n_samples* points.

    Parameters
    ----------
    target_fs_hz : float
        Fundamental frequency of the signal being captured (Hz).
    n_samples : int
        Number of samples in the capture block.

    Returns
    -------
    timebase : int
        Timebase index accepted by ps5000aRunBlock.
    dt_ns : float
        Actual sample interval in nanoseconds.
    """
    interval_ns = ctypes.c_float()
    max_samp    = ctypes.c_int32()

    min_period_ns       = 1e9 / target_fs_hz
    desired_interval_ns = (20.0 * min_period_ns) / n_samples

    for tb in range(1, 2**23):
        st = ps.ps5000aGetTimebase2(
            chandle, tb, n_samples,
            ctypes.byref(interval_ns), ctypes.byref(max_samp), 0
        )
        if st == 0 and interval_ns.value <= desired_interval_ns:
            return tb, float(interval_ns.value)
        if st == 0 and tb > 8:
            return tb, float(interval_ns.value)

    return 62, float(interval_ns.value)    # safe fallback

# ──────────────────────────────────────────────────────────────────────────────
# BLOCK CAPTURE HELPER
# ──────────────────────────────────────────────────────────────────────────────
def capture_block(freq_hz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Arm, trigger, and retrieve one block capture from channels A, B, and C.

    Parameters
    ----------
    freq_hz : float
        Current generator frequency (Hz); used to select the timebase.

    Returns
    -------
    chA_mV, chB_mV, chC_mV : np.ndarray
        Voltage waveforms in millivolts.
    dt_ns : float
        Sample interval in nanoseconds.
    """
    tb, dt_ns  = find_timebase(freq_hz, N_SAMPLES)
    RATIO_NONE = ps.PS5000A_RATIO_MODE["PS5000A_RATIO_MODE_NONE"]

    # Simple rising-edge trigger on Ch A at 0 V; auto-trigger after 1 000 ms
    status["trig"] = ps.ps5000aSetSimpleTrigger(
        chandle, 1, CH_A, 0, 2, 0, 1000
    )
    assert_pico_ok(status["trig"])

    status["runBlock"] = ps.ps5000aRunBlock(
        chandle, 0, N_SAMPLES, tb, None, 0, None, None
    )
    assert_pico_ok(status["runBlock"])

    ready = ctypes.c_int16(0)
    while ready.value == 0:
        ps.ps5000aIsReady(chandle, ctypes.byref(ready))
        time.sleep(0.001)

    bufA = (ctypes.c_int16 * N_SAMPLES)()
    bufB = (ctypes.c_int16 * N_SAMPLES)()
    bufC = (ctypes.c_int16 * N_SAMPLES)()

    for ch, buf in [(CH_A, bufA), (CH_B, bufB), (CH_C, bufC)]:
        status[f"setBuf{ch}"] = ps.ps5000aSetDataBuffer(
            chandle, ch, ctypes.byref(buf), N_SAMPLES, 0, RATIO_NONE
        )
        assert_pico_ok(status[f"setBuf{ch}"])

    n_ret    = ctypes.c_uint32(N_SAMPLES)
    overflow = ctypes.c_int16()
    status["getValues"] = ps.ps5000aGetValues(
        chandle, 0, ctypes.byref(n_ret), 1, RATIO_NONE, 0, ctypes.byref(overflow)
    )
    assert_pico_ok(status["getValues"])

    chA_mV = np.array(adc2mV(bufA, CH_RANGE, maxADC), dtype=np.float64)
    chB_mV = np.array(adc2mV(bufB, CH_RANGE, maxADC), dtype=np.float64)
    chC_mV = np.array(adc2mV(bufC, CH_RANGE, maxADC), dtype=np.float64)

    return chA_mV, chB_mV, chC_mV, dt_ns

# ──────────────────────────────────────────────────────────────────────────────
# FFT HELPER
# ──────────────────────────────────────────────────────────────────────────────
def compute_fft(signal_mV: np.ndarray, dt_ns: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a Blackman window and compute the one-sided real FFT.

    Parameters
    ----------
    signal_mV : np.ndarray
        Time-domain signal in millivolts.
    dt_ns : float
        Sample interval in nanoseconds.

    Returns
    -------
    freqs : np.ndarray
        Frequency axis in Hz.
    Y : np.ndarray
        Complex FFT coefficients (single-sided).
    """
    n   = len(signal_mV)
    win = blackman(n)
    Y   = np.fft.rfft(signal_mV * win)
    f   = np.fft.rfftfreq(n, d=dt_ns * 1e-9)
    return f, Y

# ──────────────────────────────────────────────────────────────────────────────
# MAIN SWEEP LOOP
# ──────────────────────────────────────────────────────────────────────────────
frequencies = np.geomspace(FREQ_START_HZ, FREQ_STOP_HZ, N_FREQ_POINTS)

wavetype   = ctypes.c_int32(0)    # PS5000A_SINE
sweepType  = ctypes.c_int32(0)    # PS5000A_UP (fixed freq per step)
trigType   = ctypes.c_int32(0)    # PS5000A_SIGGEN_RISING
trigSource = ctypes.c_int32(0)    # PS5000A_SIGGEN_NONE

# Accumulate one dict per frequency step; converted to DataFrame after the loop
rows: list[dict] = []

print(f"\nStarting frequency sweep: {FREQ_START_HZ/1e3:.1f} kHz → {FREQ_STOP_HZ/1e6:.0f} MHz")
print(f"{'Freq (Hz)':>14}  {'Set Freq (Hz)':>14}  {'Gain (dB)':>10}  "
      f"{'Phase (deg)':>12}  {'chA RMS (mV)':>13}  {'BC RMS (mV)':>12}  "
      f"{'dt (ns)':>9}  {'Timebase':>9}")
print("-" * 102)

for freq in frequencies:
    # ── 1. Program generator ─────────────────────────────────────────────────
    status["sigGen"] = ps.ps5000aSetSigGenBuiltInV2(
        chandle,
        GEN_OFFSET_UV,
        GEN_PK2PK_UV,
        wavetype,
        ctypes.c_double(freq),
        ctypes.c_double(freq),
        ctypes.c_double(0.0),
        ctypes.c_double(1.0),
        sweepType, 0, 0, 0,
        trigType, trigSource, 0
    )
    assert_pico_ok(status["sigGen"])
    time.sleep(SETTLE_TIME_S)

    # ── 2. Capture ───────────────────────────────────────────────────────────
    tb, dt_ns = find_timebase(freq, N_SAMPLES)
    chA, chB, chC, dt_ns = capture_block(freq)
    diff_BC = chB - chC

    # ── 3. FFT ───────────────────────────────────────────────────────────────
    freqs_A,    Y_A  = compute_fft(chA,     dt_ns)
    freqs_diff, Y_BC = compute_fft(diff_BC, dt_ns)

    # ── 4. Locate fundamental bin ────────────────────────────────────────────
    fund_bin = np.argmin(np.abs(freqs_diff - freq))
    A_fund   = Y_A[fund_bin]
    BC_fund  = Y_BC[fund_bin]

    mag_A  = np.abs(A_fund)
    mag_BC = np.abs(BC_fund)

    # ── 5. Gain and phase ────────────────────────────────────────────────────
    if mag_A < 1e-12:
        gain_db   = np.nan
        phase_deg = np.nan
    else:
        gain_db   = 20.0 * np.log10(mag_BC / mag_A)
        phase_rad = np.angle(BC_fund) - np.angle(A_fund)
        phase_deg = float(np.degrees(np.unwrap([phase_rad])[0]))

    # ── 6. Ancillary diagnostics ─────────────────────────────────────────────
    chA_rms_mV = float(np.sqrt(np.mean(chA ** 2)))
    BC_rms_mV  = float(np.sqrt(np.mean(diff_BC ** 2)))
    actual_freq = float(freqs_diff[fund_bin])

    print(
        f"{freq:14.1f}  {actual_freq:14.1f}  {gain_db:10.3f}  "
        f"{phase_deg:12.3f}  {chA_rms_mV:13.4f}  {BC_rms_mV:12.4f}  "
        f"{dt_ns:9.3f}  {tb:9d}"
    )

    # ── 7. Accumulate row ────────────────────────────────────────────────────
    rows.append({
        "timestamp":            RUN_TIMESTAMP,
        "set_freq_hz":          float(freq),
        "actual_fund_freq_hz":  actual_freq,
        "gain_dB":              gain_db,
        "phase_deg":            phase_deg,
        "chA_rms_mV":           chA_rms_mV,
        "chB_rms_mV":           float(np.sqrt(np.mean(chB ** 2))),
        "chC_rms_mV":           float(np.sqrt(np.mean(chC ** 2))),
        "diff_BC_rms_mV":       BC_rms_mV,
        "chA_peak_mV":          float(np.max(np.abs(chA))),
        "diff_BC_peak_mV":      float(np.max(np.abs(diff_BC))),
        "fund_bin_index":       int(fund_bin),
        "fft_mag_A_at_fund":    float(mag_A),
        "fft_mag_BC_at_fund":   float(mag_BC),
        "sample_interval_ns":   dt_ns,
        "timebase_index":       tb,
        "n_samples":            N_SAMPLES,
        "ch_range_index":       CH_RANGE,
        "gen_pk2pk_uV":         GEN_PK2PK_UV,
        "gen_offset_uV":        GEN_OFFSET_UV,
    })

# ──────────────────────────────────────────────────────────────────────────────
# CLOSE DEVICE
# ──────────────────────────────────────────────────────────────────────────────
status["close"] = ps.ps5000aCloseUnit(chandle)
assert_pico_ok(status["close"])
print("\nDevice closed.")

# ──────────────────────────────────────────────────────────────────────────────
# BUILD DATAFRAME AND EXPORT CSV
# ──────────────────────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)

# Enforce column dtypes explicitly for clean CSV/downstream use
df = df.astype({
    "set_freq_hz":          "float64",
    "actual_fund_freq_hz":  "float64",
    "gain_dB":              "float64",
    "phase_deg":            "float64",
    "chA_rms_mV":           "float64",
    "chB_rms_mV":           "float64",
    "chC_rms_mV":           "float64",
    "diff_BC_rms_mV":       "float64",
    "chA_peak_mV":          "float64",
    "diff_BC_peak_mV":      "float64",
    "fund_bin_index":       "int64",
    "fft_mag_A_at_fund":    "float64",
    "fft_mag_BC_at_fund":   "float64",
    "sample_interval_ns":   "float64",
    "timebase_index":       "int64",
    "n_samples":            "int64",
    "ch_range_index":       "int64",
    "gen_pk2pk_uV":         "int64",
    "gen_offset_uV":        "int64",
})

csv_path = OUTPUT_DIR / f"bode_sweep_{RUN_TIMESTAMP}.csv"
df.to_csv(csv_path, index=False, float_format="%.6f")
print(f"\nCSV saved  → {csv_path}")
print(df[["set_freq_hz", "gain_dB", "phase_deg"]].to_string(index=False))

# ──────────────────────────────────────────────────────────────────────────────
# BODE PLOT
# ──────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.suptitle(
    f"Bode Plot — (B−C) vs A  |  PicoScope 5000 Series\n"
    f"Run: {RUN_TIMESTAMP}  |  {N_FREQ_POINTS} points  |  "
    f"{GEN_PK2PK_UV/1e6:.1f} Vpp  |  12-bit",
    fontsize=11
)

ax1.semilogx(
    df["set_freq_hz"], df["gain_dB"],
    "b.-", linewidth=1.5, markersize=5, label="Gain (B−C)/A"
)
ax1.set_ylabel("Gain (dB)")
ax1.grid(True, which="both", linestyle="--", alpha=0.6)
ax1.axhline(0, color="k", linewidth=0.8)
ax1.legend(fontsize=9)

ax2.semilogx(
    df["set_freq_hz"], df["phase_deg"],
    "r.-", linewidth=1.5, markersize=5, label="Phase (B−C) − A"
)
ax2.set_ylabel("Phase (degrees)")
ax2.set_xlabel("Frequency (Hz)")
ax2.grid(True, which="both", linestyle="--", alpha=0.6)
ax2.axhline(0, color="k", linewidth=0.8)
ax2.legend(fontsize=9)

plt.tight_layout()
plot_path = OUTPUT_DIR / f"bode_plot_{RUN_TIMESTAMP}.png"
plt.savefig(plot_path, dpi=150)
print(f"Plot saved → {plot_path}")
