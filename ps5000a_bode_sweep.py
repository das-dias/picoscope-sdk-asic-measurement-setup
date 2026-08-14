"""
================================================================================
ps5000a_bode_sweep.py
PicoScope 5000 Series — Swept-frequency Bode Plot (Gain & Phase)
================================================================================
Setup
  - Generator (AWG out) → your ASIC input AND Channel D (x10 probe reference)
  - ASIC output+ → Channel B (x10 probe)
  - ASIC output- → Channel C (x10 probe)
  - Channel A     → unused (disabled)
  - (B - C)       → differential ASIC output (CMRR-rejected)
  - Gain and phase are computed as (B-C) relative to D
  - All active channels are AC coupled

What the script does
  1. Iterates over log-spaced frequencies from 10 kHz to 50 MHz
  2. At each step the built-in sine generator is programmed via
     ps5000aSetSigGenBuiltInV2 (software step-and-settle, no hardware sweep)
  3. Channels B, C, D are captured simultaneously in block mode
  4. Raw ADC counts are converted to mV then multiplied by the probe factor
  5. FFT is computed for D and (B-C)
  6. At the fundamental bin: gain [dB] and phase difference [°] of
     (B-C) relative to D are extracted
  7. All per-step results are accumulated in a pandas DataFrame and exported
     to a timestamped CSV after the sweep completes
  8. A two-panel Bode plot (gain + phase vs log frequency) is saved as PNG

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

# --- Frequency sweep parameters ----------------------------------------------
FREQ_START_HZ   = 10_000        # 10 kHz
FREQ_STOP_HZ    = 50_000_000    # 50 MHz
N_FREQ_POINTS   = 60            # number of log-spaced frequency steps
N_SAMPLES       = 8192          # samples per block capture (power-of-2 for FFT)
SETTLE_TIME_S   = 0.05          # seconds to wait after changing generator freq

# --- Probe attenuation -------------------------------------------------------
# Set the attenuation factor for each channel independently.
# 1   = 1x probe  (direct connection, no attenuation)
# 10  = 10x probe (probe tip voltage = ADC reading × 10)
# 100 = 100x probe
#
# Channel A  → disabled (not used in this configuration)
# Channel B  → ASIC output+  (10x probe)
# Channel C  → ASIC output-  (10x probe)
# Channel D  → generator monitor (10x probe) — gain/phase reference
#
# These factors are applied in software after adc2mV() conversion.
# They do NOT affect the ADC input range selection (CH_RANGE_*); choose
# the range that prevents clipping of the attenuated signal at the BNC input.
PROBE_A = 1     # Ch A disabled — value unused but declared for completeness
PROBE_B = 10    # ASIC output+
PROBE_C = 10    # ASIC output-
PROBE_D = 10    # Generator monitor (gain/phase reference)

# --- Voltage range indices ---------------------------------------------------
# Select the tightest range that does not clip the attenuated BNC input.
# With a 10x probe the BNC sees 1/10 of the probe-tip voltage, so a
# ±2 V ADC range (index 7) accommodates probe-tip signals up to ±20 V.
#
# PS5000A ranges:
#   1=±20 mV  2=±50 mV  3=±100 mV  4=±200 mV  5=±500 mV
#   6=±1 V    7=±2 V    8=±5 V     9=±10 V    10=±20 V
CH_RANGE_A      = 7             # unused — set to avoid SDK error on open
CH_RANGE_B      = 7             # ±2 V at BNC (10x probe → ±20 V at tip)
CH_RANGE_C      = 7             # ±2 V at BNC (10x probe → ±20 V at tip)
CH_RANGE_D      = 7             # ±2 V at BNC (10x probe → ±20 V at tip)

# --- Channel coupling --------------------------------------------------------
# PS5000A_AC = 0  →  AC coupled (blocks DC, high-pass ~1 Hz corner)
# PS5000A_DC = 1  →  DC coupled (full bandwidth including DC component)
#
# All channels are AC coupled to reject DC offsets and power-supply
# common-mode noise from the ASIC supply rails. Switch to PS5000A_DC
# if you need to measure absolute DC levels or signals below ~10 Hz.
COUPLING        = ps.PS5000A_COUPLING["PS5000A_AC"]

# --- Generator ---------------------------------------------------------------
GEN_OFFSET_UV   = 0             # DC offset in µV
GEN_PK2PK_UV    = 2_000_000     # 2 Vpp
RESOLUTION      = ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_12BIT"]

# --- Output ------------------------------------------------------------------
OUTPUT_DIR      = Path("outputs")

# ──────────────────────────────────────────────────────────────────────────────
# DERIVED CONSTANTS  (do not edit below this line)
# ──────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
print(
    f"  Coupling: AC  |  Probes — "
    f"Ch A: disabled  |  Ch B: {PROBE_B}x  |  "
    f"Ch C: {PROBE_C}x  |  Ch D: {PROBE_D}x (reference)"
)

# ──────────────────────────────────────────────────────────────────────────────
# CHANNEL SETUP
# A → disabled  |  B → ASIC output+  |  C → ASIC output-  |  D → reference
# All active channels: AC coupled, per COUPLING constant above.
#
# NOTE: On quad-channel models (5443B, 5444B) all four channels are available
# on USB power. On dual-channel models (5242B, 5243B, 5244B) only A and B are
# available; connect the external PSU to unlock C and D.
# ──────────────────────────────────────────────────────────────────────────────
CH_A = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
CH_B = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_B"]
CH_C = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_C"]
CH_D = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_D"]

for ch_enum, en, range_idx in [
    (CH_A, 0, CH_RANGE_A),      # disabled
    (CH_B, 1, CH_RANGE_B),      # ASIC output+,  AC coupled, 10x probe
    (CH_C, 1, CH_RANGE_C),      # ASIC output-,  AC coupled, 10x probe
    (CH_D, 1, CH_RANGE_D),      # generator ref, AC coupled, 10x probe
]:
    status[f"setCh{ch_enum}"] = ps.ps5000aSetChannel(
        chandle,
        ch_enum,
        en,
        COUPLING,       # PS5000A_AC applied to all channels uniformly
        range_idx,
        ctypes.c_float(0.0)
    )
    assert_pico_ok(status[f"setCh{ch_enum}"])

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

    return 62, float(interval_ns.value)

# ──────────────────────────────────────────────────────────────────────────────
# BLOCK CAPTURE HELPER
# Returns probe-corrected waveforms in mV (true probe-tip voltage).
# Ch D is the gain/phase reference; Ch A is not captured.
# ──────────────────────────────────────────────────────────────────────────────
def capture_block(freq_hz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """
    Arm, trigger, and retrieve one block capture from channels B, C, D.
    Applies per-channel probe attenuation after adc2mV conversion so that
    returned arrays represent the true signal at the probe tip.

    Parameters
    ----------
    freq_hz : float
        Current generator frequency (Hz); used to select the timebase.

    Returns
    -------
    chB_mV, chC_mV, chD_mV : np.ndarray
        Probe-tip voltages in millivolts (attenuation-corrected).
        chD_mV is the generator monitor (reference channel).
    dt_ns : float
        Sample interval in nanoseconds.
    tb : int
        Timebase index used for this capture.
    """
    tb, dt_ns  = find_timebase(freq_hz, N_SAMPLES)
    RATIO_NONE = ps.PS5000A_RATIO_MODE["PS5000A_RATIO_MODE_NONE"]

    # Trigger on Ch D (generator monitor), rising edge, threshold 0 V,
    # auto-trigger after 1 000 ms to prevent indefinite blocking.
    status["trig"] = ps.ps5000aSetSimpleTrigger(
        chandle, 1, CH_D, 0, 2, 0, 1000
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

    bufB = (ctypes.c_int16 * N_SAMPLES)()
    bufC = (ctypes.c_int16 * N_SAMPLES)()
    bufD = (ctypes.c_int16 * N_SAMPLES)()

    for ch_enum, buf in [(CH_B, bufB), (CH_C, bufC), (CH_D, bufD)]:
        status[f"setBuf{ch_enum}"] = ps.ps5000aSetDataBuffer(
            chandle, ch_enum, ctypes.byref(buf), N_SAMPLES, 0, RATIO_NONE
        )
        assert_pico_ok(status[f"setBuf{ch_enum}"])

    n_ret    = ctypes.c_uint32(N_SAMPLES)
    overflow = ctypes.c_int16()
    status["getValues"] = ps.ps5000aGetValues(
        chandle, 0, ctypes.byref(n_ret), 1, RATIO_NONE, 0, ctypes.byref(overflow)
    )
    assert_pico_ok(status["getValues"])

    # adc2mV converts ADC counts → BNC socket millivolts.
    # Multiplying by the probe factor recovers the true probe-tip voltage.
    chB_mV = np.array(adc2mV(bufB, CH_RANGE_B, maxADC), dtype=np.float64) * PROBE_B
    chC_mV = np.array(adc2mV(bufC, CH_RANGE_C, maxADC), dtype=np.float64) * PROBE_C
    chD_mV = np.array(adc2mV(bufD, CH_RANGE_D, maxADC), dtype=np.float64) * PROBE_D

    return chB_mV, chC_mV, chD_mV, dt_ns, tb

# ──────────────────────────────────────────────────────────────────────────────
# FFT HELPER
# ──────────────────────────────────────────────────────────────────────────────
def compute_fft(signal_mV: np.ndarray, dt_ns: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a Blackman window and compute the one-sided real FFT.

    Parameters
    ----------
    signal_mV : np.ndarray
        Time-domain signal in millivolts (probe-tip corrected).
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

rows: list[dict] = []

print(f"\nStarting frequency sweep: {FREQ_START_HZ/1e3:.1f} kHz → {FREQ_STOP_HZ/1e6:.0f} MHz")
print(
    f"  Reference channel: D ({PROBE_D}x probe)  |  "
    f"Signal: (B−C)  (B={PROBE_B}x, C={PROBE_C}x)  |  Coupling: AC"
)
print(
    f"\n{'Freq (Hz)':>14}  {'Actual (Hz)':>12}  {'Gain (dB)':>10}  "
    f"{'Phase (deg)':>12}  {'D RMS (mV)':>11}  {'BC RMS (mV)':>12}  "
    f"{'dt (ns)':>9}  {'TB':>4}"
)
print("-" * 112)

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

    # ── 2. Capture (probe-corrected) ─────────────────────────────────────────
    chB, chC, chD, dt_ns, tb = capture_block(freq)
    diff_BC = chB - chC

    # ── 3. FFT of reference (D) and differential output (B-C) ───────────────
    f_D,    Y_D  = compute_fft(chD,     dt_ns)
    f_diff, Y_BC = compute_fft(diff_BC, dt_ns)

    # ── 4. Locate fundamental bin ────────────────────────────────────────────
    fund_bin = np.argmin(np.abs(f_diff - freq))
    D_fund   = Y_D[fund_bin]
    BC_fund  = Y_BC[fund_bin]

    mag_D  = np.abs(D_fund)
    mag_BC = np.abs(BC_fund)

    # ── 5. Gain and phase of (B-C) relative to D ─────────────────────────────
    if mag_D < 1e-12:
        gain_db   = np.nan
        phase_deg = np.nan
    else:
        gain_db   = 20.0 * np.log10(mag_BC / mag_D)
        phase_rad = np.angle(BC_fund) - np.angle(D_fund)
        phase_deg = float(np.degrees(np.unwrap([phase_rad])[0]))

    actual_freq = float(f_diff[fund_bin])
    D_rms_mV    = float(np.sqrt(np.mean(chD ** 2)))
    BC_rms_mV   = float(np.sqrt(np.mean(diff_BC ** 2)))

    print(
        f"{freq:14.1f}  {actual_freq:12.1f}  {gain_db:10.3f}  "
        f"{phase_deg:12.3f}  {D_rms_mV:11.4f}  {BC_rms_mV:12.4f}  "
        f"{dt_ns:9.3f}  {tb:4d}"
    )

    # ── 6. Accumulate row ────────────────────────────────────────────────────
    rows.append({
        # Identification
        "timestamp":                RUN_TIMESTAMP,
        "set_freq_hz":              float(freq),
        "actual_fund_freq_hz":      actual_freq,
        # Primary results — (B-C) relative to D
        "gain_dB":                  gain_db,
        "phase_deg":                phase_deg,
        # Waveform statistics (probe-tip corrected)
        "chB_rms_mV":               float(np.sqrt(np.mean(chB ** 2))),
        "chC_rms_mV":               float(np.sqrt(np.mean(chC ** 2))),
        "chD_rms_mV":               D_rms_mV,
        "diff_BC_rms_mV":           BC_rms_mV,
        "chB_peak_mV":              float(np.max(np.abs(chB))),
        "chC_peak_mV":              float(np.max(np.abs(chC))),
        "chD_peak_mV":              float(np.max(np.abs(chD))),
        "diff_BC_peak_mV":          float(np.max(np.abs(diff_BC))),
        "fft_mag_D_at_fund_mV":     float(mag_D),
        "fft_mag_BC_at_fund_mV":    float(mag_BC),
        "fund_bin_index":           int(fund_bin),
        # Acquisition parameters
        "sample_interval_ns":       dt_ns,
        "timebase_index":           tb,
        "n_samples":                N_SAMPLES,
        # Probe and coupling configuration (recorded for traceability)
        "coupling":                 "AC",
        "probe_B":                  PROBE_B,
        "probe_C":                  PROBE_C,
        "probe_D":                  PROBE_D,
        "ch_range_B":               CH_RANGE_B,
        "ch_range_C":               CH_RANGE_C,
        "ch_range_D":               CH_RANGE_D,
        # Generator configuration
        "gen_pk2pk_uV":             GEN_PK2PK_UV,
        "gen_offset_uV":            GEN_OFFSET_UV,
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

df = df.astype({
    "set_freq_hz":              "float64",
    "actual_fund_freq_hz":      "float64",
    "gain_dB":                  "float64",
    "phase_deg":                "float64",
    "chB_rms_mV":               "float64",
    "chC_rms_mV":               "float64",
    "chD_rms_mV":               "float64",
    "diff_BC_rms_mV":           "float64",
    "chB_peak_mV":              "float64",
    "chC_peak_mV":              "float64",
    "chD_peak_mV":              "float64",
    "diff_BC_peak_mV":          "float64",
    "fft_mag_D_at_fund_mV":     "float64",
    "fft_mag_BC_at_fund_mV":    "float64",
    "fund_bin_index":           "int64",
    "sample_interval_ns":       "float64",
    "timebase_index":           "int64",
    "n_samples":                "int64",
    "coupling":                 "string",
    "probe_B":                  "int64",
    "probe_C":                  "int64",
    "probe_D":                  "int64",
    "ch_range_B":               "int64",
    "ch_range_C":               "int64",
    "ch_range_D":               "int64",
    "gen_pk2pk_uV":             "int64",
    "gen_offset_uV":            "int64",
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
    f"Bode Plot — (B−C) vs D  |  PicoScope 5000 Series\n"
    f"Run: {RUN_TIMESTAMP}  |  {N_FREQ_POINTS} points  |  "
    f"{GEN_PK2PK_UV/1e6:.1f} Vpp  |  12-bit  |  AC coupled  |  "
    f"Probes: B={PROBE_B}x  C={PROBE_C}x  D={PROBE_D}x (ref)",
    fontsize=11
)

ax1.semilogx(
    df["set_freq_hz"], df["gain_dB"],
    "b.-", linewidth=1.5, markersize=5, label="Gain (B−C) / D"
)
ax1.set_ylabel("Gain (dB)")
ax1.grid(True, which="both", linestyle="--", alpha=0.6)
ax1.axhline(0, color="k", linewidth=0.8)
ax1.legend(fontsize=9)

ax2.semilogx(
    df["set_freq_hz"], df["phase_deg"],
    "r.-", linewidth=1.5, markersize=5, label="Phase (B−C) − D"
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
