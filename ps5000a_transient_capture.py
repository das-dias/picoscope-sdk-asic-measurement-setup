"""
================================================================================
ps5000a_transient_capture.py
PicoScope 5000 Series — Transient Capture (B, C, D + differential B−C)
================================================================================
Setup  (identical to bode sweep and spectral metrics scripts)
  - Generator (AWG out) → your ASIC input AND Channel D (x10 probe reference)
  - ASIC output+ → Channel B (x10 probe)
  - ASIC output- → Channel C (x10 probe)
  - Channel A     → unused (disabled)
  - (B - C)       → differential ASIC output (CMRR-rejected)
  - All active channels are AC coupled

Design intent
  This script is the time-domain companion to the frequency-domain scripts.
  Rather than stepping a sine generator across frequencies, it arms the
  oscilloscope for a single triggered capture of a transient event — a pulse,
  a step response, a power-on sequence, or any brief anomaly — and exports the
  raw waveforms alongside a summary of key time-domain statistics.

  The generator can optionally be used to inject a stimulus pulse. Set
  GEN_ENABLED = False if your stimulus comes from an external source.

  Pre-trigger samples are supported so that the record starts before the
  trigger edge, giving context for the signal state prior to the event.

Trigger modes available via TRIGGER_MODE constant
  "rising"   — rising edge on Ch D above TRIGGER_THRESHOLD_MV (probe-tip)
  "falling"  — falling edge on Ch D below TRIGGER_THRESHOLD_MV (probe-tip)
  "auto"     — no trigger condition; captures immediately after arming
               (useful for free-running signals or debugging)

Output
  - outputs/transient_waveforms_<timestamp>.csv   (one row per sample)
  - outputs/transient_summary_<timestamp>.csv     (one row — run metadata)
  - outputs/transient_plot_<timestamp>.png        (4-panel time-domain plot)

Dependencies:  pip install picosdk numpy matplotlib pandas
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

from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import adc2mV, assert_pico_ok

# ──────────────────────────────────────────────────────────────────────────────
# USER CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# --- Capture parameters ------------------------------------------------------
N_SAMPLES           = 16384     # total samples in the capture block (power-of-2)
PRE_TRIGGER_SAMPLES = 2048      # samples captured before the trigger point
#                               # POST_TRIGGER = N_SAMPLES - PRE_TRIGGER_SAMPLES

# Desired sample rate — the script finds the fastest timebase whose interval
# is at or below 1/DESIRED_SAMPLE_RATE_HZ. Actual rate is printed at runtime.
DESIRED_SAMPLE_RATE_HZ = 125_000_000    # 125 MS/s → 8 ns per sample

# --- Trigger -----------------------------------------------------------------
# Channel D (generator / reference probe) is used as the trigger source,
# consistent with the bode sweep and spectral metrics scripts.
#
# TRIGGER_MODE options:
#   "rising"   — arm on rising edge of Ch D
#   "falling"  — arm on falling edge of Ch D
#   "auto"     — capture immediately (no edge condition)
TRIGGER_MODE            = "rising"

# Trigger threshold in millivolts at the PROBE TIP (after x10 correction).
# The SDK accepts ADC counts; this value is converted automatically below.
TRIGGER_THRESHOLD_MV    = 100.0     # mV at probe tip

# Auto-trigger timeout — milliseconds to wait for a trigger before capturing
# anyway. Set to 0 to wait indefinitely (only recommended if a trigger is
# guaranteed to arrive).
AUTO_TRIGGER_MS         = 2000      # 2 seconds

# --- Probe attenuation -------------------------------------------------------
# 1   = 1x probe  (no attenuation)
# 10  = 10x probe (probe tip voltage = ADC reading × 10)
# 100 = 100x probe
#
# Channel A  → disabled
# Channel B  → ASIC output+  (10x probe)
# Channel C  → ASIC output-  (10x probe)
# Channel D  → generator / reference (10x probe)
PROBE_A = 1     # unused — declared for completeness
PROBE_B = 10
PROBE_C = 10
PROBE_D = 10

# --- Voltage range indices ---------------------------------------------------
# With a 10x probe the BNC sees 1/10 of the probe-tip voltage.
# A ±2 V ADC range (index 7) accommodates probe-tip signals up to ±20 V.
#
# PS5000A ranges:
#   1=±20 mV  2=±50 mV  3=±100 mV  4=±200 mV  5=±500 mV
#   6=±1 V    7=±2 V    8=±5 V     9=±10 V    10=±20 V
CH_RANGE_A      = 7         # unused — set to avoid SDK error on open
CH_RANGE_B      = 7         # ±2 V at BNC (10x probe → ±20 V at tip)
CH_RANGE_C      = 7         # ±2 V at BNC (10x probe → ±20 V at tip)
CH_RANGE_D      = 7         # ±2 V at BNC (10x probe → ±20 V at tip)

# --- Channel coupling --------------------------------------------------------
# PS5000A_AC = 0  →  AC coupled (blocks DC, high-pass ~1 Hz corner)
# PS5000A_DC = 1  →  DC coupled (full bandwidth including DC)
#
# All channels are AC coupled to match the bode sweep and spectral metrics
# scripts. Switch to PS5000A_DC if your transient includes a DC component
# or if the signal frequency is below ~10 Hz.
COUPLING        = ps.PS5000A_COUPLING["PS5000A_AC"]

# --- Optional stimulus generator ---------------------------------------------
# Set GEN_ENABLED = True to output a burst from the built-in generator
# immediately before arming the trigger. This is useful for repeatable
# step-response or impulse-response measurements.
# Set GEN_ENABLED = False if the transient comes from an external source.
GEN_ENABLED     = False
GEN_PK2PK_UV    = 2_000_000    # 2 Vpp
GEN_OFFSET_UV   = 0
GEN_FREQ_HZ     = 1_000        # 1 kHz — used only when GEN_ENABLED = True
#                               # Set to match your stimulus frequency
GEN_WAVE_SINE   = 0            # PS5000A_SINE
GEN_WAVE_SQUARE = 1            # PS5000A_SQUARE — useful for step response
GEN_WAVETYPE    = GEN_WAVE_SINE

# --- ADC resolution ----------------------------------------------------------
RESOLUTION      = ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_12BIT"]

# --- Output ------------------------------------------------------------------
OUTPUT_DIR      = Path("outputs")

# ──────────────────────────────────────────────────────────────────────────────
# DERIVED CONSTANTS  (do not edit below this line)
# ──────────────────────────────────────────────────────────────────────────────
POST_TRIGGER_SAMPLES = N_SAMPLES - PRE_TRIGGER_SAMPLES
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Trigger direction codes used by ps5000aSetSimpleTrigger
_TRIG_DIR = {
    "rising":  2,   # PS5000A_RISING
    "falling": 3,   # PS5000A_FALLING
    "auto":    2,   # direction unused in auto mode but must be a valid value
}

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
    f"Ch C: {PROBE_C}x  |  Ch D: {PROBE_D}x (trigger/reference)"
)
print(
    f"  N samples: {N_SAMPLES}  |  Pre-trigger: {PRE_TRIGGER_SAMPLES}  |  "
    f"Post-trigger: {POST_TRIGGER_SAMPLES}  |  "
    f"Trigger: {TRIGGER_MODE.upper()} on Ch D  |  "
    f"Threshold: {TRIGGER_THRESHOLD_MV:.1f} mV (probe-tip)"
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
    (CH_D, 1, CH_RANGE_D),      # trigger/ref,   AC coupled, 10x probe
]:
    status[f"setCh{ch_enum}"] = ps.ps5000aSetChannel(
        chandle,
        ch_enum,
        en,
        COUPLING,
        range_idx,
        ctypes.c_float(0.0)
    )
    assert_pico_ok(status[f"setCh{ch_enum}"])

# Maximum ADC value for this resolution (used by adc2mV and threshold scaling)
maxADC = ctypes.c_int16()
status["maxADC"] = ps.ps5000aMaximumValue(chandle, ctypes.byref(maxADC))
assert_pico_ok(status["maxADC"])

# ──────────────────────────────────────────────────────────────────────────────
# TIMEBASE SELECTION
# Find the fastest timebase whose sample interval is at or below the
# reciprocal of DESIRED_SAMPLE_RATE_HZ.
# ──────────────────────────────────────────────────────────────────────────────
def find_timebase_for_rate(desired_rate_hz: float, n_samples: int) -> tuple[int, float]:
    """
    Walk ps5000aGetTimebase2 to find the fastest available timebase whose
    sample interval is at or below 1 / *desired_rate_hz*.

    Parameters
    ----------
    desired_rate_hz : float
        Target sample rate in Hz.
    n_samples : int
        Block capture length in samples.

    Returns
    -------
    timebase : int
        Timebase index for ps5000aRunBlock.
    dt_ns : float
        Actual sample interval in nanoseconds.
    """
    desired_interval_ns = 1e9 / desired_rate_hz
    interval_ns = ctypes.c_float()
    max_samp    = ctypes.c_int32()

    for tb in range(1, 2**23):
        st = ps.ps5000aGetTimebase2(
            chandle, tb, n_samples,
            ctypes.byref(interval_ns), ctypes.byref(max_samp), 0
        )
        if st == 0 and interval_ns.value <= desired_interval_ns:
            return tb, float(interval_ns.value)

    # Fallback: return the slowest valid timebase found
    return tb, float(interval_ns.value)

timebase, dt_ns = find_timebase_for_rate(DESIRED_SAMPLE_RATE_HZ, N_SAMPLES)
actual_sample_rate_hz = 1.0 / (dt_ns * 1e-9)
record_duration_us    = N_SAMPLES * dt_ns / 1e3
pretrig_duration_us   = PRE_TRIGGER_SAMPLES * dt_ns / 1e3

print(
    f"\n  Timebase index : {timebase}"
    f"\n  Sample interval: {dt_ns:.3f} ns  ({actual_sample_rate_hz/1e6:.1f} MS/s)"
    f"\n  Record duration: {record_duration_us:.1f} µs  "
    f"(pre={pretrig_duration_us:.1f} µs  "
    f"post={(record_duration_us - pretrig_duration_us):.1f} µs)"
)

# ──────────────────────────────────────────────────────────────────────────────
# OPTIONAL STIMULUS GENERATOR
# ──────────────────────────────────────────────────────────────────────────────
if GEN_ENABLED:
    wavetype   = ctypes.c_int32(GEN_WAVETYPE)
    sweepType  = ctypes.c_int32(0)     # PS5000A_UP — no sweep
    trigType   = ctypes.c_int32(0)     # PS5000A_SIGGEN_RISING
    trigSource = ctypes.c_int32(0)     # PS5000A_SIGGEN_NONE

    status["sigGen"] = ps.ps5000aSetSigGenBuiltInV2(
        chandle,
        GEN_OFFSET_UV,
        GEN_PK2PK_UV,
        wavetype,
        ctypes.c_double(GEN_FREQ_HZ),
        ctypes.c_double(GEN_FREQ_HZ),
        ctypes.c_double(0.0),
        ctypes.c_double(1.0),
        sweepType, 0, 0, 0,
        trigType, trigSource, 0
    )
    assert_pico_ok(status["sigGen"])
    print(f"\n  Generator: ON  |  {GEN_FREQ_HZ/1e3:.3f} kHz  |  "
          f"{GEN_PK2PK_UV/1e6:.1f} Vpp  |  "
          f"{'Sine' if GEN_WAVETYPE == GEN_WAVE_SINE else 'Square'}")
    time.sleep(0.05)    # allow generator output to stabilise
else:
    print("\n  Generator: OFF (external stimulus)")

# ──────────────────────────────────────────────────────────────────────────────
# TRIGGER SETUP
# ──────────────────────────────────────────────────────────────────────────────
# Convert probe-tip threshold (mV) → BNC threshold (mV) → ADC counts.
# adc2mV maps maxADC counts → the full-scale range voltage.
# Invert: threshold_adc = (threshold_bnc_mV / range_mv) * maxADC
_RANGE_MV = {
    1: 20, 2: 50, 3: 100, 4: 200, 5: 500,
    6: 1000, 7: 2000, 8: 5000, 9: 10000, 10: 20000,
}
threshold_bnc_mV = TRIGGER_THRESHOLD_MV / PROBE_D
range_mv_D       = _RANGE_MV[CH_RANGE_D]
threshold_adc    = int((threshold_bnc_mV / range_mv_D) * maxADC.value)
threshold_adc    = max(-maxADC.value, min(maxADC.value, threshold_adc))  # clamp

if TRIGGER_MODE == "auto":
    # Disable edge condition — scope captures as soon as it is armed.
    # AUTO_TRIGGER_MS controls how long it waits before forcing a capture.
    trig_enable    = 0
    trig_direction = _TRIG_DIR["rising"]    # value irrelevant when disabled
    trig_threshold = 0
    print(f"\n  Trigger: AUTO (no edge condition, timeout={AUTO_TRIGGER_MS} ms)")
else:
    trig_enable    = 1
    trig_direction = _TRIG_DIR[TRIGGER_MODE]
    trig_threshold = threshold_adc
    print(
        f"\n  Trigger: {TRIGGER_MODE.upper()} edge on Ch D  |  "
        f"Threshold: {TRIGGER_THRESHOLD_MV:.1f} mV probe-tip  "
        f"({threshold_bnc_mV:.2f} mV at BNC)  |  "
        f"ADC counts: {threshold_adc}  |  "
        f"Auto-timeout: {AUTO_TRIGGER_MS} ms"
    )

status["trig"] = ps.ps5000aSetSimpleTrigger(
    chandle,
    trig_enable,
    CH_D,
    trig_threshold,
    trig_direction,
    0,                  # delay (samples) — 0 = trigger is at PRE_TRIGGER boundary
    AUTO_TRIGGER_MS
)
assert_pico_ok(status["trig"])

# ──────────────────────────────────────────────────────────────────────────────
# ARM AND WAIT FOR TRIGGER
# ──────────────────────────────────────────────────────────────────────────────
print("\nArming oscilloscope — waiting for trigger...")

status["runBlock"] = ps.ps5000aRunBlock(
    chandle,
    PRE_TRIGGER_SAMPLES,
    POST_TRIGGER_SAMPLES,
    timebase,
    None,               # timeIndisposedMs — not needed here
    0,                  # segment index
    None,               # callback — use polling instead
    None
)
assert_pico_ok(status["runBlock"])

ready = ctypes.c_int16(0)
while ready.value == 0:
    ps.ps5000aIsReady(chandle, ctypes.byref(ready))
    time.sleep(0.001)

print("  Trigger received — retrieving data.")

# ──────────────────────────────────────────────────────────────────────────────
# RETRIEVE DATA
# ──────────────────────────────────────────────────────────────────────────────
RATIO_NONE = ps.PS5000A_RATIO_MODE["PS5000A_RATIO_MODE_NONE"]

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

if overflow.value != 0:
    print(
        f"  WARNING: overflow flag = {overflow.value:#06x}  "
        f"— one or more channels clipped. Reduce CH_RANGE or probe amplitude."
    )

# ──────────────────────────────────────────────────────────────────────────────
# PROBE CORRECTION
# adc2mV → BNC millivolts; × probe factor → probe-tip millivolts
# ──────────────────────────────────────────────────────────────────────────────
chB_mV    = np.array(adc2mV(bufB, CH_RANGE_B, maxADC), dtype=np.float64) * PROBE_B
chC_mV    = np.array(adc2mV(bufC, CH_RANGE_C, maxADC), dtype=np.float64) * PROBE_C
chD_mV    = np.array(adc2mV(bufD, CH_RANGE_D, maxADC), dtype=np.float64) * PROBE_D
diff_BC   = chB_mV - chC_mV

# ──────────────────────────────────────────────────────────────────────────────
# TIME AXIS
# t = 0 is defined at the trigger point (sample index PRE_TRIGGER_SAMPLES)
# ──────────────────────────────────────────────────────────────────────────────
t_us = (np.arange(N_SAMPLES) - PRE_TRIGGER_SAMPLES) * dt_ns / 1e3

# ──────────────────────────────────────────────────────────────────────────────
# CLOSE DEVICE
# ──────────────────────────────────────────────────────────────────────────────
status["close"] = ps.ps5000aCloseUnit(chandle)
assert_pico_ok(status["close"])
print("Device closed.")

# ──────────────────────────────────────────────────────────────────────────────
# TIME-DOMAIN STATISTICS
# ──────────────────────────────────────────────────────────────────────────────
def channel_stats(name: str, data: np.ndarray, t: np.ndarray) -> dict:
    """
    Compute basic time-domain statistics for a single waveform.

    Parameters
    ----------
    name : str
        Channel label used as a key prefix in the returned dict.
    data : np.ndarray
        Waveform in millivolts (probe-tip corrected).
    t : np.ndarray
        Time axis in microseconds, with t=0 at the trigger point.

    Returns
    -------
    dict
        min, max, peak-to-peak, mean (DC offset), RMS, and the time
        at which the absolute peak occurs.
    """
    abs_peak_idx = int(np.argmax(np.abs(data)))
    return {
        f"{name}_min_mV":           float(np.min(data)),
        f"{name}_max_mV":           float(np.max(data)),
        f"{name}_pk2pk_mV":         float(np.max(data) - np.min(data)),
        f"{name}_mean_mV":          float(np.mean(data)),
        f"{name}_rms_mV":           float(np.sqrt(np.mean(data ** 2))),
        f"{name}_abs_peak_mV":      float(np.abs(data[abs_peak_idx])),
        f"{name}_abs_peak_time_us": float(t[abs_peak_idx]),
    }

stats: dict = {}
stats.update(channel_stats("chB",    chB_mV,  t_us))
stats.update(channel_stats("chC",    chC_mV,  t_us))
stats.update(channel_stats("chD",    chD_mV,  t_us))
stats.update(channel_stats("diffBC", diff_BC, t_us))

print("\n── Time-domain statistics (probe-tip corrected) ─────────────────────────────")
col_w = 28
for k, v in stats.items():
    print(f"  {k:<{col_w}}: {v:>10.4f} mV" if "time" not in k
          else f"  {k:<{col_w}}: {v:>10.4f} µs")

# ──────────────────────────────────────────────────────────────────────────────
# BUILD DataFrames
# ──────────────────────────────────────────────────────────────────────────────

# ── Waveform DataFrame — one row per sample ───────────────────────────────────
df_wave = pd.DataFrame({
    "timestamp":        RUN_TIMESTAMP,
    "sample_idx":       np.arange(N_SAMPLES, dtype=np.int64),
    "time_us":          t_us.astype(np.float64),
    "chB_mV":           chB_mV,    # probe-tip corrected
    "chC_mV":           chC_mV,    # probe-tip corrected
    "chD_mV":           chD_mV,    # probe-tip corrected (reference)
    "diff_BC_mV":       diff_BC,   # probe-tip corrected
})

df_wave = df_wave.astype({
    "timestamp":  "string",
    "sample_idx": "int64",
    "time_us":    "float64",
    "chB_mV":     "float64",
    "chC_mV":     "float64",
    "chD_mV":     "float64",
    "diff_BC_mV": "float64",
})

# ── Summary DataFrame — one row summarising the entire run ───────────────────
summary = {
    # Run identification
    "timestamp":                    RUN_TIMESTAMP,
    # Trigger configuration
    "trigger_mode":                 TRIGGER_MODE,
    "trigger_channel":              "D",
    "trigger_threshold_mV":         TRIGGER_THRESHOLD_MV,
    "trigger_threshold_adc":        threshold_adc,
    "auto_trigger_ms":              AUTO_TRIGGER_MS,
    # Acquisition parameters
    "n_samples":                    N_SAMPLES,
    "pre_trigger_samples":          PRE_TRIGGER_SAMPLES,
    "post_trigger_samples":         POST_TRIGGER_SAMPLES,
    "timebase_index":               timebase,
    "sample_interval_ns":           dt_ns,
    "actual_sample_rate_hz":        actual_sample_rate_hz,
    "record_duration_us":           record_duration_us,
    "pretrig_duration_us":          pretrig_duration_us,
    "overflow_flag":                int(overflow.value),
    # Probe and coupling configuration
    "coupling":                     "AC",
    "probe_B":                      PROBE_B,
    "probe_C":                      PROBE_C,
    "probe_D":                      PROBE_D,
    "ch_range_B":                   CH_RANGE_B,
    "ch_range_C":                   CH_RANGE_C,
    "ch_range_D":                   CH_RANGE_D,
    # Generator configuration
    "gen_enabled":                  GEN_ENABLED,
    "gen_freq_hz":                  GEN_FREQ_HZ   if GEN_ENABLED else None,
    "gen_pk2pk_uV":                 GEN_PK2PK_UV  if GEN_ENABLED else None,
    "gen_offset_uV":                GEN_OFFSET_UV if GEN_ENABLED else None,
    "gen_wavetype":                 GEN_WAVETYPE  if GEN_ENABLED else None,
    # Time-domain statistics
    **stats,
}

df_summary = pd.DataFrame([summary])

# Enforce column order: identification → trigger → acquisition →
# hardware config → generator → statistics
fixed_cols = [
    "timestamp",
    "trigger_mode", "trigger_channel", "trigger_threshold_mV",
    "trigger_threshold_adc", "auto_trigger_ms",
    "n_samples", "pre_trigger_samples", "post_trigger_samples",
    "timebase_index", "sample_interval_ns", "actual_sample_rate_hz",
    "record_duration_us", "pretrig_duration_us", "overflow_flag",
    "coupling", "probe_B", "probe_C", "probe_D",
    "ch_range_B", "ch_range_C", "ch_range_D",
    "gen_enabled", "gen_freq_hz", "gen_pk2pk_uV",
    "gen_offset_uV", "gen_wavetype",
] + list(stats.keys())

df_summary = df_summary[fixed_cols]

# ──────────────────────────────────────────────────────────────────────────────
# EXPORT CSVs
# ──────────────────────────────────────────────────────────────────────────────
wave_csv_path    = OUTPUT_DIR / f"transient_waveforms_{RUN_TIMESTAMP}.csv"
summary_csv_path = OUTPUT_DIR / f"transient_summary_{RUN_TIMESTAMP}.csv"

df_wave.to_csv(wave_csv_path,    index=False, float_format="%.6f")
df_summary.to_csv(summary_csv_path, index=False, float_format="%.6f")

print(f"\nWaveform CSV saved → {wave_csv_path}")
print(f"Summary CSV  saved → {summary_csv_path}")

# ──────────────────────────────────────────────────────────────────────────────
# PLOT — four panels: Ch B, Ch C, Ch D (reference), B−C differential
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
fig.suptitle(
    f"Transient Capture  |  PicoScope 5000 Series\n"
    f"Run: {RUN_TIMESTAMP}  |  "
    f"{N_SAMPLES} samples  |  "
    f"{actual_sample_rate_hz/1e6:.1f} MS/s  |  "
    f"{dt_ns:.2f} ns/sample  |  "
    f"12-bit  |  AC coupled  |  "
    f"Trigger: {TRIGGER_MODE.upper()} on Ch D @ {TRIGGER_THRESHOLD_MV:.0f} mV  |  "
    f"Probes: B={PROBE_B}x  C={PROBE_C}x  D={PROBE_D}x",
    fontsize=10
)

_panels = [
    (axes[0], chB_mV,  "Ch B  (ASIC OUT+)",          "steelblue"),
    (axes[1], chC_mV,  "Ch C  (ASIC OUT−)",          "darkorange"),
    (axes[2], chD_mV,  "Ch D  (Generator / Ref)",    "green"),
    (axes[3], diff_BC, "B − C  (Differential out)",  "crimson"),
]

for ax, data, label, colour in _panels:
    ax.plot(t_us, data, linewidth=0.8, color=colour, label=label)
    ax.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5,
               label="Trigger point")
    ax.axhline(0, color="k", linewidth=0.5, linestyle=":", alpha=0.4)
    ax.set_ylabel("Amplitude (mV\nprobe-tip)", fontsize=8)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)

    # Annotate peak value
    abs_peak_idx = int(np.argmax(np.abs(data)))
    ax.annotate(
        f"peak = {data[abs_peak_idx]:.2f} mV",
        xy=(t_us[abs_peak_idx], data[abs_peak_idx]),
        xytext=(10, 10), textcoords="offset points",
        fontsize=7, color=colour,
        arrowprops=dict(arrowstyle="->", color=colour, lw=0.8)
    )

axes[-1].set_xlabel("Time (µs)  —  t = 0 at trigger point")

plt.tight_layout()
plot_path = OUTPUT_DIR / f"transient_plot_{RUN_TIMESTAMP}.png"
plt.savefig(plot_path, dpi=150)
print(f"Plot saved         → {plot_path}")
