"""
================================================================================
ps5000a_spectral_metrics.py
PicoScope 5000 Series — SNR, SINAD, THD, HD2, HD3 of (B−C) differential output
================================================================================
Setup
  - Generator (AWG out) → your ASIC input AND Channel D (x10 probe reference)
  - ASIC output+ → Channel B (x10 probe)
  - ASIC output- → Channel C (x10 probe)
  - Channel A     → unused (disabled)
  - (B - C)       → differential ASIC output (CMRR-rejected)
  - All active channels are AC coupled

What the script does
  1. Programs the generator at each of three test frequencies
  2. Captures channels B, C, D in block mode (A disabled)
  3. Applies per-channel probe correction to raw ADC data
  4. Computes FFT of (B - C) with Blackman-Harris windowing
  5. Computes SNR, SINAD, THD, HD2, HD3 in dB
  6. Exports a summary CSV (one row per frequency) and a waveform CSV
     (one row per sample per frequency) to the outputs/ directory

Dependencies:  pip install picosdk numpy matplotlib scipy pandas
================================================================================
"""

import ctypes
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal.windows import blackmanharris

from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import adc2mV, assert_pico_ok

# ──────────────────────────────────────────────────────────────────────────────
# USER CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# --- Test frequencies --------------------------------------------------------
TEST_FREQUENCIES_HZ = [100_000, 1_000_000, 10_000_000]   # 100 kHz, 1 MHz, 10 MHz

# --- Capture parameters ------------------------------------------------------
N_SAMPLES     = 16384       # power-of-2; larger → finer spectral bins
N_HARMONICS   = 5           # harmonics included in THD: H2 … H(N_HARMONICS+1)
GUARD_BINS    = 3           # bins excluded either side of each harmonic
SETTLE_TIME_S = 0.10        # settle after generator frequency change

# --- Probe attenuation -------------------------------------------------------
# Set the attenuation factor for each channel independently.
# 1   = 1x probe  (direct connection, no attenuation)
# 10  = 10x probe (probe tip voltage = ADC reading × 10)
# 100 = 100x probe
#
# Channel A  → disabled (not used in this configuration)
# Channel B  → ASIC output+  (10x probe)
# Channel C  → ASIC output-  (10x probe)
# Channel D  → generator monitor (10x probe) — reference channel
#
# These factors are applied in software after adc2mV() conversion.
# They do NOT affect the ADC input range selection (CH_RANGE_*); choose
# the range that prevents clipping of the attenuated signal at the BNC input.
PROBE_A = 1     # Ch A disabled — value unused but declared for completeness
PROBE_B = 10    # ASIC output+
PROBE_C = 10    # ASIC output-
PROBE_D = 10    # Generator monitor (reference)

# --- Voltage range indices ---------------------------------------------------
# Select the tightest range that does not clip the attenuated BNC input.
# With a 10x probe the BNC sees 1/10 of the probe-tip voltage, so a
# ±2 V ADC range (index 7) accommodates probe-tip signals up to ±20 V.
#
# PS5000A ranges:
#   1=±20 mV  2=±50 mV  3=±100 mV  4=±200 mV  5=±500 mV
#   6=±1 V    7=±2 V    8=±5 V     9=±10 V    10=±20 V
CH_RANGE_A    = 7           # unused — set to avoid SDK error on open
CH_RANGE_B    = 7           # ±2 V at BNC (10x probe → ±20 V at tip)
CH_RANGE_C    = 7           # ±2 V at BNC (10x probe → ±20 V at tip)
CH_RANGE_D    = 7           # ±2 V at BNC (10x probe → ±20 V at tip)

# --- Channel coupling --------------------------------------------------------
# PS5000A_AC = 0  →  AC coupled (blocks DC, high-pass ~1 Hz corner)
# PS5000A_DC = 1  →  DC coupled (full bandwidth including DC component)
#
# All channels are AC coupled to reject DC offsets and power-supply
# common-mode noise from the ASIC supply rails. Switch to PS5000A_DC
# if you need to measure absolute DC levels or signals below ~10 Hz.
COUPLING      = ps.PS5000A_COUPLING["PS5000A_AC"]

# --- Generator ---------------------------------------------------------------
GEN_PK2PK_UV  = 2_000_000  # 2 Vpp
GEN_OFFSET_UV = 0
RESOLUTION    = ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_12BIT"]

# --- Output ------------------------------------------------------------------
OUTPUT_DIR = Path("outputs")

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

maxADC = ctypes.c_int16()
status["maxADC"] = ps.ps5000aMaximumValue(chandle, ctypes.byref(maxADC))
assert_pico_ok(status["maxADC"])

# ──────────────────────────────────────────────────────────────────────────────
# TIMEBASE HELPER
# ──────────────────────────────────────────────────────────────────────────────
def find_timebase(target_fs_hz: float, n_samples: int) -> tuple[int, float]:
    """
    Walk ps5000aGetTimebase2 to find the fastest timebase that captures
    at least 20 complete cycles of *target_fs_hz*.

    Parameters
    ----------
    target_fs_hz : float
        Target signal frequency (Hz).
    n_samples : int
        Block capture length in samples.

    Returns
    -------
    timebase : int
    dt_ns : float
        Sample interval in nanoseconds.
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
# Ch D is the reference; Ch A is not captured.
# ──────────────────────────────────────────────────────────────────────────────
def capture_block(freq_hz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """
    Arm and retrieve one block capture from channels B, C, D.
    Applies per-channel probe attenuation after adc2mV conversion so that
    returned arrays represent the true signal at the probe tip.

    Parameters
    ----------
    freq_hz : float
        Current generator frequency (Hz).

    Returns
    -------
    chB_mV, chC_mV, chD_mV : np.ndarray
        Probe-tip voltages in millivolts (attenuation-corrected).
        chD_mV is the generator monitor reference.
    dt_ns : float
        Sample interval in nanoseconds.
    tb : int
        Timebase index used.
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
# SPECTRAL METRICS COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────
def spectral_metrics(
    signal_mV: np.ndarray,
    dt_ns: float,
    fund_freq_hz: float,
    n_harmonics: int = 5,
    guard_bins: int  = 3,
) -> dict:
    """
    Compute SNR, SINAD, THD, HD2, and HD3 from a windowed FFT.

    Uses a Blackman-Harris window (−92 dB sidelobes) to suppress spectral
    leakage so low-level harmonic products are not masked.

    Input signal is expected to be probe-tip corrected (true mV at DUT output).

    Parameters
    ----------
    signal_mV : np.ndarray
        Time-domain differential signal (B − C) in millivolts, probe-corrected.
    dt_ns : float
        Sample interval in nanoseconds.
    fund_freq_hz : float
        Expected fundamental frequency in Hz.
    n_harmonics : int
        Number of harmonics beyond the fundamental to include in THD.
    guard_bins : int
        FFT bins to exclude either side of each harmonic when computing noise.

    Returns
    -------
    dict
        All computed metrics and intermediate spectral data.
    """
    n  = len(signal_mV)
    fs = 1.0 / (dt_ns * 1e-9)
    df = fs / n

    win = blackmanharris(n)
    cg  = float(np.mean(win))
    Y   = np.fft.rfft(signal_mV * win) / (n * cg)
    mag = np.abs(Y)
    freqs = np.fft.rfftfreq(n, d=dt_ns * 1e-9)

    # ── Locate fundamental ───────────────────────────────────────────────────
    fund_bin = int(round(fund_freq_hz / df))
    fund_bin = max(1, min(fund_bin, len(mag) - 1))
    search   = 5
    lo = max(1,            fund_bin - search)
    hi = min(len(mag) - 1, fund_bin + search)
    fund_bin = lo + int(np.argmax(mag[lo:hi + 1]))
    H1       = float(mag[fund_bin])
    fund_hz  = float(freqs[fund_bin])

    # ── Locate harmonics ─────────────────────────────────────────────────────
    harmonic_bins    = []
    harmonic_mags_mV = []
    for h in range(2, n_harmonics + 2):
        hbin = int(round(h * fund_hz / df))
        if hbin >= len(mag):
            break
        lo_h = max(1, hbin - search)
        hi_h = min(len(mag) - 1, hbin + search)
        hbin = lo_h + int(np.argmax(mag[lo_h:hi_h + 1]))
        harmonic_bins.append(hbin)
        harmonic_mags_mV.append(float(mag[hbin]))

    H2 = harmonic_mags_mV[0] if len(harmonic_mags_mV) > 0 else 0.0
    H3 = harmonic_mags_mV[1] if len(harmonic_mags_mV) > 1 else 0.0

    # ── Signal mask (fundamental + harmonics ± guard) ────────────────────────
    n_fft    = len(mag)
    sig_mask = np.zeros(n_fft, dtype=bool)
    for hb in [fund_bin] + harmonic_bins:
        lo_g = max(0, hb - guard_bins)
        hi_g = min(n_fft - 1, hb + guard_bins)
        sig_mask[lo_g:hi_g + 1] = True

    # ── Noise floor (excludes DC, fundamental, harmonics) ────────────────────
    noise_mask              = ~sig_mask
    noise_mask[:guard_bins] = False
    noise_rms = float(np.sqrt(np.sum(mag[noise_mask] ** 2)))

    # ── Harmonic + noise (SINAD denominator) ─────────────────────────────────
    harm_mask = np.zeros(n_fft, dtype=bool)
    for hb in harmonic_bins:
        lo_g = max(0, hb - guard_bins)
        hi_g = min(n_fft - 1, hb + guard_bins)
        harm_mask[lo_g:hi_g + 1] = True

    harm_and_noise_rms = float(
        np.sqrt(np.sum(mag[harm_mask | noise_mask] ** 2))
    )

    # ── Metrics ──────────────────────────────────────────────────────────────
    eps = 1e-15
    SNR_dB   = 20.0 * np.log10(H1 / (noise_rms          + eps))
    SINAD_dB = 20.0 * np.log10(H1 / (harm_and_noise_rms + eps))
    thd_num  = float(np.sqrt(sum(h ** 2 for h in harmonic_mags_mV)))
    THD_dB   = 20.0 * np.log10(thd_num / (H1 + eps))
    HD2_dB   = 20.0 * np.log10(H2      / (H1 + eps))
    HD3_dB   = 20.0 * np.log10(H3      / (H1 + eps))

    return {
        "SNR_dB":            float(SNR_dB),
        "SINAD_dB":          float(SINAD_dB),
        "THD_dB":            float(THD_dB),
        "HD2_dB":            float(HD2_dB),
        "HD3_dB":            float(HD3_dB),
        "fund_freq_hz":      fund_hz,
        "fund_mag_mV":       H1,
        "H2_mag_mV":         H2,
        "H3_mag_mV":         H3,
        "noise_rms_mV":      noise_rms,
        "harm_noise_rms_mV": harm_and_noise_rms,
        "mag_spectrum":      mag,
        "freqs_hz":          freqs,
        "fund_bin":          fund_bin,
        "harmonic_bins":     harmonic_bins,
        "harmonic_mags_mV":  harmonic_mags_mV,
        "df_hz":             df,
        "fs_hz":             fs,
    }

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
wavetype   = ctypes.c_int32(0)
sweepType  = ctypes.c_int32(0)
trigType   = ctypes.c_int32(0)
trigSource = ctypes.c_int32(0)

summary_rows:  list[dict] = []
waveform_rows: list[dict] = []

fig, axes = plt.subplots(
    len(TEST_FREQUENCIES_HZ), 1,
    figsize=(12, 5 * len(TEST_FREQUENCIES_HZ))
)
if len(TEST_FREQUENCIES_HZ) == 1:
    axes = [axes]

print("\n" + "=" * 95)
print(
    f"{'Freq (Hz)':>12}  {'SNR':>8}  {'SINAD':>8}  {'THD':>8}  "
    f"{'HD2':>8}  {'HD3':>8}  {'H1 (mV)':>10}  {'H2 (mV)':>10}  {'H3 (mV)':>10}"
)
print(f"{'':>12}  {'(dB)':>8}  {'(dB)':>8}  {'(dB)':>8}  {'(dB)':>8}  {'(dB)':>8}")
print("=" * 95)

for ax, freq in zip(axes, TEST_FREQUENCIES_HZ):

    # ── Program generator ─────────────────────────────────────────────────────
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

    # ── Capture (probe-corrected) ─────────────────────────────────────────────
    chB, chC, chD, dt_ns, tb = capture_block(freq)
    diff_BC = chB - chC
    t_us    = np.arange(N_SAMPLES) * dt_ns / 1e3

    # ── Metrics on (B-C) ─────────────────────────────────────────────────────
    m = spectral_metrics(
        diff_BC, dt_ns, freq,
        n_harmonics=N_HARMONICS, guard_bins=GUARD_BINS
    )

    print(
        f"{freq:12.0f}  {m['SNR_dB']:8.2f}  {m['SINAD_dB']:8.2f}  "
        f"{m['THD_dB']:8.2f}  {m['HD2_dB']:8.2f}  {m['HD3_dB']:8.2f}  "
        f"{m['fund_mag_mV']:10.4f}  {m['H2_mag_mV']:10.4f}  {m['H3_mag_mV']:10.4f}"
    )

    # ── Dynamic per-harmonic columns ──────────────────────────────────────────
    harmonic_cols: dict = {}
    for k, (hm, hf) in enumerate(
        zip(m["harmonic_mags_mV"],
            [m["freqs_hz"][b] for b in m["harmonic_bins"]]),
        start=2
    ):
        harmonic_cols[f"H{k}_mag_mV"]  = hm
        harmonic_cols[f"H{k}_freq_hz"] = hf
        harmonic_cols[f"HD{k}_dB"]     = 20.0 * np.log10(hm / (m["fund_mag_mV"] + 1e-15))

    # ── Summary row ───────────────────────────────────────────────────────────
    summary_rows.append({
        # Identification
        "timestamp":                RUN_TIMESTAMP,
        "set_freq_hz":              float(freq),
        "actual_fund_freq_hz":      m["fund_freq_hz"],
        # Primary metrics (computed on B-C)
        "SNR_dB":                   m["SNR_dB"],
        "SINAD_dB":                 m["SINAD_dB"],
        "THD_dB":                   m["THD_dB"],
        "HD2_dB":                   m["HD2_dB"],
        "HD3_dB":                   m["HD3_dB"],
        # Amplitudes — probe-tip corrected
        "H1_mag_mV":                m["fund_mag_mV"],
        "H2_mag_mV":                m["H2_mag_mV"],
        "H3_mag_mV":                m["H3_mag_mV"],
        "noise_rms_mV":             m["noise_rms_mV"],
        "harm_noise_rms_mV":        m["harm_noise_rms_mV"],
        # Waveform statistics — probe-tip corrected
        "chB_rms_mV":               float(np.sqrt(np.mean(chB ** 2))),
        "chC_rms_mV":               float(np.sqrt(np.mean(chC ** 2))),
        "chD_rms_mV":               float(np.sqrt(np.mean(chD ** 2))),
        "diff_BC_rms_mV":           float(np.sqrt(np.mean(diff_BC ** 2))),
        "chB_peak_mV":              float(np.max(np.abs(chB))),
        "chC_peak_mV":              float(np.max(np.abs(chC))),
        "chD_peak_mV":              float(np.max(np.abs(chD))),
        "diff_BC_peak_mV":          float(np.max(np.abs(diff_BC))),
        "chD_dc_offset_mV":         float(np.mean(chD)),
        "diff_BC_dc_offset_mV":     float(np.mean(diff_BC)),
        # Acquisition parameters
        "sample_rate_hz":           m["fs_hz"],
        "sample_interval_ns":       dt_ns,
        "freq_resolution_hz":       m["df_hz"],
        "n_samples":                N_SAMPLES,
        "timebase_index":           tb,
        # Probe and coupling configuration — recorded for full traceability
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
        "n_harmonics":              N_HARMONICS,
        "guard_bins":               GUARD_BINS,
        # Dynamic harmonic columns
        **harmonic_cols,
    })

    # ── Waveform rows — all three captured channels per sample ────────────────
    for i in range(N_SAMPLES):
        waveform_rows.append({
            "timestamp":   RUN_TIMESTAMP,
            "set_freq_hz": float(freq),
            "sample_idx":  i,
            "time_us":     float(t_us[i]),
            "chB_mV":      float(chB[i]),       # probe-tip corrected
            "chC_mV":      float(chC[i]),       # probe-tip corrected
            "chD_mV":      float(chD[i]),       # probe-tip corrected (reference)
            "diff_BC_mV":  float(diff_BC[i]),   # probe-tip corrected
        })

    # ── Spectrum plot ─────────────────────────────────────────────────────────
    freqs_khz = m["freqs_hz"] / 1e3
    mag_dBmV  = 20.0 * np.log10(m["mag_spectrum"] + 1e-15)
    xlim_khz  = min(
        (N_HARMONICS + 2) * m["fund_freq_hz"] / 1e3,
        m["freqs_hz"][-1] / 1e3
    )

    ax.plot(freqs_khz, mag_dBmV, linewidth=0.8, color="steelblue",
            label="(B−C) spectrum")
    ax.axvline(m["fund_freq_hz"] / 1e3, color="green", linestyle="--",
               linewidth=1.3, label=f"H1 = {m['fund_mag_mV']:.3f} mV")
    for k, hbin in enumerate(m["harmonic_bins"][:5], start=2):
        hf_khz = m["freqs_hz"][hbin] / 1e3
        ax.axvline(hf_khz, color="red", linestyle=":",
                   linewidth=1.0, label=f"H{k}" if k <= 3 else "")

    ax.set_title(
        f"f = {freq/1e3:.0f} kHz  |  SNR = {m['SNR_dB']:.1f} dB  "
        f"SINAD = {m['SINAD_dB']:.1f} dB  THD = {m['THD_dB']:.1f} dB  "
        f"HD2 = {m['HD2_dB']:.1f} dB  HD3 = {m['HD3_dB']:.1f} dB  "
        f"|  AC coupled  |  Ref: D ({PROBE_D}x)  Signal: B−C ({PROBE_B}x/{PROBE_C}x)",
        fontsize=9
    )
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Amplitude (dBmV, probe-tip)")
    ax.set_xlim([0, xlim_khz])
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

print("=" * 95)

# ──────────────────────────────────────────────────────────────────────────────
# CLOSE DEVICE
# ──────────────────────────────────────────────────────────────────────────────
status["close"] = ps.ps5000aCloseUnit(chandle)
assert_pico_ok(status["close"])
print("\nDevice closed.")

# ──────────────────────────────────────────────────────────────────────────────
# BUILD DataFrames
# ──────────────────────────────────────────────────────────────────────────────

# ── Summary ───────────────────────────────────────────────────────────────────
df_summary = pd.DataFrame(summary_rows)

fixed_cols = [
    "timestamp", "set_freq_hz", "actual_fund_freq_hz",
    "SNR_dB", "SINAD_dB", "THD_dB", "HD2_dB", "HD3_dB",
    "H1_mag_mV", "H2_mag_mV", "H3_mag_mV",
    "noise_rms_mV", "harm_noise_rms_mV",
    "chB_rms_mV", "chC_rms_mV", "chD_rms_mV", "diff_BC_rms_mV",
    "chB_peak_mV", "chC_peak_mV", "chD_peak_mV", "diff_BC_peak_mV",
    "chD_dc_offset_mV", "diff_BC_dc_offset_mV",
    "sample_rate_hz", "sample_interval_ns", "freq_resolution_hz",
    "n_samples", "timebase_index",
    "coupling",
    "probe_B", "probe_C", "probe_D",
    "ch_range_B", "ch_range_C", "ch_range_D",
    "gen_pk2pk_uV", "gen_offset_uV", "n_harmonics", "guard_bins",
]
dynamic_cols = [c for c in df_summary.columns if c not in fixed_cols]
df_summary   = df_summary[fixed_cols + sorted(dynamic_cols)]

# ── Waveforms ─────────────────────────────────────────────────────────────────
df_waveforms = pd.DataFrame(waveform_rows).astype({
    "timestamp":   "string",
    "set_freq_hz": "float64",
    "sample_idx":  "int64",
    "time_us":     "float64",
    "chB_mV":      "float64",
    "chC_mV":      "float64",
    "chD_mV":      "float64",
    "diff_BC_mV":  "float64",
})

# ──────────────────────────────────────────────────────────────────────────────
# EXPORT CSVs
# ──────────────────────────────────────────────────────────────────────────────
summary_csv_path  = OUTPUT_DIR / f"spectral_metrics_summary_{RUN_TIMESTAMP}.csv"
waveform_csv_path = OUTPUT_DIR / f"spectral_metrics_waveforms_{RUN_TIMESTAMP}.csv"

df_summary.to_csv(summary_csv_path,  index=False, float_format="%.6f")
df_waveforms.to_csv(waveform_csv_path, index=False, float_format="%.6f")

print(f"\nSummary CSV  saved → {summary_csv_path}")
print(f"Waveform CSV saved → {waveform_csv_path}")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.4f}".format)
print("\n── Summary ──────────────────────────────────────────────────────────────────")
print(df_summary[[
    "set_freq_hz", "SNR_dB", "SINAD_dB", "THD_dB",
    "HD2_dB", "HD3_dB", "H1_mag_mV", "H2_mag_mV", "H3_mag_mV",
    "chD_rms_mV", "coupling", "probe_B", "probe_C", "probe_D",
]].to_string(index=False))

# ──────────────────────────────────────────────────────────────────────────────
# SAVE PLOTS
# ──────────────────────────────────────────────────────────────────────────────
plt.suptitle(
    f"Spectral Metrics — (B−C) Differential Output  |  PicoScope 5000 Series\n"
    f"Run: {RUN_TIMESTAMP}  |  {N_SAMPLES} samples  |  "
    f"{GEN_PK2PK_UV/1e6:.1f} Vpp  |  12-bit  |  Blackman-Harris  |  AC coupled  |  "
    f"Probes: B={PROBE_B}x  C={PROBE_C}x  D={PROBE_D}x (ref)",
    fontsize=11
)
plt.tight_layout()
plot_path = OUTPUT_DIR / f"spectral_metrics_{RUN_TIMESTAMP}.png"
plt.savefig(plot_path, dpi=150)
print(f"Plot saved         → {plot_path}")
