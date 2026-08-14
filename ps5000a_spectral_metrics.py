"""
================================================================================
ps5000a_spectral_metrics.py
PicoScope 5000 Series — SNR, SINAD, THD, HD2, HD3 of (B−C) differential output
================================================================================
For each of three user-defined test frequencies:
  1. Program the built-in sine generator
  2. Capture channels A, B, C in block mode
  3. Compute the FFT of the differential signal (B - C)
  4. Identify the fundamental and harmonic bins
  5. Compute SNR, SINAD, THD, HD2, HD3
  6. Accumulate all results — including per-harmonic amplitudes and raw
     waveform statistics — in a pandas DataFrame
  7. Export a summary CSV (one row per test frequency) and a waveform CSV
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
TEST_FREQUENCIES_HZ = [100_000, 1_000_000, 10_000_000]   # 100 kHz, 1 MHz, 10 MHz

N_SAMPLES    = 16384        # power-of-2; larger → finer spectral bins
N_HARMONICS  = 5            # harmonics included in THD: H2 … H(N_HARMONICS+1)
GUARD_BINS   = 3            # bins excluded either side of each harmonic
CH_RANGE     = 7            # ±2 V
GEN_PK2PK_UV = 2_000_000   # 2 Vpp
GEN_OFFSET_UV = 0
SETTLE_TIME_S = 0.10
RESOLUTION    = ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_12BIT"]

OUTPUT_DIR = Path("outputs")

# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORY & TIMESTAMP
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

# ──────────────────────────────────────────────────────────────────────────────
# CHANNEL SETUP
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
# ──────────────────────────────────────────────────────────────────────────────
def capture_block(freq_hz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """
    Arm and retrieve one block capture from channels A, B, C.

    Parameters
    ----------
    freq_hz : float
        Current generator frequency (Hz).

    Returns
    -------
    chA_mV, chB_mV, chC_mV : np.ndarray
        Waveforms in millivolts.
    dt_ns : float
        Sample interval in nanoseconds.
    tb : int
        Timebase index used.
    """
    tb, dt_ns  = find_timebase(freq_hz, N_SAMPLES)
    RATIO_NONE = ps.PS5000A_RATIO_MODE["PS5000A_RATIO_MODE_NONE"]

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

    return chA_mV, chB_mV, chC_mV, dt_ns, tb

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
    leakage so that low-level harmonic products are not masked.

    Parameters
    ----------
    signal_mV : np.ndarray
        Time-domain differential signal (B − C) in millivolts.
    dt_ns : float
        Sample interval in nanoseconds.
    fund_freq_hz : float
        Expected fundamental frequency in Hz.
    n_harmonics : int
        Number of harmonics beyond the fundamental to include in THD.
    guard_bins : int
        Number of FFT bins to exclude either side of each harmonic peak
        when computing the noise floor.

    Returns
    -------
    dict
        Keys: SNR_dB, SINAD_dB, THD_dB, HD2_dB, HD3_dB,
              fund_freq_hz, fund_mag_mV, H2_mag_mV, H3_mag_mV,
              noise_rms_mV, mag_spectrum, freqs_hz, fund_bin,
              harmonic_bins, harmonic_mags_mV, df_hz, fs_hz.
    """
    n  = len(signal_mV)
    fs = 1.0 / (dt_ns * 1e-9)
    df = fs / n

    win = blackmanharris(n)
    cg  = float(np.mean(win))                   # coherent gain
    Y   = np.fft.rfft(signal_mV * win) / (n * cg)
    mag = np.abs(Y)                              # single-sided amplitude [mV]
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
    harmonic_bins     = []
    harmonic_mags_mV  = []

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
    noise_mask             = ~sig_mask
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

    SNR_dB   = 20.0 * np.log10(H1 / (noise_rms              + eps))
    SINAD_dB = 20.0 * np.log10(H1 / (harm_and_noise_rms     + eps))
    thd_num  = float(np.sqrt(sum(h ** 2 for h in harmonic_mags_mV)))
    THD_dB   = 20.0 * np.log10(thd_num / (H1 + eps))
    HD2_dB   = 20.0 * np.log10(H2      / (H1 + eps))
    HD3_dB   = 20.0 * np.log10(H3      / (H1 + eps))

    return {
        "SNR_dB":           float(SNR_dB),
        "SINAD_dB":         float(SINAD_dB),
        "THD_dB":           float(THD_dB),
        "HD2_dB":           float(HD2_dB),
        "HD3_dB":           float(HD3_dB),
        "fund_freq_hz":     fund_hz,
        "fund_mag_mV":      H1,
        "H2_mag_mV":        H2,
        "H3_mag_mV":        H3,
        "noise_rms_mV":     noise_rms,
        "harm_noise_rms_mV": harm_and_noise_rms,
        "mag_spectrum":     mag,
        "freqs_hz":         freqs,
        "fund_bin":         fund_bin,
        "harmonic_bins":    harmonic_bins,
        "harmonic_mags_mV": harmonic_mags_mV,
        "df_hz":            df,
        "fs_hz":            fs,
    }

# ──────────────────────────────────────────────────────────────────────────────
# MAIN — iterate over test frequencies
# ──────────────────────────────────────────────────────────────────────────────
wavetype   = ctypes.c_int32(0)
sweepType  = ctypes.c_int32(0)
trigType   = ctypes.c_int32(0)
trigSource = ctypes.c_int32(0)

# ── Accumulators ─────────────────────────────────────────────────────────────
summary_rows:  list[dict] = []   # one row per test frequency
waveform_rows: list[dict] = []   # one row per sample per frequency

fig, axes = plt.subplots(
    len(TEST_FREQUENCIES_HZ), 1,
    figsize=(12, 5 * len(TEST_FREQUENCIES_HZ))
)
if len(TEST_FREQUENCIES_HZ) == 1:
    axes = [axes]

print("\n" + "=" * 90)
print(f"{'Freq (Hz)':>12}  {'SNR':>8}  {'SINAD':>8}  {'THD':>8}  "
      f"{'HD2':>8}  {'HD3':>8}  {'H1 (mV)':>10}  {'H2 (mV)':>10}  {'H3 (mV)':>10}")
print(f"{'':>12}  {'(dB)':>8}  {'(dB)':>8}  {'(dB)':>8}  "
      f"{'(dB)':>8}  {'(dB)':>8}  {'':>10}  {'':>10}  {'':>10}")
print("=" * 90)

for ax, freq in zip(axes, TEST_FREQUENCIES_HZ):

    # ── Program generator ─────────────────────────────────────────────────
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

    # ── Capture ───────────────────────────────────────────────────────────
    chA, chB, chC, dt_ns, tb = capture_block(freq)
    diff_BC = chB - chC
    fs_hz   = 1.0 / (dt_ns * 1e-9)
    t_us    = np.arange(N_SAMPLES) * dt_ns / 1e3   # time axis in µs

    # ── Metrics ───────────────────────────────────────────────────────────
    m = spectral_metrics(diff_BC, dt_ns, freq,
                         n_harmonics=N_HARMONICS, guard_bins=GUARD_BINS)

    print(
        f"{freq:12.0f}  {m['SNR_dB']:8.2f}  {m['SINAD_dB']:8.2f}  "
        f"{m['THD_dB']:8.2f}  {m['HD2_dB']:8.2f}  {m['HD3_dB']:8.2f}  "
        f"{m['fund_mag_mV']:10.4f}  {m['H2_mag_mV']:10.4f}  {m['H3_mag_mV']:10.4f}"
    )

    # ── Build per-harmonic columns dynamically ────────────────────────────
    harmonic_cols: dict = {}
    for k, (hm, hf) in enumerate(
        zip(m["harmonic_mags_mV"],
            [m["freqs_hz"][b] for b in m["harmonic_bins"]]),
        start=2
    ):
        harmonic_cols[f"H{k}_mag_mV"]  = hm
        harmonic_cols[f"H{k}_freq_hz"] = hf
        harmonic_cols[f"HD{k}_dB"]     = 20.0 * np.log10(hm / (m["fund_mag_mV"] + 1e-15))

    # ── Summary row ───────────────────────────────────────────────────────
    summary_rows.append({
        # Identification
        "timestamp":            RUN_TIMESTAMP,
        "set_freq_hz":          float(freq),
        "actual_fund_freq_hz":  m["fund_freq_hz"],
        # Primary metrics
        "SNR_dB":               m["SNR_dB"],
        "SINAD_dB":             m["SINAD_dB"],
        "THD_dB":               m["THD_dB"],
        "HD2_dB":               m["HD2_dB"],
        "HD3_dB":               m["HD3_dB"],
        # Amplitude summary
        "H1_mag_mV":            m["fund_mag_mV"],
        "H2_mag_mV":            m["H2_mag_mV"],
        "H3_mag_mV":            m["H3_mag_mV"],
        "noise_rms_mV":         m["noise_rms_mV"],
        "harm_noise_rms_mV":    m["harm_noise_rms_mV"],
        # Waveform statistics
        "chA_rms_mV":           float(np.sqrt(np.mean(chA ** 2))),
        "chB_rms_mV":           float(np.sqrt(np.mean(chB ** 2))),
        "chC_rms_mV":           float(np.sqrt(np.mean(chC ** 2))),
        "diff_BC_rms_mV":       float(np.sqrt(np.mean(diff_BC ** 2))),
        "chA_peak_mV":          float(np.max(np.abs(chA))),
        "chB_peak_mV":          float(np.max(np.abs(chB))),
        "chC_peak_mV":          float(np.max(np.abs(chC))),
        "diff_BC_peak_mV":      float(np.max(np.abs(diff_BC))),
        "chA_dc_offset_mV":     float(np.mean(chA)),
        "diff_BC_dc_offset_mV": float(np.mean(diff_BC)),
        # Acquisition parameters
        "sample_rate_hz":       m["fs_hz"],
        "sample_interval_ns":   dt_ns,
        "freq_resolution_hz":   m["df_hz"],
        "n_samples":            N_SAMPLES,
        "timebase_index":       tb,
        "ch_range_index":       CH_RANGE,
        "gen_pk2pk_uV":         GEN_PK2PK_UV,
        "gen_offset_uV":        GEN_OFFSET_UV,
        "n_harmonics":          N_HARMONICS,
        "guard_bins":           GUARD_BINS,
        # Dynamic per-harmonic columns (H2…HN)
        **harmonic_cols,
    })

    # ── Waveform rows (time-domain, all four signals per sample) ─────────
    for i in range(N_SAMPLES):
        waveform_rows.append({
            "timestamp":   RUN_TIMESTAMP,
            "set_freq_hz": float(freq),
            "sample_idx":  i,
            "time_us":     float(t_us[i]),
            "chA_mV":      float(chA[i]),
            "chB_mV":      float(chB[i]),
            "chC_mV":      float(chC[i]),
            "diff_BC_mV":  float(diff_BC[i]),
        })

    # ── Spectrum plot ─────────────────────────────────────────────────────
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
        f"HD2 = {m['HD2_dB']:.1f} dB  HD3 = {m['HD3_dB']:.1f} dB",
        fontsize=9
    )
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Amplitude (dBmV)")
    ax.set_xlim([0, xlim_khz])
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

print("=" * 90)

# ──────────────────────────────────────────────────────────────────────────────
# CLOSE DEVICE
# ──────────────────────────────────────────────────────────────────────────────
status["close"] = ps.ps5000aCloseUnit(chandle)
assert_pico_ok(status["close"])
print("\nDevice closed.")

# ──────────────────────────────────────────────────────────────────────────────
# BUILD DataFrames
# ──────────────────────────────────────────────────────────────────────────────

# ── Summary DataFrame ─────────────────────────────────────────────────────────
df_summary = pd.DataFrame(summary_rows)

# Enforce stable column order: identification → metrics → amplitudes →
# statistics → acquisition → dynamic harmonic columns
fixed_cols = [
    "timestamp", "set_freq_hz", "actual_fund_freq_hz",
    "SNR_dB", "SINAD_dB", "THD_dB", "HD2_dB", "HD3_dB",
    "H1_mag_mV", "H2_mag_mV", "H3_mag_mV",
    "noise_rms_mV", "harm_noise_rms_mV",
    "chA_rms_mV", "chB_rms_mV", "chC_rms_mV", "diff_BC_rms_mV",
    "chA_peak_mV", "chB_peak_mV", "chC_peak_mV", "diff_BC_peak_mV",
    "chA_dc_offset_mV", "diff_BC_dc_offset_mV",
    "sample_rate_hz", "sample_interval_ns", "freq_resolution_hz",
    "n_samples", "timebase_index", "ch_range_index",
    "gen_pk2pk_uV", "gen_offset_uV", "n_harmonics", "guard_bins",
]
dynamic_cols = [c for c in df_summary.columns if c not in fixed_cols]
df_summary   = df_summary[fixed_cols + sorted(dynamic_cols)]

# ── Waveform DataFrame ────────────────────────────────────────────────────────
df_waveforms = pd.DataFrame(waveform_rows).astype({
    "timestamp":   "string",
    "set_freq_hz": "float64",
    "sample_idx":  "int64",
    "time_us":     "float64",
    "chA_mV":      "float64",
    "chB_mV":      "float64",
    "chC_mV":      "float64",
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

# Pretty-print the summary table to the terminal
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.4f}".format)
print("\n── Summary ──────────────────────────────────────────────────────────────")
print(df_summary[[
    "set_freq_hz", "SNR_dB", "SINAD_dB", "THD_dB",
    "HD2_dB", "HD3_dB", "H1_mag_mV", "H2_mag_mV", "H3_mag_mV",
    "noise_rms_mV", "sample_rate_hz", "freq_resolution_hz"
]].to_string(index=False))

# ──────────────────────────────────────────────────────────────────────────────
# SAVE PLOTS
# ──────────────────────────────────────────────────────────────────────────────
plt.suptitle(
    f"Spectral Metrics — (B−C) Differential Output  |  PicoScope 5000 Series\n"
    f"Run: {RUN_TIMESTAMP}  |  {N_SAMPLES} samples  |  "
    f"{GEN_PK2PK_UV/1e6:.1f} Vpp  |  12-bit  |  Blackman-Harris window",
    fontsize=11
)
plt.tight_layout()
plot_path = OUTPUT_DIR / f"spectral_metrics_{RUN_TIMESTAMP}.png"
plt.savefig(plot_path, dpi=150)
print(f"Plot saved         → {plot_path}")
