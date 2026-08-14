"""
================================================================================
PicoScope 5000 Series — SNR, SINAD, THD, HD2, HD3 of (B−C) differential output
================================================================================
For each of three user-defined test frequencies:
  1. Program the built-in sine generator
  2. Capture channels A, B, C in block mode
  3. Compute the FFT of the differential signal (B - C)
  4. Identify the fundamental and harmonic bins
  5. Compute:
       SNR   = 20·log10( V_fundamental / V_noise_floor )
       SINAD = 20·log10( V_fundamental / V_noise_and_harmonics )
       THD   = 20·log10( sqrt(H2²+H3²+…) / H1 )
       HD2   = 20·log10( H2 / H1 )   [2nd harmonic distortion]
       HD3   = 20·log10( H3 / H1 )   [3rd harmonic distortion]
  6. Results are printed in a formatted table and saved to spectral_metrics.csv

Dependencies:  pip install picosdk numpy matplotlib scipy
================================================================================
"""

import ctypes
import time
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal.windows import blackmanharris
from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import adc2mV, assert_pico_ok

# ──────────────────────────────────────────────────────────────────────────────
# USER CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
# Three test frequencies in Hz
TEST_FREQUENCIES_HZ = [100_000, 1_000_000, 10_000_000]   # 100 kHz, 1 MHz, 10 MHz

N_SAMPLES    = 16384        # must be power-of-2; larger → better spectral resolution
N_HARMONICS  = 5            # number of harmonics to consider for THD (H2…HN)
CH_RANGE     = 7            # ±2 V  (see range table in Script 1)
GEN_PK2PK_UV = 2_000_000   # 2 Vpp
GEN_OFFSET_UV = 0
SETTLE_TIME_S = 0.10        # settle after generator frequency change
RESOLUTION    = ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_12BIT"]

# Blackman-Harris window coherent gain correction factor
BH_COHERENT_GAIN = 0.3587   # for Blackman-Harris (4-term)

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
# TIMEBASE HELPER (same logic as Script 1)
# ──────────────────────────────────────────────────────────────────────────────
def find_timebase(target_fs_hz: float, n_samples: int):
    interval_ns = ctypes.c_float()
    max_samp    = ctypes.c_int32()
    min_period_ns      = 1e9 / target_fs_hz
    desired_interval_ns = (20 * min_period_ns) / n_samples
    for tb in range(1, 2**23):
        st = ps.ps5000aGetTimebase2(
            chandle, tb, n_samples,
            ctypes.byref(interval_ns), ctypes.byref(max_samp), 0
        )
        if st == 0 and interval_ns.value <= desired_interval_ns:
            return tb, interval_ns.value
        if st == 0 and tb > 8:
            return tb, interval_ns.value
    return 62, interval_ns.value

# ──────────────────────────────────────────────────────────────────────────────
# BLOCK CAPTURE HELPER
# ──────────────────────────────────────────────────────────────────────────────
def capture_block(freq_hz: float):
    tb, dt_ns = find_timebase(freq_hz, N_SAMPLES)
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
    return chA_mV, chB_mV, chC_mV, dt_ns

# ──────────────────────────────────────────────────────────────────────────────
# SPECTRAL METRICS COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────
def spectral_metrics(signal_mV: np.ndarray, dt_ns: float, fund_freq_hz: float,
                     n_harmonics: int = 5, guard_bins: int = 3):
    """
    Compute SNR, SINAD, THD, HD2, HD3 from a time-domain signal.

    Parameters
    ----------
    signal_mV    : time-domain signal in mV
    dt_ns        : sample interval in nanoseconds
    fund_freq_hz : expected fundamental frequency in Hz
    n_harmonics  : number of harmonics to include in THD (H2 … Hn)
    guard_bins   : bins either side of each harmonic to exclude from noise

    Returns
    -------
    dict with keys: SNR_dB, SINAD_dB, THD_dB, HD2_dB, HD3_dB,
                    fund_freq_actual_Hz, fund_mag_mV
    """
    n  = len(signal_mV)
    fs = 1.0 / (dt_ns * 1e-9)
    df = fs / n                          # frequency resolution

    # Blackman-Harris window (excellent sidelobe suppression for distortion work)
    win       = blackmanharris(n)
    cg        = np.mean(win)             # coherent gain
    Y         = np.fft.rfft(signal_mV * win) / (n * cg)
    mag       = np.abs(Y)               # single-sided amplitude spectrum [mV]
    freqs     = np.fft.rfftfreq(n, d=dt_ns * 1e-9)

    # ── Locate fundamental bin ────────────────────────────────────────────────
    fund_bin  = int(round(fund_freq_hz / df))
    fund_bin  = max(1, min(fund_bin, len(mag) - 1))

    # Refine: find peak within ±5 bins of the expected bin
    search    = 5
    lo        = max(1,            fund_bin - search)
    hi        = min(len(mag) - 1, fund_bin + search)
    fund_bin  = lo + np.argmax(mag[lo:hi + 1])

    H1        = mag[fund_bin]
    fund_hz   = freqs[fund_bin]

    # ── Locate harmonic bins ─────────────────────────────────────────────────
    harmonic_bins = []
    harmonic_mags = []
    for h in range(2, n_harmonics + 2):
        hbin = int(round(h * fund_hz / df))
        if hbin >= len(mag):
            break
        lo_h  = max(1, hbin - search)
        hi_h  = min(len(mag) - 1, hbin + search)
        hbin  = lo_h + np.argmax(mag[lo_h:hi_h + 1])
        harmonic_bins.append(hbin)
        harmonic_mags.append(mag[hbin])

    H2 = harmonic_mags[0] if len(harmonic_mags) > 0 else 0.0
    H3 = harmonic_mags[1] if len(harmonic_mags) > 1 else 0.0

    # ── Build a boolean mask of "signal" bins (fund + harmonics ± guard) ────
    n_fft     = len(mag)
    sig_mask  = np.zeros(n_fft, dtype=bool)
    for hb in [fund_bin] + harmonic_bins:
        lo_g = max(0, hb - guard_bins)
        hi_g = min(n_fft - 1, hb + guard_bins)
        sig_mask[lo_g:hi_g + 1] = True

    # ── Noise power = all bins that are NOT signal and NOT DC ───────────────
    dc_bins   = guard_bins          # exclude DC and its sideskirts
    noise_mask = (~sig_mask)
    noise_mask[:dc_bins] = False    # exclude DC bin region

    noise_rms = np.sqrt(np.sum(mag[noise_mask] ** 2))  # total noise amplitude (approx RMS)

    # ── SINAD denominator: harmonics + noise ────────────────────────────────
    harm_mask  = np.zeros(n_fft, dtype=bool)
    for hb in harmonic_bins:
        lo_g = max(0, hb - guard_bins)
        hi_g = min(n_fft - 1, hb + guard_bins)
        harm_mask[lo_g:hi_g + 1] = True

    harm_and_noise_rms = np.sqrt(
        np.sum(mag[harm_mask | noise_mask] ** 2)
    )

    # ── Metrics ──────────────────────────────────────────────────────────────
    eps = 1e-15     # avoid log(0)

    SNR_dB   = 20 * np.log10(H1 / (noise_rms + eps))
    SINAD_dB = 20 * np.log10(H1 / (harm_and_noise_rms + eps))

    thd_num  = np.sqrt(sum(h ** 2 for h in harmonic_mags))
    THD_dB   = 20 * np.log10(thd_num / (H1 + eps))
    HD2_dB   = 20 * np.log10(H2 / (H1 + eps))
    HD3_dB   = 20 * np.log10(H3 / (H1 + eps))

    return {
        "SNR_dB":           SNR_dB,
        "SINAD_dB":         SINAD_dB,
        "THD_dB":           THD_dB,
        "HD2_dB":           HD2_dB,
        "HD3_dB":           HD3_dB,
        "fund_freq_hz":     fund_hz,
        "fund_mag_mV":      H1,
        "H2_mag_mV":        H2,
        "H3_mag_mV":        H3,
        "noise_rms_mV":     noise_rms,
        "mag_spectrum":     mag,
        "freqs_hz":         freqs,
        "fund_bin":         fund_bin,
        "harmonic_bins":    harmonic_bins,
    }

# ──────────────────────────────────────────────────────────────────────────────
# MAIN — iterate over the three test frequencies
# ──────────────────────────────────────────────────────────────────────────────
wavetype   = ctypes.c_int32(0)   # PS5000A_SINE
sweepType  = ctypes.c_int32(0)
trigType   = ctypes.c_int32(0)
trigSource = ctypes.c_int32(0)

results = []
fig, axes = plt.subplots(len(TEST_FREQUENCIES_HZ), 1,
                         figsize=(11, 4 * len(TEST_FREQUENCIES_HZ)))
if len(TEST_FREQUENCIES_HZ) == 1:
    axes = [axes]

print("\n" + "=" * 78)
print(f"{'Freq':>12}  {'SNR':>8}  {'SINAD':>8}  {'THD':>8}  {'HD2':>8}  {'HD3':>8}")
print(f"{'(Hz)':>12}  {'(dB)':>8}  {'(dB)':>8}  {'(dB)':>8}  {'(dB)':>8}  {'(dB)':>8}")
print("=" * 78)

for ax, freq in zip(axes, TEST_FREQUENCIES_HZ):
    # ── Program generator ────────────────────────────────────────────────────
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

    # ── Capture ───────────────────────────────────────────────────────────────
    chA, chB, chC, dt_ns = capture_block(freq)
    diff_BC = chB - chC

    # ── Compute metrics ──────────────────────────────────────────────────────
    m = spectral_metrics(diff_BC, dt_ns, freq, n_harmonics=N_HARMONICS)

    print(f"{freq:12.0f}  {m['SNR_dB']:8.2f}  {m['SINAD_dB']:8.2f}  "
          f"{m['THD_dB']:8.2f}  {m['HD2_dB']:8.2f}  {m['HD3_dB']:8.2f}")

    results.append({
        "freq_hz":    freq,
        "SNR_dB":     m["SNR_dB"],
        "SINAD_dB":   m["SINAD_dB"],
        "THD_dB":     m["THD_dB"],
        "HD2_dB":     m["HD2_dB"],
        "HD3_dB":     m["HD3_dB"],
        "fund_hz":    m["fund_freq_hz"],
        "fund_mV":    m["fund_mag_mV"],
    })

    # ── Spectrum plot for this frequency ────────────────────────────────────
    freqs_khz = m["freqs_hz"] / 1e3
    ax.plot(freqs_khz, 20 * np.log10(m["mag_spectrum"] + 1e-15),
            linewidth=0.8, color="steelblue", label="(B−C) spectrum")

    # Mark fundamental and harmonics
    ax.axvline(m["fund_freq_hz"] / 1e3, color="green", linestyle="--",
               linewidth=1.2, label=f"H1 = {m['fund_mag_mV']:.2f} mV")
    for k, hbin in enumerate(m["harmonic_bins"][:4], start=2):
        hf = m["freqs_hz"][hbin] / 1e3
        ax.axvline(hf, color="red", linestyle=":", linewidth=1.0,
                   label=f"H{k}" if k <= 3 else "")

    ax.set_title(
        f"f = {freq/1e3:.0f} kHz  |  SNR={m['SNR_dB']:.1f} dB  "
        f"SINAD={m['SINAD_dB']:.1f} dB  THD={m['THD_dB']:.1f} dB  "
        f"HD2={m['HD2_dB']:.1f} dB  HD3={m['HD3_dB']:.1f} dB",
        fontsize=9
    )
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Amplitude (dBmV)")
    ax.set_xlim([0, min(5 * freq / 1e3 * N_HARMONICS, m["freqs_hz"][-1] / 1e3)])
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

print("=" * 78)

# ──────────────────────────────────────────────────────────────────────────────
# CLOSE DEVICE
# ──────────────────────────────────────────────────────────────────────────────
status["close"] = ps.ps5000aCloseUnit(chandle)
assert_pico_ok(status["close"])
print("\nDevice closed.")

# ──────────────────────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ──────────────────────────────────────────────────────────────────────────────
plt.suptitle("Spectral Metrics — (B−C) Differential Output\nPicoScope 5000 Series",
             fontsize=12)
plt.tight_layout()
plt.savefig("spectral_metrics.png", dpi=150)
print("Spectrum plots saved to spectral_metrics.png")

csv_path = "spectral_metrics.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"Results saved to {csv_path}")
