"""
================================================================================
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
     ps5000aSetSigGenBuiltInV2 (no hardware sweep — software step-and-settle)
  3. Three channels (A, B, C) are captured in block mode
  4. FFT is computed for A and (B-C)
  5. At the fundamental bin: gain [dB] and phase difference [°] are extracted
  6. Bode plot (gain + phase) is saved as bode_plot.png

Dependencies:  pip install picosdk numpy matplotlib scipy
================================================================================
"""

import ctypes
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless-safe; change to "TkAgg" if you want a window
import matplotlib.pyplot as plt
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
    if pwr in (282, 286):           # USB-powered or non-USB3 port
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
# Returns (timebase_index, actual_sample_interval_ns)
# For ps5000a 12-bit with 3 channels: minimum timebase is typically 3
# (500 MS/s shared), but we probe to find the right one per frequency.
# ──────────────────────────────────────────────────────────────────────────────
def find_timebase(target_fs_hz: float, n_samples: int):
    """
    Walk timebase indices until we find one whose sample interval gives
    at least 10 full cycles of target_fs_hz.  Falls back to the highest
    timebase (slowest) if the signal is very low frequency.
    """
    interval_ns  = ctypes.c_float()
    max_samp     = ctypes.c_int32()
    # We want at least 10 cycles captured → minimum record length
    min_period_ns = 1e9 / target_fs_hz
    # target: capture ~20 periods or N_SAMPLES, whichever gives better resolution
    desired_interval_ns = (20 * min_period_ns) / n_samples

    for tb in range(1, 2**23):
        st = ps.ps5000aGetTimebase2(
            chandle, tb, n_samples,
            ctypes.byref(interval_ns), ctypes.byref(max_samp), 0
        )
        if st == 0 and interval_ns.value <= desired_interval_ns:
            return tb, interval_ns.value
        if st == 0 and tb > 8:
            # Accept whatever we have once we pass the fast timebases
            return tb, interval_ns.value
    return 62, interval_ns.value     # safe fallback

# ──────────────────────────────────────────────────────────────────────────────
# BLOCK CAPTURE HELPER
# Returns (chA_mV, chB_mV, chC_mV, sample_interval_ns)
# ──────────────────────────────────────────────────────────────────────────────
def capture_block(freq_hz: float):
    tb, dt_ns = find_timebase(freq_hz, N_SAMPLES)

    # Simple trigger on CH_A, threshold ≈ 0 V
    status["trig"] = ps.ps5000aSetSimpleTrigger(
        chandle, 1, CH_A, 0, 2, 0, 1000   # rising, 0 ADC counts, 1000 ms auto
    )
    assert_pico_ok(status["trig"])

    # Run block
    status["runBlock"] = ps.ps5000aRunBlock(
        chandle, 0, N_SAMPLES, tb, None, 0, None, None
    )
    assert_pico_ok(status["runBlock"])

    # Poll until ready
    ready = ctypes.c_int16(0)
    while ready.value == 0:
        status["isReady"] = ps.ps5000aIsReady(chandle, ctypes.byref(ready))
        time.sleep(0.001)

    # Allocate buffers
    bufA = (ctypes.c_int16 * N_SAMPLES)()
    bufB = (ctypes.c_int16 * N_SAMPLES)()
    bufC = (ctypes.c_int16 * N_SAMPLES)()
    RATIO_NONE = ps.PS5000A_RATIO_MODE["PS5000A_RATIO_MODE_NONE"]

    for ch, buf in [(CH_A, bufA), (CH_B, bufB), (CH_C, bufC)]:
        status[f"setBuf{ch}"] = ps.ps5000aSetDataBuffer(
            chandle, ch, ctypes.byref(buf), N_SAMPLES, 0, RATIO_NONE
        )
        assert_pico_ok(status[f"setBuf{ch}"])

    n_ret   = ctypes.c_uint32(N_SAMPLES)
    overflow = ctypes.c_int16()
    status["getValues"] = ps.ps5000aGetValues(
        chandle, 0, ctypes.byref(n_ret), 1, RATIO_NONE, 0, ctypes.byref(overflow)
    )
    assert_pico_ok(status["getValues"])

    # Convert ADC → mV (uses picosdk helper)
    chA_mV = np.array(adc2mV(bufA, CH_RANGE, maxADC), dtype=np.float64)
    chB_mV = np.array(adc2mV(bufB, CH_RANGE, maxADC), dtype=np.float64)
    chC_mV = np.array(adc2mV(bufC, CH_RANGE, maxADC), dtype=np.float64)

    return chA_mV, chB_mV, chC_mV, dt_ns

# ──────────────────────────────────────────────────────────────────────────────
# FFT HELPER — returns (freqs_Hz, complex_spectrum) with Blackman window
# ──────────────────────────────────────────────────────────────────────────────
def compute_fft(signal_mV: np.ndarray, dt_ns: float):
    n   = len(signal_mV)
    win = blackman(n)
    fs  = 1.0 / (dt_ns * 1e-9)          # sample rate in Hz
    Y   = np.fft.rfft(signal_mV * win)
    f   = np.fft.rfftfreq(n, d=dt_ns * 1e-9)
    return f, Y

# ──────────────────────────────────────────────────────────────────────────────
# MAIN SWEEP LOOP
# ──────────────────────────────────────────────────────────────────────────────
frequencies  = np.geomspace(FREQ_START_HZ, FREQ_STOP_HZ, N_FREQ_POINTS)
gain_dB      = np.zeros(N_FREQ_POINTS)
phase_deg    = np.zeros(N_FREQ_POINTS)

wavetype    = ctypes.c_int32(0)     # PS5000A_SINE
sweepType   = ctypes.c_int32(0)     # PS5000A_UP  (no sweep — single freq per step)
trigType    = ctypes.c_int32(0)     # PS5000A_SIGGEN_RISING
trigSource  = ctypes.c_int32(0)     # PS5000A_SIGGEN_NONE

print(f"\nStarting frequency sweep: {FREQ_START_HZ/1e3:.1f} kHz → {FREQ_STOP_HZ/1e6:.0f} MHz")
print(f"{'Freq (Hz)':>14}  {'Gain (dB)':>10}  {'Phase (deg)':>12}")
print("-" * 42)

for i, freq in enumerate(frequencies):
    # ── 1. Program generator at this frequency (fixed, no HW sweep) ──────────
    status["sigGen"] = ps.ps5000aSetSigGenBuiltInV2(
        chandle,
        GEN_OFFSET_UV,      # offsetVoltage [µV]
        GEN_PK2PK_UV,       # pkToPk [µV]
        wavetype,           # waveType  (sine)
        ctypes.c_double(freq),   # startFrequency
        ctypes.c_double(freq),   # stopFrequency  (= start → fixed freq)
        ctypes.c_double(0.0),    # increment
        ctypes.c_double(1.0),    # dwellTime
        sweepType,          # sweepType
        0,                  # operation
        0,                  # shots
        0,                  # sweeps
        trigType,           # triggerType
        trigSource,         # triggerSource
        0                   # extInThreshold
    )
    assert_pico_ok(status["sigGen"])
    time.sleep(SETTLE_TIME_S)   # let the DUT and generator settle

    # ── 2. Capture A, B, C ───────────────────────────────────────────────────
    chA, chB, chC, dt_ns = capture_block(freq)

    # ── 3. Differential signal ───────────────────────────────────────────────
    diff_BC = chB - chC

    # ── 4. FFT of A and (B-C) ────────────────────────────────────────────────
    freqs_A,    Y_A  = compute_fft(chA,     dt_ns)
    freqs_diff, Y_BC = compute_fft(diff_BC, dt_ns)

    # ── 5. Find fundamental bin closest to the generator frequency ───────────
    fund_bin = np.argmin(np.abs(freqs_diff - freq))

    A_fund  = Y_A[fund_bin]
    BC_fund = Y_BC[fund_bin]

    # ── 6. Gain and phase ────────────────────────────────────────────────────
    mag_A  = np.abs(A_fund)
    mag_BC = np.abs(BC_fund)

    if mag_A < 1e-12:           # guard against divide-by-zero
        gain_dB[i]  = np.nan
        phase_deg[i] = np.nan
    else:
        gain_dB[i]   = 20.0 * np.log10(mag_BC / mag_A)
        phase_rad    = np.angle(BC_fund) - np.angle(A_fund)
        phase_deg[i] = np.degrees(np.unwrap([phase_rad])[0])

    print(f"{freq:14.1f}  {gain_dB[i]:10.3f}  {phase_deg[i]:12.3f}")

# ──────────────────────────────────────────────────────────────────────────────
# CLOSE DEVICE
# ──────────────────────────────────────────────────────────────────────────────
status["close"] = ps.ps5000aCloseUnit(chandle)
assert_pico_ok(status["close"])
print("\nDevice closed.")

# ──────────────────────────────────────────────────────────────────────────────
# BODE PLOT
# ──────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.suptitle("Bode Plot — (B−C) vs A\nPicoScope 5000 Series", fontsize=13)

ax1.semilogx(frequencies, gain_dB, "b.-", linewidth=1.5, markersize=5)
ax1.set_ylabel("Gain (dB)")
ax1.grid(True, which="both", linestyle="--", alpha=0.6)
ax1.axhline(0, color="k", linewidth=0.8)
ax1.set_title("Gain")

ax2.semilogx(frequencies, phase_deg, "r.-", linewidth=1.5, markersize=5)
ax2.set_ylabel("Phase (degrees)")
ax2.set_xlabel("Frequency (Hz)")
ax2.grid(True, which="both", linestyle="--", alpha=0.6)
ax2.axhline(0, color="k", linewidth=0.8)
ax2.set_title("Phase")

plt.tight_layout()
plt.savefig("bode_plot.png", dpi=150)
print("Bode plot saved to bode_plot.png")
