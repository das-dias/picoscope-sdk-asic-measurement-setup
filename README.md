# picoscope-sdk-asic-measurement-setup
A repository of PicoSDK scripts for ASIC measurement using the PicoScope 5000 Series Digital Oscilloscope.

---

## Overview

Both scripts use **`picosdk-python-wrappers`** (`from picosdk.ps5000a import ps5000a as ps`), which is the current supported Python wrapper for the PicoScope 5000D (ps5000a drivers). The generator uses `ps5000aSetSigGenBuiltInV2` and capture uses `ps5000aRunBlock` with `ps5000aGetTimebase2`. Channel constants and trigger states follow the ps5000a SDK enumeration patterns.

Install dependencies first:
```bash
pip install picosdk numpy matplotlib scipy
```

## User Guide:

```zsh
# uv workflow (recommended)
uv sync                                  # install everything
uv sync --extra dev                      # include dev tools too
uv run ps5000a_bode_sweep.py

# plain pip workflow
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # runtime only
pip install -r requirements-dev.txt      # runtime + dev tools
```

## Key Design Notes

**Hardware setup assumed:**
- AWG out → ASIC input (and looped to **Ch A** for the reference)
- ASIC differential output+ → **Ch B**, output− → **Ch C**
- (B−C) is the CMRR-rejected differential output

**Windowing:**
- Script 1 uses a **Blackman** window — good general-purpose choice for frequency sweeps with moderate dynamic range.
- Script 2 uses a **Blackman-Harris** window — much better sidelobe suppression (−92 dB), essential for measuring low-level harmonic distortion like HD2/HD3 without spectral leakage contamination.

**Metric definitions used:**

| Metric | Formula |
|---|---|
| Gain | 20·log₁₀(|FFT(B−C)| / |FFT(A)|) at the fundamental bin |
| Phase | ∠FFT(B−C) − ∠FFT(A) at the fundamental bin |
| SNR | 20·log₁₀(H1 / V_noise_floor) |
| SINAD | 20·log₁₀(H1 / √(harmonics² + noise²)) |
| THD | 20·log₁₀(√(H2²+…+Hn²) / H1) |
| HD2 | 20·log₁₀(H2 / H1) |
| HD3 | 20·log₁₀(H3 / H1) |

**Timebase selection:** The `find_timebase()` helper queries `ps5000aGetTimebase2` iteratively to find the fastest timebase that still captures at least 20 full cycles of the generator signal, giving good spectral resolution for every frequency point.
