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

---
# PicoScope 5000 Series — ASIC Electrical Measurement Suite

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![uv](https://img.shields.io/badge/managed%20with-uv-5C4EE5.svg)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/badge/linted%20with-ruff-FCC21B.svg)](https://github.com/astral-sh/ruff)
[![PicoSDK](https://img.shields.io/badge/PicoSDK-ps5000a-00A3E0.svg)](https://github.com/picotech/picosdk-python-wrappers)

A Python measurement suite for characterising custom ASICs using the
**PicoScope 5000 Series** oscilloscope and its built-in signal generator.
The suite provides two fully automated measurement scripts:

1. **Bode Sweep** — swept-frequency gain (dB) and phase response of a
   differential output `(B − C)` referenced against the generator monitor
   channel `A`, from **10 kHz to 50 MHz**.
2. **Spectral Metrics** — single-tone distortion and noise analysis of the
   `(B − C)` differential output at three user-defined frequencies, computing
   **SNR**, **SINAD**, **THD**, **HD2**, and **HD3**.

---

## Table of Contents

- [Design Intent](#design-intent)
- [Hardware Setup](#hardware-setup)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Using uv (recommended)](#using-uv-recommended)
  - [Using pip](#using-pip)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Script 1 — Bode Sweep](#script-1--bode-sweep)
  - [Script 2 — Spectral Metrics](#script-2--spectral-metrics)
- [Output Files](#output-files)
- [Measurement Methodology](#measurement-methodology)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Cloning and Branching Model](#cloning-and-branching-model)
- [License](#license)

---

## Design Intent

This suite is designed for **hardware validation engineers** characterising
the analogue front-end of a custom ASIC in a bench or automated test
environment. The primary goals are:

- **Accuracy** — Blackman-Harris windowing minimises spectral leakage
  (−92 dB sidelobes), ensuring that low-level harmonic distortion products
  (HD2, HD3) are not masked by the measurement instrument itself.
- **Reproducibility** — all measurement parameters are declared as named
  constants at the top of each script; no magic numbers are buried in logic.
- **Traceability** — every run produces timestamped CSV output alongside
  publication-quality PNG plots, suitable for inclusion in characterisation
  reports.
- **Portability** — the code targets the official `picosdk-python-wrappers`
  low-level API (ps5000a driver), avoiding third-party abstraction layers
  that may drift out of sync with Pico Technology firmware releases.
- **Extensibility** — the helper functions (`capture_block`,
  `spectral_metrics`, `find_timebase`) are written as standalone callables
  so they can be imported and re-used in future test scripts without
  modification.

---

## Hardware Setup


**Tuning tips:**
- Increase `N_SAMPLES` (to 32768 or 65536) for finer frequency-bin resolution, particularly helpful for accurately isolating HD2/HD3 at high frequencies.
- Adjust `CH_RANGE` to the tightest range that doesn't clip your signal — this maximises the effective number of bits and lowers the quantisation noise floor.
- Increase `SETTLE_TIME_S` if your ASIC or filter has a long group delay settling time.
