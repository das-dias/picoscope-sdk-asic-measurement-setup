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

```
                        ┌─────────────────────────────┐
                        │     PicoScope 5000 Series    │
                        │                              │
  ┌──────────────┐      │  AWG ──────────────────────► │ Ch A  (generator monitor)
  │              │◄─────│  out                         │
  │  ASIC / DUT  │      │                              │
  │              │      │                              │
  │  OUT+  ──────┼─────►│ Ch B  (positive output)      │
  │  OUT−  ──────┼─────►│ Ch C  (negative output)      │
  └──────────────┘      │                              │
                        │  Ch D  (unused — disabled)   │
                        └─────────────────────────────┘
```

| Connection | Description |
|---|---|
| AWG out → ASIC input | The PicoScope's built-in sine generator drives the DUT |
| AWG out → Ch A | Ch A monitors the actual generator signal (ground truth reference) |
| ASIC OUT+ → Ch B | Positive differential output of the ASIC |
| ASIC OUT− → Ch C | Negative differential output of the ASIC |
| `(B − C)` | Computed in software — fully differential, CMRR-rejected output |

> **Note:** Ch D is explicitly disabled in both scripts to maximise the
> available timebase range and ADC resolution at high frequencies.
> If your model is a dual-channel variant (e.g. 5242), remove the Ch C/D
> setup calls and adapt the differential computation accordingly.

---

## Repository Structure

```
picoscope-asic-measurement/
│
├── ps5000a_bode_sweep.py        # Script 1 — Bode sweep (gain & phase)
├── ps5000a_spectral_metrics.py  # Script 2 — SNR, SINAD, THD, HD2, HD3
│
├── pyproject.toml               # Project metadata and dependencies (uv/pip)
├── requirements.txt             # Runtime dependencies (plain pip)
├── requirements-dev.txt         # Dev dependencies (ruff, mypy, pytest)
│
├── tests/
│   └── test_spectral_metrics.py # Unit tests for DSP helper functions
│
├── outputs/                     # Auto-created at runtime — gitignored
│   ├── bode_plot.png
│   ├── spectral_metrics.png
│   └── spectral_metrics.csv
│
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions: lint + test on push/PR
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## Requirements

### Software

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.9 | 3.11 recommended |
| PicoSDK (system driver) | Latest | Must be installed separately — see below |
| `picosdk` Python wrapper | ≥ 1.1 | Installed via pip/uv |
| `numpy` | ≥ 1.26, < 3.0 | |
| `scipy` | ≥ 1.12, < 2.0 | Windowing functions |
| `matplotlib` | ≥ 3.8, < 4.0 | Plot generation |

### System Driver (mandatory — not installed by pip)

The `picosdk` Python package is a thin wrapper around Pico Technology's
native C library. You must install the **PicoSDK system driver** for your
operating system before running any script:

| OS | Installation |
|---|---|
| **Windows** | Download and run the installer from [Pico Technology Downloads](https://www.picotech.com/downloads) |
| **Linux** | Follow the [Linux Software & Drivers guide](https://www.picotech.com/downloads/linux) — installs via apt/yum |
| **macOS** | Download the macOS SDK package from [Pico Technology Downloads](https://www.picotech.com/downloads) |

---

## Installation

### Using uv (recommended)

[`uv`](https://github.com/astral-sh/uv) resolves dependencies, creates an
isolated virtual environment, and writes a lockfile in a single command.

```bash
# 1. Install uv if you don't have it
curl -Ls https://astral.sh/uv/install.sh | sh
# Windows (PowerShell):
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Clone the repository
git clone https://github.com/<your-org>/picoscope-asic-measurement.git
cd picoscope-asic-measurement

# 3. Create the virtual environment and install runtime dependencies
uv sync

# 4. (Optional) Install development tools as well
uv sync --extra dev
```

The virtual environment is created at `.venv/` inside the project directory.

### Using pip

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/picoscope-asic-measurement.git
cd picoscope-asic-measurement

# 2. Create and activate a virtual environment
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 3. Install runtime dependencies
pip install -r requirements.txt

# 4. (Optional) Install development dependencies
pip install -r requirements-dev.txt
```

---

## Configuration

All user-tunable parameters are declared as named constants near the top of
each script. No changes to measurement logic are needed for routine
reconfiguration.

### `ps5000a_bode_sweep.py`

| Constant | Default | Description |
|---|---|---|
| `FREQ_START_HZ` | `10_000` | Sweep start frequency (Hz) |
| `FREQ_STOP_HZ` | `50_000_000` | Sweep stop frequency (Hz) |
| `N_FREQ_POINTS` | `60` | Number of log-spaced frequency steps |
| `N_SAMPLES` | `8192` | Samples captured per block (power of 2) |
| `SETTLE_TIME_S` | `0.05` | Seconds to wait after each generator step |
| `CH_RANGE` | `7` (±2 V) | Voltage range index for all channels |
| `GEN_PK2PK_UV` | `2_000_000` | Generator amplitude in µV (= 2 Vpp) |
| `GEN_OFFSET_UV` | `0` | Generator DC offset in µV |
| `RESOLUTION` | `PS5000A_DR_12BIT` | ADC resolution |

### `ps5000a_spectral_metrics.py`

| Constant | Default | Description |
|---|---|---|
| `TEST_FREQUENCIES_HZ` | `[100k, 1M, 10M]` | Three test frequencies in Hz |
| `N_SAMPLES` | `16384` | Samples per block — larger improves bin resolution |
| `N_HARMONICS` | `5` | Harmonics included in THD calculation (H2…H6) |
| `CH_RANGE` | `7` (±2 V) | Voltage range index |
| `GEN_PK2PK_UV` | `2_000_000` | Generator amplitude in µV |
| `SETTLE_TIME_S` | `0.10` | Settle time after generator frequency change |

**Voltage range index reference:**

| Index | Range |
|---|---|
| 1 | ±20 mV |
| 2 | ±50 mV |
| 3 | ±100 mV |
| 4 | ±200 mV |
| 5 | ±500 mV |
| 6 | ±1 V |
| 7 | ±2 V |
| 8 | ±5 V |
| 9 | ±10 V |
| 10 | ±20 V |

> **Tip:** Always use the tightest range that does not clip your signal.
> This maximises the effective number of bits (ENOB) and lowers the
> quantisation noise floor, which directly improves SNR and SINAD readings.

---

## Usage

### Script 1 — Bode Sweep

```bash
# uv
uv run ps5000a_bode_sweep.py

# pip / activated venv
python ps5000a_bode_sweep.py
```

**What happens:**

1. The device is opened and channels A, B, C are configured.
2. The built-in sine generator steps through 60 log-spaced frequencies
   between 10 kHz and 50 MHz, settling for 50 ms at each step.
3. All three channels are captured simultaneously in block mode.
4. The FFT of `(B − C)` and Ch A are computed using a Blackman window.
5. At the fundamental frequency bin, gain (dB) and phase (degrees) are
   extracted and printed to the terminal.
6. A two-panel Bode plot (gain over phase, log-frequency x-axis) is saved
   as `bode_plot.png`.

**Example terminal output:**

```
Device opened.

Starting frequency sweep: 10.0 kHz → 50 MHz
      Freq (Hz)   Gain (dB)   Phase (deg)
------------------------------------------
       10000.0       0.142         0.312
       13183.6       0.139         0.408
         ...
    50000000.0      -3.018       -91.245

Device closed.
Bode plot saved to bode_plot.png
```

---

### Script 2 — Spectral Metrics

```bash
# uv
uv run ps5000a_spectral_metrics.py

# pip / activated venv
python ps5000a_spectral_metrics.py
```

**What happens:**

1. The device is opened and channels A, B, C are configured.
2. For each of the three test frequencies, the generator is programmed and
   the system settles for 100 ms.
3. A block capture of 16 384 samples is taken.
4. The differential signal `(B − C)` is windowed with a Blackman-Harris
   function and transformed via FFT.
5. The fundamental bin is located by peak search; harmonic bins are located
   at integer multiples of the fundamental.
6. SNR, SINAD, THD, HD2, and HD3 are computed and printed.
7. A spectrum plot for each frequency (with annotated harmonic markers) is
   saved as `spectral_metrics.png`.
8. All numeric results are saved to `spectral_metrics.csv`.

**Example terminal output:**

```
Device opened.

==============================================================================
        Freq       SNR     SINAD       THD       HD2       HD3
        (Hz)      (dB)      (dB)      (dB)      (dB)      (dB)
==============================================================================
      100000     72.14     68.32    -71.08    -74.55    -79.21
     1000000     69.88     65.71    -68.43    -71.90    -76.12
    10000000     61.23     57.44    -60.17    -63.44    -68.99
==============================================================================

Device closed.
Spectrum plots saved to spectral_metrics.png
Results saved to spectral_metrics.csv
```

---

## Output Files

| File | Script | Description |
|---|---|---|
| `bode_plot.png` | Bode sweep | Two-panel gain and phase Bode plot, log-frequency x-axis |
| `spectral_metrics.png` | Spectral metrics | One spectrum subplot per test frequency with harmonic markers |
| `spectral_metrics.csv` | Spectral metrics | Tabular results: frequency, SNR, SINAD, THD, HD2, HD3, fund amplitude |

All output files are written to the working directory. Add `outputs/` to your
`.gitignore` (already included in this repo) to avoid committing large binary
files.

---

## Measurement Methodology

### Windowing

| Script | Window | Sidelobe level | Use case |
|---|---|---|---|
| Bode sweep | Blackman | −58 dB | General swept-frequency gain/phase |
| Spectral metrics | Blackman-Harris | −92 dB | Low-distortion harmonic identification |

The Blackman-Harris window is chosen for Script 2 because at −92 dB sidelobe
attenuation, spectral leakage from the fundamental will not mask HD2 or HD3
components that are 60–80 dB below the carrier — a common situation in
well-designed analogue front-ends.

### Metric Definitions

| Metric | Formula | Notes |
|---|---|---|
| **Gain** | 20·log₁₀(\|FFT(B−C)\| / \|FFT(A)\|) | Evaluated at the fundamental bin |
| **Phase** | ∠FFT(B−C) − ∠FFT(A) | Degrees, unwrapped |
| **SNR** | 20·log₁₀(H1 / V_noise) | Noise excludes fundamental and harmonics |
| **SINAD** | 20·log₁₀(H1 / √(harmonics² + noise²)) | Signal-to-Noise-and-Distortion |
| **THD** | 20·log₁₀(√(H2²+…+Hn²) / H1) | H2 through H6 by default |
| **HD2** | 20·log₁₀(H2 / H1) | Second harmonic distortion |
| **HD3** | 20·log₁₀(H3 / H1) | Third harmonic distortion |

### Timebase Selection

The `find_timebase()` helper iterates `ps5000aGetTimebase2` to find the
fastest available timebase that still captures at least 20 complete cycles of
the test signal within the requested sample buffer. This balances frequency
resolution (more cycles → narrower FFT bins) against acquisition time.

---

## Troubleshooting

**`PICO_NOT_FOUND` or device not opening**
- Confirm the PicoSDK system driver is installed and the device is connected
  via USB before running any script.
- On Linux, ensure your user is in the `plugdev` group:
  `sudo usermod -aG plugdev $USER` then log out and back in.

**`assert_pico_ok` raises on `openUnit` with code 286 or 282**
- These are expected power-source warnings (USB-powered without external PSU,
  or USB 3.0 device on a USB 2.0 port). Both scripts handle them
  automatically via `ps5000aChangePowerSource`.

**Generator frequency above ~20 MHz sounds distorted or amplitude drops**
- This is the bandwidth limit of the ps5000a built-in generator on some
  models (e.g. 5242B is rated to 20 MHz, 5444B to 200 MHz). Check the
  datasheet for your specific model and adjust `FREQ_STOP_HZ` accordingly.

**FFT bin does not land on the fundamental**
- Increase `N_SAMPLES` to improve frequency resolution (`df = fs / N`).
- The `find_timebase` function targets 20 cycles per record; at very high
  frequencies the actual sample rate may be constrained by the timebase
  hardware — print `dt_ns` to verify.

**HD2 / HD3 readings seem too high**
- Verify that the generator amplitude does not overdrive the ASIC input.
  Reduce `GEN_PK2PK_UV` and confirm Ch B and Ch C are not clipping
  (overflow flag printed by the SDK will be non-zero if they are).
- Verify probe ground leads are short and at the same potential to minimise
  external common-mode interference.

---

## Contributing

Contributions are welcome. Please follow these guidelines to keep the
codebase consistent and the review process efficient.

### Reporting Issues

Open a GitHub Issue and include:
- Your PicoScope model number and firmware version
- Your operating system and Python version (`python --version`)
- The full error traceback
- The values of any configuration constants you changed

### Submitting a Pull Request

1. **Fork** the repository and create a feature branch from `main`
   (see [Branching Model](#cloning-and-branching-model) below).
2. **Install dev dependencies** so linting and tests run locally:
   ```bash
   uv sync --extra dev
   # or
   pip install -r requirements-dev.txt
   ```
3. **Write your code.** Keep all user-tunable values as named constants.
   Do not introduce hard-coded literals into measurement logic.
4. **Lint and format** before committing:
   ```bash
   ruff check .
   ruff format .
   ```
5. **Add or update tests** for any new DSP helpers or utility functions
   under `tests/`. Run the full suite:
   ```bash
   pytest
   # or with uv
   uv run pytest
   ```
6. **Update `README.md`** if you add a new script, configuration constant,
   or output file.
7. Open a Pull Request against `main`. The PR description should state:
   - What the change does and why
   - Which PicoScope model and measurement scenario it was validated on
   - Before/after plots or CSV excerpts if the change affects numeric output

### Code Style

| Rule | Detail |
|---|---|
| Formatter | `ruff format` (88-char lines) |
| Linter | `ruff check` — E, F, W, I, UP rule sets |
| Docstrings | NumPy-style for all public functions |
| Type hints | Required for all new function signatures |
| Constants | `UPPER_SNAKE_CASE` at module level |
| Units | Always state units in variable names or inline comments (e.g. `dt_ns`, `freq_hz`, `amplitude_mV`) |

### What We Welcome

- Support for additional PicoScope families (ps6000a, ps3000a) via
  driver-agnostic abstractions
- Additional spectral metrics (SFDR, ENOB, IMD)
- Automated test fixtures that mock the `picosdk` C calls for CI without
  hardware
- Export to additional formats (HDF5, TDMS, JSON)
- GUI front-end for real-time parameter control

### What to Avoid

- Breaking changes to the public constant names (downstream test scripts
  may depend on them)
- Adding mandatory dependencies without discussion in an Issue first
- Committing output files (`*.png`, `*.csv`) or virtual environment
  directories

---

## Cloning and Branching Model

### Cloning

```bash
# HTTPS (default — works everywhere)
git clone https://github.com/<your-org>/picoscope-asic-measurement.git

# SSH (recommended if you have a key configured)
git clone git@github.com:<your-org>/picoscope-asic-measurement.git

cd picoscope-asic-measurement
```

### Branching Model

This repository follows a simplified **GitHub Flow**:

```
main                    # always stable and deployable
 └── feature/<topic>    # new scripts, metrics, or instrument support
 └── fix/<topic>        # bug fixes
 └── docs/<topic>       # documentation-only changes
 └── chore/<topic>      # dependency bumps, CI changes, tooling
```

| Branch prefix | Use for |
|---|---|
| `feature/` | New measurement scripts, new metrics, new hardware support |
| `fix/` | Bug fixes in existing scripts or DSP logic |
| `docs/` | README, docstring, or comment updates only |
| `chore/` | Dependency updates, CI pipeline changes, tooling config |

**Workflow:**

```bash
# 1. Always branch from an up-to-date main
git checkout main
git pull origin main

# 2. Create your branch
git checkout -b feature/sfdr-metric

# 3. Make commits — small and focused
git add ps5000a_spectral_metrics.py
git commit -m "feat: add SFDR computation to spectral_metrics helper"

# 4. Push and open a PR
git push origin feature/sfdr-metric
```

Commits should follow the
[Conventional Commits](https://www.conventionalcommits.org/) format:

| Prefix | When to use |
|---|---|
| `feat:` | A new measurement, metric, or feature |
| `fix:` | A bug fix |
| `docs:` | Documentation only |
| `chore:` | Maintenance, dependency bumps |
| `test:` | New or updated tests |
| `refactor:` | Code restructured without behaviour change |

Direct pushes to `main` are disabled. All changes enter via Pull Request
and require at least one approval before merging.

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE)
for the full text.

PicoScope® is a registered trademark of Pico Technology Ltd.
This project is not affiliated with or endorsed by Pico Technology Ltd.
