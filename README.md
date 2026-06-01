# REDSoX / GOSoX MDP Calculator

**REDSoX/GOSoX Effective Area and Minimum Detectable Polarization Calculator**

Version: `2.1`

Authors: Swati Ravi, Herman Marshall

---

## Overview

The REDSoX/GOSoX MDP Calculator is a Python tool for estimating the soft X-ray spectropolarimetric performance of the REDSoX and GOSoX missions.

The code computes:

* Effective area curves
* Modulation factor curves
* Instrument throughput diagnostics
* Source count rates
* Minimum Detectable Polarization at 99% confidence (MDP99)

for both a built-in library of astrophysical sources and user-defined source models.

The instrument response is constructed from:

* Mirror collecting area
* Grating efficiencies
* Multilayer reflectivities
* Detector quantum efficiency
* Optical blocking filter transmission

The resulting response is folded through a source spectrum to estimate count rates and polarization sensitivity.

This software is intended for scientific and educational use and comes with ABSOLUTELY NO WARRANTY.

---

## Features

### Instrument Performance

* Wavelength-dependent effective areas
* Modulation factor calculations
* Instrument throughput diagnostics
* Full-band MDP calculations
* User-defined wavelength-band MDP calculations

### Source Models

Built-in benchmark sources include:

* RX J1856
* RX J0720
* PSR B0656
* Her X-1
* Mrk 421
* PKS 2155-304
* 3C 273
* Ark 564
* Mrk 478

Custom source models include:

* Single blackbody
* Single power law
* Two blackbodies
* Two power laws
* Blackbody + power law

---

## Installation

Clone the repository:

```bash
git clone https://github.com/swati-ravi/MDP.git
cd MDP
```

Install dependencies:

```bash
pip install numpy
```

Optional (but highly recommended): pyXspec For installation instructions, see the
[official pyXspec documentation](https://heasarc.gsfc.nasa.gov/docs/software/xspec/python/html/).

If pyXspec is unavailable, the code automatically falls back to an approximate interstellar absorption model. 

---

## Repository Structure

```text
redsox-gosox-mdp/
├── mdp_redsox.py
├── README.md
├── LICENSE
├── .gitignore
├── data/
│   ├── ml_redsox_40.txt
│   ├── ml_redsox_50.txt
│   ├── ccd097.txt
│   ├── aluminum_transmission.txt
│   ├── polyimide_transmission.txt
│   └── Si_4um_deep_for_MDP.tsv
└── outputs/
```

---

## Usage

### Run Benchmark Sources

```bash
python3 mdp_redsox.py samples
```

This evaluates all built-in benchmark sources and prints:

* Zeroth-order count rate
* First-order count rate
* Source counts
* Full-band MDP
* Band-limited MDP

Before proceeding with custom source models, please check against the following validation table: 

## Validation Against Original Implementation

| Source | R₀ (cts s⁻¹) | R₁ (cts s⁻¹) | Counts | MDP₉₉,band (%) |
|---------|-------------:|-------------:|-------:|---------------:|
| RX J1856 | 0.233 | 0.0135 | 4.05 | 2.443 |
| RX J0720 | 0.173 | 0.0090 | 2.70 | 3.088 |
| PSR B0656 | 0.316 | 0.0184 | 5.52 | 2.058 |
| Her X-1 | 4.787 | 0.2321 | 69.6 | 0.553 |
| Mrk 421 | 11.747 | 0.6523 | 196 | 0.329 |
| PKS 2155-304 | 2.195 | 0.1233 | 37.0 | 0.761 |
| 3C 273 | 0.088 | 0.00465 | 1.39 | 4.652 |
| Ark 564 | 0.253 | 0.0114 | 3.41 | 2.699 |
| Mrk 478 | 0.281 | 0.0163 | 4.89 | 2.200 |

Where:

* **R₀** = zeroth-order count rate (counts s⁻¹)
* **R₁** = first-order count rate within the selected wavelength band (counts s⁻¹)
* **Counts** = total first-order source counts for the specified exposure
* **MDP₉₉,band** = 99% confidence minimum detectable polarization within the selected wavelength band

---

## Custom Source Models

General syntax:

```bash
python3 mdp_redsox.py custom [options]
```

Display all available options:

```bash
python3 mdp_redsox.py custom --help
```

---

### Single Blackbody

Model:

`F_λ = Ω B_λ(T)`

Example:

```bash
python3 mdp_redsox.py custom \
    --name "RXJ-like" \
    --model blackbody \
    --nh 8e19 \
    --kt1 0.062 \
    --omega1 1e-29
```

Parameters:

| Parameter  | Description                       |
| ---------- | --------------------------------- |
| `--kt1`    | Blackbody temperature (keV)       |
| `--omega1` | Geometric normalization ((R/d)^2) |
| `--nh`     | Hydrogen column density (cm⁻²)    |

---

### Single Power Law

Model:

`F(E) = N E^{-Γ}`

Example:

```bash
python3 mdp_redsox.py custom \
    --name "AGN-like" \
    --model powerlaw \
    --nh 1e20 \
    --norm1 0.05 \
    --slope1 2.7
```

Parameters:

| Parameter  | Description             |
| ---------- | ----------------------- |
| `--norm1`  | Power-law normalization |
| `--slope1` | Power-law photon index  |

---

### Two Blackbodies

Model:

`F_λ = Ω₁ B_λ(T₁) + Ω₂ B_λ(T₂)`

Example:

```bash
python3 mdp_redsox.py custom \
    --model bb+bb \
    --nh 1e20 \
    --kt1 0.07 \
    --omega1 1e-29 \
    --kt2 0.17 \
    --omega2 1e-31
```

---

### Two Power Laws

Model:

`F(E) = N₁ E^{-Γ₁} + N₂ E^{-Γ₂}`

Example:

```bash
python3 mdp_redsox.py custom \
    --model pl+pl \
    --nh 5e20 \
    --norm1 0.01 \
    --slope1 2.5 \
    --norm2 0.003 \
    --slope2 3.6
```

---

### Blackbody + Power Law

Model:

`F(E) = Ω B(E,T) + N E^{-Γ}`

Example:

```bash
python3 mdp_redsox.py custom \
    --model bb+pl \
    --nh 1e20 \
    --kt1 0.1 \
    --omega1 1e-30 \
    --norm1 0.01 \
    --slope1 2.0
```

---

## Output

The code writes:

### Effective Area Table

```text
outputs/effective_area.txt
```

Columns:

* Wavelength (Å)
* Lower-order effective area (cm²)
* Upper-order effective area (cm²)

### Modulation Factor Table

```text
outputs/modulation_factor.txt
```

Columns:

* Wavelength (Å)
* Lower-order modulation factor
* Upper-order modulation factor

### Console Output

For each source:

```text
rate0
rate1
counts
mdp
mdp_band
```

---

## Data Files

The `data/` directory contains:

| File                       | Purpose                     |
| -------------------------- | --------------------------- |
| ml_redsox_40.txt           | Multilayer reflectivity     |
| ml_redsox_50.txt           | Multilayer reflectivity     |
| ccd097.txt                 | Detector quantum efficiency |
| aluminum_transmission.txt  | Aluminum transmission       |
| polyimide_transmission.txt | Polyimide transmission      |
| Si_4um_deep_for_MDP.tsv    | Grating efficiencies        |

---

## License

This is free research software provided as-is for scientific use. You are welcome to redistribute it under certain conditions. Licensed under the MIT License (see LICENSE file for details).

---

## Disclaimer

This software is provided as-is for scientific use.

ABSOLUTELY NO WARRANTY is provided.

---

## Contact

For questions, please contact

[swatir@mit.edu](mailto:swatir@mit.edu)

---

## Citation

If you use this software in a publication, please acknowledge the use of this tool!
