#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
REDSoX / GOSoX-style Effective Area + Minimum Detectable Polarization Calculator 

This script builds a wavelength-dependent REDSoX instrument response by
combining mirror throughput, multilayer reflectivities, detector quantum
efficiency, optical blocking filter transmission, and grating efficiencies.
It then writes effective-area and modulation-factor tables and evaluates
count rates and 99% confidence MDPs for a set of representative sources.

Required response files are expected in a ``data/`` subdirectory, and
generated effective-area products are written to an ``outputs/`` subdirectory.

Run built-in samples with:

    python3 mdp_redsox.py samples

Run a custom source with:

    python3 mdp_redsox.py custom --help
"""

import math
import warnings
import numpy as np
from pathlib import Path
import argparse

HC_KEV_ANG = 12.3984193       # keV * Angstrom
CM_PER_PC = 3.0856776e18      # cm

__version__ = "2.1"

WELCOME_BANNER = f"""
============================================================
  Welcome to the REDSoX Sensitivity Calculator
  Version: {__version__}
------------------------------------------------------------
  Authors: Swati Ravi and Herman Marshall
------------------------------------------------------------
  Effective Area and Minimum Detectable Polarization (MDP)
  Calculator for Soft X-ray Spectropolarimetry

  Copyright (C) 2026, the REDSoX Team.

  This software computes:
    • Effective areas
    • Modulation factors
    • Count rates
    • Minimum Detectable Polarizations (MDPs)

  License: This is free research software provided as-is for scientific use. 
  You are welcome to redistribute it under certain conditions. 
  Licensed under the MIT License (see LICENSE file for details).

  Disclaimer: This software is provided as-is with
  ABSOLUTELY NO WARRANTY.

  Contact: swatir@mit.edu
============================================================
"""

def load_two_cols_forgiving(fname):
    """
    Load the first two numeric columns from a text file.

    Lines that are empty, contain fewer than two columns, or cannot be
    parsed as floats are skipped automatically.

    Parameters
    ----------
    fname : str
        Path to the input file.

    Returns
    -------
    xs, ys : numpy.ndarray
        Arrays containing the first and second numeric columns.

    Raises
    ------
    ValueError
        If no valid numeric rows are found.
    """
    xs, ys = [], []
    with open(fname, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
            except Exception:
                continue
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size == 0:
        raise ValueError(f"{fname}: no numeric rows found.")
    return xs, ys


def idl_interpol(y, x, xnew):
    """
    Perform 1D linear interpolation in the style of IDL's INTERPOL function 
    Everyone Say: "Thanks, Herman"!!

    The input coordinates are sorted before interpolation to ensure
    monotonic ordering. Values requested outside the input range are
    assigned the nearest endpoint value rather than being extrapolated.

    Parameters
    ----------
    y : array-like
        Values defined at coordinates ``x``.
    x : array-like
        Input coordinates.
    xnew : array-like
        Coordinates at which to evaluate the interpolated values.

    Returns
    -------
    numpy.ndarray
        Interpolated values at ``xnew``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xnew = np.asarray(xnew, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    return np.interp(xnew, x, y, left=y[0], right=y[-1])

def bbspec(wave_angstrom, kT_keV):
    """
    Compute the blackbody photon flux spectrum as a function of wavelength.

    Returns the hemispherically integrated blackbody photon flux per unit
    wavelength at the emitting surface, assuming a unit geometric
    normalization (omega = 1). The spectrum is expressed in photon units
    rather than energy units.

    Parameters
    ----------
    wave_angstrom : array-like
        Wavelengths in Angstroms.
    kT_keV : float
        Blackbody temperature in keV.

    Returns
    -------
    numpy.ndarray
        Photon flux density in units of photons cm^-2 s^-1 Å^-1.

    Notes
    -----
    The geometric dilution factor (R/d)^2 is not included and should be
    applied separately when modeling a source at finite distance.
    """
    wave = np.asarray(wave_angstrom, dtype=float)

    # cgs constants
    c_cms = 2.99792458e10              # cm/s
    hc_erg_cm = 1.98644586e-16         # erg*cm
    kev_to_erg = 1.602176634e-9        # erg/keV

    lam_cm = wave * 1e-8               # Å -> cm
    kT_erg = float(kT_keV) * kev_to_erg

    x = hc_erg_cm / (lam_cm * kT_erg)

    out = np.empty_like(x)
    large = x > 50.0

    # Photon spectral radiance per wavelength per steradian:
    # N_λ = (2c/λ^4)/(exp(x)-1) [photons s^-1 cm^-2 sr^-1 cm^-1]
    # Flux over hemisphere: π N_λ
    # Convert per cm to per Å: multiply by 1e-8
    out[large] = np.pi * (2.0 * c_cms / lam_cm[large]**4) * np.exp(-x[large]) * 1e-8
    out[~large] = np.pi * (2.0 * c_cms / lam_cm[~large]**4) / np.expm1(x[~large]) * 1e-8

    out[~np.isfinite(out)] = 0.0
    return out

def ism_tb_pyxspec(E_keV, NH_cm2, abund="wilm", xsect=None):
    """
    Compute the interstellar transmission using XSPEC's TBabs model.

    A temporary TBabs*powerlaw model with photon index 0 and normalization 1
    is evaluated through PyXspec so that the model output corresponds to the
    energy-dependent transmission fraction.

    Parameters
    ----------
    E_keV : array-like
        Energies in keV.
    NH_cm2 : float
        Hydrogen column density in cm^-2.
    abund : str, optional
        XSPEC abundance table (default: "wilm").
    xsect : str, optional
        XSPEC photoelectric cross-section table. If None, XSPEC's current
        setting is used.

    Returns
    -------
    numpy.ndarray
        Transmission fraction at each energy, ranging from 0 to 1.
    """
    import xspec
    xspec.Xset.abund = "wilm"
    xspec.Xset.xsect = "vern"

    E_keV = np.asarray(E_keV, dtype=float)
    if E_keV.ndim != 1:
        raise ValueError("E_keV must be a 1D array")

    xspec.Xset.chatter = 0
    xspec.Xset.logChatter = 0

    if abund is not None:
        xspec.Xset.abund = abund
    if xsect is not None:
        xspec.Xset.xsect = xsect

    try:
        xspec.AllModels.clear()
    except Exception:
        pass

    m = xspec.Model("TBabs*powerlaw")
    nh_1e22 = float(NH_cm2) / 1e22
    m.setPars(nh_1e22, 0.0, 1.0)

    Emin = float(np.min(E_keV))
    Emax = float(np.max(E_keV))
    npts = int(len(E_keV))

    Emin = max(Emin, 1e-4)
    Emax = max(Emax, Emin * 1.0001)

    xspec.AllModels.setEnergies(f"{Emin} {Emax} {npts} lin")
    xspec.Plot.xAxis = "keV"
    xspec.Plot("model")

    mod_x = np.array(xspec.Plot.x(), dtype=float)
    mod_y = np.array(xspec.Plot.model(), dtype=float)

    trans = np.interp(E_keV, mod_x, mod_y, left=mod_y[0], right=mod_y[-1])
    trans[~np.isfinite(trans)] = 0.0
    return np.clip(trans, 0.0, 1.0)


def ism_tb_approx(E_keV, NH_cm2):
    """
    Compute an approximate interstellar transmission curve.

    Uses a simple photoelectric absorption model with
    sigma(E) = 3e-22 E^-3 cm^2, giving a transmission of
    exp(-NH * sigma). This is intended as a fast approximation
    and is less accurate than a full TBabs calculation.

    Parameters
    ----------
    E_keV : array-like
        Energies in keV.
    NH_cm2 : float
        Hydrogen column density in cm^-2.

    Returns
    -------
    numpy.ndarray
        Transmission fraction at each energy.
    """
    E_keV = np.asarray(E_keV, dtype=float)
    E_keV = np.clip(E_keV, 1e-3, None)
    sigma = 3.0e-22 * (E_keV ** -3.0)
    return np.exp(-NH_cm2 * sigma)


def ism_tb(E_keV, NH_cm2):
    """
    Compute interstellar absorption transmission.

    Uses the XSPEC TBabs model via PyXspec when available. If PyXspec
    cannot be imported or evaluated, falls back to a simple analytic
    approximation and issues a warning.

    Parameters
    ----------
    E_keV : array-like
        Energies in keV.
    NH_cm2 : float
        Hydrogen column density in cm^-2.

    Returns
    -------
    numpy.ndarray
        Transmission fraction at each energy.
    """
    try:
        import xspec  # noqa: F401
        return ism_tb_pyxspec(E_keV, NH_cm2)
    except Exception as e:
        warnings.warn(f"pyXspec TBabs failed — using approximate ISM model. Reason: {e}")
        return ism_tb_approx(E_keV, NH_cm2)

def read_eff_tsv(fname):
    """
    Read diffraction efficiency tables from a TSV file.

    The input file is assumed to contain wavelength-angle grids with
    diffraction efficiencies for orders 0, +1, and +2. The data are
    reshaped into 2D arrays indexed by wavelength and diffraction angle.

    Parameters
    ----------
    fname : str
        Path to the efficiency table.

    Returns
    -------
    wave : numpy.ndarray
        Wavelength grid in Angstroms.
    theta : numpy.ndarray
        Diffraction angle grid in degrees.
    eff0, eff1, eff2 : numpy.ndarray
        Efficiency arrays for orders 0, +1, and +2 with shape
        (nwave, nangle).

    Raises
    ------
    ValueError
        If the file format is invalid or the wavelength/angle grids
        cannot be inferred.
    """
    data = np.genfromtxt(fname, dtype=float)
    if data.ndim != 2 or data.shape[1] < 6:
        raise ValueError(f"{fname}: expected >= 6 numeric columns")

    wave_nm = data[:, 0]
    angle0 = data[:, 1]
    o0 = data[:, 3]
    o1 = data[:, 4]
    o2 = data[:, 5]

    wave = 10.0 * wave_nm[angle0 == 0.0]                 # Å
    theta = angle0[np.isclose(wave_nm, 1.5)]              # deg

    nwave = wave.size
    nangle = theta.size
    if nwave == 0 or nangle == 0:
        raise ValueError(f"{fname}: could not infer wave/theta axes (check file format)")

    eff0 = np.zeros((nwave, nangle), dtype=float)
    eff1 = np.zeros((nwave, nangle), dtype=float)
    eff2 = np.zeros((nwave, nangle), dtype=float)

    for i in range(nwave):
        ii = i * nangle
        eff0[i, :] = o0[ii:ii + nangle]
        eff1[i, :] = o1[ii:ii + nangle]
        eff2[i, :] = o2[ii:ii + nangle]

    return wave, theta, eff0, eff1, eff2

def mdp_redsox(wave, nlam, lam1, lam2, area1_lam_lo, area1_lam_hi, area0_lam, modfactor_lo, modfactor_hi, exptime, bg, src_name):
    """
    Compute REDSoX count rates and minimum detectable polarization (MDP).

    Calculates the zeroth-order and first-order count rates, total source
    counts, and the 99% confidence MDP using the wavelength-dependent
    effective areas and modulation factors. MDPs are reported for both the
    full wavelength range and a user-specified wavelength band.

    Parameters
    ----------
    wave : array-like
        Wavelength grid in Angstroms.
    nlam : array-like
        Source photon spectrum in photons cm^-2 s^-1 Å^-1.
    lam1, lam2 : float
        Lower and upper wavelength limits for the band-limited MDP.
    area1_lam_lo, area1_lam_hi : array-like
        Effective areas for the -1 and +1 diffraction orders.
    area0_lam : array-like
        Effective area for the zeroth order.
    modfactor_lo, modfactor_hi : array-like
        Modulation factors for the -1 and +1 diffraction orders.
    exptime : float
        Exposure time in seconds.
    bg : float
        Background count rate in counts s^-1.
    src_name : str
        Source name used in the printed summary.

    Returns
    -------
    rate0 : float
        Zeroth-order count rate (counts s^-1).
    rate1 : float
        First-order count rate within the selected wavelength band
        (counts s^-1).
    counts : float
        Total first-order source counts in the selected band.
    mdp : float
        Full-band 99% confidence minimum detectable polarization.
    mdp_band : float
        Band-limited 99% confidence minimum detectable polarization.
    """
    wave = np.asarray(wave, dtype=float)
    nlam = np.asarray(nlam, dtype=float)
    area1_lam_lo = np.asarray(area1_lam_lo, dtype=float)
    area1_lam_hi = np.asarray(area1_lam_hi, dtype=float)
    area0_lam = np.asarray(area0_lam, dtype=float)
    modfactor_lo = np.asarray(modfactor_lo, dtype=float)
    modfactor_hi = np.asarray(modfactor_hi, dtype=float)

    oklam = np.where((wave >= lam1) & (wave < lam2))[0]
    dlam = wave[1] - wave[0]

    rate0 = np.sum(nlam * area0_lam)

    rate1_lam_lo = nlam * area1_lam_lo
    rate1_lam_hi = nlam * area1_lam_hi
    rate1_tot = np.sum(rate1_lam_lo + rate1_lam_hi)
    rate1 = np.sum(rate1_lam_lo[oklam] + rate1_lam_hi[oklam])
    counts = exptime * rate1

    # Full-band MDP
    denom_full = np.sum(modfactor_lo**2 * rate1_lam_lo + modfactor_hi**2 * rate1_lam_hi)
    if rate1_tot <= 0 or denom_full <= 0:
        mdp = np.inf
    else:
        bg_factor = math.sqrt(1.0 + bg / rate1_tot)
        mdp_const = 4.29 * bg_factor / math.sqrt(exptime)
        mdp = mdp_const / math.sqrt(denom_full)

    # Band-limited MDP
    denom_band = np.sum(
        modfactor_lo[oklam]**2 * rate1_lam_lo[oklam] +
        modfactor_hi[oklam]**2 * rate1_lam_hi[oklam]
    )
    if rate1 <= 0 or denom_band <= 0:
        mdp_band = np.inf
    else:
        bg_factor = math.sqrt(1.0 + bg / rate1)
        mdp_const = 4.29 * bg_factor / math.sqrt(exptime)
        mdp_band = mdp_const / math.sqrt(denom_band)

    print(f"; {src_name}, Rate0, rate1, Cnt, MDP_full, MDP_band")
    print(f"  rate0={rate0:.6e}  rate1={rate1:.6e}  counts={counts:.6e}  mdp={mdp:.6g}  mdp_band={mdp_band:.6g}")

    return rate0, rate1, counts, mdp, mdp_band

def build_effective_areas(data_dir: Path, wave):
    """
    Construct REDSoX effective area and modulation factor arrays.

    Combines mirror geometric area, multilayer reflectivities, detector
    quantum efficiency, optical blocking filter transmission, and grating
    diffraction efficiencies to compute the wavelength-dependent effective
    areas for the zeroth and first diffraction orders.

    Parameters
    ----------
    data_dir : pathlib.Path
        Directory containing the REDSoX response and efficiency files.
    wave : array-like
        Wavelength grid in Angstroms.

    Returns
    -------
    area1_lam_lo, area1_lam_hi : numpy.ndarray
        Effective area contributions for the lower and upper polarized
        first-order beams.
    area0_lam : numpy.ndarray
        Zeroth-order effective area.
    modfactor_lo, modfactor_hi : numpy.ndarray
        Modulation factors for the lower and upper polarized beams.
    geom_area : float
        Net geometric collecting area after obscuration losses.
    detqe_filt : numpy.ndarray
        Detector quantum efficiency including optical blocking filter
        transmission.
    """
    wave = np.asarray(wave, dtype=float)
    dlam = wave[1] - wave[0]
    nrg = HC_KEV_ANG / wave  # keV

    # modulation factor arrays
    modfactor_lo = np.minimum(0.95 + 0.01 * (wave - 50.0) / 20.0, 1.0)
    modfactor_hi = np.minimum(0.923 - 0.006 * (wave - 50.0) / 20.0, 1.0)

    # geometric area scalar
    cat_l1_obscur = 0.78
    cat_l2_obscur = 0.81
    cat_obscur = cat_l1_obscur * cat_l2_obscur
    l3_obscur = 0.83

    mirror_baseline = 409.0
    mirror_area = (0.89**2) * mirror_baseline
    mirror_mount_transmission = (1.0 - 0.07) * 0.95

    geom_area = l3_obscur * cat_obscur * mirror_mount_transmission * mirror_area  # cm^2

    # ML reflectivities (read, interpolate to wave)
    lam_hi, refl_hi = load_two_cols_forgiving(data_dir / "ml_redsox_40.txt")
    lam_lo, refl_lo = load_two_cols_forgiving(data_dir / "ml_redsox_50.txt")

    ml_eff_lo = idl_interpol(refl_lo, lam_lo, wave)
    ml_eff_hi = idl_interpol(refl_hi, lam_hi, wave)

    # 3/15/25 correction from pol'd beam to unpol'd reflectivity 
    rs_over_rp_lo = (1.0 - modfactor_lo) / (1.0 + modfactor_lo)
    ref_factor_lo = 0.97 * (1.0 + rs_over_rp_lo) / (0.97 + 0.03 * rs_over_rp_lo)
    ml_eff_lo = ref_factor_lo * ml_eff_lo

    rs_over_rp_hi = (1.0 - modfactor_hi) / (1.0 + modfactor_hi)
    ref_factor_hi = 0.97 * (1.0 + rs_over_rp_hi) / (0.97 + 0.03 * rs_over_rp_hi)
    ml_eff_hi = ref_factor_hi * ml_eff_hi

    # losses/jitter are currently zeroed in IDL (ml_refl_* = ml_eff_*)
    ml_refl_lo = ml_eff_lo.copy()
    ml_refl_hi = ml_eff_hi.copy()

    # 3/15/25 linearity losses 
    linearity_loss = 0.147
    ml_refl_lo *= (1.0 - linearity_loss)
    ml_refl_hi *= (1.0 - linearity_loss)

    # detector QE 
    nrg_kev_qe, qe = load_two_cols_forgiving(data_dir / "ccd097.txt")
    detqe = idl_interpol(qe, nrg_kev_qe, nrg)

    # OBF transmission
    nrg_al, trans_al = load_two_cols_forgiving(data_dir / "aluminum_transmission.txt")
    nrg_poly, trans_poly = load_two_cols_forgiving(data_dir / "polyimide_transmission.txt")

    al_thick_nm = 25.0
    poly_thick_nm = 45.0
    al_tau = al_thick_nm / 100.0       # referenced to 0.1 micron = 100 nm
    poly_tau = poly_thick_nm / 100.0

    trans_al_on = idl_interpol(trans_al, 0.001 * nrg_al, nrg)
    trans_poly_on = idl_interpol(trans_poly, 0.001 * nrg_poly, nrg)

    # mesh factor 0.82 
    trans_obf = 0.82 * (trans_al_on ** al_tau) * (trans_poly_on ** poly_tau)
    detqe_filt = trans_obf * detqe

    # grating efficiencies and blaze angle selection
    wave_eff, theta, eff0, eff1, eff2 = read_eff_tsv(data_dir / "Si_4um_deep_for_MDP.tsv")
    target_theta = 0.7
    iangle = int(np.argmin(np.abs(theta - target_theta)))

    eff0_theta = eff0[:, iangle]
    eff1_theta = eff1[:, iangle]

    # interpolate efficiencies onto wave grid (IDL interpol(eff[*,iangle], wave_eff, wave))
    eff0_on = idl_interpol(eff0_theta, wave_eff, wave)
    eff1_on = idl_interpol(eff1_theta, wave_eff, wave)

    # IDL " > 0" clamps to 0 for negatives
    eff0_on = np.maximum(eff0_on, 0.0)
    eff1_on = np.maximum(eff1_on, 0.0)

    # assemble EA integrand arrays 
    area1_lam_lo = 0.5 * dlam * geom_area * detqe_filt * ml_refl_lo * eff1_on
    area1_lam_hi = 0.5 * dlam * geom_area * detqe_filt * ml_refl_hi * eff1_on
    area0_lam = dlam * geom_area * detqe_filt * eff0_on

    # clamp (IDL >0)
    area1_lam_lo = np.maximum(area1_lam_lo, 0.0)
    area1_lam_hi = np.maximum(area1_lam_hi, 0.0)
    area0_lam = np.maximum(area0_lam, 0.0)

    return area1_lam_lo, area1_lam_hi, area0_lam, modfactor_lo, modfactor_hi, geom_area, detqe_filt

def run_all_sources(wave, nrg, lam1, lam2, area1_lam_lo, area1_lam_hi, area0_lam, modfactor_lo, modfactor_hi, exptime, bg):
    """
    Evaluate REDSoX performance for a set of representative astrophysical sources.

    For each source, a spectral model and interstellar absorption are applied
    to generate a photon spectrum, which is then passed to the REDSoX MDP
    calculator. Results include count rates, total counts, and full-band and
    band-limited minimum detectable polarizations.

    Parameters
    ----------
    wave, nrg : array-like
        Wavelength (Å) and energy (keV) grids.
    lam1, lam2 : float
        Wavelength limits for the band-limited MDP calculation.
    area1_lam_lo, area1_lam_hi, area0_lam : array-like
        REDSoX effective area arrays.
    modfactor_lo, modfactor_hi : array-like
        REDSoX modulation factor arrays.
    exptime : float
        Exposure time in seconds.
    bg : float
        Background count rate in counts s^-1.

    Returns
    -------
    dict
        Dictionary keyed by source name. Each entry contains
        ``(rate0, rate1, counts, mdp, mdp_band)``.
    """
    results = {}

    # RX J1856
    src_name = "RX J1856"
    nh = 8.0e19
    kT_keV = 0.06228
    omega = (4.95e5 / (130.0 * CM_PER_PC)) ** 2
    ism = ism_tb(nrg, nh)
    nlam = omega * bbspec(wave, kT_keV) * ism
    results[src_name] = mdp_redsox(
        wave, nlam, lam1, lam2,
        area1_lam_lo, area1_lam_hi, area0_lam,
        modfactor_lo, modfactor_hi,
        exptime, bg, src_name
    )

    # RX J0720
    src_name = "RX J0720"
    nh = 0.886e20
    kT1_keV = 0.0924
    omega1 = (4.5e5 / (300.0 * CM_PER_PC)) ** 2
    omega2 = 0.0
    ism = ism_tb(nrg, nh)
    bb1 = omega1 * bbspec(wave, kT1_keV)
    bb2 = omega2 * bbspec(wave, 0.0 + 1e-6)  # omega2=0 => harmless
    nlam = (bb1 + bb2) * ism
    results[src_name] = mdp_redsox(
        wave, nlam, lam1, lam2,
        area1_lam_lo, area1_lam_hi, area0_lam,
        modfactor_lo, modfactor_hi,
        exptime, bg, src_name
    )

    # PSR B0656 (note: IDL label says 1e6 s, but IDL exptime var is set earlier)
    src_name = "PSR B0656, 1e6 s"
    nh = 0.12e20
    kT1_keV = 0.0679
    kT2_keV = 0.170
    omega1 = (9.53e5 / (300.0 * CM_PER_PC)) ** 2
    omega2 = (0.373e5 / (300.0 * CM_PER_PC)) ** 2
    ism = ism_tb(nrg, nh)
    bb1 = omega1 * bbspec(wave, kT1_keV)
    bb2 = omega2 * bbspec(wave, kT2_keV)
    nlam = (bb1 + bb2) * ism
    results[src_name] = mdp_redsox(
        wave, nlam, lam1, lam2,
        area1_lam_lo, area1_lam_hi, area0_lam,
        modfactor_lo, modfactor_hi,
        exptime, bg, src_name
    )

    # Her X-1
    src_name = "Her X-1"
    norm = 0.0
    slope = 1.5
    nh = 1.5e20
    bbnorm = 440.0
    kt_keV = 0.11
    ism = ism_tb(nrg, nh)
    pl_eflux = norm * (nrg ** (-slope))
    bb_eflux = bbnorm * (nrg ** 2) / np.expm1(np.clip(nrg / kt_keV, 1e-12, 700.0))
    eflux = (pl_eflux + bb_eflux) * ism
    nlam = (nrg ** 2) * eflux / HC_KEV_ANG
    results[src_name] = mdp_redsox(
        wave, nlam, lam1, lam2,
        area1_lam_lo, area1_lam_hi, area0_lam,
        modfactor_lo, modfactor_hi,
        exptime, bg, src_name
    )

    # Mk 421
    src_name = "Mk 421"
    norm = 0.25
    slope = 2.7
    nh = 1.45e20
    ism = ism_tb(nrg, nh)
    eflux = norm * (nrg ** (-slope)) * ism
    nlam = (nrg ** 2) * eflux / HC_KEV_ANG
    results[src_name] = mdp_redsox(
        wave, nlam, lam1, lam2,
        area1_lam_lo, area1_lam_hi, area0_lam,
        modfactor_lo, modfactor_hi,
        exptime, bg, src_name
    )

    # PKS 2155
    src_name = "PKS 2155"
    norm = 0.04
    slope = 2.8
    nh = 1.36e20
    ism = ism_tb(nrg, nh)
    eflux = norm * (nrg ** (-slope)) * ism
    nlam = (nrg ** 2) * eflux / HC_KEV_ANG
    results[src_name] = mdp_redsox(
        wave, nlam, lam1, lam2,
        area1_lam_lo, area1_lam_hi, area0_lam,
        modfactor_lo, modfactor_hi,
        exptime, bg, src_name
    )

    # 3C 273
    src_name = "3C 273"
    norm1 = 0.00996
    slope1 = 2.30
    norm2 = 0.0110
    slope2 = 1.44
    nh = 1.13e20
    ism = ism_tb(nrg, nh)
    pl1 = norm1 * (nrg ** (-slope1))
    pl2 = norm2 * (nrg ** (-slope2))
    eflux = ism / (1.0 / np.clip(pl1, 1e-300, None) + 1.0 / np.clip(pl2, 1e-300, None))
    nlam = (nrg ** 2) * eflux / HC_KEV_ANG
    results[src_name] = mdp_redsox(
        wave, nlam, lam1, lam2,
        area1_lam_lo, area1_lam_hi, area0_lam,
        modfactor_lo, modfactor_hi,
        exptime, bg, src_name
    )

    # Ark 564
    src_name = "Ark 564"
    norm1 = 10.0 * 0.001
    slope1 = 2.5
    norm2 = 3.3 * 0.001
    slope2 = 3.6
    nh = 5.6e20
    ism = ism_tb(nrg, nh)
    pl1 = norm1 * (nrg ** (-slope1))
    pl2 = norm2 * (nrg ** (-slope2))
    eflux = (pl1 + pl2) * ism
    nlam = (nrg ** 2) * eflux / HC_KEV_ANG
    results[src_name] = mdp_redsox(
        wave, nlam, lam1, lam2,
        area1_lam_lo, area1_lam_hi, area0_lam,
        modfactor_lo, modfactor_hi,
        exptime, bg, src_name
    )

    # Mk 478
    src_name = "Mk 478"
    norm1 = 3.3 * 0.001
    slope1 = 3.03
    norm2 = 0.28 * 0.001
    slope2 = 1.4
    nh = 9.8e19
    ism = ism_tb(nrg, nh)
    pl1 = norm1 * (nrg ** (-slope1))
    pl2 = norm2 * (nrg ** (-slope2))
    eflux = (pl1 + pl2) * ism
    nlam = (nrg ** 2) * eflux / HC_KEV_ANG
    results[src_name] = mdp_redsox(
        wave, nlam, lam1, lam2,
        area1_lam_lo, area1_lam_hi, area0_lam,
        modfactor_lo, modfactor_hi,
        exptime, bg, src_name
    )

    return results

def parse_args():
    """
    Parse command-line arguments for REDSoX sensitivity calculations.

    Supports two operating modes:

    - ``samples``: run the built-in benchmark source suite.
    - ``custom``: evaluate a user-defined source model.

    Custom sources may be specified as single- or two-component
    blackbody and/or power-law models, together with observing
    parameters such as wavelength band, exposure time, and
    background rate.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute REDSoX/GOSoX effective areas, count rates, and "
            "99% confidence minimum detectable polarizations (MDPs)."
        ),
        epilog="""
    Examples:

      Run benchmark sources:
          python3 mdp_redsox.py samples

      Run a blackbody source:
          python3 mdp_redsox.py custom --name RXJ \
              --model blackbody \
              --nh 8e19 \
              --kt1 0.06 \
              --omega1 1e-29

      Run a power-law source:
          python3 mdp_redsox.py custom --name AGN \
              --model powerlaw \
              --nh 1e20 \
              --norm1 0.05 \
              --slope1 2.7
    """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    subparsers.add_parser(
        "samples",
        help="Run the built-in benchmark source library.",
        description=(
            "Evaluate the built-in suite of neutron stars, AGN, "
            "blazars, and X-ray binaries."
        )
    )

    custom = subparsers.add_parser(
        "custom",
        help="Evaluate a user-defined source spectrum.",
        description=(
            "Construct an absorbed source spectrum and compute "
            "REDSoX count rates and minimum detectable polarizations."
        )
    )

    custom.add_argument("--name", type=str, default="Custom Source")
    custom.add_argument("--nh", type=float, required=True)
    custom.add_argument(
        "--model",
        choices=["blackbody", "powerlaw", "bb+bb", "pl+pl", "bb+pl"],
        required=True
    )

    # Blackbody component 1
    custom.add_argument("--kt1", type=float)
    custom.add_argument("--omega1", type=float)

    # Blackbody component 2
    custom.add_argument("--kt2", type=float)
    custom.add_argument("--omega2", type=float)

    # Power-law component 1
    custom.add_argument("--norm1", type=float)
    custom.add_argument("--slope1", type=float)

    # Power-law component 2
    custom.add_argument("--norm2", type=float)
    custom.add_argument("--slope2", type=float)

    custom.add_argument("--lam1", type=float, default=32.0)
    custom.add_argument("--lam2", type=float, default=62.0)
    custom.add_argument("--exptime", type=float, default=300.0)
    custom.add_argument("--bg", type=float, default=0.002)

    return parser.parse_args()

def make_custom_source_spectrum(wave, nrg, args):
    """
    Construct a user-defined absorbed source spectrum.

    Generates a photon spectrum from one of several supported spectral
    models (blackbody, power law, two-blackbody, two-power-law, or
    blackbody plus power law) and applies interstellar absorption using
    the specified hydrogen column density.

    Parameters
    ----------
    wave : array-like
        Wavelength grid in Angstroms.
    nrg : array-like
        Energy grid in keV corresponding to ``wave``.
    args : argparse.Namespace
        Parsed command-line arguments containing the source model
        parameters and hydrogen column density.

    Returns
    -------
    numpy.ndarray
        Absorbed photon flux density in units of
        photons cm^-2 s^-1 Å^-1.

    Raises
    ------
    ValueError
        If the selected model is unknown or required model
        parameters are missing.
    """
    ism = ism_tb(nrg, args.nh)

    if args.model == "blackbody":
        if args.kt1 is None or args.omega1 is None:
            raise ValueError("blackbody requires --kt1 and --omega1")
        nlam = args.omega1 * bbspec(wave, args.kt1)

    elif args.model == "powerlaw":
        if args.norm1 is None or args.slope1 is None:
            raise ValueError("powerlaw requires --norm1 and --slope1")
        eflux = args.norm1 * (nrg ** (-args.slope1))
        nlam = (nrg ** 2) * eflux / HC_KEV_ANG

    elif args.model == "bb+bb":
        if None in (args.kt1, args.omega1, args.kt2, args.omega2):
            raise ValueError("bb+bb requires --kt1 --omega1 --kt2 --omega2")
        bb1 = args.omega1 * bbspec(wave, args.kt1)
        bb2 = args.omega2 * bbspec(wave, args.kt2)
        nlam = bb1 + bb2

    elif args.model == "pl+pl":
        if None in (args.norm1, args.slope1, args.norm2, args.slope2):
            raise ValueError("pl+pl requires --norm1 --slope1 --norm2 --slope2")
        pl1 = args.norm1 * (nrg ** (-args.slope1))
        pl2 = args.norm2 * (nrg ** (-args.slope2))
        eflux = pl1 + pl2
        nlam = (nrg ** 2) * eflux / HC_KEV_ANG

    elif args.model == "bb+pl":
        if None in (args.kt1, args.omega1, args.norm1, args.slope1):
            raise ValueError("bb+pl requires --kt1 --omega1 --norm1 --slope1")
        bb = args.omega1 * bbspec(wave, args.kt1)
        pl_eflux = args.norm1 * (nrg ** (-args.slope1))
        pl_nlam = (nrg ** 2) * pl_eflux / HC_KEV_ANG
        nlam = bb + pl_nlam

    else:
        raise ValueError(f"Unknown model: {args.model}")

    return nlam * ism

def main():
    """
    Generate REDSoX effective area products and evaluate source performance.

    Builds the instrument response on a wavelength grid, computes effective
    areas and modulation factors, writes response tables to disk, reports
    grasp diagnostics, and evaluates count rates and minimum detectable
    polarizations for a set of representative astrophysical sources.
    """
    args = parse_args()
    print(WELCOME_BANNER)
    
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    # wavelength grid
    nwave = 1000
    wave = np.arange(nwave, dtype=float) * 0.001 * 32.0 + 32.0
    dlam = wave[1] - wave[0]
    nrg = HC_KEV_ANG / wave

    # Build EA arrays 
    area1_lam_lo, area1_lam_hi, area0_lam, modfactor_lo, modfactor_hi, geom_area, detqe_filt = \
        build_effective_areas(data_dir, wave)

    lam1 = getattr(args, "lam1", 32.0)
    lam2 = getattr(args, "lam2", 62.0)
    oklam = np.where((wave >= lam1) & (wave < lam2))[0]

    # "grasp" diagnostics 
    grasp0 = float(np.sum(area0_lam[oklam]))
    grasp1_tot = float(np.sum(area1_lam_lo + area1_lam_hi))
    grasp1 = float(np.sum((area1_lam_lo + area1_lam_hi)[oklam]))

    print(f"; Computing EA for angle ~0.7 deg")
    print(f"; Geom_area scalar (cm^2): {geom_area:.6f}")
    print(f"; detqe_filt min/max: {float(detqe_filt.min()):.6g} / {float(detqe_filt.max()):.6g}")
    print(f"; Grasp0, Grasp1_tot, Grasp1: {grasp0:.6f}  {grasp1_tot:.6f}  {grasp1:.6f}")

    # effective_area.txt columns: wave, area1_lo/dlam, area1_hi/dlam
    np.savetxt(
        output_dir / "effective_area.txt",
        np.column_stack([wave[oklam], area1_lam_lo[oklam] / dlam, area1_lam_hi[oklam] / dlam]),
        fmt="%.6f  %.8e  %.8e",
        header="wave_A  EA1_lo_cm2  EA1_hi_cm2",
        comments=""
    )

    np.savetxt(
        output_dir / "modulation_factor.txt",
        np.column_stack([wave[oklam], modfactor_lo[oklam], modfactor_hi[oklam]]),
        fmt="%.6f  %.8f  %.8f",
        header="wave_A  modfactor_lo  modfactor_hi",
        comments=""
    )

    bg = getattr(args, "bg", 0.002)
    exptime = getattr(args, "exptime", 300.0)

    if args.mode == "samples":
        print("; Running all source blocks from IDL driver...")
        _ = run_all_sources(
            wave, nrg, lam1, lam2,
            area1_lam_lo, area1_lam_hi, area0_lam,
            modfactor_lo, modfactor_hi,
            exptime, bg
        )

    elif args.mode == "custom":
        nlam = make_custom_source_spectrum(wave, nrg, args)

        print(f"; Running custom source: {args.name}")
        _ = mdp_redsox(
            wave, nlam, lam1, lam2,
            area1_lam_lo, area1_lam_hi, area0_lam,
            modfactor_lo, modfactor_hi,
            exptime, bg, args.name
        )


if __name__ == "__main__":
    main()
