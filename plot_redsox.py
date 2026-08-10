#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plotting utilities for REDSoX analysis and presentation figures.

Run the full testing/presentation figure set with:

    python3 plot_redsox.py testing

Expected repository layout:

    MDP/
    ├── mdp_redsox.py
    ├── plot_redsox.py
    ├── data/
    └── outputs/

Figures are written to:

    outputs/presentation_figures/
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np

import mdp_redsox as mdp


def save_figure(fig, output_path):
    """Save a figure at presentation-friendly resolution."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def sorted_xy(x, y):
    """Return x and y sorted by increasing x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    return x[order], y[order]


def interpolate_to_grid(x_source, y_source, x_target):
    """
    Interpolate y(x) onto x_target.

    Points outside the source x range are returned as NaN rather than
    extrapolated.
    """
    x_source, y_source = sorted_xy(x_source, y_source)
    x_target = np.asarray(x_target, dtype=float)

    return np.interp(
        x_target,
        x_source,
        y_source,
        left=np.nan,
        right=np.nan,
    )


def build_old_zeroth_order_model(data_dir):
    """
    Reproduce the OLD zeroth-order effective-area model.

    The legacy build_effective_areas() function in mdp_redsox.py still uses:
      - the constant mirror-area prescription
      - Si_4um_deep_for_MDP.tsv
      - the original 32--64 Angstrom wavelength grid

    area0_lam returned by build_effective_areas() already contains dlam, so
    divide by dlam to recover A_eff in cm^2.
    """
    nwave = 1000
    wave_old = np.arange(nwave, dtype=float) * 0.001 * 32.0 + 32.0
    dlam_old = wave_old[1] - wave_old[0]
    energy_old = mdp.HC_KEV_ANG / wave_old

    (
        _area1_lam_lo,
        _area1_lam_hi,
        area0_lam_old,
        _modfactor_lo,
        _modfactor_hi,
        _geom_area,
        _detqe_filt,
    ) = mdp.build_effective_areas(data_dir, wave_old)

    aeff_old = area0_lam_old / dlam_old

    return wave_old, energy_old, aeff_old


def build_updated_zeroth_order_model(data_dir):
    """
    Build the UPDATED zeroth-order response from the energy-dependent mirror
    area and extended XLSX grating efficiencies.
    """
    (
        wave_new,
        energy_new,
        area0_lam_new,
        dlam_new,
        ea_stages,
        detqe_filt_new,
        selected_theta,
    ) = mdp.build_zeroth_order_effective_areas(data_dir)

    aeff_new = np.asarray(ea_stages["detector_qe"], dtype=float)

    return {
        "wave": np.asarray(wave_new, dtype=float),
        "energy": np.asarray(energy_new, dtype=float),
        "area0_lam": np.asarray(area0_lam_new, dtype=float),
        "dlam": np.asarray(dlam_new, dtype=float),
        "ea_stages": ea_stages,
        "detqe_filt": np.asarray(detqe_filt_new, dtype=float),
        "selected_theta": float(selected_theta),
        "aeff": aeff_new,
    }


def plot_old_vs_new_ea(energy_old, aeff_old, energy_new, aeff_new, output_path):
    """Compare old and updated zeroth-order effective areas."""
    new_on_old = interpolate_to_grid(energy_new, aeff_new, energy_old)

    mask = (
        np.isfinite(energy_old)
        & np.isfinite(aeff_old)
        & np.isfinite(new_on_old)
    )

    energy = energy_old[mask]
    old = aeff_old[mask]
    new = new_on_old[mask]

    order = np.argsort(energy)
    energy = energy[order]
    old = old[order]
    new = new[order]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(energy, old, linewidth=2.0, label="Old model")
    ax.plot(energy, new, linewidth=2.0, label="Updated model")

    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel(r"Zeroth-order effective area (cm$^2$)")
    ax.set_title("REDSoX Zeroth-Order Effective Area: Old vs. Updated Model")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    save_figure(fig, output_path)


def plot_old_over_new_ratio(energy_old, aeff_old, energy_new, aeff_new, output_path):
    """Plot A_eff,old(E) / A_eff,new(E) over the common energy range."""
    new_on_old = interpolate_to_grid(energy_new, aeff_new, energy_old)

    mask = (
        np.isfinite(energy_old)
        & np.isfinite(aeff_old)
        & np.isfinite(new_on_old)
        & (new_on_old > 0.0)
    )

    energy = energy_old[mask]
    ratio = aeff_old[mask] / new_on_old[mask]

    energy, ratio = sorted_xy(energy, ratio)

    fig, ax = plt.subplots(figsize=(9, 5.0))

    ax.plot(energy, ratio, linewidth=2.0)
    ax.axhline(1.0, linewidth=1.0, linestyle="--")

    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel(r"$A_{\rm eff,old}(E)/A_{\rm eff,new}(E)$")
    ax.set_title("Change in REDSoX Zeroth-Order Effective Area")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    save_figure(fig, output_path)


def plot_final_ea_with_polarimetry_band(energy, aeff, output_path):
    """Plot final updated zeroth-order EA and highlight 0.2--0.4 keV."""
    energy, aeff = sorted_xy(energy, aeff)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(energy, aeff, linewidth=2.0, label="Final zeroth-order EA")
    ax.axvspan(
        0.2,
        0.4,
        alpha=0.15,
        label="Polarimetry band: 0.2--0.4 keV",
    )

    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel(r"Effective area (cm$^2$)")
    ax.set_title("REDSoX Final Zeroth-Order Effective Area")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    save_figure(fig, output_path)


def get_grating_efficiency_curves(data_dir, target_theta=0.7):
    """
    Return zeroth-order, summed nonzero-order, and total grating efficiency.

    The nonzero-order curve explicitly excludes diffraction order 0.
    No weighting is applied: efficiencies are simply summed across orders.
    """
    grat_wave, grat_theta, eff_by_order = mdp.read_eff_xlsx(
        data_dir / "Si_4um_deep_30pct_dc_extended.xlsx"
    )

    iangle = int(np.argmin(np.abs(grat_theta - target_theta)))
    selected_theta = float(grat_theta[iangle])

    if 0 not in eff_by_order:
        raise KeyError("Diffraction order 0 is missing from the XLSX efficiency table.")

    eff_zero = np.asarray(eff_by_order[0][:, iangle], dtype=float)

    nonzero_curves = [
        np.asarray(efficiency[:, iangle], dtype=float)
        for diffraction_order, efficiency in eff_by_order.items()
        if diffraction_order != 0
    ]

    if not nonzero_curves:
        raise ValueError("No nonzero diffraction orders were found.")

    eff_nonzero = np.sum(nonzero_curves, axis=0)
    eff_total = eff_zero + eff_nonzero

    grat_energy = mdp.HC_KEV_ANG / np.asarray(grat_wave, dtype=float)

    return grat_energy, eff_zero, eff_nonzero, eff_total, selected_theta


def plot_grating_efficiencies(
    energy,
    eff_zero,
    eff_nonzero,
    eff_total,
    selected_theta,
    output_path,
    xlim=None,
    title_suffix="",
):
    """Plot zeroth, all nonzero, and total summed grating efficiency."""
    order = np.argsort(energy)
    energy = np.asarray(energy)[order]
    eff_zero = np.asarray(eff_zero)[order]
    eff_nonzero = np.asarray(eff_nonzero)[order]
    eff_total = np.asarray(eff_total)[order]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(energy, eff_zero, linewidth=2.0, label="Zeroth order")
    ax.plot(
        energy,
        eff_nonzero,
        linewidth=2.0,
        label="All nonzero orders",
    )
    ax.plot(
        energy,
        eff_total,
        linewidth=2.0,
        label="All orders / total grating throughput",
    )

    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Grating efficiency")
    ax.set_title(
        "CAT Grating Efficiency "
        f"(blaze angle = {selected_theta:.3f} deg){title_suffix}"
    )
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    save_figure(fig, output_path)


def build_mrk421_spectrum(energy):
    """
    Build exactly the Mrk 421 spectrum already used by mdp_redsox.py.

    N_E(E) = K E^{-Gamma} T_ISM(E)

    with
        K = 0.25
        Gamma = 2.7
        N_H = 1.45e20 cm^-2
    """
    energy = np.asarray(energy, dtype=float)

    norm = 0.25
    gamma = 2.7
    nh = 1.45e20

    transmission = mdp.ism_tb(energy, nh)
    photon_flux_E = norm * energy**(-gamma) * transmission

    return photon_flux_E, norm, gamma, nh


def plot_mrk421_source_spectrum(energy, photon_flux_E, norm, gamma, nh, output_path):
    """Plot the absorbed Mrk 421 photon spectrum N_E(E)."""
    energy, photon_flux_E = sorted_xy(energy, photon_flux_E)

    positive = photon_flux_E > 0.0

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        energy[positive],
        photon_flux_E[positive],
        linewidth=2.0,
    )

    ax.set_yscale("log")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel(
        r"$N_E(E)$ (photons cm$^{-2}$ s$^{-1}$ keV$^{-1}$)"
    )
    ax.set_title("Mrk 421 Absorbed Source Spectrum")

    equation = (
        r"$N_E(E)=K E^{-\Gamma} T_{\rm ISM}(E)$"
        "\n"
        + rf"$K={norm:g},\ \Gamma={gamma:g},\ N_H={nh:.2e}\ "
        + r"\mathrm{cm}^{-2}$"
    )

    ax.text(
        0.97,
        0.95,
        equation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
    )

    ax.grid(alpha=0.25)

    fig.tight_layout()
    save_figure(fig, output_path)


def plot_mrk421_count_rate_spectrum(energy, photon_flux_E, aeff, output_path):
    """Plot dR/dE = N_E(E) A_eff(E) for the updated zeroth-order response."""
    energy = np.asarray(energy, dtype=float)
    photon_flux_E = np.asarray(photon_flux_E, dtype=float)
    aeff = np.asarray(aeff, dtype=float)

    dR_dE = photon_flux_E * aeff

    energy, dR_dE = sorted_xy(energy, dR_dE)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(energy, dR_dE, linewidth=2.0)
    ax.axvspan(0.2, 0.4, alpha=0.15, label="Polarimetry band: 0.2--0.4 keV")

    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel(r"$dR/dE$ (count s$^{-1}$ keV$^{-1}$)")
    ax.set_title("Mrk 421 Zeroth-Order Count-Rate Spectrum")
    ax.grid(alpha=0.25)
    ax.legend()

    ax.text(
        0.97,
        0.95,
        r"$\frac{dR}{dE}=N_E(E)\,A_{\rm eff}(E)$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
    )

    fig.tight_layout()
    save_figure(fig, output_path)


def plot_stage_count_rate_bars(stage_rates, output_path):
    """Plot cumulative Mrk 421 count rate after each instrument stage."""
    stage_order = [
        "mirror",
        "mirror_mount",
        "grating_supports",
        "grating_efficiency",
        "obf",
        "detector_qe",
    ]

    labels = [
        "Mirror",
        "Mirror mount",
        "Grating supports",
        "0th-order grating",
        "OBF",
        "Detector QE",
    ]

    rates = np.asarray(
        [stage_rates[stage_name] for stage_name in stage_order],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bars = ax.bar(labels, rates)

    ax.set_ylabel(r"Mrk 421 count rate (count s$^{-1}$)")
    ax.set_title("Mrk 421 Zeroth-Order Count Rate Through the Instrument")
    ax.grid(axis="y", alpha=0.25)

    ax.tick_params(axis="x", labelrotation=20)

    initial_rate = rates[0]

    for bar, rate in zip(bars, rates):
        percent = 100.0 * rate / initial_rate if initial_rate > 0.0 else np.nan
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{rate:.2g}\n({percent:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    save_figure(fig, output_path)


def plot_effective_area_stages(
    energy,
    ea_stages,
    selected_theta,
    stage_rates,
    output_path,
):
    """Plot six cumulative zeroth-order EA stages in a 2 x 3 grid."""
    energy = np.asarray(energy, dtype=float)
    plot_order = np.argsort(energy)
    energy_plot = energy[plot_order]

    stage_labels = {
        "mirror": "1. Mirror",
        "mirror_mount": "2. Mirror mount",
        "grating_supports": "3. Grating supports",
        "grating_efficiency": "4. Zeroth-order grating efficiency",
        "obf": "5. Optical blocking filter",
        "detector_qe": "6. Detector QE",
    }

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(13, 7),
        sharex=True,
        sharey=True,
    )

    axes = axes.flatten()
    common_ymax = 1.05 * np.nanmax(ea_stages["mirror"])

    for ax, (stage_name, area) in zip(axes, ea_stages.items()):
        area = np.asarray(area, dtype=float)
        area_plot = area[plot_order]

        ax.plot(energy_plot, area_plot, linewidth=1.5)
        ax.set_title(stage_labels[stage_name], loc="left", fontsize=10)
        ax.set_ylim(0.0, common_ymax)
        ax.grid(alpha=0.25)

        rate = stage_rates[stage_name]
        annotation = (
            rf"max EA = {np.nanmax(area_plot):.3g} cm$^2$"
            "\n"
            + rf"Mrk 421 = {rate:.2g} count s$^{{-1}}$"
        )

        ax.text(
            0.97,
            0.93,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )

    fig.supxlabel("Energy (keV)", fontsize=11)
    fig.supylabel(r"Effective area (cm$^2$)", fontsize=11)

    fig.suptitle(
        "REDSoX Zeroth-Order Effective Area Throughput\n"
        f"CAT blaze angle = {selected_theta:.3f} deg",
        fontsize=13,
    )

    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.94))
    save_figure(fig, output_path)


def run_testing_figures():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"

    output_dir = base_dir / "outputs" / "presentation_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Current build_zeroth_order_effective_areas() still contains temporary
    # debug plots that write to outputs/, so make sure that directory exists.
    (base_dir / "outputs").mkdir(parents=True, exist_ok=True)

    print("\nBuilding old zeroth-order model...")
    _wave_old, energy_old, aeff_old = build_old_zeroth_order_model(data_dir)

    print("Building updated zeroth-order model...")
    updated = build_updated_zeroth_order_model(data_dir)

    energy_new = updated["energy"]
    aeff_new = updated["aeff"]
    dlam_new = updated["dlam"]
    ea_stages = updated["ea_stages"]
    selected_theta = updated["selected_theta"]

    plot_old_vs_new_ea(
        energy_old,
        aeff_old,
        energy_new,
        aeff_new,
        output_dir / "01_zeroth_order_ea_old_vs_new.png",
    )

    plot_old_over_new_ratio(
        energy_old,
        aeff_old,
        energy_new,
        aeff_new,
        output_dir / "02_zeroth_order_ea_old_over_new.png",
    )

    plot_final_ea_with_polarimetry_band(
        energy_new,
        aeff_new,
        output_dir / "03_final_zeroth_order_ea_polarimetry_band.png",
    )

    (
        grat_energy,
        eff_zero,
        eff_nonzero,
        eff_total,
        grating_theta,
    ) = get_grating_efficiency_curves(data_dir)

    plot_grating_efficiencies(
        grat_energy,
        eff_zero,
        eff_nonzero,
        eff_total,
        grating_theta,
        output_dir / "04_grating_efficiencies_all_orders.png",
    )

    plot_grating_efficiencies(
        grat_energy,
        eff_zero,
        eff_nonzero,
        eff_total,
        grating_theta,
        output_dir / "05_grating_efficiencies_zoom_1p7_1p9_keV.png",
        xlim=(1.7, 1.9),
        title_suffix=" -- 1.7--1.9 keV Zoom",
    )

    photon_flux_E, norm, gamma, nh = build_mrk421_spectrum(energy_new)

    plot_mrk421_source_spectrum(
        energy_new,
        photon_flux_E,
        norm,
        gamma,
        nh,
        output_dir / "06_mrk421_source_spectrum.png",
    )

    plot_mrk421_count_rate_spectrum(
        energy_new,
        photon_flux_E,
        aeff_new,
        output_dir / "07_mrk421_count_rate_spectrum.png",
    )

    # Convert N_E to n_lambda for the same wavelength-space count-rate
    # integration used in mdp_redsox.py:
    #
    #     n_lambda(lambda) = (E^2 / hc) N_E(E)
    #
    #     R = integral n_lambda(lambda) A_eff(lambda) d lambda
    nlam_mrk421 = (
        energy_new**2
        * photon_flux_E
        / mdp.HC_KEV_ANG
    )

    stage_rates = mdp.calculate_stage_count_rates(
        nlam=nlam_mrk421,
        dlam=dlam_new,
        ea_stages=ea_stages,
    )

    plot_stage_count_rate_bars(
        stage_rates,
        output_dir / "08_mrk421_stage_count_rates.png",
    )

    plot_effective_area_stages(
        energy_new,
        ea_stages,
        selected_theta,
        stage_rates,
        output_dir / "09_zeroth_order_effective_area_stages_mrk421.png",
    )

    print("\nMrk 421 zeroth-order count rates")
    print("--------------------------------")
    for stage_name, rate in stage_rates.items():
        print(f"{stage_name:<22} {rate:.6e} count/s")

    print(
        "\nFinal Mrk 421 zeroth-order count rate "
        f"(wavelength integration) = "
        f"{stage_rates['detector_qe']:.6e} count/s"
    )

    print(f"\nAll presentation figures are in:\n  {output_dir}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate REDSoX diagnostic and presentation figures."
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    subparsers.add_parser(
        "testing",
        help="Generate the complete testing/presentation figure set.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "testing":
        run_testing_figures()


if __name__ == "__main__":
    main()
