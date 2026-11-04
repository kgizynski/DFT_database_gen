from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import connect_db, load_config, project_root


HARTREE_TO_KJ_MOL = 2625.4996394799
R_KJ_MOL_K = 0.00831446261815324


def fwhm_to_sigma(fwhm: float) -> float:
    return fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def gaussian_kernel_grid(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def laser_wavenumber_cm1(laser_nm: float) -> float:
    return 1e7 / laser_nm


def activity_to_chemcraft_intensity_cm2_sr(
    freq_cm1: float,
    activity_A4_per_amu: float,
    laser_nm: float,
    temperature_K: float,
) -> float:
    if freq_cm1 <= 0.0:
        return 0.0

    nu0 = laser_wavenumber_cm1(laser_nm)
    nu = float(freq_cm1)
    nu_s = nu0 - nu
    if nu_s <= 0.0:
        return 0.0

    amu_kg = 1.66053878e-27
    h = 6.62606896e-34
    c = 2.99792458e10
    k = 1.3806504e-23

    activity_cm4_per_kg = float(activity_A4_per_amu) * 1e-32 / amu_kg
    bose = 1.0 / (1.0 - math.exp(-(h * c * nu) / (k * float(temperature_K))))
    pref = ((2.0 * math.pi) ** 4) / 45.0
    geom = h / (8.0 * (math.pi ** 2) * c * nu)
    return pref * (nu_s ** 4) * geom * bose * activity_cm4_per_kg


def broaden_spectrum(
    peaks_int: List[Tuple[float, float]],
    wn_min: float,
    wn_max: float,
    wn_step: float,
    fwhm: float,
    scale_factor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.arange(wn_min, wn_max + 0.5 * wn_step, wn_step, dtype=float)
    y = np.zeros_like(x)

    sigma = fwhm_to_sigma(fwhm)
    for freq, intensity in peaks_int:
        scaled_freq = freq * scale_factor
        if intensity == 0.0:
            continue
        if scaled_freq < wn_min - 5 * fwhm or scaled_freq > wn_max + 5 * fwhm:
            continue
        y += intensity * gaussian_kernel_grid(x, scaled_freq, sigma)
    return x, y


def parse_orca_raman_activity(out_path: Path) -> List[Tuple[float, float]]:
    lines = out_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    header_idx = None
    for idx, line in enumerate(lines):
        if "RAMAN SPECTRUM" in line.upper():
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError(f"Missing RAMAN SPECTRUM section: {out_path.name}")

    row_re = re.compile(r"^\s*(\d+)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)")
    peaks: List[Tuple[float, float]] = []
    in_table = False

    for line in lines[header_idx + 1:]:
        upper = line.upper()
        if ("MODE" in upper) and ("FREQ" in upper) and ("ACTIV" in upper):
            in_table = True
            continue

        if set(line.strip()) <= {"-"} and line.strip():
            continue

        if in_table and (
            "THE FIRST FREQUENCY" in upper
            or "TOTAL NUMBER OF VIBRATIONS" in upper
            or "ORCA TERMINATED" in upper
            or "TIMINGS" in upper
        ):
            break

        if not in_table:
            continue

        match = row_re.match(line)
        if not match:
            continue

        freq = float(match.group(2))
        activity = float(match.group(3))
        if freq > 0.0:
            peaks.append((freq, activity))

    if not peaks:
        raise ValueError(f"No Raman peaks parsed from {out_path.name}")
    return peaks


def parse_gibbs_free_energy_hartree(out_path: Path) -> float:
    text = out_path.read_text(encoding="utf-8", errors="ignore")
    patterns = [
        r"Final Gibbs free energy\s*\.\.\.\s*([-+]?\d+\.\d+)\s*Eh",
        r"Final Gibbs free energy\s*:\s*([-+]?\d+\.\d+)\s*Eh",
        r"Gibbs free energy\s*\.\.\.\s*([-+]?\d+\.\d+)\s*Eh",
        r"Gibbs free energy\s*:\s*([-+]?\d+\.\d+)\s*Eh",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return float(matches[-1])
    raise ValueError(f"Gibbs free energy not found in {out_path.name}")


def already_done(out_path: Path) -> bool:
    if not out_path.exists():
        return False
    text = out_path.read_text(encoding="utf-8", errors="ignore")
    return "ORCA TERMINATED NORMALLY" in text


def extract_conformer_prefix(stem: str) -> str:
    marker = "_conf_"
    pos = stem.find(marker)
    if pos == -1:
        return stem
    return stem[:pos]


def extract_conformer_suffix(stem: str) -> str:
    marker = "_conf_"
    pos = stem.find(marker)
    if pos == -1:
        return stem
    return stem[pos + 1:]


def fetch_molecule_name(inchikey: str) -> str:
    cfg = load_config()
    db_path = Path(cfg["paths"]["db_path"]).resolve()
    con = connect_db(db_path)
    try:
        row = con.execute(
            "SELECT name FROM molecule WHERE inchikey = ?",
            (inchikey,),
        ).fetchone()
    finally:
        con.close()

    if row and row[0]:
        return str(row[0])
    return inchikey


def boltzmann_weights(gibbs_hartree: List[float], temperature_k: float) -> List[float]:
    min_g = min(gibbs_hartree)
    exponents = []
    for g in gibbs_hartree:
        delta_kj_mol = (g - min_g) * HARTREE_TO_KJ_MOL
        exponents.append(math.exp(-delta_kj_mol / (R_KJ_MOL_K * temperature_k)))
    norm = sum(exponents)
    return [value / norm for value in exponents]


def write_weights_csv(
    path: Path,
    labels: List[str],
    gibbs_hartree: List[float],
    weights: List[float],
) -> None:
    min_g = min(gibbs_hartree)
    lines = ["conformer,gibbs_hartree,delta_g_kj_mol,boltzmann_weight"]
    for label, g, w in zip(labels, gibbs_hartree, weights):
        delta_kj_mol = (g - min_g) * HARTREE_TO_KJ_MOL
        lines.append(f"{label},{g:.12f},{delta_kj_mol:.8f},{w:.10f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_spectrum_csv(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    lines = ["wn_cm-1,intensity_cm2_sr,intensity_1e-38_cm2_sr"]
    for xi, yi in zip(x, y):
        lines.append(f"{xi:.6f},{yi:.10g},{(yi * 1e38):.10g}")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_overlay_plot(
    path: Path,
    x: np.ndarray,
    conformer_curves: List[np.ndarray],
    labels: List[str],
    weights: List[float],
    weighted_y: np.ndarray,
    laser_nm: float,
    temperature_k: float,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)

    for label, weight, curve in zip(labels, weights, conformer_curves):
        ax.plot(
            x,
            curve * 1e38,
            linewidth=0.9,
            alpha=0.45,
            label=f"{label} (w={weight:.3f})",
        )

    ax.plot(
        x,
        weighted_y * 1e38,
        color="black",
        linewidth=2.4,
        label="Weighted sum",
        zorder=10,
    )

    ax.set_xlabel("Raman shift (cm^-1)")
    ax.set_ylabel("Raman intensity (x10^38 cm^2/sr)")
    ax.set_title(f"Conformer Raman spectra and weighted sum | {int(round(laser_nm))} nm | T={temperature_k:.2f} K")
    ax.grid(alpha=0.25)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=7, ncols=2, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_conformers_plot(
    path: Path,
    x: np.ndarray,
    conformer_curves: List[np.ndarray],
    labels: List[str],
    weights: List[float],
    laser_nm: float,
    temperature_k: float,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    max_intensity = max(float(np.max(curve)) for curve in conformer_curves) if conformer_curves else 0.0
    offset_step = max_intensity * 1.2 if max_intensity > 0.0 else 1.0

    for idx, (label, weight, curve) in enumerate(zip(labels, weights, conformer_curves)):
        y_offset = idx * offset_step
        ax.plot(
            x,
            (curve + y_offset) * 1e38,
            linewidth=1.0,
            alpha=0.6,
            label=f"{label} (w={weight:.3f}, offset={idx})",
        )

    ax.set_xlabel("Raman shift (cm^-1)")
    ax.set_ylabel("Raman intensity (x10^38 cm^2/sr)")
    ax.set_title(f"Conformer Raman spectra (stacked in Y) | {int(round(laser_nm))} nm | T={temperature_k:.2f} K")
    ax.grid(alpha=0.25)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=7, ncols=2, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    post_cfg = cfg.get("postprocess", {})

    wn_min = 300.0
    wn_max = 1800.0
    wn_step = float(post_cfg.get("wn_step", 1.0))
    fwhm = float(post_cfg.get("fwhm_cm-1", 15.0))
    temperature_k = float(post_cfg.get("temperature_K", 298.15))
    scale_orca = float(post_cfg.get("scale_orca", 1.0))

    cfg_lasers = post_cfg.get("laser_nms") or post_cfg.get("laser_nm")
    if isinstance(cfg_lasers, (list, tuple)):
        lasers_nm = sorted({float(value) for value in cfg_lasers})
    elif cfg_lasers is not None:
        lasers_nm = [float(cfg_lasers)]
    else:
        lasers_nm = [633.0]

    input_dir = project_root() / "testConformers"
    out_paths = sorted(path for path in input_dir.glob("*.out") if already_done(path))
    if not out_paths:
        raise RuntimeError(f"No finished ORCA .out files found in {input_dir}")

    labels = [path.stem for path in out_paths]
    prefixes = {extract_conformer_prefix(label) for label in labels}
    base_name = prefixes.pop() if len(prefixes) == 1 else "testConformers"
    molecule_name = fetch_molecule_name(base_name) if base_name != "testConformers" else base_name
    legend_labels = [f"{molecule_name} {extract_conformer_suffix(label)}" for label in labels]

    gibbs_values = [parse_gibbs_free_energy_hartree(path) for path in out_paths]
    weights = boltzmann_weights(gibbs_values, temperature_k)

    weights_path = input_dir / f"{base_name}_boltzmann_weights.csv"
    write_weights_csv(weights_path, labels, gibbs_values, weights)

    for laser_nm in lasers_nm:
        weighted_y = None
        x_ref = None
        conformer_curves = []

        for out_path, weight in zip(out_paths, weights):
            peaks_activity = parse_orca_raman_activity(out_path)
            peaks_intensity = [
                (
                    freq,
                    activity_to_chemcraft_intensity_cm2_sr(freq, activity, laser_nm, temperature_k),
                )
                for freq, activity in peaks_activity
            ]

            x, y = broaden_spectrum(
                peaks_int=peaks_intensity,
                wn_min=wn_min,
                wn_max=wn_max,
                wn_step=wn_step,
                fwhm=fwhm,
                scale_factor=scale_orca,
            )

            if weighted_y is None:
                x_ref = x
                weighted_y = np.zeros_like(y)
            conformer_curves.append(y.copy())
            weighted_y += weight * y

        spectrum_path = input_dir / f"{base_name}_weighted_raman_spectrum_{int(round(laser_nm))}nm.csv"
        plot_path = input_dir / f"{base_name}_weighted_raman_overlay_{int(round(laser_nm))}nm.png"
        conformers_plot_path = input_dir / f"{base_name}_conformers_raman_{int(round(laser_nm))}nm.png"
        write_spectrum_csv(spectrum_path, x_ref, weighted_y)
        write_overlay_plot(
            path=plot_path,
            x=x_ref,
            conformer_curves=conformer_curves,
            labels=legend_labels,
            weights=weights,
            weighted_y=weighted_y,
            laser_nm=laser_nm,
            temperature_k=temperature_k,
        )
        write_conformers_plot(
            path=conformers_plot_path,
            x=x_ref,
            conformer_curves=conformer_curves,
            labels=legend_labels,
            weights=weights,
            laser_nm=laser_nm,
            temperature_k=temperature_k,
        )
        print(f"[TEST] Wrote weighted spectrum: {spectrum_path}")
        print(f"[TEST] Wrote overlay plot: {plot_path}")
        print(f"[TEST] Wrote conformers plot: {conformers_plot_path}")

    print(f"[TEST] Wrote Boltzmann weights: {weights_path}")
    print(f"[TEST] Used {len(out_paths)} conformers at T={temperature_k:.2f} K")


if __name__ == "__main__":
    main()
