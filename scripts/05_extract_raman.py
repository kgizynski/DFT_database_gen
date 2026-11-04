import argparse
import math
import re
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    from scripts.common import load_config, get_paths, connect_db
except ModuleNotFoundError:
    # Allow running as: python scripts/05_extract_raman.py
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts.common import load_config, get_paths, connect_db


# -----------------------------
# DB helpers
# -----------------------------
def ensure_calc_columns(con: sqlite3.Connection) -> None:
    cols = {row[1] for row in con.execute("PRAGMA table_info(calculation)").fetchall()}
    if "raman_intensity_peaks_path" not in cols:
        con.execute("ALTER TABLE calculation ADD COLUMN raman_intensity_peaks_path TEXT")
    if "raman_intensity_spectrum_path" not in cols:
        con.execute("ALTER TABLE calculation ADD COLUMN raman_intensity_spectrum_path TEXT")
    if "raman_intensity_plot_path" not in cols:
        con.execute("ALTER TABLE calculation ADD COLUMN raman_intensity_plot_path TEXT")


def ensure_conformer_energy_columns(con: sqlite3.Connection) -> None:
    cols = {row[1] for row in con.execute("PRAGMA table_info(conformer)").fetchall()}
    if "mmff_energy" not in cols:
        con.execute("ALTER TABLE conformer ADD COLUMN mmff_energy REAL")
    if "gibbs_hartree" not in cols:
        con.execute("ALTER TABLE conformer ADD COLUMN gibbs_hartree REAL")


# -----------------------------
# Spectrum math
# -----------------------------
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
    """
    Chemcraft Raman intensity (Stokes) in cm^2/sr.

    Implements Chemcraft manual formula:

      R_i = ((2π)^4 / 45) * (ν0 - ν)^4 * [ h / (8π^2 c ν) ] *
            [ 1 / (1 - exp(-h c ν / (k T))) ] * S_i

    where:
      ν, ν0 in cm^-1
      c in cm/s
      h in J*s
      k in J/K
      S_i is Raman activity converted from Å^4/amu -> cm^4/kg

    Returns:
      R_i in cm^2/sr (Chemcraft y-axis is often shown as ×10^-38 cm^2/sr).
    """
    if freq_cm1 <= 0.0:
        return 0.0

    # Laser wavenumber ν0 [cm^-1]
    nu0 = laser_wavenumber_cm1(laser_nm)
    nu = float(freq_cm1)
    nu_s = nu0 - nu
    if nu_s <= 0.0:
        return 0.0

    # Fundamental constants (as in Chemcraft manual)
    amu_kg = 1.66053878e-27  # kg
    h = 6.62606896e-34       # J*s
    c = 2.99792458e10        # cm/s
    k = 1.3806504e-23        # J/K

    # Convert activity: Å^4/amu -> cm^4/kg
    # 1 Å = 1e-8 cm => Å^4 = 1e-32 cm^4
    S = float(activity_A4_per_amu) * 1e-32 / amu_kg  # cm^4/kg

    # Bose factor: 1 / (1 - exp(-h c ν / (k T)))
    expo = -(h * c * nu) / (k * float(temperature_K))
    bose = 1.0 / (1.0 - math.exp(expo))

    pref = ((2.0 * math.pi) ** 4) / 45.0
    geom = h / (8.0 * (math.pi ** 2) * c * nu)

    R = pref * (nu_s ** 4) * geom * bose * S
    return R


def broaden_spectrum(
    peaks_int: List[Tuple[float, float]],
    wn_min: float,
    wn_max: float,
    wn_step: float,
    fwhm: float,
    scale_factor: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.arange(wn_min, wn_max + 0.5 * wn_step, wn_step, dtype=float)
    y = np.zeros_like(x)

    sigma = fwhm_to_sigma(fwhm)
    for freq, inten in peaks_int:
        f = freq * scale_factor
        if inten == 0.0:
            continue
        if f < wn_min - 5 * fwhm or f > wn_max + 5 * fwhm:
            continue
        y += inten * gaussian_kernel_grid(x, f, sigma)

    return x, y


def write_peaks_csv(path: Path, peaks: List[Tuple[float, float]]) -> None:
    """
    Writes peaks as Chemcraft-style absolute intensities:
      - raman_intensity_cm2_sr: cm^2/sr
      - raman_intensity_1e-38_cm2_sr: the same scaled by 1e38 (matches Chemcraft axis label style)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["freq_cm-1,raman_intensity_cm2_sr,raman_intensity_1e-38_cm2_sr"]
    for f, inten in peaks:
        lines.append(f"{f:.6f},{inten:.10g},{(inten * 1e38):.10g}")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_spectrum_csv(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    """
    Writes broadened spectrum as Chemcraft-style absolute intensities:
      - intensity_cm2_sr: cm^2/sr
      - intensity_1e-38_cm2_sr: scaled by 1e38
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["wn_cm-1,intensity_cm2_sr,intensity_1e-38_cm2_sr"]
    for xi, yi in zip(x, y):
        lines.append(f"{xi:.6f},{yi:.10g},{(yi * 1e38):.10g}")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_spectrum_png(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    peaks: List[Tuple[float, float]],
    laser_nm: float,
    temperature_K: float,
    title: str,
    show_sticks: bool = True,
    curve_color: str = "C0",
    stick_color: str = "black",
    curve_width: float = 1.4,
    stick_width: float = 1.0,
    use_1e38: bool = True,
) -> None:
    """
    Writes Raman intensity plot (optionally with stick peaks) using default Matplotlib style/colors.
    """
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    y_plot = y * (1e38 if use_1e38 else 1.0)
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=160)
    ax.plot(x, y_plot, color=curve_color, linewidth=curve_width)

    if show_sticks and peaks:
        px = np.array([p[0] for p in peaks], dtype=float)
        py = np.array([p[1] for p in peaks], dtype=float)
        py_plot = py * (1e38 if use_1e38 else 1.0)
        ax.vlines(px, 0.0, py_plot, color=stick_color, linewidth=stick_width)

    ax.set_xlabel("Raman shift (cm^-1)")
    if use_1e38:
        ax.set_ylabel("Raman intensity (x10^38 cm^2/sr)")
    else:
        ax.set_ylabel("Raman intensity (cm^2/sr)")
    ax.set_title(f"{title} | {int(round(laser_nm))} nm | T={temperature_K:.2f} K")
    ax.grid(alpha=0.25)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# -----------------------------
# Engine inference
# -----------------------------
def infer_engine(engine_db: Optional[str], output_path: Path) -> str:
    if engine_db:
        e = str(engine_db).lower()
        if "orca" in e:
            return "orca"
        if "gauss" in e:
            return "gaussian"

    ext = output_path.suffix.lower()
    if ext == ".out":
        return "orca"
    if ext == ".log":
        return "gaussian"

    # fallback by file header sniff
    head = output_path.read_text(errors="ignore")[:5000]
    if "ORCA" in head:
        return "orca"
    if "Entering Gaussian System" in head or "Gaussian" in head:
        return "gaussian"

    raise ValueError(f"Cannot infer engine for output: {output_path}")


# -----------------------------
# Parsers -> return (freq, activity)
# -----------------------------
def parse_gaussian_raman_activity(log_path: Path) -> List[Tuple[float, float]]:
    text = log_path.read_text(errors="ignore")

    freq_re = re.compile(r"Frequencies --\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)")
    ract_re = re.compile(r"Raman Activ --\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)")

    freqs_blocks = freq_re.findall(text)
    ract_blocks = ract_re.findall(text)

    if not freqs_blocks:
        raise ValueError("Gaussian: no 'Frequencies --' blocks found.")
    if not ract_blocks:
        raise ValueError("Gaussian: no 'Raman Activ --' blocks found. Did you run freq=raman?")

    n = min(len(freqs_blocks), len(ract_blocks))
    peaks: List[Tuple[float, float]] = []
    for i in range(n):
        f1, f2, f3 = map(float, freqs_blocks[i])
        a1, a2, a3 = map(float, ract_blocks[i])
        for f, a in [(f1, a1), (f2, a2), (f3, a3)]:
            if f > 0:
                peaks.append((f, a))

    if not peaks:
        raise ValueError("Gaussian: parsed blocks, but no positive frequencies with Raman activity.")
    return peaks


def parse_orca_raman_activity(out_path: Path) -> List[Tuple[float, float]]:
    lines = out_path.read_text(errors="ignore").splitlines()

    # find Raman header
    header_idx = None
    for i, ln in enumerate(lines):
        if "RAMAN SPECTRUM" in ln.upper():
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("ORCA: 'RAMAN SPECTRUM' section not found.")

    # parse table rows: "  6:  47.75  0.563786  0.747907"
    row_re = re.compile(
        r"^\s*(\d+)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)"
    )

    peaks: List[Tuple[float, float]] = []
    in_table = False

    for ln in lines[header_idx + 1 :]:
        u = ln.upper()

        if ("MODE" in u) and ("FREQ" in u) and ("ACTIV" in u):
            in_table = True
            continue

        if set(ln.strip()) <= set("-") and ln.strip() != "":
            continue

        if in_table and (
            "THE FIRST FREQUENCY" in u
            or "TOTAL NUMBER OF VIBRATIONS" in u
            or "ORCA TERMINATED" in u
            or "TIMINGS" in u
        ):
            break

        if not in_table:
            continue

        m = row_re.match(ln)
        if not m:
            continue

        freq = float(m.group(2))
        act = float(m.group(3))
        if freq > 0:
            peaks.append((freq, act))

    if not peaks:
        raise ValueError("ORCA: Raman section found but no activity peaks parsed.")
    return peaks


def parse_orca_gibbs_hartree(out_path: Path) -> Optional[float]:
    text = out_path.read_text(errors="ignore")
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
    return None


def parse_gaussian_gibbs_hartree(log_path: Path) -> Optional[float]:
    text = log_path.read_text(errors="ignore")
    patterns = [
        r"Sum of electronic and thermal Free Energies=\s*([-+]?\d+\.\d+)",
        r"Thermal Free Energy\s*=\s*([-+]?\d+\.\d+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return float(matches[-1])
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Raman intensities from Gaussian/ORCA outputs and write spectra CSV files."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional single-file mode: path to Gaussian .log or ORCA .out.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory for single-file mode (default: same directory as --input).",
    )
    parser.add_argument(
        "--laser-cm1",
        type=float,
        default=None,
        help="Incident laser wavenumber in cm^-1 (overrides --laser-nm).",
    )
    parser.add_argument(
        "--laser-nm",
        type=float,
        default=None,
        help="Incident laser wavelength in nm.",
    )
    parser.add_argument("--temperature-k", type=float, default=None, help="Temperature in Kelvin.")
    parser.add_argument("--wn-min", type=float, default=None, help="Minimum Raman shift for broadened spectrum.")
    parser.add_argument("--wn-max", type=float, default=None, help="Maximum Raman shift for broadened spectrum.")
    parser.add_argument("--wn-step", type=float, default=None, help="Grid step for broadened spectrum.")
    parser.add_argument("--fwhm", type=float, default=None, help="Gaussian broadening FWHM in cm^-1.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate PNG spectrum plot in single-file mode.",
    )
    return parser.parse_args()


def run_single_file_mode(args: argparse.Namespace, cfg: dict) -> None:
    post_cfg = cfg.get("postprocess", {})
    wn_min = float(args.wn_min if args.wn_min is not None else post_cfg.get("wn_min", 100.0))
    wn_max = float(args.wn_max if args.wn_max is not None else post_cfg.get("wn_max", 3500.0))
    wn_step = float(args.wn_step if args.wn_step is not None else post_cfg.get("wn_step", 1.0))
    fwhm = float(args.fwhm if args.fwhm is not None else post_cfg.get("fwhm_cm-1", 15.0))
    temperature_K = float(
        args.temperature_k if args.temperature_k is not None else post_cfg.get("temperature_K", 298.15)
    )

    out_path = Path(args.input)
    if not out_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {out_path}")

    engine = infer_engine(None, out_path)
    scale_orca = float(post_cfg.get("scale_orca", 1.0))
    scale_gaussian = float(post_cfg.get("scale_gaussian", 1.0))
    scale_by_engine = scale_gaussian if engine == "gaussian" else scale_orca

    if args.laser_cm1 is not None:
        laser_nm = 1e7 / float(args.laser_cm1)
    elif args.laser_nm is not None:
        laser_nm = float(args.laser_nm)
    else:
        laser_nm = float(post_cfg.get("laser_nm", 633.0))

    if engine == "gaussian":
        peaks_act = parse_gaussian_raman_activity(out_path)
    else:
        peaks_act = parse_orca_raman_activity(out_path)

    peaks_int = [
        (f, activity_to_chemcraft_intensity_cm2_sr(f, a, laser_nm, temperature_K))
        for f, a in peaks_act
    ]

    uniq = {}
    for f, inten in peaks_int:
        key = round(f, 4)
        uniq[key] = max(inten, uniq.get(key, -1e99))
    peaks_int_sorted = sorted([(float(k), v) for k, v in uniq.items()], key=lambda t: t[0])

    x, y = broaden_spectrum(
        peaks_int=peaks_int_sorted,
        wn_min=wn_min,
        wn_max=wn_max,
        wn_step=wn_step,
        fwhm=fwhm,
        scale_factor=scale_by_engine,
    )

    out_dir = Path(args.output_dir) if args.output_dir else out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_path.name
    peaks_csv = out_dir / f"{stem}_raman_intensity_peaks_{int(round(laser_nm))}nm.csv"
    spec_csv = out_dir / f"{stem}_raman_intensity_spectrum_{int(round(laser_nm))}nm.csv"
    plot_png = out_dir / f"{stem}_raman_intensity_plot_{int(round(laser_nm))}nm.png"

    write_peaks_csv(peaks_csv, peaks_int_sorted)
    write_spectrum_csv(spec_csv, x, y)
    print(f"[05] OK(single): peaks -> {peaks_csv}")
    print(f"[05] OK(single): spectrum -> {spec_csv}")

    if args.plot:
        write_spectrum_png(
            path=plot_png,
            x=x,
            y=y,
            peaks=peaks_int_sorted,
            laser_nm=laser_nm,
            temperature_K=temperature_K,
            title=f"{out_path.stem} ({engine})",
            show_sticks=True,
            curve_color="C0",
            stick_color="black",
            use_1e38=True,
        )
        print(f"[05] OK(single): plot -> {plot_png}")


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    cfg = load_config()

    if args.input:
        run_single_file_mode(args, cfg)
        return

    root, work_dir, db_path = get_paths(cfg)
    con = connect_db(db_path)

    ensure_calc_columns(con)
    ensure_conformer_energy_columns(con)

    post_cfg = cfg.get("postprocess", {})
    wn_min = float(post_cfg.get("wn_min", 100.0))
    wn_max = float(post_cfg.get("wn_max", 3500.0))
    wn_step = float(post_cfg.get("wn_step", 1.0))
    fwhm = float(post_cfg.get("fwhm_cm-1", 15.0))

    temperature_K = float(post_cfg.get("temperature_K", 298.15))

    scale_orca = float(post_cfg.get("scale_orca", 1.0))
    scale_gaussian = float(post_cfg.get("scale_gaussian", 1.0))
    plot_png = bool(post_cfg.get("plot_png", False))
    plot_sticks = bool(post_cfg.get("plot_sticks", True))
    plot_curve_color = str(post_cfg.get("plot_curve_color", "C0"))
    plot_stick_color = str(post_cfg.get("plot_stick_color", "black"))
    plot_curve_width = float(post_cfg.get("plot_curve_width", 1.4))
    plot_stick_width = float(post_cfg.get("plot_stick_width", 1.0))
    plot_use_1e38 = bool(post_cfg.get("plot_use_1e38", True))

    # collect configured lasers (single value or list)
    cfg_lasers = post_cfg.get("laser_nms") or post_cfg.get("laser_nm")
    if isinstance(cfg_lasers, (list, tuple)):
        lasers = sorted({float(x) for x in cfg_lasers})
    else:
        lasers = [float(cfg_lasers)] if cfg_lasers is not None else [633.0]

    cols = {row[1] for row in con.execute("PRAGMA table_info(calculation)").fetchall()}
    has_engine = "engine" in cols

    if has_engine:
        rows = con.execute(
            "SELECT calc_id, engine, output_path FROM calculation WHERE status='DONE'"
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT calc_id, NULL as engine, output_path FROM calculation WHERE status='DONE'"
        ).fetchall()

    ok = 0
    fail = 0

    for calc_id, engine_db, outp in rows:
        out_path = Path(outp)
        if not out_path.exists():
            print(f"[05] MISSING OUTPUT: {calc_id} -> {out_path}")
            fail += 1
            continue

        try:
            engine = infer_engine(engine_db, out_path)
            gibbs_hartree = None

            # Parse activity peaks once per output
            if engine == "gaussian":
                peaks_act = parse_gaussian_raman_activity(out_path)
                scale_by_engine = scale_gaussian
                gibbs_hartree = parse_gaussian_gibbs_hartree(out_path)
            else:
                peaks_act = parse_orca_raman_activity(out_path)
                scale_by_engine = scale_orca
                gibbs_hartree = parse_orca_gibbs_hartree(out_path)

            # For DB: collect produced paths
            produced_peaks = []
            produced_specs = []
            produced_plots = []

            # Process for each laser wavelength
            for laser_nm in lasers:
                try:
                    peaks_int = [
                        (f, activity_to_chemcraft_intensity_cm2_sr(f, a, laser_nm, temperature_K))
                        for f, a in peaks_act
                    ]

                    # Deduplicate by frequency (keep max intensity)
                    uniq = {}
                    for f, inten in peaks_int:
                        key = round(f, 4)
                        uniq[key] = max(inten, uniq.get(key, -1e99))
                    peaks_int_sorted = sorted([(float(k), v) for k, v in uniq.items()], key=lambda t: t[0])

                    peaks_path = out_path.with_suffix(out_path.suffix + f"_raman_intensity_peaks_{int(laser_nm)}nm.csv")
                    spec_path = out_path.with_suffix(out_path.suffix + f"_raman_intensity_spectrum_{int(laser_nm)}nm.csv")
                    plot_path = out_path.with_suffix(out_path.suffix + f"_raman_intensity_plot_{int(laser_nm)}nm.png")

                    write_peaks_csv(peaks_path, peaks_int_sorted)

                    x, y = broaden_spectrum(
                        peaks_int=peaks_int_sorted,
                        wn_min=wn_min,
                        wn_max=wn_max,
                        wn_step=wn_step,
                        fwhm=fwhm,
                        scale_factor=scale_by_engine,
                    )
                    write_spectrum_csv(spec_path, x, y)

                    produced_peaks.append(str(peaks_path))
                    produced_specs.append(str(spec_path))

                    if plot_png:
                        try:
                            write_spectrum_png(
                                path=plot_path,
                                x=x,
                                y=y,
                                peaks=peaks_int_sorted,
                                laser_nm=laser_nm,
                                temperature_K=temperature_K,
                                title=f"{calc_id} ({engine})",
                                show_sticks=plot_sticks,
                                curve_color=plot_curve_color,
                                stick_color=plot_stick_color,
                                curve_width=plot_curve_width,
                                stick_width=plot_stick_width,
                                use_1e38=plot_use_1e38,
                            )
                            produced_plots.append(str(plot_path))
                            print(
                                f"[05] OK: {calc_id} ({engine}, {int(laser_nm)}nm) -> "
                                f"{peaks_path.name}, {spec_path.name}, {plot_path.name}"
                            )
                        except Exception as e_plot:
                            print(f"[05] WARN (plot {int(laser_nm)}nm): {calc_id} -> {e_plot}")
                    else:
                        print(f"[05] OK: {calc_id} ({engine}, {int(laser_nm)}nm) -> {peaks_path.name}, {spec_path.name}")

                except Exception as e_laser:
                    print(f"[05] FAIL (laser {int(laser_nm)}nm): {calc_id} -> {e_laser}")
                    # continue to next laser without aborting the calculation

            peaks_db_val = ";".join(produced_peaks) if produced_peaks else None
            specs_db_val = ";".join(produced_specs) if produced_specs else None
            plots_db_val = ";".join(produced_plots) if produced_plots else None

            if produced_peaks and produced_specs:
                con.execute(
                    "UPDATE calculation SET raman_intensity_peaks_path=?, raman_intensity_spectrum_path=?, raman_intensity_plot_path=? WHERE calc_id=?",
                    (peaks_db_val, specs_db_val, plots_db_val, calc_id),
                )
                if gibbs_hartree is not None:
                    con.execute(
                        """
                        UPDATE conformer
                        SET gibbs_hartree=?
                        WHERE conformer_id=(SELECT conformer_id FROM calculation WHERE calc_id=?)
                        """,
                        (gibbs_hartree, calc_id),
                    )
                con.commit()
                ok += 1
            else:
                con.execute(
                    "UPDATE calculation SET raman_intensity_peaks_path=NULL, raman_intensity_spectrum_path=NULL, raman_intensity_plot_path=NULL WHERE calc_id=?",
                    (calc_id,),
                )
                con.commit()
                print(f"[05] FAIL: {calc_id} -> no Raman outputs generated")
                fail += 1

        except Exception as e:
            print(f"[05] FAIL: {calc_id} -> {e}")
            fail += 1

    con.close()
    print(f"[05] DONE. ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
