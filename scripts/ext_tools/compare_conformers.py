# 06_plot_conformers.py
import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt

from scripts.common import load_config, get_paths, connect_db


def resolve_inchikey(con: sqlite3.Connection, *, inchikey: Optional[str], name: Optional[str]) -> str:
    if inchikey:
        row = con.execute("SELECT inchikey FROM molecule WHERE inchikey = ?", (inchikey,)).fetchone()
        if row:
            return row[0]
        raise ValueError(f"inchikey not found in DB: {inchikey}")

    if name:
        row = con.execute("SELECT inchikey FROM molecule WHERE name = ?", (name,)).fetchone()
        if row:
            return row[0]
        # fallback: LIKE match
        rows = con.execute("SELECT inchikey, name FROM molecule WHERE name LIKE ?", (f"%{name}%",)).fetchall()
        if len(rows) == 1:
            return rows[0][0]
        if len(rows) > 1:
            opts = ", ".join([f"{ik}({nm})" for ik, nm in rows[:10]])
            raise ValueError(f"Ambiguous name. Matches (up to 10): {opts}")
        raise ValueError(f"name not found in DB: {name}")

    raise ValueError("Provide either inchikey or name.")


def has_column(con: sqlite3.Connection, table: str, col: str) -> bool:
    cols = {r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()}
    return col in cols


def _pick_field(names, candidates):
    """Return the first matching field name from names, using exact or 'normalized' match."""
    if not names:
        return None

    # exact match first
    for c in candidates:
        if c in names:
            return c

    # normalized match: lowercase, remove non-alnum
    def norm(s: str) -> str:
        return "".join(ch.lower() for ch in s if ch.isalnum())

    norm_map = {norm(n): n for n in names}
    for c in candidates:
        key = norm(c)
        if key in norm_map:
            return norm_map[key]

    return None


def read_spectrum_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust CSV reader for spectra exported by your pipeline.
    Handles sanitized headers (e.g., 'wn_cm-1' -> 'wn_cm_1') and alternative column names.
    """
    data = np.genfromtxt(
        str(csv_path),
        delimiter=",",
        names=True,
        dtype=float,
        encoding="utf-8",
        invalid_raise=False,
    )

    names = list(data.dtype.names) if data.dtype.names else []
    if not names:
        raise ValueError(f"CSV has no header / no named columns: {csv_path}")

    # Common variants for x-axis
    x_candidates = [
        "wn_cm-1", "wn_cm_1", "wavenumber", "wavenumber_cm-1", "wavenumber_cm_1",
        "raman_shift", "raman_shift_cm-1", "raman_shift_cm_1",
        "freq_cm-1", "freq_cm_1", "frequency", "frequency_cm-1", "frequency_cm_1",
        "x",
    ]

    # Common variants for y-axis
    y_candidates = [
        "intensity_cm2_sr", "intensity_cm2sr", "raman_intensity_cm2_sr",
        "intensity", "y",
        "intensity_1e-38_cm2_sr", "intensity_1e38_cm2_sr",
        "raman_intensity_1e-38_cm2_sr", "raman_intensity_1e38_cm2_sr",
    ]

    x_field = _pick_field(names, x_candidates)
    y_field = _pick_field(names, y_candidates)

    if x_field is None or y_field is None:
        raise ValueError(
            "Could not identify spectrum columns.\n"
            f"File: {csv_path}\n"
            f"Detected columns: {names}\n"
            "Expected something like wavenumber (wn_cm-1 / wn_cm_1 / raman_shift / freq_cm-1) "
            "and intensity (intensity_cm2_sr / intensity_1e-38_cm2_sr / intensity)."
        )

    x = data[x_field]
    y = data[y_field]

    # If the only intensity available is scaled 1e-38, convert it back to cm^2/sr
    y_norm = "".join(ch.lower() for ch in y_field if ch.isalnum())
    if "1e38" in y_norm or "1e038" in y_norm:
        y = y / 1e38

    return x, y


# python
def main():
    # -------------------------
    # User settings (edit here)
    # -------------------------
    MOLECULE_INCHIKEY = None       # e.g. "XXXXXXXXXXXXXX-YYYYYYYYYY-Z"
    MOLECULE_NAME = "thiabendazole"      # exact name from DB, or partial (then must be unique)

    PLOT_MODE = "overlay"            # "overlay" or "stack"
    NORMALIZE = False               # normalize each conformer spectrum to max=1
    USE_SCALED_AXIS_1E38 = True    # display y as (cm^2/sr)*1e38 to match Chemcraft axis style

    X_MIN, X_MAX = 300.0, 1800.0
    Y_OFFSET = 1.25                # only used in stack mode (vertical separation)
    LINEWIDTH = 1.0

    SAVE_DIR = None                # e.g. Path("A:/plots"); if None -> saves next to DB/workdir
    OUT_NAME = None                # if None -> auto
    DPI = 200

    LASERS = [633, 532]            # lasers to compare (order defines subplot order)
    # -------------------------

    cfg = load_config()
    root, work_dir, db_path = get_paths(cfg)
    con = connect_db(db_path)

    if not has_column(con, "calculation", "raman_intensity_spectrum_path"):
        con.close()
        raise RuntimeError(
            "calculation.raman_intensity_spectrum_path missing. "
            "Run your 05_extract_raman.py postprocess that writes spectrum CSVs first."
        )

    inchikey = resolve_inchikey(con, inchikey=MOLECULE_INCHIKEY, name=MOLECULE_NAME)

    mol_row = con.execute("SELECT name FROM molecule WHERE inchikey=?", (inchikey,)).fetchone()
    mol_name = mol_row[0] if mol_row and mol_row[0] else inchikey

    rows = con.execute(
        """
        SELECT
            c.conformer_id,
            c.conf_index,
            calc.calc_id,
            calc.engine,
            calc.method,
            calc.raman_intensity_spectrum_path
        FROM conformer c
        JOIN calculation calc ON calc.conformer_id = c.conformer_id
        WHERE c.inchikey = ?
          AND calc.status = 'DONE'
          AND calc.raman_intensity_spectrum_path IS NOT NULL
        ORDER BY c.conf_index ASC
        """,
        (inchikey,),
    ).fetchall()
    con.close()

    if not rows:
        raise RuntimeError(
            f"No spectra found for molecule={mol_name} ({inchikey}). "
            "Check that calculations are DONE and 05 wrote raman_intensity_spectrum_path."
        )

    # Prepare per-laser spectra collections
    per_laser_spectra = {int(l): [] for l in LASERS}

    for conformer_id, conf_index, calc_id, engine, method, spec_path in rows:
        # parse semicolon-separated paths
        candidates = [p.strip() for p in (spec_path or "").split(";") if p.strip()]
        if not candidates:
            print(f"[WARN] no spectrum paths for {calc_id}")
            continue

        for laser in LASERS:
            chosen = None
            # prefer files with explicit "_{nm}nm" token
            for p in candidates:
                if f"_{int(laser)}nm" in p or f"{int(laser)}nm.csv" in p:
                    chosen = p
                    break
            # fallback: if only one candidate use it
            if chosen is None and len(candidates) == 1:
                chosen = candidates[0]

            if chosen is None:
                # no match for this laser for this calc
                continue

            p = Path(chosen)
            if not p.exists():
                print(f"[WARN] missing spectrum file for {calc_id} (laser={laser}): {p}")
                continue

            try:
                x, y = read_spectrum_csv(p)
            except Exception as e:
                print(f"[WARN] failed reading spectrum for {calc_id} (laser={laser}): {e}")
                continue

            # window
            m = (x >= X_MIN) & (x <= X_MAX)
            x_sel = x[m]
            y_sel = y[m]

            if NORMALIZE:
                ymax = float(np.nanmax(y_sel)) if y_sel.size else 0.0
                if ymax > 0:
                    y_sel = y_sel / ymax

            if USE_SCALED_AXIS_1E38 and not NORMALIZE:
                y_sel = y_sel * 1e38

            per_laser_spectra[int(laser)].append(
                dict(
                    conformer_id=conformer_id,
                    conf_index=int(conf_index) if conf_index is not None else None,
                    calc_id=calc_id,
                    engine=engine,
                    method=method,
                    path=str(p),
                    x=x_sel,
                    y=y_sel,
                )
            )

    # Plot side-by-side subplots, one per laser
    n_lasers = len(LASERS)
    if n_lasers == 0:
        raise RuntimeError("No lasers configured.")

    plt.figure(figsize=(6 * n_lasers, 6))
    for i, laser in enumerate(LASERS, start=1):
        spectra = per_laser_spectra[int(laser)]
        ax = plt.subplot(1, n_lasers, i)

        if not spectra:
            ax.text(0.5, 0.5, f"No spectra for {laser} nm", ha="center", va="center")
            ax.set_title(f"{laser} nm")
            ax.set_xlim(X_MIN, X_MAX)
            continue

        if PLOT_MODE.lower() == "overlay":
            for s in spectra:
                label = f"conf {s['conf_index']} | {s['engine']} | {s['calc_id']}"
                ax.plot(s["x"], s["y"], linewidth=LINEWIDTH, label=label)
            ax.legend(fontsize=8, ncol=1, frameon=False)

        elif PLOT_MODE.lower() == "stack":
            for idx, s in enumerate(spectra):
                y = s["y"] + idx * Y_OFFSET
                label = f"conf {s['conf_index']} | {s['engine']} | {s['calc_id']}"
                ax.plot(s["x"], y, linewidth=LINEWIDTH, label=label)
            ax.set_yticks([])
            ax.legend(fontsize=8, ncol=1, frameon=False)
        else:
            raise ValueError("PLOT_MODE must be 'overlay' or 'stack'.")

        ax.set_xlim(X_MIN, X_MAX)
        ax.set_xlabel("Wavenumber (cm$^{-1}$)")
        if NORMALIZE:
            ax.set_ylabel("Intensity (normalized)")
        else:
            if USE_SCALED_AXIS_1E38:
                ax.set_ylabel("Raman intensity (cm$^2$/sr) × 10$^{38}$")
            else:
                ax.set_ylabel("Raman intensity (cm$^2$/sr)")
        ax.set_title(f"{mol_name} ({inchikey}) | {laser} nm | mode={PLOT_MODE}")

    plt.tight_layout()

    # Save
    save_dir = Path(SAVE_DIR) if SAVE_DIR else (work_dir / inchikey / "plots")
    save_dir.mkdir(parents=True, exist_ok=True)

    if OUT_NAME is None:
        lasers_stub = "_".join(str(int(l)) for l in LASERS)
        out_stub = f"{mol_name}_{inchikey}_conformers_{PLOT_MODE}_{lasers_stub}nm"
        if NORMALIZE:
            out_stub += "_norm"
        if USE_SCALED_AXIS_1E38 and (not NORMALIZE):
            out_stub += "_1e38"
        OUT_NAME = out_stub + ".png"

    out_path = save_dir / OUT_NAME
    plt.savefig(out_path, dpi=DPI)
    print(f"[OK] saved: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()