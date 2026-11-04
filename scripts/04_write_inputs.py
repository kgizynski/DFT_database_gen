from pathlib import Path

from scripts.common import load_config, get_paths, connect_db


def normalize_spaces(s: str) -> str:
    return " ".join((s or "").split())


def normalize_lines(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        line = value.strip()
        return [line] if line else []
    if isinstance(value, list):
        out = []
        for item in value:
            line = str(item).rstrip()
            if line.strip():
                out.append(line)
        return out
    raise ValueError("Expected string/list/None for ORCA block config.")


def table_has_column(con, table: str, col: str) -> bool:
    cols = {row[1] for row in con.execute(f"PRAGMA table_info({table})").fetchall()}
    return col in cols


def read_xyz_geometry(xyz_path: Path) -> list[str]:
    lines = xyz_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid XYZ file (too short): {xyz_path}")

    try:
        n_atoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"Invalid XYZ atom count in {xyz_path}") from exc

    geometry = [line.rstrip() for line in lines[2 : 2 + n_atoms] if line.strip()]
    if len(geometry) != n_atoms:
        raise ValueError(f"Invalid XYZ geometry block in {xyz_path}")
    return geometry


def main():
    cfg = load_config()
    root, work_dir, db_path = get_paths(cfg)
    con = connect_db(db_path)

    engine = cfg["engine"].lower().strip()
    if engine not in {"orca", "gaussian"}:
        con.close()
        raise ValueError("cfg['engine'] must be 'orca' or 'gaussian'.")

    # If molecule.conformer_status exists, use only molecules where conformer generation succeeded
    has_conf_status = table_has_column(con, "molecule", "conformer_status")

    if has_conf_status:
        rows = con.execute(
            """
            SELECT c.conformer_id, c.xyz_path
            FROM conformer c
            JOIN molecule m ON m.inchikey = c.inchikey
            WHERE m.conformer_status = 'OK'
            """
        ).fetchall()
    else:
        rows = con.execute("SELECT conformer_id, xyz_path FROM conformer").fetchall()

    if engine == "orca":
        orca_cfg = cfg["orca"]
        # Backward compatibility: prefer explicit "input_line", fallback to legacy "method_basis".
        method = normalize_spaces(str(orca_cfg.get("input_line", orca_cfg["method_basis"])))
        charge = orca_cfg["charge"]
        mult = orca_cfg["multiplicity"]
        pal_block = normalize_lines(orca_cfg.get("pal_block", "%pal nprocs {{NPROCS}} end"))
        maxcore_block = normalize_lines(orca_cfg.get("maxcore_block", "%maxcore {{MAXCORE}}"))
        elprop_block = normalize_lines(orca_cfg.get("elprop_block", ["%elprop", "  polar 1", "end"]))
    else:
        route = normalize_spaces(str(cfg["gaussian"]["route"]))
        charge = cfg["gaussian"]["charge"]
        mult = cfg["gaussian"]["multiplicity"]
        nproc = int(cfg["gaussian"]["nproc"])
        mem = str(cfg["gaussian"]["mem"])
        checkpoint = bool(cfg["gaussian"].get("checkpoint", True))

    written_inputs = 0
    inserted_calcs = 0
    skipped_existing = 0
    skipped_missing_xyz = 0

    for conf_id, xyz_path in rows:
        xyz_path = Path(xyz_path) if xyz_path else None
        if xyz_path is None or (not xyz_path.exists()):
            skipped_missing_xyz += 1
            continue

        if engine == "orca":
            inp = xyz_path.with_suffix(".inp")
            outp = xyz_path.with_suffix(".out")
            content = [method, ""]
            content.extend(pal_block)
            content.extend(maxcore_block)
            if pal_block or maxcore_block:
                content.append("")
            content.extend(elprop_block)
            if elprop_block:
                content.append("")
            content.extend(
                [
                    f"* xyzfile {charge} {mult} {xyz_path.name}",
                    "",
                ]
            )
            method_name = "DFT"
        else:
            inp = xyz_path.with_suffix(".gjf")
            outp = xyz_path.with_suffix(".log")
            geometry = read_xyz_geometry(xyz_path)
            content = []
            if checkpoint:
                content.append(f"%chk={inp.stem}.chk")
            content.extend(
                [
                    f"%nprocshared={nproc}",
                    f"%mem={mem}",
                    route,
                    "",
                    conf_id,
                    "",
                    f"{charge} {mult}",
                    *geometry,
                    "",
                ]
            )
            method_name = route

        # Always write input file (cheap). If you prefer not to overwrite, tell me.
        inp.write_text("\n".join(content), encoding="utf-8")
        written_inputs += 1

        calc_id = f"{conf_id}_{engine}"

        # (2) Idempotent insert: don't create calculation if it already exists
        exists = con.execute("SELECT 1 FROM calculation WHERE calc_id = ? LIMIT 1", (calc_id,)).fetchone()
        if exists:
            skipped_existing += 1
            continue

        with con:
            con.execute(
                """
                INSERT INTO calculation (
                    calc_id, conformer_id, engine, method, input_path, output_path, status,
                    raman_peaks_path, raman_spectrum_path,
                    raman_intensity_peaks_path, raman_intensity_spectrum_path
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    calc_id,
                    conf_id,
                    engine,
                    method_name,
                    str(inp),
                    str(outp),
                    "READY",
                    None,
                    None,
                    None,
                    None,
                ),
            )
        inserted_calcs += 1

    con.close()
    print(
        "Inputs written.\n"
        f" - ORCA inputs written: {written_inputs}\n"
        f" - New calculation rows inserted: {inserted_calcs}\n"
        f" - Skipped (calculation exists): {skipped_existing}\n"
        f" - Skipped (missing xyz): {skipped_missing_xyz}"
    )


if __name__ == "__main__":
    main()
