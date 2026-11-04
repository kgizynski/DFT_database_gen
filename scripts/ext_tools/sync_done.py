from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path

from scripts.common import connect_db, get_paths, load_config


def is_finished(engine: str, out_path: Path) -> bool:
    if not out_path.exists():
        return False

    try:
        txt = out_path.read_text(errors="ignore")
    except Exception:
        return False

    engine = (engine or "").lower()
    if engine == "orca":
        return "ORCA TERMINATED NORMALLY" in txt
    if engine == "gaussian":
        return "Normal termination of Gaussian" in txt
    return False


def guess_molecules_dir(work_dir: Path) -> Path:
    cand = work_dir / "molecules"
    return cand if cand.exists() else work_dir


def resolve_output_path(output_path: str | None, work_dir: Path, molecules_dir: Path) -> Path | None:
    if not output_path:
        return None

    outp = Path(output_path)
    if outp.is_absolute():
        return outp

    candidate_molecules = molecules_dir / outp
    candidate_work = work_dir / outp
    return candidate_molecules if candidate_molecules.exists() else candidate_work


def load_batch_result_map(batch_root: Path, batch_id: str) -> dict[str, Path]:
    batch_dir = batch_root / batch_id
    manifest_path = batch_dir / "manifest.csv"
    results_dir = batch_dir / "results"

    if not manifest_path.exists() or not results_dir.exists():
        return {}

    result_map: dict[str, Path] = {}
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            calc_id = (row.get("calc_id") or "").strip()
            output_file = (row.get("output_file") or "").strip()
            if calc_id and output_file:
                result_map[calc_id] = results_dir / output_file
    return result_map


def sync_pc2_batches(con, cfg: dict, work_dir: Path, molecules_dir: Path) -> tuple[int, int, int, int]:
    batch_root = Path(cfg.get("dispatch", {}).get("onedrive_batches_dir", "")).expanduser()
    if not str(batch_root):
        return 0, 0, 0, 0

    rows = con.execute(
        """
        SELECT calc_id, output_path, engine, batch_id
        FROM calculation
        WHERE upper(coalesce(claimed_by, ''))='PC2'
          AND status IN ('DISPATCHED', 'RUNNING')
          AND batch_id IS NOT NULL
          AND trim(batch_id) <> ''
        """
    ).fetchall()

    batch_maps: dict[str, dict[str, Path]] = {}
    updated_done = 0
    copied_outputs = 0
    missing_results = 0
    not_finished = 0

    for calc_id, output_path, eng, batch_id in rows:
        if batch_id not in batch_maps:
            batch_maps[batch_id] = load_batch_result_map(batch_root, batch_id)

        result_path = batch_maps[batch_id].get(calc_id)
        if result_path is None or not result_path.exists():
            missing_results += 1
            continue

        engine = (eng or cfg["engine"]).lower()
        if not is_finished(engine, result_path):
            not_finished += 1
            continue

        target_out = resolve_output_path(output_path, work_dir, molecules_dir)
        if target_out is None:
            missing_results += 1
            continue

        target_out.parent.mkdir(parents=True, exist_ok=True)
        if not target_out.exists() or target_out.read_bytes() != result_path.read_bytes():
            shutil.copy2(result_path, target_out)
            copied_outputs += 1

        con.execute("UPDATE calculation SET status='DONE' WHERE calc_id=?", (calc_id,))
        updated_done += 1

    return updated_done, copied_outputs, missing_results, not_finished


def main():
    print(f"\n========== SYNC DONE STARTED: {datetime.now()} ==========\n")

    cfg = load_config()
    _root, work_dir, db_path = get_paths(cfg)
    con = connect_db(db_path)

    try:
        con.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        pass
    try:
        con.execute("PRAGMA busy_timeout=10000;")
    except Exception:
        pass

    engine = str(cfg["engine"]).lower()
    molecules_dir = guess_molecules_dir(work_dir)

    print(f"[DEBUG] engine={engine}")
    print(f"[DEBUG] db_path={db_path}")
    print(f"[DEBUG] molecules_dir={molecules_dir}")

    pc2_done, pc2_copied, pc2_missing, pc2_not_finished = sync_pc2_batches(
        con=con,
        cfg=cfg,
        work_dir=work_dir,
        molecules_dir=molecules_dir,
    )

    rows = con.execute(
        "SELECT calc_id, output_path, status, engine FROM calculation"
    ).fetchall()

    updated_done = 0
    already_done = 0
    missing = 0
    not_finished = 0

    for calc_id, output_path, status, eng in rows:
        eng_eff = (eng or engine).lower()
        outp = resolve_output_path(output_path, work_dir, molecules_dir)
        if outp is None:
            missing += 1
            continue

        if not outp.exists():
            missing += 1
            continue

        if is_finished(eng_eff, outp):
            if status != "DONE":
                con.execute("UPDATE calculation SET status='DONE' WHERE calc_id=?", (calc_id,))
                updated_done += 1
            else:
                already_done += 1
        else:
            not_finished += 1

    con.commit()

    exts = [".out"] if engine == "orca" else [".log"]
    found_outputs = sum(1 for p in molecules_dir.rglob("*") if p.suffix.lower() in exts)
    print(f"[DEBUG] filesystem outputs found (by ext {exts}): {found_outputs}")

    con.close()

    print("\n========== SUMMARY ==========")
    print(f"PC2 synced to DONE:         {pc2_done}")
    print(f"PC2 outputs copied:         {pc2_copied}")
    print(f"PC2 missing batch results:  {pc2_missing}")
    print(f"PC2 batch unfinished:       {pc2_not_finished}")
    print(f"UPDATED to DONE:            {updated_done}")
    print(f"ALREADY DONE:               {already_done}")
    print(f"NOT FINISHED:               {not_finished}")
    print(f"MISSING output:             {missing}")
    print("Sync finished.\n")


if __name__ == "__main__":
    main()
