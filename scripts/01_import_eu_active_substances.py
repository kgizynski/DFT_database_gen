# 06_import_eu_active_substances.py
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from scripts.common import load_config, get_paths, connect_db


# -----------------------------
# Helpers
# -----------------------------
def slugify_col(s: str) -> str:
    """
    Turn arbitrary Excel header into safe SQLite column name (snake_case).
    """
    s = str(s).strip()
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"<[^>]+>", "", s)  # strip HTML tags if any
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "col"
    # avoid leading digit
    if s[0].isdigit():
        s = "c_" + s
    return s


def find_latest_export(root: Path) -> Optional[Path]:
    """
    If you keep EU exports in project root, pick the latest ActiveSubstanceExport_*.xlsx.
    """
    candidates = sorted(root.glob("ActiveSubstanceExport_*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def read_eu_export(xlsx_path: Path) -> pd.DataFrame:
    """
    Robust reader for the EU export.
    The file has a title row and then a row containing headers (e.g., 'Active Substance ID').
    We locate that header row and rebuild the DataFrame from there.
    """
    raw = pd.read_excel(xlsx_path, header=None, dtype=str)
    raw = raw.fillna("")

    # Find the row that contains "Active Substance ID"
    header_row_idx = None
    for i in range(min(len(raw), 50)):  # search first 50 rows
        row_vals = [str(v).strip() for v in raw.iloc[i].tolist()]
        if any(v == "Active Substance ID" for v in row_vals):
            header_row_idx = i
            break

    if header_row_idx is None:
        raise ValueError(
            f"Could not find header row containing 'Active Substance ID' in: {xlsx_path}\n"
            "Open the file and confirm the column label exists."
        )

    headers = [str(v).strip() for v in raw.iloc[header_row_idx].tolist()]
    df = raw.iloc[header_row_idx + 1 :].copy()
    df.columns = headers
    df = df.reset_index(drop=True)

    # Drop fully-empty rows
    df = df.loc[~(df.apply(lambda r: all(str(x).strip() == "" for x in r), axis=1))].copy()

    return df


def ensure_table(con: sqlite3.Connection) -> None:
    """
    Create the target table if missing.
    We store key EU fields + mapping/pipeline fields.
    The EU export contains many columns; we import all of them dynamically too (as text).
    """
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS eu_active_substance (
            eu_active_substance_id INTEGER PRIMARY KEY,
            substance_name TEXT,
            cas_number TEXT,
            status_1107_2009 TEXT,
            date_of_approval TEXT,
            expiration_of_approval TEXT,
            classification_clp TEXT,
            authorised TEXT,
            functions TEXT,

            -- mapping to structure
            pubchem_cid INTEGER,
            inchikey TEXT,

            -- pipeline bookkeeping
            structure_status TEXT,   -- UNRESOLVED / RESOLVED / AMBIGUOUS / FAILED
            structure_notes TEXT,
            source_file TEXT,
            imported_at TEXT
        )
        """
    )

    # Indices for fast selection & joins
    con.execute("CREATE INDEX IF NOT EXISTS idx_eu_active_status ON eu_active_substance(status_1107_2009)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_eu_active_cas ON eu_active_substance(cas_number)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_eu_active_inchikey ON eu_active_substance(inchikey)")


def add_dynamic_columns(con: sqlite3.Connection, df_cols: List[str]) -> Dict[str, str]:
    """
    Add any missing columns from the EU export (as TEXT) so we keep the raw metadata.
    Returns mapping from original Excel column -> sqlite column name.
    """
    existing = {row[1] for row in con.execute("PRAGMA table_info(eu_active_substance)").fetchall()}
    colmap: Dict[str, str] = {}

    for c in df_cols:
        sc = slugify_col(c)
        # avoid collisions with our fixed columns by prefixing if needed
        if sc in existing:
            # if it's already present, just map to it
            colmap[c] = sc
            continue

        # if slug collides with another Excel column slug, make unique
        base = sc
        k = 2
        while sc in colmap.values() or sc in existing:
            sc = f"{base}_{k}"
            k += 1

        con.execute(f"ALTER TABLE eu_active_substance ADD COLUMN {sc} TEXT")
        existing.add(sc)
        colmap[c] = sc

    return colmap


def pick(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].astype(str).str.strip() if col in df.columns else pd.Series([""] * len(df), dtype=str)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    cfg = load_config()
    root, work_dir, db_path = get_paths(cfg)
    con = connect_db(db_path)

    # ---- choose input file
    # Option A: hardcode a path
    # xlsx_path = Path(r"D:\path\to\ActiveSubstanceExport_27-02-2026.xlsx")

    # Option B: auto-detect latest export in project root
    xlsx_path = find_latest_export(root)
    if xlsx_path is None:
        con.close()
        raise FileNotFoundError(
            f"No ActiveSubstanceExport_*.xlsx found in project root: {root}\n"
            "Place the EU export there or hardcode xlsx_path in this script."
        )

    df = read_eu_export(xlsx_path)

    # Create base table + add dynamic columns
    with con:
        ensure_table(con)
        colmap = add_dynamic_columns(con, list(df.columns))

    # Normalize core fields we care about for selection & mapping
    # (these column names match your EU file as of 27/02/2026)
    df_core = pd.DataFrame(
        {
            "eu_active_substance_id": pd.to_numeric(pick(df, "Active Substance ID"), errors="coerce").astype("Int64"),
            "substance_name": pick(df, "Substance"),
            "cas_number": pick(df, "CAS Number"),
            "status_1107_2009": pick(df, "Status under Reg. (EC) No 1107/2009"),
            "date_of_approval": pick(df, "Date of approval"),
            "expiration_of_approval": pick(df, "Expiration of approval"),
            "classification_clp": pick(df, "Classification (Reg. 1272/2008)"),
            "authorised": pick(df, "Authorised"),
            "functions": pick(df, "Functions"),
        }
    )

    # Drop rows without ID
    before = len(df_core)
    valid_rows = df_core["eu_active_substance_id"].notna()
    df_core = df_core.loc[valid_rows].copy().reset_index(drop=True)
    df = df.loc[valid_rows].copy().reset_index(drop=True)
    after = len(df_core)

    imported_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # Build upsert statement: fixed columns + all dynamic columns
    fixed_cols = [
        "eu_active_substance_id",
        "substance_name",
        "cas_number",
        "status_1107_2009",
        "date_of_approval",
        "expiration_of_approval",
        "classification_clp",
        "authorised",
        "functions",
        "pubchem_cid",
        "inchikey",
        "structure_status",
        "structure_notes",
        "source_file",
        "imported_at",
    ]

    # dynamic: store raw EU columns too
    dyn_cols = [colmap[c] for c in df.columns]

    all_cols = fixed_cols + dyn_cols
    placeholders = ",".join(["?"] * len(all_cols))

    sql = f"""
        INSERT INTO eu_active_substance ({",".join(all_cols)})
        VALUES ({placeholders})
        ON CONFLICT(eu_active_substance_id) DO UPDATE SET
        {",".join([f"{c}=excluded.{c}" for c in all_cols if c != "eu_active_substance_id"])}
    """

    rows = []
    for i in range(len(df_core)):
        rid = int(df_core.iloc[i]["eu_active_substance_id"])
        # fixed
        row = [
            rid,
            str(df_core.iloc[i]["substance_name"]),
            str(df_core.iloc[i]["cas_number"]),
            str(df_core.iloc[i]["status_1107_2009"]),
            str(df_core.iloc[i]["date_of_approval"]),
            str(df_core.iloc[i]["expiration_of_approval"]),
            str(df_core.iloc[i]["classification_clp"]),
            str(df_core.iloc[i]["authorised"]),
            str(df_core.iloc[i]["functions"]),
            None,   # pubchem_cid
            None,   # inchikey
            "UNRESOLVED",
            None,
            str(xlsx_path.name),
            imported_at,
        ]

        # dynamic raw columns
        for original_col in df.columns:
            val = str(df.iloc[i].get(original_col, "")).strip()
            row.append(val if val != "" else None)

        rows.append(row)

    with con:
        con.executemany(sql, rows)

    # Summary
    total = con.execute("SELECT COUNT(*) FROM eu_active_substance").fetchone()[0]
    by_status = con.execute(
        "SELECT status_1107_2009, COUNT(*) FROM eu_active_substance GROUP BY status_1107_2009 ORDER BY COUNT(*) DESC"
    ).fetchall()

    con.close()

    print(f"[OK] Imported EU export: {xlsx_path.name}")
    print(f" - Rows in file (core): {before} -> kept with ID: {after}")
    print(f" - Total rows in eu_active_substance: {total}")
    print(" - Status counts (top):")
    for s, n in by_status[:10]:
        print(f"   {s or '(NULL)'}: {n}")


if __name__ == "__main__":
    main()
