import sqlite3
from scripts.common import load_config, get_paths


def main():
    cfg = load_config()
    root, work_dir, db_path = get_paths(cfg)

    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON;")
    con.execute("PRAGMA journal_mode = WAL;")  # lepsze dla wieloprocesowości lokalnie
    con.execute("PRAGMA synchronous = NORMAL;")

    with con:
        # -------- molecule --------
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS molecule (
                inchikey TEXT PRIMARY KEY,
                name TEXT,
                cid INTEGER,
                smiles TEXT,

                -- filled by conformer generation patch
                conformer_status TEXT,
                conformer_notes TEXT
            )
            """
        )

        # -------- conformer --------
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS conformer (
                conformer_id TEXT PRIMARY KEY,
                inchikey TEXT NOT NULL,
                conf_index INTEGER NOT NULL,
                xyz_path TEXT NOT NULL,
                energy REAL,          -- legacy ambiguous field; keep for backward compatibility
                mmff_energy REAL,
                gibbs_hartree REAL,

                FOREIGN KEY(inchikey) REFERENCES molecule(inchikey) ON DELETE CASCADE
            )
            """
        )

        # helpful uniqueness (optional but recommended)
        con.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_conformer_inchikey_index
            ON conformer(inchikey, conf_index)
            """
        )

        # -------- calculation --------
        # schema exactly compatible with what you showed (11 cols) + dispatch fields
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS calculation (
                calc_id TEXT PRIMARY KEY,
                conformer_id TEXT NOT NULL,
                engine TEXT NOT NULL,
                method TEXT NOT NULL,
                input_path TEXT NOT NULL,
                output_path TEXT NOT NULL,
                status TEXT NOT NULL,

                raman_peaks_path TEXT,
                raman_spectrum_path TEXT,
                raman_intensity_peaks_path TEXT,
                raman_intensity_spectrum_path TEXT,

                -- for multi-executor scheduling (PC1/PC2/HPC)
                claimed_by TEXT,
                claimed_at TEXT,
                batch_id TEXT,

                FOREIGN KEY(conformer_id) REFERENCES conformer(conformer_id) ON DELETE CASCADE
            )
            """
        )

        # -------- EU source table (core fields only; dynamic columns can be added by importer) --------
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

                pubchem_cid INTEGER,
                inchikey TEXT,

                structure_status TEXT,
                structure_notes TEXT,
                source_file TEXT,
                imported_at TEXT
            )
            """
        )

        # -------- indices --------
        con.execute("CREATE INDEX IF NOT EXISTS idx_conformer_inchikey ON conformer(inchikey)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_calc_conformer ON calculation(conformer_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_calc_status ON calculation(status)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_calc_engine ON calculation(engine)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_calc_claimed ON calculation(claimed_by, status)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_eu_status ON eu_active_substance(status_1107_2009)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_eu_cas ON eu_active_substance(cas_number)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_eu_inchikey ON eu_active_substance(inchikey)")

    con.close()
    print(f"[OK] DB initialized: {db_path}")


if __name__ == "__main__":
    main()
