from __future__ import annotations

from typing import Optional, Tuple, Set

from rdkit import Chem
from rdkit.Chem import AllChem

from scripts.common import load_config, get_paths, connect_db


ALLOWED_ELEMENTS: Set[str] = {
    "H", "B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Si"
}
# Jeśli chcesz dopuścić np. Na/K/Mg/Ca (sole) – dodaj, ale to zwykle psuje sens DFT “molekułowego”:
# ALLOWED_ELEMENTS |= {"Na", "K", "Mg", "Ca"}


def ensure_molecule_status_columns(con) -> None:
    cols = {row[1] for row in con.execute("PRAGMA table_info(molecule)").fetchall()}
    if "conformer_status" not in cols:
        con.execute("ALTER TABLE molecule ADD COLUMN conformer_status TEXT")
    if "conformer_notes" not in cols:
        con.execute("ALTER TABLE molecule ADD COLUMN conformer_notes TEXT")


def ensure_conformer_energy_columns(con) -> None:
    cols = {row[1] for row in con.execute("PRAGMA table_info(conformer)").fetchall()}
    if "mmff_energy" not in cols:
        con.execute("ALTER TABLE conformer ADD COLUMN mmff_energy REAL")
    if "gibbs_hartree" not in cols:
        con.execute("ALTER TABLE conformer ADD COLUMN gibbs_hartree REAL")


def has_only_allowed_elements(mol: Chem.Mol) -> Tuple[bool, Optional[str]]:
    for a in mol.GetAtoms():
        sym = a.GetSymbol()
        if sym not in ALLOWED_ELEMENTS:
            return False, sym
    return True, None


def parse_mol(smiles: str) -> Optional[Chem.Mol]:
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        # sanity: kekulize issues etc. -> keep as-is; AddHs later
        return m
    except Exception:
        return None


def get_mmff_energy(mol: Chem.Mol, conf_id: int) -> Optional[float]:
    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is None:
        return None

    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=int(conf_id))
    if ff is None:
        return None
    return float(ff.CalcEnergy())


def main():
    cfg = load_config()
    root, work_dir, db_path = get_paths(cfg)
    con = connect_db(db_path)
    conf_cfg = cfg.get("conformers", {})
    n_conformers = int(conf_cfg.get("n_conformers", 3))
    max_keep = int(conf_cfg.get("max_conformers_keep", n_conformers))
    embed_seed = int(conf_cfg.get("embed_seed", 42))
    mmff_max_iters = int(conf_cfg.get("mmff_max_iters", 800))
    prune_rms_thresh = float(conf_cfg.get("prune_rms_thresh", -1.0))
    use_mmff = bool(conf_cfg.get("use_mmff", True))
    write_sdf = bool(conf_cfg.get("write_sdf", False))

    if "write_xyz" in conf_cfg and not bool(conf_cfg.get("write_xyz")):
        raise ValueError("conformers.write_xyz=false is not supported; the pipeline requires XYZ files.")

    with con:
        ensure_molecule_status_columns(con)
        ensure_conformer_energy_columns(con)

    rows = con.execute("SELECT inchikey, smiles FROM molecule").fetchall()

    # ETKDG settings
    params = AllChem.ETKDGv3()
    params.randomSeed = embed_seed
    params.useSmallRingTorsions = True
    params.useExpTorsionAnglePrefs = True
    if prune_rms_thresh > 0.0:
        params.pruneRmsThresh = prune_rms_thresh

    total = len(rows)
    made = 0
    skipped = 0

    for inchikey, smiles in rows:
        if not smiles:
            skipped += 1
            with con:
                con.execute(
                    "UPDATE molecule SET conformer_status=?, conformer_notes=? WHERE inchikey=?",
                    ("SKIPPED", "empty_smiles", inchikey),
                )
            continue

        mol0 = parse_mol(smiles)
        if mol0 is None:
            skipped += 1
            with con:
                con.execute(
                    "UPDATE molecule SET conformer_status=?, conformer_notes=? WHERE inchikey=?",
                    ("SKIPPED", "smiles_parse_failed", inchikey),
                )
            continue

        ok, bad_el = has_only_allowed_elements(mol0)
        if not ok:
            skipped += 1
            with con:
                con.execute(
                    "UPDATE molecule SET conformer_status=?, conformer_notes=? WHERE inchikey=?",
                    ("SKIPPED", f"contains_disallowed_element:{bad_el}", inchikey),
                )
            continue

        mol = Chem.AddHs(mol0)

        try:
            ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
        except Exception as e:
            skipped += 1
            with con:
                con.execute(
                    "UPDATE molecule SET conformer_status=?, conformer_notes=? WHERE inchikey=?",
                    ("FAILED", f"embed_exception:{e}", inchikey),
                )
            continue

        if not ids:
            skipped += 1
            with con:
                con.execute(
                    "UPDATE molecule SET conformer_status=?, conformer_notes=? WHERE inchikey=?",
                    ("FAILED", "embed_returned_no_confs", inchikey),
                )
            continue

        ids = list(ids)
        energy_by_conf = {int(cid): None for cid in ids}

        if use_mmff:
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is None:
                skipped += 1
                with con:
                    con.execute(
                        "UPDATE molecule SET conformer_status=?, conformer_notes=? WHERE inchikey=?",
                        ("FAILED", "mmff_properties_unavailable", inchikey),
                    )
                continue

            try:
                AllChem.MMFFOptimizeMoleculeConfs(
                    mol,
                    mmffVariant="MMFF94",
                    maxIters=mmff_max_iters,
                    numThreads=0,
                )
            except Exception as e:
                skipped += 1
                with con:
                    con.execute(
                        "UPDATE molecule SET conformer_status=?, conformer_notes=? WHERE inchikey=?",
                        ("FAILED", f"mmff_optimize_exception:{e}", inchikey),
                    )
                continue

            for cid in ids:
                energy_by_conf[int(cid)] = get_mmff_energy(mol, int(cid))

        ranked = []
        for cid in ids:
            cid_int = int(cid)
            energy = energy_by_conf[cid_int]
            sort_energy = energy if energy is not None else float("inf")
            ranked.append((sort_energy, cid_int, energy))

        ranked.sort(key=lambda item: (item[0], item[1]))
        selected = ranked[: max(1, min(max_keep, len(ranked)))]

        # Save conformers
        out_dir = work_dir / inchikey
        out_dir.mkdir(parents=True, exist_ok=True)

        # Regeneration should replace previous conformers and their calculations.
        stale_prefix = f"{inchikey}_conf_"
        for path in out_dir.glob(f"{stale_prefix}*"):
            if path.is_file():
                path.unlink(missing_ok=True)

        with con:
            con.execute("DELETE FROM conformer WHERE inchikey=?", (inchikey,))

        for i, (_sort_energy, cid, energy) in enumerate(selected):
            conf_id = f"{inchikey}_conf_{i}"
            xyz_path = out_dir / f"{conf_id}.xyz"
            Chem.MolToXYZFile(mol, str(xyz_path), confId=cid)
            if write_sdf:
                sdf_path = out_dir / f"{conf_id}.sdf"
                writer = Chem.SDWriter(str(sdf_path))
                try:
                    writer.write(mol, confId=cid)
                finally:
                    writer.close()

            with con:
                con.execute(
                    """
                    INSERT OR REPLACE INTO conformer (
                        conformer_id, inchikey, conf_index, xyz_path, energy, mmff_energy, gibbs_hartree
                    ) VALUES (?,?,?,?,?,?,?)
                    """,
                    (conf_id, inchikey, i, str(xyz_path), None, energy, None),
                )

        made += 1
        with con:
            con.execute(
                "UPDATE molecule SET conformer_status=?, conformer_notes=? WHERE inchikey=?",
                ("OK", f"generated={len(ids)}; kept={len(selected)}; mmff={'yes' if use_mmff else 'no'}", inchikey),
            )

    con.close()
    print(f"Conformers generated. Molecules total={total}, OK={made}, skipped/failed={skipped}")


if __name__ == "__main__":
    main()
