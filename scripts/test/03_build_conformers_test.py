from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional, Set, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import connect_db, load_config, project_root


TARGET_INCHIKEY = "WJCNZQLZVWNLKY-UHFFFAOYSA-N"
N_CONFORMERS = 20
CHARGE = 0
MULTIPLICITY = 1

ORCA_INPUT_TEMPLATE = """! LibXC(B3LYP) def2-SVP OPT FREQ RIJCOSX TightSCF

%pal nprocs 4 end
%maxcore 1000

%elprop
  polar 1
end

* xyzfile 0 1 {xyz_name}
"""

ALLOWED_ELEMENTS: Set[str] = {
    "H", "B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Si"
}


def has_only_allowed_elements(mol: Chem.Mol) -> Tuple[bool, Optional[str]]:
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in ALLOWED_ELEMENTS:
            return False, symbol
    return True, None


def parse_mol(smiles: str) -> Optional[Chem.Mol]:
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def fetch_smiles(inchikey: str) -> str:
    cfg = load_config()
    db_path = Path(cfg["paths"]["db_path"]).resolve()
    con = connect_db(db_path)
    try:
        row = con.execute(
            "SELECT smiles FROM molecule WHERE inchikey = ?",
            (inchikey,),
        ).fetchone()
    finally:
        con.close()

    if row is None:
        raise ValueError(f"Molecule not found for InChIKey: {inchikey}")
    if not row[0]:
        raise ValueError(f"Empty SMILES for InChIKey: {inchikey}")
    return str(row[0])


def build_conformers(smiles: str) -> tuple[Chem.Mol, list[int]]:
    mol0 = parse_mol(smiles)
    if mol0 is None:
        raise ValueError("SMILES parsing failed")

    ok, bad_el = has_only_allowed_elements(mol0)
    if not ok:
        raise ValueError(f"Disallowed element in molecule: {bad_el}")

    mol = Chem.AddHs(mol0)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.useSmallRingTorsions = True
    params.useExpTorsionAnglePrefs = True

    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=N_CONFORMERS, params=params))
    if not ids:
        raise RuntimeError("EmbedMultipleConfs returned no conformers")

    return mol, ids


def write_orca_input(inp_path: Path, xyz_name: str) -> None:
    content = ORCA_INPUT_TEMPLATE.format(xyz_name=xyz_name)
    inp_path.write_text(content, encoding="utf-8")


def main() -> None:
    smiles = fetch_smiles(TARGET_INCHIKEY)
    mol, conformer_ids = build_conformers(smiles)

    out_dir = project_root() / "testConformers"
    out_dir.mkdir(parents=True, exist_ok=True)

    for index, conf_id in enumerate(conformer_ids, start=1):
        stem = f"{TARGET_INCHIKEY}_conf_{index:03d}"
        xyz_path = out_dir / f"{stem}.xyz"
        inp_path = out_dir / f"{stem}.inp"

        Chem.MolToXYZFile(mol, str(xyz_path), confId=int(conf_id))
        write_orca_input(inp_path, xyz_path.name)

    print(
        f"Generated {len(conformer_ids)} conformers for {TARGET_INCHIKEY} "
        f"in {out_dir}"
    )


if __name__ == "__main__":
    main()
