import re
import time
from typing import Optional, Tuple, List
from urllib.parse import quote

import requests
import pubchempy as pcp
from tqdm import tqdm

from scripts.common import load_config, get_paths, connect_db


CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")
PUG_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def normalize_cas(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = str(s).strip()
    if not s:
        return None
    parts = re.split(r"[;,/]\s*", s)
    for p in parts:
        p = p.strip()
        if CAS_RE.match(p):
            return p
    return None


def clean_name(name: str) -> str:
    n = (name or "").strip()

    # Keep first chunk before mixture delimiter
    if " / " in n:
        n = n.split(" / ")[0].strip()

    # Remove SCLP label parentheses
    n = re.sub(r"\s*\(SCLP[^)]*\)\s*", " ", n, flags=re.IGNORECASE).strip()

    # Remove other parentheses content
    n = re.sub(r"\s*\([^)]*\)\s*", " ", n).strip()

    # Remove trailing qualifiers
    n = re.sub(r"\baka\b.*$", "", n, flags=re.IGNORECASE).strip()
    n = re.sub(r"\bformerly\b.*$", "", n, flags=re.IGNORECASE).strip()
    n = re.sub(r"\bincl\.\b.*$", "", n, flags=re.IGNORECASE).strip()

    # Normalize dashes/whitespace
    n = n.replace("–", "-").replace("—", "-")
    n = re.sub(r"\s+", " ", n).strip()
    return n


def is_non_molecule(name: str, cas_raw: Optional[str]) -> Tuple[bool, str]:
    """
    Broader SKIP set: anything that is a class, mixture, organism, polymer, mineral, or undefined.
    """
    n = (name or "").lower()
    cas = (cas_raw or "").lower()

    # organisms/strains/plants
    bio_terms = [
        " strain", " strains", "aureobasidium", "trichoderma", "verticillium", "akanthomyces",
        "bacillus", "pseudomonas", "streptomyces", "beauveria", "metarhizium",
        "equisetum", " l.", " spp.", " species",
    ]
    if any(t in n for t in bio_terms):
        return True, "biological_agent_or_plant"

    # mixtures/materials/natural products
    mixture_terms = [
        " oil", " oils", "extract", "ext.", "meal", "blood meal", "bordeaux mixture",
        "mixture", "compounds", "repellents", "by smell",
        "fatty acids", "salts", "potassium salts", "unsaturated",
        "residues", "distillation", "hydrolysed protein", "hydrolyzed protein", "proteins",
        "cos-oga", "cerevisane",
    ]
    if any(t in n for t in mixture_terms):
        return True, "mixture_or_class"

    # polymers/undefined composition
    polymer_terms = ["maltodextrin", "starch", "cellulose", "lignin"]
    if any(t in n for t in polymer_terms):
        return True, "polymer_or_undefined"

    # minerals/materials/inorganics: (you generally don’t want for molecular DFT Raman)
    material_terms = [
        "kieselgur", "diatomaceous", "diatomite", "earth", "clayed charcoal", "charcoal",
        "kaolin", "aluminium silicate", "copper compounds", "tribasic copper", "copper", "sulphate",
        "carbonate", "phosphonate", "phosphites", "phosphonates",
    ]
    if any(t in n for t in material_terms) and "acetate" not in n and "ol" not in n:
        return True, "material_or_inorganic_class"

    # mixture delimiter
    if " / " in (name or ""):
        return True, "mixture_delimiter"

    # CAS explicitly says “No unique CAS” → likely a class
    if "no unique cas" in cas or "see note" in cas:
        if any(t in n for t in ["compounds", "mixture", "salts", "fatty acids", "phosphon"]):
            return True, "no_unique_cas_class"

    return False, ""


def pug_get_json(url: str, timeout: float = 20.0) -> Tuple[Optional[dict], str]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None, f"http_{r.status_code}"
        return r.json(), ""
    except Exception as e:
        return None, f"error:{e}"


def cas_to_cids_pugrest(cas: str) -> Tuple[Optional[List[int]], str]:
    url = f"{PUG_BASE}/compound/xref/RN/{quote(cas)}/cids/JSON"
    js, err = pug_get_json(url)
    if err:
        return None, f"cas_pugrest:{err}"
    cids = js.get("IdentifierList", {}).get("CID", None)
    if not cids:
        return [], "cas_pugrest:no_match"
    return [int(x) for x in cids], ""


def name_to_cids_pugrest(name: str) -> Tuple[Optional[List[int]], str]:
    # PubChem allows /compound/name/<name>/cids/JSON
    url = f"{PUG_BASE}/compound/name/{quote(name)}/cids/JSON"
    js, err = pug_get_json(url)
    if err:
        return None, f"name_pugrest:{err}"
    cids = js.get("IdentifierList", {}).get("CID", None)
    if not cids:
        return [], "name_pugrest:no_match"
    return [int(x) for x in cids], ""


def cid_to_compound(cid: int) -> Tuple[Optional[pcp.Compound], str]:
    try:
        c = pcp.Compound.from_cid(cid)
        return c, ""
    except Exception as e:
        return None, f"cid_fetch_error:{e}"


def pick_best_from_cids(cids: List[int]) -> Tuple[Optional[pcp.Compound], str]:
    """
    Fetch first CID that yields inchikey + smiles.
    If none do, fetch first as fallback.
    """
    if not cids:
        return None, "no_cids"

    first_comp = None
    for cid in cids[:10]:  # cap fetching to avoid too many calls
        comp, err = cid_to_compound(cid)
        if err or comp is None:
            continue
        if first_comp is None:
            first_comp = comp
        if getattr(comp, "inchikey", None) and getattr(comp, "smiles", None):
            return comp, f"picked_cid={cid}"
    if first_comp is not None:
        return first_comp, f"picked_first_available_cid={getattr(first_comp,'cid',None)}"
    return None, "cid_fetch_failed"


def map_one(name: str, cas: Optional[str]) -> Tuple[Optional[pcp.Compound], str]:
    notes = []

    # 1) CAS -> CIDs
    if cas:
        cids, err = cas_to_cids_pugrest(cas)
        if err:
            notes.append(err)
        elif cids:
            comp, pick_note = pick_best_from_cids(cids)
            if comp:
                if len(cids) > 1:
                    notes.append(f"resolved_by=CAS; ambiguous_cids={len(cids)}; {pick_note}")
                else:
                    notes.append(f"resolved_by=CAS; {pick_note}")
                return comp, "; ".join(notes)
        else:
            notes.append("cas_no_match")

    # 2) Name -> CIDs (PUG REST)  (better for IUPAC-ish names)
    cids, err = name_to_cids_pugrest(name)
    if err:
        notes.append(err)
    elif cids:
        comp, pick_note = pick_best_from_cids(cids)
        if comp:
            if len(cids) > 1:
                notes.append(f"resolved_by=NAME_PUGREST; ambiguous_cids={len(cids)}; {pick_note}")
            else:
                notes.append(f"resolved_by=NAME_PUGREST; {pick_note}")
            return comp, "; ".join(notes)
    else:
        notes.append("name_no_match")

    # 3) Last resort: PubChemPy name search
    try:
        comps = pcp.get_compounds(name, "name")
    except Exception as e:
        notes.append(f"name_pubchempy_error:{e}")
        return None, "; ".join(notes)

    if not comps:
        notes.append("name_pubchempy_no_match")
        return None, "; ".join(notes)

    best = comps[0]
    notes.append(f"resolved_by=NAME_PUBCHEMPY; ambiguous={len(comps)}")
    return best, "; ".join(notes)


def main():
    cfg = load_config()
    root, work_dir, db_path = get_paths(cfg)
    con = connect_db(db_path)

    TARGET_STATUSES = ("Approved", "Pending", "Not approved")
    ONLY_STRUCTURE_STATUS = ("UNRESOLVED", "FAILED")

    SLEEP_SEC = 0.10

    q = f"""
        SELECT eu_active_substance_id, substance_name, cas_number, status_1107_2009
        FROM eu_active_substance
        WHERE status_1107_2009 IN ({",".join(["?"] * len(TARGET_STATUSES))})
          AND (structure_status IS NULL OR structure_status IN ({",".join(["?"] * len(ONLY_STRUCTURE_STATUS))}))
        ORDER BY eu_active_substance_id
    """
    targets = con.execute(q, list(TARGET_STATUSES) + list(ONLY_STRUCTURE_STATUS)).fetchall()
    print(f"[INFO] Targets to map: {len(targets)} (statuses={TARGET_STATUSES}, structure_status={ONLY_STRUCTURE_STATUS})")

    mapped = ambiguous = failed = skipped = 0
    failures_preview = []

    for eu_id, name, cas_raw, _status in tqdm(targets):
        name = (name or "").strip()
        cas = normalize_cas(cas_raw)

        if not name:
            failed += 1
            with con:
                con.execute(
                    "UPDATE eu_active_substance SET structure_status=?, structure_notes=? WHERE eu_active_substance_id=?",
                    ("FAILED", "empty_name", eu_id),
                )
            continue

        skip, reason = is_non_molecule(name, cas_raw)
        if skip:
            skipped += 1
            with con:
                con.execute(
                    "UPDATE eu_active_substance SET structure_status=?, structure_notes=? WHERE eu_active_substance_id=?",
                    ("SKIPPED", reason, eu_id),
                )
            continue

        # Try raw then cleaned
        comp, note = map_one(name=name, cas=cas)
        if comp is None:
            name2 = clean_name(name)
            if name2 and name2 != name:
                comp2, note2 = map_one(name=name2, cas=cas)
                if comp2 is not None:
                    comp, note = comp2, f"{note}; name_cleaned={name2}; {note2}"

        if comp is None:
            failed += 1
            if len(failures_preview) < 20:
                failures_preview.append((eu_id, name, cas_raw, note))
            with con:
                con.execute(
                    """
                    UPDATE eu_active_substance
                    SET pubchem_cid=NULL, inchikey=NULL, structure_status=?, structure_notes=?
                    WHERE eu_active_substance_id=?
                    """,
                    ("FAILED", note, eu_id),
                )
            time.sleep(SLEEP_SEC)
            continue

        inchikey = getattr(comp, "inchikey", None)
        cid = getattr(comp, "cid", None)
        smiles = getattr(comp, "smiles", None)

        if not inchikey or not cid:
            failed += 1
            note2 = f"{note}; missing_inchikey_or_cid"
            if len(failures_preview) < 20:
                failures_preview.append((eu_id, name, cas_raw, note2))
            with con:
                con.execute(
                    """
                    UPDATE eu_active_substance
                    SET pubchem_cid=NULL, inchikey=NULL, structure_status=?, structure_notes=?
                    WHERE eu_active_substance_id=?
                    """,
                    ("FAILED", note2, eu_id),
                )
            time.sleep(SLEEP_SEC)
            continue

        status_out = "RESOLVED"
        # Mark ambiguous if multiple candidates indicated
        if "ambiguous" in note:
            status_out = "AMBIGUOUS"
            ambiguous += 1

        with con:
            con.execute(
                """
                UPDATE eu_active_substance
                SET pubchem_cid=?, inchikey=?, structure_status=?, structure_notes=?
                WHERE eu_active_substance_id=?
                """,
                (int(cid), str(inchikey), status_out, note, eu_id),
            )
            con.execute(
                """
                INSERT INTO molecule (inchikey, name, cid, smiles)
                VALUES (?,?,?,?)
                ON CONFLICT(inchikey) DO UPDATE SET
                    name=excluded.name,
                    cid=excluded.cid,
                    smiles=excluded.smiles
                """,
                (str(inchikey), name, int(cid), smiles),
            )

        mapped += 1
        time.sleep(SLEEP_SEC)

    total_resolved = con.execute("SELECT COUNT(*) FROM eu_active_substance WHERE structure_status='RESOLVED'").fetchone()[0]
    total_amb = con.execute("SELECT COUNT(*) FROM eu_active_substance WHERE structure_status='AMBIGUOUS'").fetchone()[0]
    total_failed = con.execute("SELECT COUNT(*) FROM eu_active_substance WHERE structure_status='FAILED'").fetchone()[0]
    total_skipped = con.execute("SELECT COUNT(*) FROM eu_active_substance WHERE structure_status='SKIPPED'").fetchone()[0]
    con.close()

    print("[OK] PubChem mapping done.")
    print(f" - Mapped this run: {mapped}")
    print(f" - Ambiguous this run: {ambiguous}")
    print(f" - Failed this run: {failed}")
    print(f" - Skipped this run: {skipped}")
    print(" - Totals in DB:")
    print(f"   RESOLVED:  {total_resolved}")
    print(f"   AMBIGUOUS: {total_amb}")
    print(f"   FAILED:    {total_failed}")
    print(f"   SKIPPED:   {total_skipped}")

    if failures_preview:
        print("Some failures (up to 20):")
        for eu_id, name, cas_raw, reason in failures_preview:
            print(f" - EU_ID={eu_id} | {name} | CAS={cas_raw} | {reason}")


if __name__ == "__main__":
    main()