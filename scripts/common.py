from pathlib import Path
import sqlite3
import yaml

def project_root():
    return Path(__file__).resolve().parents[1]

def load_config():
    root = project_root()
    cfg_path = root / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["project"]["root_dir"] = str(root)
    return cfg

def get_paths(cfg):
    root = Path(cfg.get("paths", {}).get("root", ".")).resolve()

    work_dir = Path(cfg["paths"]["work_dir"]).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    db_path = Path(cfg["paths"]["db_path"]).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    return root, work_dir, db_path

def connect_db(db_path):
    con = sqlite3.connect(str(db_path), timeout=30)
    # Robustness for concurrent access (single machine / LAN)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA busy_timeout=10000;")  # 10 s
    con.execute("PRAGMA foreign_keys=ON;")     # IMPORTANT
    # Optional performance tweaks
    con.execute("PRAGMA temp_store=MEMORY;")
    con.execute("PRAGMA cache_size=-20000;")   # ~20k pages in memory (negative => KB)
    return con