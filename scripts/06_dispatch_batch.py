import csv
import shutil
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Dict, Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import load_config, get_paths, connect_db


# PyCharm run configuration: edit these values instead of passing CLI arguments.
TARGET = "PC2"
JOB_COUNT = 1000


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def get_target_profile(cfg: Dict[str, Any], target: str) -> Dict[str, Any]:
    dispatch_cfg = cfg.get("dispatch", {})
    targets_cfg = dispatch_cfg.get("targets", {})
    profile = dict(targets_cfg.get(target.upper(), {}))

    if not profile:
        if target.upper() == "PC2":
            profile = {
                "nprocs": 8,
                "maxcore": 2000,
                "remove_maxcore_line": False,
                "generate_slurm": False,
            }
        elif target.upper() == "HPC":
            profile = {
                "nprocs": 8,
                "maxcore": 1500,
                "remove_maxcore_line": False,
                "generate_slurm": True,
            }
        else:
            raise ValueError(f"Unknown target '{target}'. Configure dispatch.targets.{target.upper()} in config.yaml")

    profile.setdefault("nprocs", 8)
    profile.setdefault("remove_maxcore_line", False)
    profile.setdefault("generate_slurm", target.upper() == "HPC")
    return profile


def rewrite_orca_input_for_target(inp_text: str, profile: Dict[str, Any]) -> str:
    nprocs = int(profile.get("nprocs", 8))
    remove_maxcore = bool(profile.get("remove_maxcore_line", False))
    maxcore = profile.get("maxcore")

    out_lines = []
    for raw_line in inp_text.splitlines():
        line = raw_line
        if "{{NPROCS}}" in line:
            line = line.replace("{{NPROCS}}", str(nprocs))

        if "{{MAXCORE}}" in line:
            if remove_maxcore:
                continue
            if maxcore is None:
                raise ValueError("Missing 'maxcore' for target profile with remove_maxcore_line=false")
            line = line.replace("{{MAXCORE}}", str(int(maxcore)))

        out_lines.append(line)

    rendered = "\n".join(out_lines)
    if inp_text.endswith("\n"):
        rendered += "\n"
    return rendered


def render_slurm_script(cfg: Dict[str, Any], profile: Dict[str, Any], batch_id: str, job_count: int) -> str:
    slurm_cfg = cfg.get("dispatch", {}).get("hpc_slurm", {})
    job_name_prefix = str(slurm_cfg.get("job_name_prefix", "Orca_job"))
    job_name = f"{job_name_prefix}_{batch_id}"
    ntasks = int(profile.get("nprocs", 8))
    cpus_per_task = int(slurm_cfg.get("cpus_per_task", 1))
    mem = str(slurm_cfg.get("mem", "12GB"))
    walltime = str(slurm_cfg.get("time", "72:00:00"))
    account = str(slurm_cfg.get("account", "plgdftraman-cpu"))
    partition = str(slurm_cfg.get("partition", "plgrid"))
    array_expr = str(slurm_cfg.get("array", "1-{job_count}")).format(job_count=job_count)
    out_path = str(slurm_cfg.get("output", "logs/out_%A_%a.out"))
    err_path = str(slurm_cfg.get("error", "logs/err_%A_%a.err"))
    module_load = str(slurm_cfg.get("module_load", "orca")).strip()
    orca_cmd = str(slurm_cfg.get("orca_command", "$ORCA")).strip()

    lines = [
        "#!/bin/bash -l",
        f"#SBATCH -J {job_name}",
        f"#SBATCH --ntasks={ntasks}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={walltime}",
        f"#SBATCH -A {account}",
        f"#SBATCH -p {partition}",
        f"#SBATCH --array={array_expr}",
        f"#SBATCH --output={out_path}",
        f"#SBATCH --error={err_path}",
        "",
        "cd \"$SLURM_SUBMIT_DIR\" || exit 1",
        "",
        "mkdir -p logs outputs",
    ]

    if module_load:
        lines.append(f"module load {module_load}")

    lines.extend(
        [
            "",
            "INPUT=$(sed -n \"${SLURM_ARRAY_TASK_ID}p\" input_list.txt)",
            "BASENAME=$(basename \"$INPUT\" .inp)",
            "",
            "cd inputs || exit 1",
            "",
            f"{orca_cmd} \"$INPUT\" > " + "\"../outputs/${BASENAME}.out\"",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    cfg = load_config()
    root, work_dir, db_path = get_paths(cfg)
    con = connect_db(db_path)

    target = TARGET.strip().upper()
    N = int(JOB_COUNT)
    profile = get_target_profile(cfg, target)

    batch_root = Path(cfg.get("dispatch", {}).get("onedrive_batches_dir", "")).expanduser().resolve()
    if not str(batch_root):
        raise ValueError("Set dispatch.onedrive_batches_dir in config.yaml")
    batch_root.mkdir(parents=True, exist_ok=True)

    batch_id = f"{target.lower()}_{utc_ts()}"
    batch_dir = batch_root / batch_id
    in_dir = batch_dir / "inputs"
    in_dir.mkdir(parents=True, exist_ok=True)

    # 1) claim READY jobs atomically; only mark DISPATCHED after files are written
    con.execute("BEGIN IMMEDIATE;")
    try:
        rows = con.execute(
            """
            SELECT calc_id, input_path, output_path
            FROM calculation
            WHERE status='READY'
              AND engine='orca'
              AND (claimed_by IS NULL OR claimed_by='')
            LIMIT ?
            """,
            (N,),
        ).fetchall()
        calc_ids = [r[0] for r in rows]

        if not calc_ids:
            con.execute("COMMIT;")
            con.close()
            print("[INFO] No READY jobs to dispatch.")
            return

        ts = datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z"
        con.executemany(
            """
            UPDATE calculation
            SET claimed_by=?, claimed_at=?, batch_id=?
            WHERE calc_id=?
            """,
            [(target, ts, batch_id, cid) for cid in calc_ids],
        )

        con.execute("COMMIT;")
    except Exception:
        con.execute("ROLLBACK;")
        con.close()
        raise

    # 2) copy inp + referenced xyz into OneDrive batch, write manifest
    try:
        manifest_path = batch_dir / "manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["calc_id", "input_file", "output_file"])
            for calc_id, input_path, output_path in rows:
                inp = Path(input_path)
                inp_dst = in_dir / inp.name
                inp_src_text = inp.read_text(encoding="utf-8")
                inp_dst.write_text(rewrite_orca_input_for_target(inp_src_text, profile), encoding="utf-8")

                # find xyzfile line in ORCA input
                xyz_name = None
                for line in inp_src_text.splitlines():
                    line = line.strip()
                    if line.lower().startswith("* xyzfile"):
                        xyz_name = line.split()[-1]
                        break
                if not xyz_name:
                    raise RuntimeError(f"Cannot find '* xyzfile' line in {inp}")

                xyz_src = inp.parent / xyz_name
                if not xyz_src.exists():
                    raise RuntimeError(f"Missing xyz file: {xyz_src}")
                shutil.copy2(xyz_src, in_dir / xyz_src.name)

                w.writerow([calc_id, inp.name, Path(output_path).name])

        if bool(profile.get("generate_slurm", False)):
            input_list = batch_dir / "input_list.txt"
            inputs = sorted(p.name for p in in_dir.glob("*.inp"))
            if not inputs:
                raise RuntimeError("No .inp files found for HPC SLURM generation.")
            input_list.write_text(
                "\n".join(inputs) + "\n",
                encoding="utf-8",
                newline="\n",
            )

            logs_dir = batch_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            slurm_cfg = cfg.get("dispatch", {}).get("hpc_slurm", {})
            slurm_name = str(slurm_cfg.get("filename", "submit_hpc.slurm"))
            slurm_path = batch_dir / slurm_name
            slurm_path.write_text(
                render_slurm_script(cfg, profile=profile, batch_id=batch_id, job_count=len(inputs)),
                encoding="utf-8",
                newline="\n",
            )
    except Exception:
        with con:
            con.execute(
                """
                UPDATE calculation
                SET claimed_by=NULL, claimed_at=NULL, batch_id=NULL
                WHERE batch_id=? AND status='READY'
                """,
                (batch_id,),
            )
        con.close()
        raise

    with con:
        con.execute(
            """
            UPDATE calculation
            SET status='DISPATCHED'
            WHERE batch_id=? AND claimed_by=? AND status='READY'
            """,
            (batch_id, target),
        )

    con.close()
    print(f"[OK] Dispatched {len(rows)} jobs to OneDrive batch: {batch_dir}")
    if bool(profile.get("generate_slurm", False)):
        print("HPC mode: generated input_list.txt and SLURM script in batch folder.")
    else:
        print("PC2 mode: inputs prepared for local batch runner after sync.")


if __name__ == "__main__":
    main()
