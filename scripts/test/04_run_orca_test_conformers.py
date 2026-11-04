from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import load_config, project_root


N_PARALLEL_JOBS = 4
NPROCS_SEQUENCE = (8, 4, 1)


def already_done(out_path: Path) -> bool:
    if not out_path.exists():
        return False
    text = out_path.read_text(encoding="utf-8", errors="ignore")
    return "ORCA TERMINATED NORMALLY" in text


def rewrite_pal_block(text: str, nprocs: int) -> str:
    return re.sub(
        r"%pal\s+nprocs\s+\d+\s+end",
        f"%pal nprocs {nprocs} end",
        text,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    )


def write_runtime_input(inp_path: Path, nprocs: int) -> Path:
    runtime_inp = inp_path.with_suffix(f".n{nprocs}.run.inp")
    text = inp_path.read_text(encoding="utf-8")
    runtime_inp.write_text(rewrite_pal_block(text, nprocs), encoding="utf-8")
    return runtime_inp


def is_properties_mpi_crash(out_path: Path) -> bool:
    if not out_path.exists():
        return False
    text = out_path.read_text(encoding="utf-8", errors="ignore")
    markers = [
        "ORCA finished by error termination in PROPERTIES",
        "orca_prop_mpi.exe ended prematurely",
        "exit code 0xc0000005",
    ]
    return all(marker in text for marker in markers)


def run_orca(orca_cmd: str, inp_path: Path, out_path: Path, nprocs: int) -> None:
    runtime_inp = write_runtime_input(inp_path, nprocs)
    try:
        with open(out_path, "w", encoding="utf-8") as handle:
            subprocess.run(
                [orca_cmd, runtime_inp.name],
                cwd=str(inp_path.parent),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
    finally:
        runtime_inp.unlink(missing_ok=True)


def run_one(args: tuple[Path, str]) -> tuple[str, str, float]:
    inp_path, orca_cmd = args
    out_path = inp_path.with_suffix(".out")

    start_time = time.time()
    pid = os.getpid()
    job_name = inp_path.stem

    if already_done(out_path):
        print(f"[{datetime.now()}] PID {pid} | {job_name} -> SKIPPED")
        return job_name, "SKIPPED", 0.0

    tmp_out = out_path.with_suffix(".out.tmp")
    print(f"[{datetime.now()}] PID {pid} | {job_name} -> STARTED")

    try:
        for attempt_index, nprocs in enumerate(NPROCS_SEQUENCE):
            run_orca(orca_cmd, inp_path, tmp_out, nprocs)

            if already_done(tmp_out):
                break

            if attempt_index == len(NPROCS_SEQUENCE) - 1:
                break

            retry_log = out_path.with_suffix(f".out.failed_n{nprocs}")
            tmp_out.replace(retry_log)

            next_nprocs = NPROCS_SEQUENCE[attempt_index + 1]
            if is_properties_mpi_crash(retry_log):
                print(
                    f"[{datetime.now()}] PID {pid} | {job_name} -> RETRY "
                    f"(PROPERTIES crash, nprocs {nprocs} -> {next_nprocs})"
                )
            else:
                print(
                    f"[{datetime.now()}] PID {pid} | {job_name} -> RETRY "
                    f"(unfinished run, nprocs {nprocs} -> {next_nprocs})"
                )

        tmp_out.replace(out_path)
        duration = time.time() - start_time

        if already_done(out_path):
            print(f"[{datetime.now()}] PID {pid} | {job_name} -> DONE ({duration/60:.2f} min)")
            return job_name, "DONE", duration

        print(f"[{datetime.now()}] PID {pid} | {job_name} -> FAILED ({duration/60:.2f} min)")
        return job_name, "FAILED", duration
    except Exception as exc:
        duration = time.time() - start_time
        print(f"[{datetime.now()}] PID {pid} | {job_name} -> ERROR: {exc}")
        return job_name, "FAILED", duration


def main() -> None:
    cfg = load_config()
    orca_cmd = shutil.which(cfg["orca"]["orca_command"])
    if not orca_cmd:
        raise RuntimeError("ORCA not found in PATH; set full path in config.yaml")

    input_dir = project_root() / "testConformers"
    inputs = sorted(input_dir.glob("*.inp"))
    if not inputs:
        raise RuntimeError(f"No .inp files found in {input_dir}")

    print(f"\n========== ORCA TEST RUN STARTED: {datetime.now()} ==========\n")
    print(f"[INFO] input_dir={input_dir}")
    print(f"[INFO] jobs={len(inputs)}")
    print(f"[INFO] parallel_jobs={N_PARALLEL_JOBS}")
    print(f"[INFO] nprocs_sequence={NPROCS_SEQUENCE}")
    print(f"[INFO] orca_cmd={orca_cmd}")

    joblist = [(inp_path, orca_cmd) for inp_path in inputs]

    done = 0
    failed = 0
    skipped = 0

    with Pool(N_PARALLEL_JOBS) as pool:
        for _, status, _ in pool.imap_unordered(run_one, joblist, chunksize=1):
            if status == "DONE":
                done += 1
            elif status == "SKIPPED":
                skipped += 1
            else:
                failed += 1

    print("\n========== SUMMARY ==========")
    print(f"DONE:    {done}")
    print(f"FAILED:  {failed}")
    print(f"SKIPPED: {skipped}")


if __name__ == "__main__":
    main()
