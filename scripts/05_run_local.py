import os
import signal
import shutil
import subprocess
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

from scripts.common import connect_db, get_paths, load_config


def already_done(engine: str, outp: Path) -> bool:
    if not outp.exists():
        return False
    txt = outp.read_text(errors="ignore")
    if engine == "orca":
        return "ORCA TERMINATED NORMALLY" in txt
    if engine == "gaussian":
        return "Normal termination of Gaussian" in txt
    return False


def render_orca_inp(template_inp: Path, nprocs: int, maxcore: int) -> Path:
    txt = template_inp.read_text(encoding="utf-8")

    if "{{NPROCS}}" not in txt:
        raise ValueError(f"Template missing {{NPROCS}}: {template_inp}")
    if "{{MAXCORE}}" not in txt:
        raise ValueError(f"Template missing {{MAXCORE}}: {template_inp}")

    txt = txt.replace("{{NPROCS}}", str(int(nprocs)))
    txt = txt.replace("{{MAXCORE}}", str(int(maxcore)))

    rendered = template_inp.with_suffix(".run.inp")
    rendered.write_text(txt, encoding="utf-8")
    return rendered


def install_interrupt_handlers():
    def handle_interrupt(signum, _frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    for signame in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, signame, None)
        if sig is None:
            continue
        signal.signal(sig, handle_interrupt)


def reconcile_claimed_jobs(con, engine: str, rows) -> tuple[int, int]:
    done_ids = []
    ready_ids = []

    for calc_id, _inp, outp in rows:
        if already_done(engine, Path(outp)):
            done_ids.append((calc_id,))
        else:
            ready_ids.append((calc_id,))

    with con:
        if done_ids:
            con.executemany(
                """
                UPDATE calculation
                SET status='DONE'
                WHERE calc_id=?
                  AND TRIM(status)='RUNNING'
                """,
                done_ids,
            )
        if ready_ids:
            con.executemany(
                """
                UPDATE calculation
                SET status='READY'
                WHERE calc_id=?
                  AND TRIM(status)='RUNNING'
                """,
                ready_ids,
            )

    return len(done_ids), len(ready_ids)


def recover_stale_running_jobs(con, engine: str) -> tuple[int, int]:
    running_rows = con.execute(
        """
        SELECT calc_id, input_path, output_path
        FROM calculation
        WHERE engine=?
          AND TRIM(status)='RUNNING'
        """,
        (engine,),
    ).fetchall()

    if not running_rows:
        return 0, 0

    return reconcile_claimed_jobs(con, engine, running_rows)


def run_one(args):
    (
        calc_id,
        inp,
        outp,
        engine,
        cmd,
        mpi_ranks,
        omp_threads,
        maxcore_mb,
        gaussian_threads,
    ) = args

    inp = Path(inp)
    outp = Path(outp)

    start_time = time.time()
    pid = os.getpid()

    if already_done(engine, outp):
        print(f"[{datetime.now()}] PID {pid} | {calc_id} -> SKIPPED (already finished)")
        return (calc_id, "SKIPPED", 0.0)

    tmp_out = outp.with_suffix(outp.suffix + ".tmp")
    print(f"[{datetime.now()}] PID {pid} | {calc_id} -> STARTED")

    try:
        if engine == "orca":
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(int(omp_threads))

            inp_run = render_orca_inp(inp, nprocs=mpi_ranks, maxcore=maxcore_mb)

            with open(tmp_out, "w") as f:
                subprocess.run(
                    [cmd, inp_run.name],
                    cwd=str(inp_run.parent),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                )

            try:
                inp_run.unlink(missing_ok=True)
            except Exception:
                pass

        else:
            env = os.environ.copy()
            if gaussian_threads is not None:
                env["OMP_NUM_THREADS"] = str(int(gaussian_threads))

            with open(inp, "r") as fin, open(tmp_out, "w") as fout:
                subprocess.run(
                    [cmd],
                    cwd=str(inp.parent),
                    stdin=fin,
                    stdout=fout,
                    stderr=subprocess.STDOUT,
                    env=env,
                )

        tmp_out.replace(outp)

        duration = time.time() - start_time

        if already_done(engine, outp):
            print(f"[{datetime.now()}] PID {pid} | {calc_id} -> DONE ({duration / 60:.2f} min)")
            return (calc_id, "DONE", duration)

        print(f"[{datetime.now()}] PID {pid} | {calc_id} -> FAILED ({duration / 60:.2f} min)")
        return (calc_id, "FAILED", duration)

    except Exception as e:
        duration = time.time() - start_time
        print(f"[{datetime.now()}] PID {pid} | {calc_id} -> ERROR: {e}")
        return (calc_id, "FAILED", duration)


def main():
    print(f"\n========== DFT RUN STARTED: {datetime.now()} ==========\n")
    install_interrupt_handlers()

    cfg = load_config()
    root, work_dir, db_path = get_paths(cfg)
    con = connect_db(db_path)

    engine = str(cfg["engine"]).lower()
    recovered_done, recovered_ready = recover_stale_running_jobs(con, engine)
    if recovered_done or recovered_ready:
        print(
            "[INFO] Recovered stale RUNNING jobs from a previous interrupted run: "
            f"DONE={recovered_done}, READY={recovered_ready}"
        )

    rows = con.execute(
        """
        SELECT calc_id, input_path, output_path
        FROM calculation
        WHERE engine=?
          AND TRIM(status)='READY'
        """,
        (engine,),
    ).fetchall()

    n_parallel_jobs = 8

    mpi_ranks = 8
    omp_threads = 1
    maxcore_mb = 1500

    gaussian_threads = None

    if engine == "orca":
        cmd = shutil.which(cfg["orca"]["orca_command"])
        if not cmd:
            raise RuntimeError("ORCA not found in PATH; set full path in config.yaml")
    else:
        cmd = cfg["gaussian"]["g16_command"]

    print(f"[DEBUG] engine={engine}")
    print(f"[DEBUG] cmd={cmd}")
    print(f"[DEBUG] which(cmd)={shutil.which(cmd)}")
    if engine == "orca":
        print(f"[DEBUG] ORCA runtime: mpi_ranks={mpi_ranks}, omp_threads={omp_threads}, maxcore_mb={maxcore_mb}")
        print(f"[DEBUG] cores_per_job ~= {mpi_ranks * omp_threads}")

    joblist = [
        (
            calc_id,
            Path(inp),
            Path(outp),
            engine,
            cmd,
            mpi_ranks,
            omp_threads,
            maxcore_mb,
            gaussian_threads,
        )
        for calc_id, inp, outp in rows
    ]

    if not joblist:
        con.close()
        print(f"[INFO] No READY {engine} jobs to run.")
        return

    selected_ids = [calc_id for calc_id, _, _ in rows]
    with con:
        con.executemany(
            """
            UPDATE calculation
            SET status='RUNNING'
            WHERE calc_id=?
              AND TRIM(status)='READY'
            """,
            [(calc_id,) for calc_id in selected_ids],
        )

    done = failed = skipped = 0
    commit_every = 10
    processed = 0
    interrupted = False
    fatal_error = None
    pool = Pool(n_parallel_jobs)

    try:
        for calc_id, status, duration in pool.imap_unordered(run_one, joblist, chunksize=1):
            if status == "SKIPPED":
                con.execute(
                    "UPDATE calculation SET status='DONE' WHERE calc_id=?",
                    (calc_id,),
                )
                skipped += 1
            else:
                con.execute(
                    "UPDATE calculation SET status=? WHERE calc_id=?",
                    (status, calc_id),
                )
                if status == "DONE":
                    done += 1
                else:
                    failed += 1

            processed += 1

            if processed % commit_every == 0:
                con.commit()
                print(f"[DB] committed {processed}/{len(joblist)} updates")
    except KeyboardInterrupt:
        interrupted = True
        print("\n[WARN] Interrupted by user. Terminating worker processes...")
        pool.terminate()
    except Exception as exc:
        fatal_error = exc
        print(f"\n[ERROR] Runner aborted: {exc}")
        pool.terminate()
    else:
        pool.close()
    finally:
        pool.join()

    if interrupted or fatal_error is not None:
        recovered_done, recovered_ready = reconcile_claimed_jobs(con, engine, rows)
        con.close()

        print("\n========== SUMMARY ==========")
        print(f"DONE:        {done}")
        print(f"FAILED:      {failed}")
        print(f"SKIPPED:     {skipped}")
        print(f"DONE(sync):  {recovered_done}")
        print(f"READY(sync): {recovered_ready}")
        if interrupted:
            print("Run stopped by user; unfinished jobs were returned to READY")
            return
        raise fatal_error

    con.commit()
    con.close()

    print("\n========== SUMMARY ==========")
    print(f"DONE:    {done}")
    print(f"FAILED:  {failed}")
    print(f"SKIPPED: {skipped}")
    print("Parallel calculations finished")


if __name__ == "__main__":
    main()
