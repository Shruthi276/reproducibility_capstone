"""
Rewrites absolute artifact paths in mlflow.db to proxied mlflow-artifacts:/ URIs.

WHY THIS IS NEEDED
------------------
MLflow stores artifact_location per experiment at creation time, and every run
inherits it as artifact_uri. If the server originally ran without
--serve-artifacts, those values are absolute paths on the server's own
filesystem, e.g. file:///home/shruthi-rathod/.../mlruns/1/<run>/artifacts.

A remote client then tries to mkdir that path on ITS OWN machine and fails with
PermissionError. Restarting the server with --serve-artifacts fixes new
experiments but leaves the existing rows untouched, which is why the error
persists after a restart.

This script rewrites them. Run with the MLflow server STOPPED.

    python fix_artifact_paths.py --db mlflow.db --dry-run
    python fix_artifact_paths.py --db mlflow.db

Then restart with:

    mlflow server \
      --backend-store-uri sqlite:///mlflow.db \
      --artifacts-destination ./mlartifacts \
      --serve-artifacts \
      --host 0.0.0.0 --port 5000 \
      --allowed-hosts '*' --cors-allowed-origins '*'
"""

import argparse
import os
import shutil
import sqlite3
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="mlflow.db")
    parser.add_argument("--artifacts-dir", default="./mlartifacts")
    parser.add_argument("--old-dir", default="./mlruns")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        sys.exit(f"ERROR: {args.db} not found. Run this from the directory "
                 f"containing mlflow.db.")

    if not args.dry_run and not args.no_backup:
        backup = args.db + ".backup"
        shutil.copy2(args.db, backup)
        print(f"Backed up {args.db} -> {backup}")

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()

    # ---- show current state -------------------------------------------
    print("\n=== BEFORE ===")
    print("experiments:")
    for row in cur.execute(
        "SELECT experiment_id, name, artifact_location FROM experiments"
    ):
        print(f"  {row[0]}  {row[1]:<32} {row[2]}")

    cur.execute("SELECT COUNT(*) FROM runs WHERE artifact_uri LIKE 'file:%'")
    n_runs = cur.fetchone()[0]
    print(f"runs with absolute file: URIs: {n_runs}")

    cur.execute(
        "SELECT COUNT(*) FROM experiments WHERE artifact_location LIKE 'file:%'"
    )
    n_exp = cur.fetchone()[0]
    print(f"experiments with absolute file: locations: {n_exp}")

    if n_runs == 0 and n_exp == 0:
        print("\nNothing to fix. Paths are already proxied.")
        conn.close()
        return

    if args.dry_run:
        print("\nDRY RUN. Nothing written. Re-run without --dry-run to apply.")
        conn.close()
        return

    # ---- rewrite -------------------------------------------------------
    cur.execute(
        "UPDATE experiments "
        "SET artifact_location = 'mlflow-artifacts:/' || experiment_id "
        "WHERE artifact_location LIKE 'file:%'"
    )
    print(f"\nUpdated {cur.rowcount} experiment rows.")

    cur.execute(
        "UPDATE runs "
        "SET artifact_uri = 'mlflow-artifacts:/' || experiment_id "
        "                   || '/' || run_uuid || '/artifacts' "
        "WHERE artifact_uri LIKE 'file:%'"
    )
    print(f"Updated {cur.rowcount} run rows.")

    conn.commit()

    # ---- show new state ------------------------------------------------
    print("\n=== AFTER ===")
    print("experiments:")
    for row in cur.execute(
        "SELECT experiment_id, name, artifact_location FROM experiments"
    ):
        print(f"  {row[0]}  {row[1]:<32} {row[2]}")

    print("sample runs:")
    for row in cur.execute("SELECT run_uuid, artifact_uri FROM runs LIMIT 5"):
        print(f"  {row[0]}  {row[1]}")

    conn.close()

    # ---- move existing artifact files ----------------------------------
    if os.path.isdir(args.old_dir):
        os.makedirs(args.artifacts_dir, exist_ok=True)
        moved = 0
        for entry in os.listdir(args.old_dir):
            src = os.path.join(args.old_dir, entry)
            dst = os.path.join(args.artifacts_dir, entry)
            if os.path.isdir(src) and not os.path.exists(dst):
                shutil.copytree(src, dst)
                moved += 1
        print(f"\nCopied {moved} experiment directories "
              f"{args.old_dir} -> {args.artifacts_dir}")
    else:
        os.makedirs(args.artifacts_dir, exist_ok=True)
        print(f"\n{args.old_dir} not found; created empty {args.artifacts_dir}")

    print("\nDone. Restart the server with --serve-artifacts.")


if __name__ == "__main__":
    main()
