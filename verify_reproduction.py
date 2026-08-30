"""
AIOps Module 1 - Question 4, Partner B

Compares Partner B's reproduction run with Partner A's original run and
writes a note into Partner A's MLflow run description.

The note goes into the reserved tag mlflow.note.content, which MLflow renders
as the run's Description panel. Any existing note is preserved.
"""

import argparse
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import mlflow
from mlflow import MlflowClient


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


p = argparse.ArgumentParser()
p.add_argument("--original-run-id", required=True)
p.add_argument("--my-run-id", required=True)
p.add_argument("--metric", default=None)
p.add_argument("--tolerance", type=float, default=0.005)
p.add_argument("--explanation", default="")
p.add_argument("--deviation", default="")
p.add_argument("--tracking-uri",
               default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
args = p.parse_args()

mlflow.set_tracking_uri(args.tracking_uri)
client = MlflowClient()
original = client.get_run(args.original_run_id)
mine = client.get_run(args.my_run_id)

metric = args.metric
if not metric:
    shared = set(original.data.metrics) & set(mine.data.metrics)
    if not shared:
        raise SystemExit(
            "No metric names in common.\n"
            "  Partner A: " + str(sorted(original.data.metrics)) + "\n"
            "  Partner B: " + str(sorted(mine.data.metrics))
        )
    preferred = [m for m in shared if "acc" in m.lower()]
    metric = sorted(preferred or shared)[0]
    print("Auto-selected metric: " + metric + "   (shared: " + str(sorted(shared)) + ")")

a_val = original.data.metrics[metric]
b_val = mine.data.metrics[metric]
delta = abs(a_val - b_val)
matched = delta <= args.tolerance
status = "MATCH" if matched else "MISMATCH"

tags = original.data.tags
env_rows = []
for key, mine_val in [("python_version", sys.version.split()[0]),
                      ("platform", platform.platform())]:
    a_v = tags.get(key, "not logged")
    same = "same" if a_v == mine_val else "DIFFERENT"
    env_rows.append("| " + key + " | `" + a_v + "` | `" + mine_val + "` | " + same + " |")

param_diffs = []
for k, v in sorted(original.data.params.items()):
    mine_v = mine.data.params.get(k, "<missing>")
    if str(v) != str(mine_v):
        param_diffs.append("| " + k + " | `" + str(v) + "` | `" + str(mine_v) + "` |")

nl = chr(10)
stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

if not param_diffs:
    params_block = "None. All parameters identical."
else:
    params_block = ("| Param | A | B |" + nl + "|---|---|---|" + nl
                    + nl.join(param_diffs))

if matched:
    conclusion = ("The metric reproduced within the stated tolerance. Code (Git), "
                  "data (DVC) and run configuration (MLflow) were all pinned "
                  "successfully by the protocol.")
else:
    conclusion = args.explanation or "DISCREPANCY. Root cause analysis required."

deviation_block = ""
if args.deviation:
    deviation_block = nl + "**Deviation from the protocol:** " + args.deviation + nl

note = (
    "### Reproduction check, Partner B (raghava137), " + stamp + nl + nl
    + "**Result: " + status + "**" + nl + nl
    + "| | Partner A | Partner B |" + nl
    + "|---|---|---|" + nl
    + "| `" + metric + "` | " + format(a_val, ".6f") + " | " + format(b_val, ".6f") + " |" + nl + nl
    + "- Absolute delta: **" + format(delta, ".6f") + "**" + nl
    + "- Stated tolerance: +/-" + str(args.tolerance) + nl
    + "- Verification run_id: `" + mine.info.run_id + "`" + nl
    + "- Partner A git_commit: `" + tags.get("git_commit", "not logged") + "`" + nl
    + "- Partner B checked-out commit: `" + git_commit() + "`" + nl + nl
    + "**Environment comparison**" + nl + nl
    + "| Item | Partner A | Partner B | |" + nl
    + "|---|---|---|---|" + nl
    + nl.join(env_rows) + nl + nl
    + "**Parameter differences**" + nl + nl
    + params_block + nl + nl
    + "**Reproduction method, no communication with Partner A about environment or data:**" + nl + nl
    + "```" + nl
    + "git clone https://github.com/Shruthi276/reproducibility_capstone.git" + nl
    + "git checkout " + tags.get("git_commit", "<commit>") + nl
    + "dvc remote modify --local myremote profile partnera" + nl
    + "dvc pull && dvc checkout" + nl
    + "conda env create -f environment.yml" + nl
    + "python src/train.py" + nl
    + "```" + nl
    + deviation_block + nl
    + "**Conclusion:** " + conclusion
)

existing = tags.get("mlflow.note.content", "")
if existing:
    client.set_tag(args.original_run_id, "mlflow.note.content",
                   existing + nl + nl + "---" + nl + nl + note)
else:
    client.set_tag(args.original_run_id, "mlflow.note.content", note)

for k, v in [("repro.verified_by", "partner_b_raghava137"),
             ("repro.status", status),
             ("repro.delta", format(delta, ".6f")),
             ("repro.tolerance", str(args.tolerance)),
             ("repro.metric", metric),
             ("repro.run_id", mine.info.run_id)]:
    client.set_tag(args.original_run_id, k, v)

client.set_tag(mine.info.run_id, "repro.source_run_id", args.original_run_id)
client.set_tag(mine.info.run_id, "role", "partner_b_reproduction")

bar = "=" * 62
print(bar)
print("  " + status)
print("  Partner A " + metric + ": " + format(a_val, ".6f"))
print("  Partner B " + metric + ": " + format(b_val, ".6f"))
print("  delta = " + format(delta, ".6f") + "   (tolerance +/-" + str(args.tolerance) + ")")
print(bar)

with open("REPRODUCTION_NOTE.md", "w") as f:
    f.write(note + chr(10))
print("Saved REPRODUCTION_NOTE.md")
