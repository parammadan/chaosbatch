"""Survival matrix: every config against every fault, each in its own process.

A run that doesn't finish within TIMEOUT seconds is recorded as "hung".

    python matrix.py                  # stock Ray behaviour
    python matrix.py --guard          # with the ChaosBatch guard
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

TIMEOUT = 180  # hard kill; run_one reports "hung" itself at --deadline (120 s)
CELLS = [
    ("none", False), ("transient", False), ("poison", False),
    ("poison", True), ("bad_gpu", True), ("bad_gpu_full", True), ("node_loss", True),
]


def run(config, fault, coe, guard):
    cmd = [sys.executable, "-m", "chaosbatch.run_one", "--config", config, "--fault", fault]
    cmd += ["--continue-on-error"] * coe + ["--guard"] * guard
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        res = json.loads(line[7:]) if line else {"outcome": "crashed", "error": p.stderr.strip().splitlines()[-1][:200] if p.stderr else ""}
    except subprocess.TimeoutExpired:
        res = {"outcome": "hung", "secs": TIMEOUT}
    subprocess.run([str(Path(sys.executable).parent / "ray"), "stop", "--force"], capture_output=True)
    res.update(config=config, fault=fault, continue_on_error=coe, guard=guard, wall=round(time.time() - t0, 1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guard", action="store_true")
    ap.add_argument("--configs", default="default,fixed,tolerant")
    args = ap.parse_args()
    out = Path("results") / f"matrix-{'guard' if args.guard else 'stock'}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out.parent.mkdir(exist_ok=True)
    for config in args.configs.split(","):
        for fault, coe in CELLS:
            r = run(config, fault, coe, args.guard)
            with open(out, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"{config:9} {fault:10} coe={int(coe)}  {r['outcome']:9} rows={r.get('unique', '-')}"
                  f" missing={r.get('missing', '-')} dup={r.get('duplicates', '-')} err_rows={r.get('error_rows', '-')}"
                  f" deaths={r.get('engine_deaths', '-')} starts={r.get('actor_starts', '-')} {r.get('secs', '-')}s"
                  + (f"  [{r['error']}]" if r.get("error") else ""), flush=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
