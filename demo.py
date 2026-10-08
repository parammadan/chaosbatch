"""Live view of one run: prints engine deaths and restarts as they happen.

    python demo.py --fault poison [--guard] [--deadline 20]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

RED, GREEN, DIM, BOLD, OFF = "\033[31m", "\033[32m", "\033[2m", "\033[1m", "\033[0m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fault", default="poison")
    ap.add_argument("--guard", action="store_true")
    ap.add_argument("--deadline", type=float, default=20)
    a = ap.parse_args()
    ev = tempfile.mktemp(suffix=".jsonl")
    cmd = [sys.executable, "-m", "chaosbatch.run_one", "--config", "fixed", "--fault", a.fault,
           "--continue-on-error", "--deadline", str(a.deadline)] + (["--guard"] if a.guard else [])
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                            env={**os.environ, "CB_EVENTS": ev})
    print(f"{BOLD}ray batch job · 2,000 rows · 3 vLLM engines · {'guard ON' if a.guard else 'stock Ray 2.59'}{OFF}")
    seen, deaths, t0 = 0, 0, None  # clock starts when the first engine starts, like the job's own timer
    while proc.poll() is None:
        if os.path.exists(ev):
            lines = open(ev).read().splitlines()
            for l in lines[seen:]:
                e = json.loads(l)
                if t0 is None:
                    t0 = e["t"]
                t = e["t"] - t0
                if e["kind"] == "engine_dead":
                    deaths += 1
                    what = "canary" if e.get("row") == -1 and "canary" in e.get("err", "") else (
                        f"row {e['row']}" if e.get("row", -1) >= 0 else "batch")
                    print(f"{RED}  ✕ engine crashed{OFF}  {DIM}({what}){OFF}   crashes: {BOLD}{deaths}{OFF}   {DIM}{t:4.1f}s{OFF}")
                elif e["kind"] == "actor_start" and deaths:
                    print(f"{DIM}  ↻ engine restarted{OFF}")
            seen = len(lines)
        time.sleep(0.1)
    out = proc.stdout.read()
    r = json.loads(next(l for l in out.splitlines() if l.startswith("RESULT "))[7:])
    if r["outcome"] == "completed":
        q = r.get("guard_report", {}).get("quarantined", [])
        print(f"{GREEN}{BOLD}  ✓ completed{OFF}  {r['unique']} rows · {r['error_rows']} error row"
              f"{f' (row {q[0]} quarantined)' if q else ''} · {r['engine_deaths']} crashes · {r['secs']}s")
    else:
        print(f"{RED}{BOLD}  ✕ {r['outcome'].upper()}{OFF}  {r['engine_deaths']} crashes in {r['secs']}s · still not finished")


if __name__ == "__main__":
    main()
