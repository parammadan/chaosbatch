"""Turn results/*.jsonl into the tables used by the README and docs/data.json.
Numbers are only ever read from runs, never typed in.

    python report.py
"""

import json
from pathlib import Path

R = Path("results")
FAULT_NAMES = {"none": "No fault", "transient": "Transient engine crash", "poison": "Poison row",
               "bad_gpu": "Bad GPU (spare node)", "bad_gpu_full": "Bad GPU (packed cluster)",
               "node_loss": "Node loss"}


def latest(kind):
    return [json.loads(l) for l in sorted(R.glob(f"matrix-{kind}-*.jsonl"))[-1].read_text().splitlines()]


def cell(r):
    if r["outcome"] != "completed":
        return f"**{r['outcome'].upper()}** ({r.get('engine_deaths', '?')} engine deaths)"
    s = f"✓ {r['unique']} rows"
    if r.get("error_rows"):
        s += f", {r['error_rows']} error row"
    return s + f" · {r['engine_deaths']} deaths"


def main():
    stock, guard = latest("stock"), latest("guard")
    key = lambda r: (r["config"], r["fault"], r["continue_on_error"])
    g = {key(r): r for r in guard}
    lines = ["| Config | Fault | continue_on_error | Stock Ray 2.59 | With guard |", "|---|---|---|---|---|"]
    for r in stock:
        lines.append(f"| {r['config']} | {FAULT_NAMES[r['fault']]} | {'yes' if r['continue_on_error'] else 'no'} "
                     f"| {cell(r)} | {cell(g[key(r)])} |")
    long_runs = [json.loads(l) for l in (R / "badgpu_long.jsonl").read_text().splitlines()]
    data = {"stock": stock, "guard": guard, "bad_gpu_long": long_runs}
    gpu = sorted(R.glob("gpu-*.json"))
    if gpu:
        data["gpu"] = [json.loads(p.read_text()) for p in gpu]
    Path("docs").mkdir(exist_ok=True)
    Path("docs/data.json").write_text(json.dumps(data, indent=1))
    print("\n".join(lines))
    print()
    for r in long_runs:
        print(f"bad GPU, 20k rows, {'guard' if r['guard'] else 'stock'}: deaths on bad GPU={r['deaths_on_bad_node']}"
              f" restarts={r['actor_starts']} {r['secs']}s rows={r['unique']}")
    hung = [r for r in stock if r["outcome"] == "hung"]
    print(f"\nstock hung cells: {len(hung)}/{len(stock)}; guard completed: "
          f"{sum(r['outcome'] == 'completed' for r in guard)}/{len(guard)}")


if __name__ == "__main__":
    main()
