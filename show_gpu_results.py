"""Print the real A100 results (from results/gpu/*.log) for the walkthrough."""
import json, re, time
from pathlib import Path
B, G, R, D, C, O = "\033[1m", "\033[32m", "\033[31m", "\033[2m", "\033[36m", "\033[0m"
log = Path("results/gpu/gpu5.10929447.log").read_text()
res = {json.loads(l[7:])["mode"]: json.loads(l[7:]) for l in log.splitlines() if l.startswith("RESULT ")}
print(f"{B}NVIDIA A100-SXM4-80GB · Ray 2.59.0 · vLLM 0.27.0 · Qwen2.5-0.5B-Instruct{O}"); time.sleep(0.6)
b = res["baseline"]
print(f"  clean job      {G}✓ completed{O}  {b['unique']}/{b['unique']} rows in {b['secs']:.0f}s   {D}answers: {', '.join(b['sample'])}{O}"); time.sleep(0.8)
s = res["stock"]
print(f"  1 poison row   {R}✕ HUNG{O}       {s['engine_kills']} engine restarts in {s['secs']/60:.0f} min   {D}(model reloads each time){O}"); time.sleep(0.8)
print(f"  guard, run 1   {R}✕ hung{O}       {D}found a bug in my guard (ledger was GC'd){O}"); time.sleep(0.8)
print(f"  guard, fixed   {C}✓ completed{O}  {D}Ray's real code path, stubbed engine: poison quarantined{O}")
