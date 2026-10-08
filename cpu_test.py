"""Real Ray Data LLM pipeline + real guard, fake engine, on CPU.

    python cpu_test.py --mode stock|guard [--deadline 120]
"""

import argparse
import json
import os
import threading
import time

import ray
from ray.data.llm import build_processor, vLLMEngineProcessorConfig

MARKER, N, POISON_ID = "__CHAOS_POISON__", 300, 137
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["baseline", "stock", "guard"])
    ap.add_argument("--deadline", type=float, default=120)
    a = ap.parse_args()
    events = f"/tmp/cb-cpu-{a.mode}-{int(time.time())}.events"
    env = {"PYTHONPATH": HERE, "CHAOS_EVENTS": events, "CB_GUARD": "1" if a.mode == "guard" else "0"}
    if a.mode != "baseline":
        env["CHAOS_POISON_MARKER"] = MARKER
    os.environ.update(env)
    rt = {"env_vars": env, "worker_process_setup_hook": "chaosbatch.cpu_harness.install"}
    ray.init(num_gpus=1, runtime_env=rt, log_to_driver=bool(os.environ.get("CB_VERBOSE")))
    if a.mode == "guard":
        from chaosbatch.guard import start_ledger
        ledger = start_ledger()  # keep the handle, or Ray garbage-collects the actor
    rows = [{"id": i, "prompt": (MARKER + " " if i == POISON_ID else "") + f"question {i}"} for i in range(N)]
    cfg = vLLMEngineProcessorConfig(
        model_source="Qwen/Qwen2.5-0.5B-Instruct", concurrency=1, batch_size=16, should_continue_on_error=True,
        chat_template_stage=False, tokenize_stage=False, detokenize_stage=False, runtime_env=rt)
    proc = build_processor(cfg,
        preprocess=lambda r: dict(id=r["id"], prompt=r["prompt"], sampling_params=dict(max_tokens=8)),
        postprocess=lambda r: dict(id=r["id"], out=r.get("generated_text", ""), err=r.get("__inference_error__", ""),
                                   keys=",".join(sorted(k for k in r if not k.startswith("_")))[:300]))
    t0 = time.time()

    def report(outcome, out=None, err=None):
        kills = sum(1 for _ in open(events)) if os.path.exists(events) else 0
        res = {"mode": a.mode, "outcome": outcome, "secs": round(time.time() - t0, 1), "engine_kills": kills, "error": err}
        if out is not None:
            ids = [r["id"] for r in out]
            res.update(rows=len(out), unique=len(set(ids)), error_rows=[r["id"] for r in out if r.get("err") or r.get("__inference_error__")],
                       poison_row=next((r for r in out if r["id"] == POISON_ID), None))
        if a.mode == "guard":
            try:
                res["ledger"] = ray.get(ray.get_actor("chaosbatch-ledger").report.remote(), timeout=5)
            except Exception as e:  # noqa: BLE001
                res["ledger"] = repr(e)[:100]
        print("RESULT " + json.dumps(res), flush=True)
    threading.Timer(a.deadline, lambda: (report("hung"), os._exit(0))).start()
    try:
        report("completed", out=proc(ray.data.from_items(rows)).take_all())
    except Exception as e:  # noqa: BLE001
        report("failed", err=f"{type(e).__name__}: {str(e).splitlines()[0][:300]}")
    os._exit(0)


if __name__ == "__main__":
    main()
