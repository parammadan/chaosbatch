"""Real Ray Data LLM + vLLM on GPUs: does one poison prompt hang the job, and
does the ChaosBatch guard fix it?  Runs on 1 GPU (Explorer allows one per job).

    python gpu_test.py --mode baseline|stock|guard --deadline 900
"""

import argparse
import json
import os
import threading
import time

import ray
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig

MARKER = "__CHAOS_POISON__"
N, POISON_ID = 400, 137
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["baseline", "stock", "guard"])
    ap.add_argument("--deadline", type=float, default=900)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = ap.parse_args()

    events = f"{HERE}/results/gpu-{args.mode}-{int(time.time())}.events"
    os.makedirs(f"{HERE}/results", exist_ok=True)
    env = {"PYTHONPATH": HERE, "CHAOS_EVENTS": events}
    if args.mode != "baseline":
        env["CHAOS_POISON_MARKER"] = MARKER
    runtime_env = {"env_vars": env}
    # the poison injector lives in the same hook; stock mode installs ONLY the injector
    os.environ.update(env)
    runtime_env["worker_process_setup_hook"] = ("chaosbatch.ray_patch.install" if args.mode == "guard"
                                                else "chaosbatch.ray_patch.install_injector_only")
    ray.init(runtime_env=runtime_env)
    if args.mode == "guard":
        from chaosbatch.guard import start_ledger
        start_ledger()

    rows = [{"id": i, "prompt": (MARKER + " " if i == POISON_ID else "") + f"In one word, what color is fruit number {i}?"}
            for i in range(N)]
    cfg = vLLMEngineProcessorConfig(
        model_source=args.model, concurrency=1, batch_size=16, should_continue_on_error=True,
        engine_kwargs={"max_model_len": 1024, "enforce_eager": True, "gpu_memory_utilization": 0.7},
        runtime_env=runtime_env)
    proc = build_llm_processor(
        cfg,
        preprocess=lambda r: dict(id=r["id"], prompt=r["prompt"],
                                  sampling_params=dict(max_tokens=8, temperature=0)),
        postprocess=lambda r: dict(id=r["id"], out=r.get("generated_text", ""),
                                   err=r.get("__inference_error__", "")))

    t0 = time.time()

    def report(outcome, out=None, err=None):
        kills = sum(1 for _ in open(events)) if os.path.exists(events) else 0
        res = {"mode": args.mode, "outcome": outcome, "secs": round(time.time() - t0, 1),
               "engine_kills": kills, "error": err}
        if out is not None:
            ids = [r["id"] for r in out]
            res.update(rows=len(out), unique=len(set(ids)), missing=N - len(set(ids)),
                       error_rows=[r["id"] for r in out if r["err"]],
                       sample=[r["out"] for r in out[:3]])
        print("RESULT " + json.dumps(res), flush=True)

    def on_deadline():
        report("hung")
        os._exit(0)
    threading.Timer(args.deadline, on_deadline).start()
    try:
        report("completed", out=proc(ray.data.from_items(rows)).take_all())
    except Exception as e:  # noqa: BLE001
        report("failed", err=f"{type(e).__name__}: {str(e).splitlines()[0][:300]}")
    os._exit(0)


if __name__ == "__main__":
    main()
