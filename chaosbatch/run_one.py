"""Run one (config, fault) cell on a local multi-node Ray cluster; print JSON.

Each worker node has one "gpu_slot" resource, so each engine actor sits on its
own node, like one GPU per node. A spare node is added so a restarted actor
*could* move away from a bad node if the scheduler chose to.

    python -m chaosbatch.run_one --config fixed --fault poison
"""

import argparse
import json
import os
import tempfile
import threading
import time
from collections import Counter

import ray
from ray.cluster_utils import Cluster

from chaosbatch.engine import FakeEngineStage, FaultSpec

ROWS = 2000
ENGINES = 3          # actors the job asks for
WORKER_NODES = 4     # one spare slot


CONFIGS = {
    # what ray.data.llm does by default: int concurrency -> autoscaling pool (1, n)
    "default": dict(min_size=1, max_size=ENGINES, ctx={}),
    # advice from #59522: fixed pool, start as soon as one actor is ready, retry init forever
    "fixed": dict(min_size=ENGINES, max_size=ENGINES,
                  ctx={"wait_for_min_actors_s": -1, "actor_init_max_retries": -1}),
    # advice from #59522: also retry tasks on errors and tolerate errored blocks
    "tolerant": dict(min_size=ENGINES, max_size=ENGINES,
                     ctx={"wait_for_min_actors_s": -1, "actor_init_max_retries": -1,
                          "actor_task_retry_on_errors": True, "max_errored_blocks": -1}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default", choices=CONFIGS)
    ap.add_argument("--fault", default="none",
                    choices=["none", "transient", "poison", "bad_gpu", "bad_gpu_full", "node_loss"])
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--guard", action="store_true", help="use the ChaosBatch guard")
    ap.add_argument("--rows", type=int, default=ROWS)
    ap.add_argument("--deadline", type=float, default=120,
                    help="seconds before the run is declared hung (events are still reported)")
    args = ap.parse_args()

    cluster = Cluster(initialize_head=True, head_node_args={"num_cpus": 4})
    # bad_gpu_full: every GPU slot is taken by an engine, as on a packed paid cluster
    n_workers = ENGINES if args.fault == "bad_gpu_full" else WORKER_NODES
    workers = [cluster.add_node(num_cpus=2, resources={"gpu_slot": 1}) for _ in range(n_workers)]
    cluster.wait_for_nodes()
    ray.init(address=cluster.address, log_to_driver=False)

    events = tempfile.mktemp(suffix=".jsonl")
    faults = FaultSpec(events_path=events)
    if args.fault == "transient":
        faults.transient_p = 0.05
    elif args.fault == "poison":
        faults.poison_ids = [777]
    elif args.fault in ("bad_gpu", "bad_gpu_full"):
        faults.bad_nodes = [workers[0].node_id]
    if args.fault == "node_loss":
        threading.Timer(1.5, lambda: cluster.remove_node(workers[1], allow_graceful=False)).start()

    cfg = CONFIGS[args.config]
    ctx = ray.data.DataContext.get_current()
    for k, v in cfg["ctx"].items():
        setattr(ctx, k, v)

    stage = FakeEngineStage
    kwargs = dict(faults=faults.to_dict(), should_continue_on_error=args.continue_on_error)
    if args.guard:
        from chaosbatch.guard import GuardedEngineStage, start_ledger
        ledger = start_ledger()
        stage, kwargs = GuardedEngineStage, dict(kwargs, ledger_name="chaosbatch-ledger")

    t0 = time.time()
    out, err = None, None

    def summarize(outcome, err=None, out=None):
        ev = [json.loads(l) for l in open(events)] if os.path.exists(events) else []
        res = {"config": args.config, "fault": args.fault, "continue_on_error": args.continue_on_error,
               "guard": args.guard, "outcome": outcome, "error": err, "secs": round(time.time() - t0, 1),
               "actor_starts": sum(e["kind"] == "actor_start" for e in ev),
               "engine_deaths": sum(e["kind"] == "engine_dead" for e in ev),
               "deaths_on_bad_node": sum(e["kind"] == "engine_dead" and e["node"] in faults.bad_nodes for e in ev)}
        if out is not None:
            ids = Counter(r["id"] for r in out)
            res.update(rows_out=len(out), unique=len(ids), missing=args.rows - len(ids),
                       duplicates=sum(c - 1 for c in ids.values() if c > 1),
                       error_rows=sum(1 for r in out if r["__inference_error__"]))
        if args.guard:
            try:
                res["guard_report"] = ray.get(ray.get_actor("chaosbatch-ledger").report.remote(), timeout=5)
            except Exception:  # noqa: BLE001
                pass
        return res

    def on_deadline():
        print("RESULT " + json.dumps(summarize("hung")), flush=True)
        os._exit(0)
    threading.Timer(args.deadline, on_deadline).start()
    try:
        ds = ray.data.range(args.rows, override_num_blocks=max(40, args.rows // 50)).map_batches(
            stage,
            fn_constructor_kwargs=kwargs,
            batch_size=50,
            compute=ray.data.ActorPoolStrategy(min_size=cfg["min_size"], max_size=cfg["max_size"],
                                               max_tasks_in_flight_per_actor=4),
            max_concurrency=4,
            resources={"gpu_slot": 1},
            num_cpus=0.5,
        )
        out = ds.take_all()
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    res = summarize("completed" if err is None else "failed", err, out)
    print("RESULT " + json.dumps(res), flush=True)
    os._exit(0)  # skip slow teardown; the matrix runs `ray stop` after each cell


if __name__ == "__main__":
    main()
