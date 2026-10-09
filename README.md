# ChaosBatch

▶ **Walkthrough video (3.5 min):** [https://youtu.be/pjW7D6AzjsU](https://youtu.be/pjW7D6AzjsU) · **Results:** [parammadan.github.io/chaosbatch](https://parammadan.github.io/chaosbatch/)

**One bad row can hang a Ray batch-LLM job forever.** ChaosBatch reproduces that, measures it
across the configurations Ray users are told to try, and ships a small guard that fixes it
without forking Ray.

## The problem

Ray Data LLM runs vLLM engines as a pool of actors. When an engine dies (vLLM's
`EngineDeadError`, usually a CUDA fault), Ray 2.59 makes the actor exit with `os._exit(1)` so it
restarts, and it retries the batch. That fixed one-off crashes ([#59522](https://github.com/ray-project/ray/issues/59522)).

But if the crash is **caused by a row** (a request that kills the engine every time), the retry
sends the same row back and the engine dies again, in a loop with no exit. That fatal path
runs *before* `should_continue_on_error` is checked
([`vllm_engine_stage.py`](https://github.com/ray-project/ray/blob/master/python/ray/llm/_internal/batch/stages/vllm_engine_stage.py)),
so the setting users turn on to "skip bad rows, data loss is fine" can't help.

Users in the resiliency thread asked for exactly that: *"Is there any way to make the job
continue even if it loses a vLLM actor? Data loss is fine."*
A related open Core issue ([#65862](https://github.com/ray-project/ray/issues/65862), Sept 2026)
covers the device side: retries can land back on the node with the broken GPU.

## Findings (Ray 2.59, simulated engine, local multi-node cluster)

Three configurations: Ray Data LLM's default, the fixed pool recommended in #59522, and a
"tolerant" setup that also enables task retries and `max_errored_blocks=-1`. Each was run against
each fault in its own process, and a run counts as hung if it hasn't finished within 120 s.

| Fault | Stock Ray 2.59 (all 3 configs) | With the ChaosBatch guard |
|---|---|---|
| **Poison row** (with or without `should_continue_on_error`) | **hung in 6 of 6 runs**: 24 engine deaths in 120 s, never finishes | **completed 6 of 6**: all 2,000 rows, exactly 1 error row (the poison), 3 engine deaths |
| Bad GPU, long job (20k rows, packed cluster) | completes, but the broken GPU crash-loops **7–9×**; 17.6–19.5 s | fenced after **2** deaths; 12.5–13.1 s |
| Transient crash · node loss · no fault | all complete | all complete (the guard breaks nothing) |

Full 21-cell matrix: `python report.py`. Raw runs are in `results/`.

On real GPUs every restart also reloads the model weights, so a crash loop that costs
seconds here costs minutes per iteration in production.

### Real vLLM on a GPU (NVIDIA A100, NU Explorer)

Ray 2.59.0 + vLLM 0.27.0 + Qwen2.5-0.5B-Instruct. Explorer's A100 driver is CUDA 12.8 and vLLM 0.27
ships CUDA 13 builds only, so the runs use NVIDIA's CUDA forward-compatibility library.

| Run | Outcome |
|---|---|
| Clean job, no fault | **completed**: 400/400 rows in 148 s (real answers: "Green.", "Red.", …) |
| One poison prompt, stock Ray | **hung**: 2 engine restarts in the 4-minute cap; each restart reloads the model (~2 min) |
| Guard, first GPU run | **hung**: exposed a bug in my guard (below) |

The poison is real: a test-only hook kills vLLM's own EngineCore process when it sees a marker
prompt, so vLLM raises a genuine `EngineDeadError` and Ray's own handler takes it from there.

**The guard has not yet been confirmed on a real GPU.** It is verified on Ray's real code path
with only the engine stubbed (`chaosbatch/cpu_harness.py`, below). Further GPU runs were stopped
because this stack (Ray startup, the data pipeline, and vLLM initialization) leaves the GPU idle
for minutes before any work, which triggered the cluster's idle-GPU warnings.

### Ray's real code path, engine stubbed (CPU harness)

`cpu_test.py` runs the real `ray.data.llm` pipeline, the real `vLLMEngineStageUDF` and the real
guard patch, with a fake engine that raises `EngineDeadError` on the poison and stays dead, like a
killed EngineCore. Ray gets one logical GPU so the stage schedules on a laptop.

| Mode | Outcome |
|---|---|
| baseline | completed: 300/300 rows |
| stock | **hung**: 10–17 engine kills in 90 s |
| guard | **completed**: 300/300 rows, row 137 returned as an error row, 4 engine kills, ledger quarantined "137" |

## The guard

When an engine dies, every in-flight request fails with it, so the error alone can't say
whether a row or the device is at fault. The guard assigns blame with two cheap experiments:

- **Canary.** Each (re)started engine first answers a harmless warm-up prompt. If that kills it,
  the device is at fault, since no real row was involved. After 2 canary deaths the node's GPU slot
  is **fenced**: a zero-CPU placeholder actor reserves it, so the scheduler can't put an engine
  there again. That's the "prefer a different node" behavior #65862 asks for, built from
  existing Ray primitives.
- **Isolate.** Rows in flight during a death become suspects and are re-run one at a time. A row
  that is alone in flight for 2 deaths is the poison: it's emitted as an error row in Ray's own
  error-row schema, and the job carries on.

Blame state lives in a `Ledger` actor pinned to the head node. Every Ledger call is fail-safe:
if it's unreachable, the guard behaves exactly like stock Ray.

Two implementations:
- `chaosbatch/guard.py`: on the simulated engine, used for the matrix
- `chaosbatch/ray_patch.py`: on Ray's real `vLLMEngineStageUDF`, installed with
  `runtime_env={"worker_process_setup_hook": "chaosbatch.ray_patch.install"}` and
  `should_continue_on_error=True`

## Mistakes caught along the way

- My first guess was that a bad GPU kills the job. It doesn't: healthy engines absorb the work.
  Its real cost is a crash-reload loop for the rest of the job, so that's what the guard's
  GPU fence is measured against.
- The guarded engine first skipped the transient-fault injection (it overrode the method that
  injects it), which made the guard look better than it was. Fixed, and both matrices re-run.
- A node-loss test removed the node after the job had already finished, so it measured nothing.
  The timing was fixed.
- The guard itself caused one failure stock Ray didn't have: its Ledger landed on the node that
  was removed. The Ledger is now pinned to the head node and every call is fail-safe. Then 9 of 9
  repeat runs passed under node loss.
- **The fail-safe hid a bug.** On the first GPU run the guard hung. The CPU harness showed why: the
  test scripts never kept a handle to the Ledger actor, so Ray garbage-collected it immediately, and
  the fail-safe quietly fell back to stock behaviour. The simulator kept the handle, which is why it
  passed there. The fail-safe now logs `chaosbatch guard DEGRADED` instead of failing silently.
- **Yield under a lock.** The second GPU run failed with `RuntimeError: release unlocked lock`: the
  guard yielded from Ray's async-generator UDF while holding an asyncio lock, and Ray finalized the
  generator from another task. It now computes inside the lock and yields after releasing it.
- **Error rows skip postprocess.** My test reported no error rows although the poison was labelled
  correctly: Ray deliberately passes error rows through without running the user's postprocess.

## What this is not

- The matrix uses a simulated engine. The failure semantics (actor exit, restart, task retry,
  Ray Data's pool) are real Ray 2.59, but the model is a sleep.
- The guard is verified on Ray's real code path with a stubbed engine, not yet on a real GPU.
- Anyscale's GPU Health Observability (private preview) targets hardware faults through DCGM
  telemetry. The canary is a behavioral signal and would complement it, not replace it.
- 2 runs per long bad-GPU configuration: treat the speedup as indicative.

## Run it

```bash
python3.12 -m venv .venv && .venv/bin/pip install "ray[data]==2.59.0" pandas pyarrow pytest
.venv/bin/python -m pytest tests                        # blame logic
.venv/bin/python -m chaosbatch.run_one --config fixed --fault poison --continue-on-error            # hangs
.venv/bin/python -m chaosbatch.run_one --config fixed --fault poison --continue-on-error --guard    # completes
.venv/bin/python matrix.py && .venv/bin/python matrix.py --guard && .venv/bin/python report.py
```
