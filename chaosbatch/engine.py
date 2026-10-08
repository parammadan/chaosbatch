"""A stand-in for Ray Data LLM's vLLM engine stage, with injectable faults.

The failure handling mirrors ray/llm/_internal/batch/stages/vllm_engine_stage.py
(Ray 2.59): a *fatal* engine error (vLLM's EngineDeadError) makes the actor
call os._exit(1) so Ray restarts it and retries the task; any other error is
either raised or, with should_continue_on_error, turned into an error row.
Only the model is fake: each "generation" is a short sleep.

Faults (set per run through FaultSpec, shipped to actors as a dict):
  poison_ids    rows that kill the engine every time they are generated
                (a request that deterministically triggers a CUDA fault)
  bad_nodes     node ids whose "GPU" kills the engine on first use, every time
                an actor lands there (a broken device)
  transient_p   chance that a batch kills the engine once (a random CUDA error)
"""

import asyncio
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field

import ray


class FakeEngineDeadError(RuntimeError):
    """Plays the role of vllm.v1.engine.exceptions.EngineDeadError."""


@dataclass
class FaultSpec:
    poison_ids: list = field(default_factory=list)
    bad_nodes: list = field(default_factory=list)
    transient_p: float = 0.0
    gen_ms: int = 20
    events_path: str = ""          # JSONL log of actor starts and engine deaths

    def to_dict(self):
        return asdict(self)


def log_event(path, **kw):
    if path:
        with open(path, "a") as f:
            f.write(json.dumps({"t": time.time(), "pid": os.getpid(), **kw}) + "\n")


class FakeEngineStage:
    """Callable class for ds.map_batches, like vLLMEngineStageUDF."""

    def __init__(self, faults: dict, should_continue_on_error: bool = False):
        self.f = FaultSpec(**faults)
        self.should_continue_on_error = should_continue_on_error
        self.node = ray.get_runtime_context().get_node_id()
        self.poison = set(self.f.poison_ids)
        log_event(self.f.events_path, kind="actor_start", node=self.node)

    def _check_faults(self, row_id):
        if self.node in self.f.bad_nodes:
            raise FakeEngineDeadError("CUDA error: an illegal memory access was encountered")
        if row_id in self.poison:
            raise FakeEngineDeadError(f"engine crashed while generating row {row_id}")

    def generate_sync(self, row_id):
        time.sleep(self.f.gen_ms / 1000)
        self._check_faults(row_id)
        return f"answer-{row_id}"

    async def _generate(self, row_id):
        await asyncio.sleep(self.f.gen_ms / 1000)
        if self.node in self.f.bad_nodes:
            raise FakeEngineDeadError("CUDA error: an illegal memory access was encountered")
        if row_id in self.poison:
            raise FakeEngineDeadError(f"engine crashed while generating row {row_id}")
        return f"answer-{row_id}"

    async def _one(self, row_id):
        try:
            return {"id": row_id, "output": await self._generate(row_id), "__inference_error__": ""}
        except FakeEngineDeadError as e:
            # Same as Ray: exit the actor so it restarts with a fresh engine.
            log_event(self.f.events_path, kind="engine_dead", node=self.node, row=int(row_id), err=str(e))
            os._exit(1)
        except Exception as e:  # noqa: BLE001
            if not self.should_continue_on_error:
                raise
            return {"id": row_id, "output": "", "__inference_error__": f"{type(e).__name__}: {e}"}

    def maybe_transient_crash(self):
        """A random CUDA error: kills the engine once, not tied to any row or device."""
        if self.f.transient_p and random.random() < self.f.transient_p:
            log_event(self.f.events_path, kind="engine_dead", node=self.node, row=-1, err="transient")
            os._exit(1)

    async def __call__(self, batch):
        self.maybe_transient_crash()
        outs = await asyncio.gather(*(self._one(int(i)) for i in batch["id"]))
        return {k: [o[k] for o in outs] for k in ("id", "output", "__inference_error__")}
