"""ChaosBatch guard: keep a batch-LLM job alive through poison rows and bad GPUs.

When a vLLM engine dies, every in-flight request fails with it, so the error
alone can't say whether a row or the device is at fault. The guard assigns
blame with two cheap experiments:

  Canary   Each (re)started engine first answers a harmless warm-up prompt.
           If the canary kills it, the device is at fault: no real row was
           involved. After FENCE_AFTER canary deaths, the node's GPU slot is
           reserved by a placeholder actor, so the scheduler can't put an
           engine there again (the "prefer a different node" retry that
           ray-project/ray#65862 asks for, done with existing primitives).

  Isolate  Rows in flight during a death become suspects and are re-run one
           at a time. A row that is alone in flight for QUARANTINE_AFTER
           deaths is the poison: it is emitted as an error row instead of
           being generated, and the job carries on.

Blame state lives in a named Ledger actor on the head node, so it survives
engine restarts.
"""

import asyncio
import os

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from chaosbatch.engine import FakeEngineDeadError, FakeEngineStage, log_event

FENCE_AFTER = 2
QUARANTINE_AFTER = 2
CANARY_ID = -1


@ray.remote(num_cpus=0, resources={"gpu_slot": 1})
class Fence:
    """Holds a GPU slot so no engine can be scheduled onto a broken device."""

    def ping(self):
        return True


class BlameBook:
    """Pure blame logic, kept separate from Ray so it can be unit-tested."""

    def __init__(self):
        self.suspicion, self.alone_deaths, self.canary_deaths = {}, {}, {}
        self.quarantined, self.fenced = set(), set()

    def state(self):
        return {"suspects": [r for r, n in self.suspicion.items() if n and r not in self.quarantined],
                "quarantined": sorted(self.quarantined)}

    def report_death(self, node, rows):
        """An engine died with `rows` in flight (after its canary had passed)."""
        for r in rows:
            self.suspicion[r] = self.suspicion.get(r, 0) + 1
        if len(rows) == 1:
            r = rows[0]
            self.alone_deaths[r] = self.alone_deaths.get(r, 0) + 1
            if self.alone_deaths[r] >= QUARANTINE_AFTER:
                self.quarantined.add(r)

    def report_canary_death(self, node):
        """The warm-up request killed a fresh engine: blame the device. Returns True
        the first time the node crosses the fence threshold."""
        self.canary_deaths[node] = self.canary_deaths.get(node, 0) + 1
        if self.canary_deaths[node] >= FENCE_AFTER and node not in self.fenced:
            self.fenced.add(node)
            return True
        return False


@ray.remote(num_cpus=0)
class Ledger:
    """Named actor on the head node: blame state that survives engine restarts."""

    def __init__(self):
        self.book, self.fences = BlameBook(), {}

    def state(self):
        return self.book.state()

    def report_death(self, node, rows):
        self.book.report_death(node, rows)

    def report_canary_death(self, node):
        if self.book.report_canary_death(node):
            self.fences[node] = Fence.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node, soft=False)).remote()
        return node in self.fences

    def report(self):
        return {"quarantined": sorted(self.book.quarantined), "fenced_nodes": list(self.fences),
                "canary_deaths": self.book.canary_deaths}


class GuardedEngineStage(FakeEngineStage):
    def __init__(self, faults, should_continue_on_error=False, ledger_name="chaosbatch-ledger"):
        super().__init__(faults, should_continue_on_error)
        self.ledger = ray.get_actor(ledger_name)
        self.inflight = set()
        self.alone = asyncio.Lock()
        try:  # canary: a harmless request no real row depends on
            self.generate_sync(CANARY_ID)
        except FakeEngineDeadError:
            fenced = ray.get(self.ledger.report_canary_death.remote(self.node))
            log_event(self.f.events_path, kind="engine_dead", node=self.node, row=CANARY_ID,
                      err=f"canary (fenced={fenced})")
            os._exit(1)

    async def _die(self, rows):
        await self.ledger.report_death.remote(self.node, sorted(rows))
        log_event(self.f.events_path, kind="engine_dead", node=self.node, row=int(sorted(rows)[0]) if rows else -2,
                  err=f"{len(rows)} in flight")
        os._exit(1)

    async def _guarded(self, row_id):
        self.inflight.add(row_id)
        try:
            out = await self._generate(row_id)
            return {"id": row_id, "output": out, "__inference_error__": ""}
        except FakeEngineDeadError:
            await self._die(set(self.inflight))
        finally:
            self.inflight.discard(row_id)

    async def __call__(self, batch):
        self.maybe_transient_crash()  # same fault injection as the stock stage
        st = await self.ledger.state.remote()
        quarantined, suspects = set(st["quarantined"]), set(st["suspects"])
        ids = [int(i) for i in batch["id"]]
        outs = {}
        for r in ids:
            if r in quarantined:
                outs[r] = {"id": r, "output": "",
                           "__inference_error__": f"quarantined: engine died {QUARANTINE_AFTER}x with this row alone"}
        # suspects run one at a time, with nothing else in flight on this engine
        for r in [r for r in ids if r in suspects and r not in outs]:
            async with self.alone:
                while self.inflight:
                    await asyncio.sleep(0.005)
                outs[r] = await self._guarded(r)
        rest = [r for r in ids if r not in outs]
        async with self.alone:
            pass  # don't start normal work while a suspect is running alone
        for o in await asyncio.gather(*(self._guarded(r) for r in rest)):
            outs[o["id"]] = o
        return {k: [outs[r][k] for r in ids] for k in ("id", "output", "__inference_error__")}
