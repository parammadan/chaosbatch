"""The ChaosBatch guard for the REAL Ray Data LLM vLLM stage (Ray 2.59).

Installed in each engine actor via runtime_env={"worker_process_setup_hook":
"chaosbatch.ray_patch.install"}. It replaces vLLMEngineStageUDF.udf and adds
nothing to the request path except bookkeeping:

  first batch   wrap self.llm.generate_async once, so a fatal EngineDeadError
                reports the in-flight rows to the Ledger before Ray's own
                handler calls os._exit(1); then run a canary request
  every batch   quarantined rows -> error rows in Ray's own error-row schema;
                suspects run one at a time; everything else runs as before

Use with should_continue_on_error=True (that's what adds the error column).
Row identity across retries comes from an "id" column in the input.

Test-only: if CHAOS_POISON_MARKER is set, a request whose prompt contains it
kills vLLM's EngineCore child process before submission, so vLLM raises a
genuine EngineDeadError, the same path a CUDA fault takes.
"""

import asyncio
import logging
import os
import uuid

log = logging.getLogger("chaosbatch")

LEDGER = "chaosbatch-ledger"
CANARY_PROMPT = "Hi"


def _kill_engine_core():
    import psutil
    for child in psutil.Process().children(recursive=True):
        try:
            if "EngineCore" in child.name() or "EngineCore" in " ".join(child.cmdline()):
                child.kill()
        except psutil.Error:
            pass


def install_injector_only():
    """Test harness: the deterministic engine killer, without the guard."""
    from ray.llm._internal.batch.stages import vllm_engine_stage as ves
    marker = os.environ.get("CHAOS_POISON_MARKER")
    if marker and not getattr(ves.vLLMEngineWrapper, "_cb_injected", False):
        ves.vLLMEngineWrapper._cb_injected = True
        orig_wrapper_generate = ves.vLLMEngineWrapper.generate_async

        async def poisoned_generate(self, row):
            if marker in str(row.get("prompt", "")):
                ev = os.environ.get("CHAOS_EVENTS")
                if ev:
                    with open(ev, "a") as f:
                        f.write(f"engine_killed pid={os.getpid()}\n")
                _kill_engine_core()
                await asyncio.sleep(0.5)
            return await orig_wrapper_generate(self, row)

        ves.vLLMEngineWrapper.generate_async = poisoned_generate


def install():
    import ray
    from ray.llm._internal.batch.stages import vllm_engine_stage as ves
    from ray.llm._internal.common.errors import VLLM_FATAL_ERRORS

    install_injector_only()
    UDF = ves.vLLMEngineStageUDF
    orig_handle = UDF._generate_with_error_handling

    async def ledger_call(self, method, *args, default=None):
        """Never let the guard break the job: if the Ledger is unreachable,
        behave like stock Ray."""
        try:
            if self._cb_ledger is None:
                raise RuntimeError("ledger actor not found")
            return await asyncio.wait_for(getattr(self._cb_ledger, method).remote(*args), timeout=5)
        except Exception as e:  # noqa: BLE001
            # fail-safe, but never silent: a degraded guard must be visible
            if not getattr(self, "_cb_warned", False):
                log.warning("chaosbatch guard DEGRADED (behaving like stock Ray): %s", e)
                self._cb_warned = True
            return default

    def key_of(row):
        return str(row.get("id", row.get("prompt")))

    async def first_use(self, template_row):
        self._cb_inflight, self._cb_alone = set(), asyncio.Lock()
        self._cb_node = ray.get_runtime_context().get_node_id()
        try:
            self._cb_ledger = ray.get_actor(LEDGER)
        except ValueError:
            self._cb_ledger = None
        raw_generate = self.llm.generate_async

        async def reporting_generate(row):
            try:
                return await raw_generate(row)
            except VLLM_FATAL_ERRORS:
                await ledger_call(self, "report_death", self._cb_node, sorted(self._cb_inflight))
                raise                       # Ray's handler then exits the actor

        self.llm.generate_async = reporting_generate
        # the engine prefers pre-tokenized fields over the prompt text, so drop them too
        canary = {k: v for k, v in template_row.items() if k != "id" and "token" not in k and k != "messages"}
        canary["prompt"] = CANARY_PROMPT
        try:
            await raw_generate(canary)
        except VLLM_FATAL_ERRORS:
            await ledger_call(self, "report_canary_death", self._cb_node)
            os._exit(1)

    async def run_row(self, row, batch_uuid):
        key = key_of(row)
        self._cb_inflight.add(key)
        try:
            return await orig_handle(self, row, batch_uuid)
        finally:
            self._cb_inflight.discard(key)

    def quarantined_row(self, row, batch_uuid):
        out = ves.vLLMOutputData(prompt=str(row.get("prompt", ""))[:200], prompt_token_ids=None,
                                 num_input_tokens=0)
        return {**out.model_dump(), "request_id": -1,
                self.IDX_IN_BATCH_COLUMN: row[self.IDX_IN_BATCH_COLUMN],
                "batch_uuid": batch_uuid.hex, "time_taken_llm": -1, "params": "",
                "__inference_error__": "quarantined by chaosbatch: the engine died twice "
                                       "with only this row in flight"}

    async def udf(self, batch):
        if not hasattr(self, "_cb_inflight"):
            await first_use(self, batch[0])
        batch_uuid = uuid.uuid4()
        st = await ledger_call(self, "state", default={"suspects": [], "quarantined": []})
        quarantined, suspects = set(st["quarantined"]), set(st["suspects"])
        normal = []
        for row in batch:
            k = key_of(row)
            if k in quarantined:
                yield quarantined_row(self, row, batch_uuid)
            elif k in suspects:
                # Run the suspect alone, but never yield while holding the lock: Ray may
                # finalize this async generator from another task, and releasing the lock
                # there raises "release unlocked lock" (seen on A100, 2026-10-08).
                async with self._cb_alone:           # nothing else in flight on this engine
                    while self._cb_inflight:
                        await asyncio.sleep(0.01)
                    result = await run_row(self, row, batch_uuid)
                yield result
            else:
                normal.append(row)
        async with self._cb_alone:                   # don't overlap a running suspect
            pass
        tasks = [asyncio.create_task(run_row(self, r, batch_uuid)) for r in normal]
        for t in asyncio.as_completed(tasks):
            yield await t

    UDF.udf = udf
