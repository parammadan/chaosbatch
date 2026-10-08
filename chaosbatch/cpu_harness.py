"""CPU harness: Ray's REAL vLLMEngineStageUDF with only the vLLM engine stubbed.

Lets the real-Ray guard (ray_patch.py) be debugged on a laptop, without holding
a GPU. Installed as the worker setup hook; it

  * defines FakeEngineDeadError and registers it as Ray's fatal engine error,
  * replaces vLLMEngineWrapper with a fake whose engine "dies" on a prompt
    containing CHAOS_POISON_MARKER and stays dead (like a killed EngineCore),
  * then installs the guard if CB_GUARD=1.

Ray's own stage code (retries, os._exit on fatal errors, should_continue_on_error)
runs unchanged.
"""

import asyncio
import os
import types


class FakeEngineDeadError(RuntimeError):
    pass


def install():
    from ray.llm._internal.batch.stages import vllm_engine_stage as ves
    from ray.llm._internal.common import errors

    errors.VLLM_FATAL_ERRORS = (FakeEngineDeadError,)
    ves._VLLM_FATAL_ERRORS = (FakeEngineDeadError,)
    marker = os.environ.get("CHAOS_POISON_MARKER")
    events = os.environ.get("CHAOS_EVENTS")

    def init(self, *args, **kwargs):
        self.idx_in_batch_column = kwargs.get("idx_in_batch_column")
        self.max_pending_requests = kwargs.get("max_pending_requests") or 1000
        self.request_id, self.dead = 0, False
        self.engine = types.SimpleNamespace(do_log_stats=_noop)

    async def generate_async(self, row):
        prompt = row.pop("prompt", None)
        row.pop("sampling_params", None)
        row.pop("tokenized_prompt", None)
        idx = row[self.idx_in_batch_column]
        await asyncio.sleep(0.02)
        if self.dead or (marker and marker in str(prompt)):
            if not self.dead and events:
                with open(events, "a") as f:
                    f.write(f"engine_killed pid={os.getpid()}\n")
            self.dead = True
            raise FakeEngineDeadError("EngineCore encountered an issue (fake)")
        req = types.SimpleNamespace(request_id=self.request_id, idx_in_batch=idx, params="fake")
        self.request_id += 1
        out = ves.vLLMOutputData(prompt=str(prompt)[:80], prompt_token_ids=None, num_input_tokens=1,
                                 generated_text="ok").model_dump()
        return req, out, 0.02

    ves.vLLMEngineWrapper.__init__ = init
    ves.vLLMEngineWrapper.generate_async = generate_async
    ves.vLLMEngineWrapper.get_scheduler_config = lambda self: types.SimpleNamespace(max_num_seqs=256)
    ves.vLLMEngineWrapper.shutdown = lambda self: None

    if os.environ.get("CB_GUARD") == "1":
        from chaosbatch import ray_patch
        ray_patch.install()


async def _noop(*a, **k):
    return None
