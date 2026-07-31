# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Transparent micro-batching for embedding calls.

The embedding pipeline is message-per-chunk end to end: every queue worker,
reindex request, and session-commit extraction embeds one text per HTTP call.
Measured against the live provider, that request framing is almost the entire
cost -- one text takes ~213ms wall clock while the provider's own engine time
is ~12ms, and 64 texts in one request cost ~8ms per text.

Rather than rewrite every call site to accumulate chunks, the batcher sits
inside the embedder: concurrent single-text calls coalesce into one array
request. A submission waits at most ``max_wait_ms`` for company; whatever has
accumulated by then (or as soon as ``max_batch_size`` is reached) is flushed
as one request and the results are fanned back out to the waiting callers.
Sequential callers therefore pay ``max_wait_ms`` of extra latency per call,
which is noise against the request round trip; concurrent callers -- the queue
runs up to ten -- collapse N requests into one.
"""

import asyncio
from typing import Awaitable, Callable, List, Sequence, Tuple

from openviking.models.embedder.base import EmbedResult

FlushFn = Callable[[Sequence[str]], Awaitable[List[EmbedResult]]]


class EmbedMicroBatcher:
    """Coalesces concurrent single-text embed calls into array requests.

    Instances are bound to the event loop they are first used on: pending
    futures belong to a loop, so an embedder shared across loops must hold one
    batcher per loop (see the WeakKeyDictionary pattern used for the async
    embed semaphores).
    """

    def __init__(
        self,
        flush_fn: FlushFn,
        *,
        max_batch_size: int = 32,
        max_wait_ms: float = 10.0,
    ):
        self._flush_fn = flush_fn
        self._max_batch_size = max(1, int(max_batch_size))
        self._max_wait_s = max(0.0, float(max_wait_ms)) / 1000.0
        self._pending: List[Tuple[str, asyncio.Future]] = []
        self._timer: asyncio.Task | None = None
        # Flush tasks need a strong reference until done or the loop may GC them.
        self._inflight: set = set()

    async def submit(self, text: str) -> EmbedResult:
        """Queue one text and return its embedding when the batch resolves."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending.append((text, future))
        if len(self._pending) >= self._max_batch_size:
            self._flush(loop)
        elif self._timer is None:
            self._timer = loop.create_task(self._flush_after_wait())
        return await future

    async def _flush_after_wait(self) -> None:
        try:
            await asyncio.sleep(self._max_wait_s)
        except asyncio.CancelledError:
            # A size-triggered flush already took the pending batch.
            raise
        self._timer = None
        self._flush(asyncio.get_running_loop())

    def _flush(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        batch, self._pending = self._pending, []
        if not batch:
            return
        task = loop.create_task(self._run_flush(batch))
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    async def _run_flush(self, batch: List[Tuple[str, asyncio.Future]]) -> None:
        texts = [text for text, _ in batch]
        try:
            results = await self._flush_fn(texts)
            if len(results) != len(batch):
                raise RuntimeError(
                    f"Batch embedding returned {len(results)} results for {len(batch)} inputs"
                )
        except Exception as exc:
            await self._degrade_to_singles(batch, exc)
            return
        for (_, future), result in zip(batch, results):
            if not future.done():
                future.set_result(result)

    async def _degrade_to_singles(
        self, batch: List[Tuple[str, asyncio.Future]], batch_exc: Exception
    ) -> None:
        """Re-issue a failed batch one text at a time so poison cannot spread.

        Batching correlates failures: one rejected input -- an oversized chunk
        drawing a 400, say -- would fail every coalesced neighbour, and
        retrying the batch re-sends the same poison each time. Degrading to
        single requests isolates the bad input while its neighbours succeed.

        The degradation is bounded by a two-failure abort: one individual
        failure looks like a poison text, but a second means the provider
        itself is unhappy, and hammering it with the rest of the batch as
        singles would only add load to an outage. Remaining callers then get
        the original batch error.
        """
        if len(batch) == 1:
            _, future = batch[0]
            if not future.done():
                future.set_exception(batch_exc)
            return
        failures = 0
        for text, future in batch:
            if future.done():
                continue
            if failures >= 2:
                future.set_exception(batch_exc)
                continue
            try:
                results = await self._flush_fn([text])
                if len(results) != 1:
                    raise RuntimeError(
                        f"Batch embedding returned {len(results)} results for 1 input"
                    )
                future.set_result(results[0])
            except Exception as item_exc:
                failures += 1
                future.set_exception(item_exc)
