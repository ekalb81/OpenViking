"""Micro-batching for embedding calls.

Request framing dominates the cost of a single-text embedding (measured
~213ms per request vs ~8ms per text at batch size 64), so concurrent
single-text calls must coalesce into one array request — without changing
what any individual caller receives.
"""

import asyncio

import pytest

from openviking.models.embedder.base import EmbedResult
from openviking.models.embedder.micro_batcher import EmbedMicroBatcher


class FlushRecorder:
    """Stands in for the provider call; records each flushed batch."""

    def __init__(self, delay: float = 0.0, fail_with: Exception | None = None):
        self.batches: list[list[str]] = []
        self.delay = delay
        self.fail_with = fail_with

    async def __call__(self, texts):
        self.batches.append(list(texts))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail_with is not None:
            raise self.fail_with
        return [EmbedResult(dense_vector=[float(hash(t) % 97)]) for t in texts]


@pytest.mark.asyncio
async def test_concurrent_submissions_coalesce_into_one_request():
    flush = FlushRecorder()
    batcher = EmbedMicroBatcher(flush, max_batch_size=32, max_wait_ms=20.0)

    results = await asyncio.gather(*(batcher.submit(f"text {i}") for i in range(8)))

    assert len(flush.batches) == 1, "eight concurrent submits should be one request"
    assert flush.batches[0] == [f"text {i}" for i in range(8)]
    # Each caller gets the vector for its own text, not someone else's.
    for i, result in enumerate(results):
        assert result.dense_vector == [float(hash(f"text {i}") % 97)]


@pytest.mark.asyncio
async def test_size_cap_flushes_immediately_without_waiting():
    flush = FlushRecorder()
    # A wait long enough that hitting it would fail the test's own timeout.
    batcher = EmbedMicroBatcher(flush, max_batch_size=4, max_wait_ms=60_000)

    results = await asyncio.wait_for(
        asyncio.gather(*(batcher.submit(f"t{i}") for i in range(4))), timeout=5.0
    )

    assert len(results) == 4
    assert [len(b) for b in flush.batches] == [4]


@pytest.mark.asyncio
async def test_lone_submission_flushes_after_the_wait():
    flush = FlushRecorder()
    batcher = EmbedMicroBatcher(flush, max_batch_size=32, max_wait_ms=5.0)

    result = await asyncio.wait_for(batcher.submit("solo"), timeout=5.0)

    assert result.dense_vector == [float(hash("solo") % 97)]
    assert flush.batches == [["solo"]]


@pytest.mark.asyncio
async def test_overflow_spills_into_a_second_batch():
    flush = FlushRecorder()
    batcher = EmbedMicroBatcher(flush, max_batch_size=4, max_wait_ms=5.0)

    results = await asyncio.gather(*(batcher.submit(f"t{i}") for i in range(6)))

    assert len(results) == 6
    assert sorted(len(b) for b in flush.batches) == [2, 4]


@pytest.mark.asyncio
async def test_provider_failure_reaches_every_caller():
    boom = RuntimeError("provider exploded")
    flush = FlushRecorder(fail_with=boom)
    batcher = EmbedMicroBatcher(flush, max_batch_size=8, max_wait_ms=5.0)

    outcomes = await asyncio.gather(
        *(batcher.submit(f"t{i}") for i in range(3)), return_exceptions=True
    )

    assert all(isinstance(o, RuntimeError) for o in outcomes)
    # The failed batch degrades to singles, aborting after two failed probes:
    # one batch call plus two individual attempts, never one per caller.
    assert [len(b) for b in flush.batches] == [3, 1, 1]


@pytest.mark.asyncio
async def test_result_count_mismatch_recovers_via_singles_without_misassignment():
    """A short batch response must never distribute wrong vectors.

    The batch-level mismatch is refused outright; degradation then re-issues
    each text alone, where a one-for-one response is well-formed, so callers
    recover with vectors that are provably their own.
    """
    calls: list[list[str]] = []

    async def per_text_flush(texts):
        calls.append(list(texts))
        if len(texts) > 1:
            return [EmbedResult(dense_vector=[1.0])]  # short: one row for N inputs
        return [EmbedResult(dense_vector=[float(hash(texts[0]) % 97)])]

    batcher = EmbedMicroBatcher(per_text_flush, max_batch_size=8, max_wait_ms=5.0)
    results = await asyncio.gather(*(batcher.submit(f"t{i}") for i in range(3)))

    assert calls[0] == ["t0", "t1", "t2"] and len(calls) == 4
    for i, result in enumerate(results):
        assert result.dense_vector == [float(hash(f"t{i}") % 97)]


@pytest.mark.asyncio
async def test_batches_keep_flowing_after_a_failure():
    flush = FlushRecorder()
    batcher = EmbedMicroBatcher(flush, max_batch_size=8, max_wait_ms=5.0)

    flush.fail_with = RuntimeError("transient")
    bad = await asyncio.gather(batcher.submit("a"), return_exceptions=True)
    assert isinstance(bad[0], RuntimeError)

    flush.fail_with = None
    good = await batcher.submit("b")
    assert good.dense_vector == [float(hash("b") % 97)]


# --- poison isolation: a failed batch degrades to singles, bounded ----------


class PoisonFlush:
    """Fails any batch containing 'poison'; individual calls succeed except poison."""

    def __init__(self):
        self.calls: list[list[str]] = []

    async def __call__(self, texts):
        self.calls.append(list(texts))
        if any(t == "poison" for t in texts):
            raise RuntimeError("400 input rejected")
        return [EmbedResult(dense_vector=[float(hash(t) % 97)]) for t in texts]


@pytest.mark.asyncio
async def test_poison_text_fails_alone_and_neighbours_succeed():
    flush = PoisonFlush()
    batcher = EmbedMicroBatcher(flush, max_batch_size=8, max_wait_ms=5.0)

    texts = ["a", "b", "poison", "c"]
    outcomes = await asyncio.gather(
        *(batcher.submit(t) for t in texts), return_exceptions=True
    )

    # The batch call failed, then each text was re-issued individually.
    assert flush.calls[0] == texts
    assert isinstance(outcomes[2], RuntimeError), "the poison text itself must fail"
    for i in (0, 1, 3):
        assert isinstance(outcomes[i], EmbedResult), f"neighbour {texts[i]} must succeed"
        assert outcomes[i].dense_vector == [float(hash(texts[i]) % 97)]


@pytest.mark.asyncio
async def test_outage_aborts_degradation_after_two_failures():
    calls = []

    async def always_down(texts):
        calls.append(list(texts))
        raise RuntimeError("provider down")

    batcher = EmbedMicroBatcher(always_down, max_batch_size=8, max_wait_ms=5.0)
    outcomes = await asyncio.gather(
        *(batcher.submit(f"t{i}") for i in range(6)), return_exceptions=True
    )

    assert all(isinstance(o, RuntimeError) for o in outcomes)
    # One batch attempt plus exactly two individual probes -- not six.
    assert len(calls) == 3, f"expected bounded degradation, saw calls: {calls}"


@pytest.mark.asyncio
async def test_single_item_failure_does_not_degrade():
    calls = []

    async def failing(texts):
        calls.append(list(texts))
        raise RuntimeError("boom")

    batcher = EmbedMicroBatcher(failing, max_batch_size=8, max_wait_ms=5.0)
    outcome = await asyncio.gather(batcher.submit("solo"), return_exceptions=True)

    assert isinstance(outcome[0], RuntimeError)
    assert len(calls) == 1, "a batch of one has nothing to degrade to"
