"""A retryable handler failure must be retried in-process, not parked.

Regression cover for the case where a message is dequeued, the handler hits a
transient conflict (an orphaned path lock that has not yet expired), and the
message keeps its 'processing' row. The backend only resets those rows at
process start, and a restart re-runs the same race, so without an in-process
retry the message can stay parked across any number of restarts.
"""

import asyncio

import pytest

from openviking.storage.errors import ResourceBusyError
from openviking.storage.queuefs.queue_manager import QueueManager


class _FakeQueue:
    """Minimal NamedQueue stand-in recording what the worker did to it."""

    name = "SessionCommit"

    def __init__(self, failures, exc=None):
        self._remaining_failures = failures
        self._exc = exc or ResourceBusyError("Resource is busy: x", uri="x")
        self.attempts = 0
        self.acked = []
        self.errors = []

    async def process_dequeued(self, data):
        self.attempts += 1
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise self._exc
        return data

    async def ack(self, msg_id):
        self.acked.append(msg_id)

    def _on_process_error(self, message, data):
        self.errors.append(message)


@pytest.fixture
def manager(monkeypatch):
    mgr = QueueManager.__new__(QueueManager)
    # Keep the test fast: the production backoff is minutes.
    monkeypatch.setattr(QueueManager, "PROCESS_RETRY_BASE_DELAY", 0.0)
    monkeypatch.setattr(QueueManager, "PROCESS_RETRY_MAX_DELAY", 0.0)
    return mgr


@pytest.mark.asyncio
async def test_retryable_error_is_retried_until_it_succeeds(manager):
    queue = _FakeQueue(failures=3)

    result = await manager._process_with_retry(queue, {"id": "m1"})

    assert result == {"id": "m1"}
    assert queue.attempts == 4, "should have retried the three transient failures"


@pytest.mark.asyncio
async def test_retryable_error_gives_up_after_the_budget(manager):
    queue = _FakeQueue(failures=99)

    with pytest.raises(ResourceBusyError):
        await manager._process_with_retry(queue, {"id": "m1"})

    assert queue.attempts == QueueManager.PROCESS_RETRY_LIMIT + 1


@pytest.mark.asyncio
async def test_non_retryable_error_is_not_retried(manager):
    queue = _FakeQueue(failures=99, exc=ValueError("malformed payload"))

    with pytest.raises(ValueError):
        await manager._process_with_retry(queue, {"id": "m1"})

    assert queue.attempts == 1, "a bad message must fail fast, not burn the budget"


@pytest.mark.asyncio
async def test_resource_busy_error_advertises_retryable():
    # The retry path keys off this flag; if it ever stops being set by default
    # the fix silently stops working.
    assert ResourceBusyError("busy").retryable is True


@pytest.mark.asyncio
async def test_cancellation_is_not_swallowed_by_the_retry_loop(manager):
    class _Cancelling(_FakeQueue):
        async def process_dequeued(self, data):
            self.attempts += 1
            raise asyncio.CancelledError()

    queue = _Cancelling(failures=0)
    with pytest.raises(asyncio.CancelledError):
        await manager._process_with_retry(queue, {"id": "m1"})
    assert queue.attempts == 1


@pytest.mark.asyncio
async def test_hung_handler_is_cancelled_and_releases_the_slot(monkeypatch):
    """A handler that never returns must not hold its concurrency slot forever.

    SessionCommit runs four workers. Four hangs starve the queue for the life
    of the process, and the only recovery is a restart -- which re-runs the
    same work and can hang again.
    """
    monkeypatch.setattr(QueueManager, "PROCESS_TIMEOUT_SECONDS", 0.05)
    mgr = QueueManager.__new__(QueueManager)

    started = asyncio.Event()

    class _HangingQueue(_FakeQueue):
        async def process_dequeued(self, data):
            self.attempts += 1
            started.set()
            await asyncio.Event().wait()  # never resolves

    queue = _HangingQueue(failures=0)
    with pytest.raises(asyncio.TimeoutError):
        await mgr._process_with_retry(queue, {"id": "m1"})

    assert started.is_set(), "the handler should have run"
    assert queue.attempts == 1, "a hang must not be retried into another full timeout"
    assert queue.acked == [], "a timed-out message must not be acked"


@pytest.mark.asyncio
async def test_slow_handler_within_the_ceiling_still_succeeds(monkeypatch):
    monkeypatch.setattr(QueueManager, "PROCESS_TIMEOUT_SECONDS", 5.0)
    mgr = QueueManager.__new__(QueueManager)

    class _SlowQueue(_FakeQueue):
        async def process_dequeued(self, data):
            self.attempts += 1
            await asyncio.sleep(0.05)
            return data

    queue = _SlowQueue(failures=0)
    assert await mgr._process_with_retry(queue, {"id": "m1"}) == {"id": "m1"}
