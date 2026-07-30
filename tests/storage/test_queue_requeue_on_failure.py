"""A failed message must go back on the queue, not keep its 'processing' row.

The backend resets 'processing' rows only when it is constructed, so a row left
behind by a failed handler is invisible for the rest of the process lifetime, and
the restart that would recover it re-dequeues the message into the same
conditions that just failed it. These tests pin the settle contract: every exit
either deletes the row (ack) or puts the message back (enqueue then ack), and
the one case that legitimately parks -- an exhausted attempt budget -- never
destroys the message.
"""

import asyncio

import pytest

from openviking.storage.queuefs.queue_manager import QueueManager

PAYLOAD = '{"task_id": "t1", "session_uri": "viking://user/default/sessions/s1"}'


def message(payload: str = PAYLOAD, msg_id: str = "m1") -> dict:
    """The envelope shape the queuefs dequeue file returns."""
    return {"id": msg_id, "data": payload}


class _FakeQueue:
    """Minimal NamedQueue stand-in recording what the worker did to it."""

    name = "SessionCommit"

    def __init__(self, exc=None, enqueue_exc=None):
        self._exc = exc
        self._enqueue_exc = enqueue_exc
        self.processed = 0
        self.acked: list = []
        self.enqueued: list = []
        self.errors: list = []
        self.cancelled = 0

    async def process_dequeued(self, data):
        self.processed += 1
        if self._exc is not None:
            raise self._exc
        return data

    async def ack(self, msg_id):
        self.acked.append(msg_id)

    async def enqueue(self, payload):
        if self._enqueue_exc is not None:
            raise self._enqueue_exc
        self.enqueued.append(payload)
        return "new-id"

    def _on_process_error(self, message, data):
        self.errors.append(message)

    def _on_process_cancelled(self):
        self.cancelled += 1


@pytest.fixture
def manager():
    mgr = QueueManager.__new__(QueueManager)
    mgr._requeue_attempts = {}
    return mgr


async def test_successful_message_is_acked_and_not_requeued(manager):
    queue = _FakeQueue()

    await manager._process_and_settle(queue, message())

    assert queue.acked == ["m1"]
    assert queue.enqueued == []
    assert queue.errors == []


async def test_timeout_is_requeued_rather_than_parked(manager):
    """The observed production failure: wait_for cancels the handler at the cap."""
    queue = _FakeQueue(exc=asyncio.TimeoutError())

    await manager._process_and_settle(queue, message())

    assert queue.enqueued == [PAYLOAD], "the payload must go back on the queue verbatim"
    assert queue.acked == ["m1"], "the original row must be deleted, not left parked"
    # str(asyncio.TimeoutError()) is empty, which is why the old log line was blank.
    assert queue.errors[0] != "", "the recorded error must not be an empty string"


class _PredecessorPending(RuntimeError):
    """Stands in for session.ArchivePredecessorPendingError's opt-in marker."""

    requeue = True


async def test_exception_opting_into_requeue_is_requeued(manager):
    queue = _FakeQueue(exc=_PredecessorPending("archive_009 not done"))

    await manager._process_and_settle(queue, message())

    assert queue.enqueued == [PAYLOAD]
    assert queue.acked == ["m1"]


async def test_ordinary_failure_is_not_requeued(manager):
    """Re-delivering a failure caused by the message itself just repeats the work,
    and for SessionCommit each attempt can hold a slot for the full ceiling."""
    queue = _FakeQueue(exc=RuntimeError("bad payload"))

    await manager._process_and_settle(queue, message())

    assert queue.enqueued == [], "only load and ordering conditions should be re-queued"
    assert queue.acked == [], "and it must not be destroyed either"
    assert len(queue.errors) == 1


async def test_requeue_stops_after_the_attempt_limit(manager):
    queue = _FakeQueue(exc=asyncio.TimeoutError())

    for _ in range(QueueManager.REQUEUE_ATTEMPT_LIMIT):
        await manager._process_and_settle(queue, message())
    assert len(queue.enqueued) == QueueManager.REQUEUE_ATTEMPT_LIMIT

    # One more failure must stop the cycle instead of looping forever.
    await manager._process_and_settle(queue, message())
    assert len(queue.enqueued) == QueueManager.REQUEUE_ATTEMPT_LIMIT, "should stop re-queueing"
    assert len(queue.acked) == QueueManager.REQUEUE_ATTEMPT_LIMIT, "must not ack what it cannot requeue"


async def test_success_clears_the_attempt_budget(manager):
    failing = _FakeQueue(exc=asyncio.TimeoutError())
    await manager._process_and_settle(failing, message())
    assert manager._requeue_attempts

    healthy = _FakeQueue()
    await manager._process_and_settle(healthy, message())
    assert not manager._requeue_attempts, "a message that succeeds must not keep a failure budget"


async def test_undecodable_message_is_parked_never_acked(manager):
    """Acking a message we cannot re-enqueue would destroy it outright."""
    queue = _FakeQueue(exc=asyncio.TimeoutError())

    await manager._process_and_settle(queue, {"id": "m1", "data": None})

    assert queue.enqueued == []
    assert queue.acked == [], "parking is recoverable at next start; acking is not"


async def test_failed_enqueue_does_not_ack(manager):
    queue = _FakeQueue(exc=asyncio.TimeoutError(), enqueue_exc=OSError("storage down"))

    await manager._process_and_settle(queue, message())

    assert queue.acked == [], "the row must survive if the replacement was not written"


async def test_cancellation_propagates_without_requeue(manager):
    """Shutdown cancels in-flight tasks; the backend re-queues those rows at start."""
    queue = _FakeQueue(exc=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await manager._process_and_settle(queue, message())

    assert queue.cancelled == 1
    assert queue.enqueued == []
    assert queue.acked == []


async def test_distinct_payloads_get_independent_budgets(manager):
    queue = _FakeQueue(exc=asyncio.TimeoutError())

    for _ in range(QueueManager.REQUEUE_ATTEMPT_LIMIT + 1):
        await manager._process_and_settle(queue, message())
    exhausted = len(queue.enqueued)

    await manager._process_and_settle(queue, message(payload='{"task_id": "other"}', msg_id="m2"))

    assert len(queue.enqueued) == exhausted + 1, "a different message must not inherit the budget"
