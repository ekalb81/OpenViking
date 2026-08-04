# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Regression tests for NamedQueue read normalisation and in_progress accounting.

QueueStatus.is_complete is `pending == 0 and in_progress == 0`, and every drain
wait polls it. pending comes only from size(); in_progress is incremented by the
worker before dispatch and decremented only when a handler reports success or
error. A skipped decrement is therefore permanent -- nothing resets the counter
short of a process restart -- and wedges every wait on that queue while the log
stays clean. These tests pin the two paths that skipped it.

The str and .content branches of _coerce_read_bytes are defensive only; the live
binding client returns bytes. They are covered to lock the contract of the shared
normaliser, not because any backend produces them.
"""

import asyncio

import pytest

from openviking.pyagfs.exceptions import AGFSNotFoundError
from openviking.storage.queuefs.named_queue import NamedQueue


class _Response:
    """A read result that carries its payload on .content rather than inline."""

    def __init__(self, content):
        self.content = content


class _FakeAsyncAGFS:
    """Minimal AsyncAGFSClient stand-in returning a scripted read result."""

    def __init__(self, result):
        self._result = result

    async def mkdir(self, path):
        return None

    async def read(self, path, *args, **kwargs):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _queue(read_result=None) -> NamedQueue:
    queue = NamedQueue.__new__(NamedQueue)
    queue.name = "Semantic"
    queue.path = "/queue/Semantic"
    queue._async_agfs = _FakeAsyncAGFS(read_result)
    queue._initialized = True
    # __new__ skips __init__, so the task work index the error path consults
    # has to be set explicitly.
    queue._task_work_index = None
    return queue


# --- in_progress accounting ---------------------------------------------------


def _counting_queue() -> NamedQueue:
    import threading

    queue = _queue()
    queue._lock = threading.Lock()
    queue._in_progress = 0
    queue._processed = 0
    queue._requeue_count = 0
    queue._error_count = 0
    queue._errors = []
    return queue


def test_cancellation_releases_in_progress():
    """CancelledError is a BaseException and escaped the worker's except Exception.

    Handlers re-raise it deliberately (session_commit_processor,
    add_resource_processor) and the shutdown drain cancels in-flight tasks, so
    this is reachable outside shutdown too.
    """
    queue = _counting_queue()
    queue._on_dequeue_start()
    queue._on_process_cancelled()
    assert queue._in_progress == 0
    # Cancellation is not a processing failure.
    assert queue._error_count == 0


def test_decrements_clamp_at_zero():
    """A double decrement must not drive in_progress negative.

    Negative is as fatal as leaked-positive: is_complete requires exactly 0, so
    either way the queue never reports complete again.
    """
    queue = _counting_queue()
    queue._on_dequeue_start()
    queue._on_process_success()
    queue._on_process_cancelled()
    queue._on_process_error("late failure")
    assert queue._in_progress == 0


# --- _read_queue_message ------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "read_result",
    [
        None,
        b"",
        b"{}",
        b"{}\n",
        b"{ }",
        b"[]",
        b"null",
        b'""',
        _Response(b"{}"),
        _Response(b""),
    ],
)
async def test_empty_payloads_are_reported_as_no_message(read_result):
    """Must be None, not a falsy value.

    The concurrent worker only breaks on None. A falsy-but-not-None result is
    counted by _on_dequeue_start() and dispatched, then every handler's
    `if not data: return None` guard returns without reporting -- leaking
    in_progress permanently. A byte-exact b"{}" check missed b"{}\\n", b"{ }"
    and b"[]".
    """
    assert await _queue(read_result)._read_queue_message() is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "read_result",
    [b'{"id": "m1"}', '{"id": "m1"}', _Response(b'{"id": "m1"}'), bytearray(b'{"id": "m1"}')],
)
async def test_read_queue_message_parses_every_supported_shape(read_result):
    assert await _queue(read_result)._read_queue_message() == {"id": "m1"}


# --- _coerce_read_bytes -------------------------------------------------------


@pytest.mark.parametrize(
    "content,expected",
    [
        (None, None),
        (b"7", b"7"),
        ("7", b"7"),
        (bytearray(b"7"), b"7"),
        (memoryview(b"7"), b"7"),
        (_Response(b"7"), b"7"),
        (_Response("7"), b"7"),
    ],
)
def test_coerce_read_bytes_normalises_known_shapes(content, expected):
    assert NamedQueue._coerce_read_bytes(content) == expected


@pytest.mark.parametrize("content", [object(), _Response(None), {"size": 7}, 7])
def test_coerce_read_bytes_rejects_unrecoverable_shapes(content):
    with pytest.raises(TypeError, match="Unexpected AGFS read response"):
        NamedQueue._coerce_read_bytes(content)


# --- size() -------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "read_result,expected",
    [
        (b"7", 7),
        ("7", 7),
        (b" 7\n", 7),
        (bytearray(b"7"), 7),
        (_Response(b"7"), 7),
        (b"0", 0),
        (b"", 0),
        (None, 0),
    ],
)
async def test_size_reads_every_supported_shape(read_result, expected):
    assert await _queue(read_result).size() == expected


@pytest.mark.asyncio
async def test_size_reports_empty_when_size_file_missing():
    assert await _queue(AGFSNotFoundError("missing")).size() == 0
    assert await _queue(FileNotFoundError("missing")).size() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "read_result,exc",
    [
        (RuntimeError("backend down"), RuntimeError),
        (b"not-a-number", ValueError),
        (object(), TypeError),
    ],
)
async def test_size_raises_rather_than_reporting_empty(read_result, exc):
    """#3417's intent: a storage or decode fault must not look like a drained queue."""
    with pytest.raises(exc):
        await _queue(read_result).size()


# --- size-failure logging throttle --------------------------------------------


def _manager():
    from openviking.storage.queuefs.queue_manager import QueueManager

    mgr = QueueManager.__new__(QueueManager)
    mgr._size_error_last_log_at = {}
    mgr._size_error_failing = set()
    return mgr


class _RecordingLogger:
    """Captures log lines. The project logs via loguru, which caplog does not see."""

    def __init__(self):
        self.lines = []

    def error(self, msg):
        self.lines.append(("error", msg))

    def info(self, msg):
        self.lines.append(("info", msg))

    def warning(self, msg):
        self.lines.append(("warning", msg))


@pytest.fixture
def recorded_logger(monkeypatch):
    from openviking.storage.queuefs import queue_manager as qm

    recorder = _RecordingLogger()
    monkeypatch.setattr(qm, "logger", recorder)
    return recorder


def test_flapping_size_reads_do_not_defeat_the_log_throttle(recorded_logger):
    """An intermittent fault is the realistic case and must not spam.

    Resetting the throttle on every success emitted an ERROR/INFO pair per poll
    cycle -- around ten lines a second -- which is worse than not throttling.
    """
    mgr = _manager()
    for _ in range(10):
        mgr._log_size_failure("Semantic", RuntimeError("flap"))
        mgr._clear_size_failure("Semantic")
    assert len(recorded_logger.lines) == 1
    assert "Size read failed" in recorded_logger.lines[0][1]


def test_persistent_size_failure_logs_once_per_interval(recorded_logger):
    mgr = _manager()
    for _ in range(10):
        mgr._log_size_failure("Semantic", RuntimeError("down"))
    assert len(recorded_logger.lines) == 1


def test_first_size_failure_logs_and_recovery_is_reported_after_the_interval(recorded_logger):
    mgr = _manager()
    mgr._log_size_failure("Semantic", RuntimeError("down"))
    # Backdate past the throttle so the recovery notice is not suppressed.
    mgr._size_error_last_log_at["Semantic"] -= mgr.SIZE_ERROR_LOG_INTERVAL + 1
    mgr._clear_size_failure("Semantic")
    assert [level for level, _ in recorded_logger.lines] == ["error", "info"]
    assert "Size read failed" in recorded_logger.lines[0][1]
    assert "recovered" in recorded_logger.lines[1][1]


def test_separate_queues_throttle_independently(recorded_logger):
    mgr = _manager()
    mgr._log_size_failure("Semantic", RuntimeError("down"))
    mgr._log_size_failure("Embedding", RuntimeError("down"))
    assert len(recorded_logger.lines) == 2


def test_clear_size_failure_is_silent_when_nothing_failed(recorded_logger):
    mgr = _manager()
    for _ in range(100):
        mgr._clear_size_failure("Semantic")
    assert recorded_logger.lines == []


def test_worker_reports_cancellation_without_counting_an_error():
    """End-to-end: a handler that raises CancelledError must not wedge the queue."""
    from openviking.storage.queuefs import queue_manager as qm

    queue = _counting_queue()

    async def scenario():
        queue._on_dequeue_start()
        try:
            raise asyncio.CancelledError()
        except Exception as e:  # mirrors process_one's ordering
            queue._on_process_error(str(e), None)
        except BaseException:
            queue._on_process_cancelled()

    asyncio.run(scenario())
    assert queue._in_progress == 0
    assert queue._error_count == 0
    assert qm.QueueManager.SIZE_ERROR_LOG_INTERVAL > 0
