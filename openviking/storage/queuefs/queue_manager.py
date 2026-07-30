# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
QueueManager: Encapsulates AGFS QueueFS plugin operations.
All queues are managed through NamedQueue.
"""

import asyncio
import atexit
import threading
import time
import traceback
from typing import Any, Dict, Optional, Set, Union

from openviking_cli.utils.logger import get_logger

from .embedding_queue import EmbeddingQueue
from .named_queue import DequeueHandlerBase, EnqueueHookBase, NamedQueue, QueueStatus
from .semantic_queue import SemanticQueue

logger = get_logger(__name__)

# ========== Singleton Pattern ==========
_instance: Optional["QueueManager"] = None


def init_queue_manager(
    agfs: Any,
    timeout: int = 10,
    mount_point: str = "/queue",
    max_concurrent_embedding: int = 10,
    max_concurrent_semantic: int = 64,
    max_concurrent_external_parse: int = 4,
) -> "QueueManager":
    """Initialize QueueManager singleton.

    Args:
        agfs: Pre-initialized AGFS client (HTTP or Binding).
        timeout: Request timeout in seconds.
        mount_point: Path where QueueFS is mounted.
        max_concurrent_embedding: Max concurrent embedding tasks.
        max_concurrent_semantic: Max concurrent semantic node work.
    """
    global _instance
    _instance = QueueManager(
        agfs=agfs,
        timeout=timeout,
        mount_point=mount_point,
        max_concurrent_embedding=max_concurrent_embedding,
        max_concurrent_semantic=max_concurrent_semantic,
        max_concurrent_external_parse=max_concurrent_external_parse,
    )
    return _instance


def get_queue_manager() -> "QueueManager":
    """Get QueueManager singleton."""
    if _instance is None:
        raise RuntimeError("QueueManager is not initialized. Call init_queue_manager() first.")
    return _instance


class QueueManager:
    """
    QueueManager: Encapsulates AGFS QueueFS plugin operations.
    Integrates NamedQueue to manage multiple named queues.
    """

    # Standard queue names
    EMBEDDING = "Embedding"
    SEMANTIC = "Semantic"
    # Keep the on-disk name stable so pre-upgrade jobs remain recoverable.
    EXTERNAL_PARSE = "ExternalParse"
    ADD_RESOURCE = "AddResource"
    SESSION_COMMIT = "SessionCommit"

    # Throttle interval for repeated size-read failure logs (seconds).
    SIZE_ERROR_LOG_INTERVAL = 30.0

    # In-process retry budget for handler failures that advertise retryable=True.
    # The backoff is capped so a retrying task cannot hold its concurrency slot
    # for much longer than a path lock takes to expire.
    PROCESS_RETRY_LIMIT = 5
    PROCESS_RETRY_BASE_DELAY = 5.0
    PROCESS_RETRY_MAX_DELAY = 60.0

    # Ceiling on a single handler run. A handler that never returns holds its
    # concurrency slot for the life of the process, and SessionCommit only has
    # four, so a handful of hangs starves the queue outright.
    #
    # This is a hang detector, so it must sit well outside the normal runtime
    # distribution. It previously did not: at 900s, per-handler instrumentation
    # on 2026-07-29 measured four concurrent SessionCommit Phase-2 jobs at 38s,
    # 446s, 775s and >900s, and the 775s run (a 3316KB / 2038-message archive)
    # cleared the cap by only 125s. Runtime scales with concurrent load -- four
    # worker slots, competing reindex traffic, and 60s embedding-provider retry
    # backoff -- so the same message timed out on one run and succeeded on the
    # next. That produced steady cancellation of work that would have completed.
    PROCESS_TIMEOUT_SECONDS = 3600.0

    def __init__(
        self,
        agfs: Any,
        timeout: int = 10,
        mount_point: str = "/queue",
        max_concurrent_embedding: int = 10,
        max_concurrent_semantic: int = 64,
        max_concurrent_external_parse: int = 4,
    ):
        """Initialize QueueManager."""
        self._agfs = agfs
        self.timeout = timeout
        self.mount_point = mount_point
        self._max_concurrent_embedding = max_concurrent_embedding
        self._max_concurrent_semantic = max_concurrent_semantic
        self._max_concurrent_external_parse = max_concurrent_external_parse
        self._queues: Dict[str, NamedQueue] = {}
        self._started = False
        self._queue_threads: Dict[str, threading.Thread] = {}
        self._queue_stop_events: Dict[str, threading.Event] = {}
        self._poll_interval = 0.2
        self._size_error_last_log_at: Dict[str, float] = {}
        self._size_error_failing: Set[str] = set()

        atexit.register(self.stop)
        logger.info(
            f"[QueueManager] Initialized with agfs={type(agfs).__name__}, mount_point={mount_point}"
        )

    def start(self) -> None:
        """Start QueueManager workers."""
        if self._started:
            return

        self._started = True

        # Start queue workers for existing queues
        for queue in list(self._queues.values()):
            self._start_queue_worker(queue)

        logger.info(f"[QueueManager] mount_point={self.mount_point} Started")

    def setup_standard_queues(self, vector_store: Any, start: bool = True) -> None:
        """
        Setup standard queues (Embedding and Semantic) with their handlers.

        Args:
            vector_store: Vector store instance for handlers to write results.
            start: Whether to start worker threads immediately (default True).
                   Pass False when the consumer depends on resources that are
                   not yet initialized (e.g. VikingFS); call start() manually
                   after those resources are ready.
        """
        # Import handlers here to avoid circular dependencies
        from openviking.storage.collection_schemas import TextEmbeddingHandler
        from openviking.storage.queuefs import SemanticProcessor

        # Embedding Queue
        embedding_handler = TextEmbeddingHandler(vector_store)
        self.get_queue(
            self.EMBEDDING,
            dequeue_handler=embedding_handler,
            allow_create=True,
        )
        logger.info("Embedding queue initialized with TextEmbeddingHandler")

        # Semantic Queue
        semantic_processor = SemanticProcessor(max_concurrent_llm=self._max_concurrent_semantic)
        self.get_queue(
            self.SEMANTIC,
            dequeue_handler=semantic_processor,
            allow_create=True,
        )
        logger.info("Semantic queue initialized with SemanticProcessor")

        if start:
            self.start()

    def _start_queue_worker(self, queue: NamedQueue) -> None:
        """Start a dedicated worker thread for a queue if not already running."""
        if queue.name in self._queue_threads:
            thread = self._queue_threads[queue.name]
            if thread.is_alive():
                return

        if queue.name == self.EMBEDDING:
            max_concurrent = self._max_concurrent_embedding
        elif queue.name in {self.EXTERNAL_PARSE, self.SESSION_COMMIT}:
            max_concurrent = self._max_concurrent_external_parse
        else:
            max_concurrent = self._max_concurrent_semantic
        stop_event = threading.Event()
        self._queue_stop_events[queue.name] = stop_event
        thread = threading.Thread(
            target=self._queue_worker_loop,
            args=(queue, stop_event, max_concurrent),
            daemon=True,
        )
        self._queue_threads[queue.name] = thread
        thread.start()

    def _queue_worker_loop(
        self, queue: NamedQueue, stop_event: threading.Event, max_concurrent: int = 1
    ) -> None:
        """Worker loop for a single queue.

        When max_concurrent > 1, items are fetched and processed in parallel
        (up to max_concurrent at a time). Otherwise items are processed one by one.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            if max_concurrent > 1:
                loop.run_until_complete(
                    self._worker_async_concurrent(queue, stop_event, max_concurrent)
                )
            else:
                while not stop_event.is_set():
                    try:
                        queue_size = loop.run_until_complete(queue.size())
                        if queue.has_dequeue_handler() and queue_size > 0:
                            data = loop.run_until_complete(queue.dequeue())
                            if data is not None:
                                logger.debug(
                                    f"[QueueManager] Dequeued message from {queue.name}: {data}"
                                )
                        else:
                            stop_event.wait(self._poll_interval)
                    except Exception as e:
                        logger.error(f"[QueueManager] Worker error for {queue.name}: {e}")
                        traceback.print_exc()
                        stop_event.wait(self._poll_interval)
        finally:
            loop.close()

    def _claim_size_log_slot(self, queue_name: str) -> bool:
        """Rate-limit size-read logging to one line per queue per interval.

        The throttle timestamp deliberately survives recovery. A flapping backend
        alternates failure and success at the poll rate, so resetting it on every
        success would emit a failure/recovery pair several times a second — worse
        than no throttle at all.
        """
        now = time.monotonic()
        last = self._size_error_last_log_at.get(queue_name)
        if last is not None and (now - last) < self.SIZE_ERROR_LOG_INTERVAL:
            return False
        self._size_error_last_log_at[queue_name] = now
        return True

    def _log_size_failure(self, queue_name: str, exc: Exception) -> None:
        """Report a size-read failure that has stalled a queue's drain loop."""
        self._size_error_failing.add(queue_name)
        if not self._claim_size_log_slot(queue_name):
            return
        logger.error(
            f"[QueueManager] Size read failed for {queue_name}; drain loop is "
            f"stalled until it recovers: {exc!r}"
        )

    def _clear_size_failure(self, queue_name: str) -> None:
        """Report recovery after a size-read failure. No-op on the happy path."""
        if queue_name not in self._size_error_failing:
            return
        self._size_error_failing.discard(queue_name)
        if not self._claim_size_log_slot(queue_name):
            return
        logger.info(f"[QueueManager] Size read recovered for {queue_name}, resuming drain")

    async def _worker_async_concurrent(
        self, queue: NamedQueue, stop_event: threading.Event, max_concurrent: int
    ) -> None:
        """Concurrent worker: drains the queue and processes items in parallel.

        A Semaphore caps inflight tasks at max_concurrent.
        """
        sem = asyncio.Semaphore(max_concurrent)
        active_tasks: Set[asyncio.Task] = set()

        async def process_one(data: Dict[str, Any]) -> None:
            async with sem:
                msg_id = data.get("id", "") if isinstance(data, dict) else ""
                try:
                    await self._process_with_retry(queue, data)
                    # Ack after successful processing (delete from persistent storage).
                    await queue.ack(msg_id)
                except Exception as e:
                    # Handler did not call report_error; decrement in_progress manually.
                    # Do NOT ack — let RecoverStale re-queue on next startup.
                    queue._on_process_error(str(e), data)
                    logger.error(f"[QueueManager] Concurrent worker error for {queue.name}: {e}")
                except BaseException:
                    # CancelledError is a BaseException, so it escapes the clause
                    # above. Handlers re-raise it deliberately (see
                    # session_commit_processor / add_resource_processor), and the
                    # shutdown drain cancels in-flight tasks outright. Without this
                    # the _on_dequeue_start() increment is never undone and
                    # is_complete stays false for the life of the process.
                    queue._on_process_cancelled()
                    raise

        while not stop_event.is_set():
            # Prune completed tasks
            active_tasks = {t for t in active_tasks if not t.done()}

            # While capacity remains, keep draining the queue
            while len(active_tasks) < max_concurrent:
                try:
                    queue_size = await queue.size()
                except Exception as e:
                    # A failing size read stalls this drain loop entirely. Make it
                    # visible rather than polling silently forever.
                    self._log_size_failure(queue.name, e)
                    break
                self._clear_size_failure(queue.name)
                if not queue.has_dequeue_handler() or queue_size == 0:
                    break
                data = await queue.dequeue_raw()
                if data is None:
                    break
                # Increment before task creation to close the race window where
                # size=0 and in_progress=0 between dequeue_raw() and task execution.
                queue._on_dequeue_start()
                task = asyncio.create_task(process_one(data))
                active_tasks.add(task)
                logger.debug(
                    f"[QueueManager] Dispatched concurrent task for {queue.name} "
                    f"(active={len(active_tasks)})"
                )

            await asyncio.sleep(self._poll_interval)

        # Drain remaining in-flight tasks on shutdown (with timeout)
        if active_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*active_tasks, return_exceptions=True),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"[QueueManager] Drain timeout for {queue.name}, "
                    f"cancelling {len(active_tasks)} in-flight task(s)"
                )
                for t in active_tasks:
                    t.cancel()
                await asyncio.gather(*active_tasks, return_exceptions=True)

    async def _process_with_retry(self, queue: NamedQueue, data: Dict[str, Any]) -> Any:
        """Run the dequeue handler, retrying failures that advertise retryable=True.

        A retryable failure is not a bad message. ResourceBusyError is the one
        that matters here: a path lock orphaned by a crashed writer stays
        "fresh" until it expires, so a write that lands inside that window is
        rejected even though the condition clears itself minutes later.

        Without an in-process retry such a message keeps its 'processing' row
        and only the backend's RecoverStale, which runs at process start, puts
        it back. That is not enough on its own: a restart re-runs the same race
        against the same not-yet-expired lock, so the message can be parked
        again immediately and stay parked across any number of restarts.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(self.PROCESS_RETRY_LIMIT + 1):
            try:
                return await asyncio.wait_for(
                    queue.process_dequeued(data), timeout=self.PROCESS_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                # Deliberately not retried: a hang is not a transient conflict,
                # and re-running it would tie the slot up for another full
                # timeout. Free the slot and let RecoverStale re-queue the
                # message at the next process start.
                logger.error(
                    f"[QueueManager] Handler for {queue.name} exceeded "
                    f"{self.PROCESS_TIMEOUT_SECONDS:.0f}s and was cancelled; "
                    f"releasing the worker slot"
                )
                raise
            except Exception as exc:
                last_exc = exc
                if attempt >= self.PROCESS_RETRY_LIMIT or not getattr(exc, "retryable", False):
                    raise
                delay = min(
                    self.PROCESS_RETRY_MAX_DELAY,
                    self.PROCESS_RETRY_BASE_DELAY * (2**attempt),
                )
                logger.warning(
                    f"[QueueManager] Retryable error for {queue.name} "
                    f"(attempt {attempt + 1}/{self.PROCESS_RETRY_LIMIT}, "
                    f"retrying in {delay:.0f}s): {exc}"
                )
                await asyncio.sleep(delay)
        # Unreachable: the loop either returns or raises.
        raise last_exc  # type: ignore[misc]

    def stop(self) -> None:
        """Stop QueueManager and release resources."""
        global _instance
        if not self._started:
            return

        # Stop queue workers
        for stop_event in self._queue_stop_events.values():
            stop_event.set()
        for name, thread in self._queue_threads.items():
            thread.join(timeout=10.0)
            if thread.is_alive():
                logger.warning(f"[QueueManager] Worker thread {name} did not exit in time")
        self._queue_threads.clear()
        self._queue_stop_events.clear()
        self._size_error_last_log_at.clear()
        self._size_error_failing.clear()

        self._agfs = None
        self._queues.clear()
        self._started = False

        if _instance is self:
            _instance = None

        logger.info("[QueueManager] Stopped")

    def is_running(self) -> bool:
        """Check if QueueManager is running."""
        return self._started

    def get_queue(
        self,
        name: str,
        enqueue_hook: Optional[EnqueueHookBase] = None,
        dequeue_handler: Optional[DequeueHandlerBase] = None,
        allow_create: bool = False,
    ) -> NamedQueue:
        """Get or create a named queue object."""
        if name not in self._queues:
            if not allow_create:
                raise RuntimeError(f"Queue {name} does not exist and allow_create is False")
            if name == self.EMBEDDING:
                self._queues[name] = EmbeddingQueue(
                    self._agfs,
                    self.mount_point,
                    name,
                    enqueue_hook=enqueue_hook,
                    dequeue_handler=dequeue_handler,
                )
            elif name == self.SEMANTIC:
                self._queues[name] = SemanticQueue(
                    self._agfs,
                    self.mount_point,
                    name,
                    enqueue_hook=enqueue_hook,
                    dequeue_handler=dequeue_handler,
                )
            else:
                self._queues[name] = NamedQueue(
                    self._agfs,
                    self.mount_point,
                    name,
                    enqueue_hook=enqueue_hook,
                    dequeue_handler=dequeue_handler,
                )
            if self._started:
                self._start_queue_worker(self._queues[name])
        elif self._started:
            # Ensure existing queue has a worker running
            self._start_queue_worker(self._queues[name])
        return self._queues[name]

    # ========== Compatibility convenience methods ==========

    async def enqueue(self, queue_name: str, data: Union[str, Dict[str, Any]]) -> str:
        """Send message to queue (enqueue)."""
        return await self.get_queue(queue_name).enqueue(data)

    async def dequeue(self, queue_name: str) -> Optional[Dict[str, Any]]:
        """Get message from specified queue."""
        return await self.get_queue(queue_name).dequeue()

    async def peek(self, queue_name: str) -> Optional[Dict[str, Any]]:
        """Peek at the head message of specified queue."""
        return await self.get_queue(queue_name).peek()

    async def size(self, queue_name: str) -> int:
        """Get the size of specified queue."""
        return await self.get_queue(queue_name).size()

    async def clear(self, queue_name: str) -> bool:
        """Clear specified queue."""
        return await self.get_queue(queue_name).clear()

    # ========== Status check interface ==========

    async def check_status(self, queue_name: Optional[str] = None) -> Dict[str, QueueStatus]:
        """Check queue status."""
        if queue_name:
            if queue_name not in self._queues:
                return {}
            return {queue_name: await self._queues[queue_name].get_status()}
        return {name: await q.get_status() for name, q in self._queues.items()}

    def has_errors(self, queue_name: Optional[str] = None) -> bool:
        """Check if there are errors."""
        if queue_name:
            if queue_name not in self._queues:
                return False
            return self._queues[queue_name]._error_count > 0
        return any(q._error_count > 0 for q in self._queues.values())

    async def is_all_complete(self, queue_name: Optional[str] = None) -> bool:
        """Check if all processing is complete."""
        statuses = await self.check_status(queue_name)
        return all(s.is_complete for s in statuses.values())

    async def wait_complete(
        self,
        queue_name: Optional[str] = None,
        timeout: Optional[float] = None,
        poll_interval: float = 0.5,
    ) -> Dict[str, QueueStatus]:
        """Wait for completion and return final status."""
        start = time.time()
        while True:
            if await self.is_all_complete(queue_name):
                return await self.check_status(queue_name)
            if timeout and (time.time() - start) > timeout:
                raise TimeoutError(f"Queue processing not complete after {timeout}s")
            await asyncio.sleep(poll_interval)
