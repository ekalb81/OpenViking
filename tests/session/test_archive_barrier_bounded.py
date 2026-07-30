"""The archive ordering barrier must not wait forever inside a worker slot.

An earlier archive only becomes terminal when its own Phase-2 job runs, and that
job needs a slot from the same small pool the wait is occupying. Waiting
indefinitely therefore cannot make progress in the case that matters, and it
removes a slot from the pool while it tries, so a chain of archives can take the
whole pool and stall the queue.
"""

import pytest

from openviking.session.session import (
    ArchivePredecessorPendingError,
    ArchiveState,
    Session,
)


def state(index: int, kind: str) -> ArchiveState:
    archive_id = f"archive_{index:03d}"
    return ArchiveState(
        archive_id=archive_id,
        archive_uri=f"viking://user/default/sessions/s1/history/{archive_id}",
        index=index,
        state=kind,
    )


def make_session(states, phase1=None):
    session = Session.__new__(Session)
    session._viking_fs = object()

    async def _scan():
        return list(states)

    async def _read_phase1_meta(_uri):
        # "ready" means there is nothing to reconcile, so the wait is genuine.
        return phase1 if phase1 is not None else {"status": "ready"}

    async def _ensure_phase1_ready(_uri):
        # Reconciliation that never actually makes the archive ready.
        return False

    session._scan_archive_states = _scan
    session._read_phase1_meta = _read_phase1_meta
    session._ensure_phase1_ready = _ensure_phase1_ready
    return session


@pytest.fixture(autouse=True)
def fast_barrier(monkeypatch):
    monkeypatch.setattr("openviking.session.session._ARCHIVE_WAIT_POLL_SECONDS", 0.001)
    monkeypatch.setattr("openviking.session.session._ARCHIVE_WAIT_MAX_SECONDS", 0.05)


async def test_first_archive_never_waits():
    session = make_session([])
    assert await session._wait_for_previous_archive_done(1) is True


async def test_returns_when_all_earlier_archives_are_terminal():
    session = make_session([state(1, "completed"), state(2, "failed")])
    assert await session._wait_for_previous_archive_done(3) is True


async def test_pending_predecessor_raises_instead_of_waiting_forever():
    """The production wedge: archive_010 waiting on a parked archive_009."""
    session = make_session([state(9, "pending")])

    with pytest.raises(ArchivePredecessorPendingError) as excinfo:
        await session._wait_for_previous_archive_done(10)

    assert "archive_009" in str(excinfo.value)


async def test_unsuccessful_reconcile_still_hits_the_deadline():
    """The reconcile branch `continue`s past the sleep, so it must not skip the
    deadline test as well -- otherwise a reconcile that never makes an archive
    ready spins the loop forever with no bound at all."""
    session = make_session([state(9, "pending")], phase1={"status": "not-ready"})

    with pytest.raises(ArchivePredecessorPendingError):
        await session._wait_for_previous_archive_done(10)


async def test_later_pending_archives_do_not_block():
    """Only earlier archives gate this one."""
    session = make_session([state(1, "completed"), state(11, "pending")])
    assert await session._wait_for_previous_archive_done(10) is True
