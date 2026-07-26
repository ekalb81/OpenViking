"""Near-duplicate suppression for add_only memories.

Dedupe is a silent discard: nothing surfaces to the user when a record is
dropped. That makes two properties worth pinning hard — that it only fires on
genuine duplicates, and that a suppressed write is reported as *satisfied*
rather than failed. Downstream code gates on whether a declared URI landed, and
a dedupe that reads as a failed write refuses the record's links and can roll
back an unrelated batch of experiences.
"""

from types import SimpleNamespace

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.session.memory.dataclass import ResolvedOperation
from openviking.session.memory.memory_type_registry import create_default_registry
from openviking.session.memory.memory_updater import MemoryUpdater, MemoryUpdateResult
from openviking_cli.session.user_id import UserIdentifier

TRAJ_DIR = "viking://user/u/memories/trajectories"
TIMESTAMP = "20260725120000"

BODY = """## Preconditions
1. The scheduled task is registered and the server answers on its health port.

## Procedure
1. Stop the scheduled task and wait for the process to exit.
2. Kill any orphaned worker that outlived the task.
3. Start the task again and poll health until it answers.

## Result
- The server answers health checks after the restart.
"""

UNRELATED_BODY = """## Preconditions
1. A pull request exists and its review comments have been read.

## Procedure
1. Group the review comments by the mechanism each one touches.
2. Audit every grouped claim against the code that actually shipped.
3. Reply with the audit outcome and push the batched fixes together.

## Result
- Review rounds collapse from many small pushes into one reviewed batch.
"""


def _ctx():
    return RequestContext(user=UserIdentifier("account", "u"), role=Role.USER)


def _traj(name, content, timestamp=TIMESTAMP):
    return ResolvedOperation(
        memory_type="trajectories",
        memory_fields={"trajectory_name": name, "content": content},
        uris=[f"{TRAJ_DIR}/{name}_{timestamp}.md"],
        page_id=1,
    )


def _updater(files=None):
    updater = MemoryUpdater(registry=create_default_registry())
    updater._viking_fs = _fs(files or {})
    return updater


def _fs(files, ls_error=None):
    """A viking_fs whose directories serve ``files`` mapping uri -> raw markdown."""

    async def ls(parent, ctx=None):
        if ls_error is not None:
            raise ls_error
        return [
            {"name": uri.rsplit("/", 1)[1], "uri": uri, "isDir": False}
            for uri in files
            if uri.rsplit("/", 1)[0] == parent
        ]

    async def read_file(uri, ctx=None):
        return files[uri]

    return SimpleNamespace(ls=ls, read_file=read_file)


@pytest.mark.asyncio
async def test_same_batch_near_duplicate_is_dropped_and_recorded_against_the_keeper():
    result = MemoryUpdateResult()
    first = _traj("restart_openviking_server", BODY)
    second = _traj("restart_the_openviking_server", BODY)

    kept = await _updater()._drop_duplicate_add_only([first, second], _ctx(), result)

    assert [op.uris[0] for op in kept] == [first.uris[0]]
    assert result.deduplicated_uris == {second.uris[0]: first.uris[0]}
    # The whole point: a suppressed duplicate is satisfied, not failed.
    assert second.uris[0] in result.satisfied_uris()


@pytest.mark.asyncio
async def test_the_richer_of_two_duplicates_is_the_one_kept():
    result = MemoryUpdateResult()
    thin = _traj("restart_server", BODY)
    rich = _traj("restart_server_fully", BODY + "\n- Orphaned workers must be killed first.\n")

    kept = await _updater()._drop_duplicate_add_only([thin, rich], _ctx(), result)

    assert [op.uris[0] for op in kept] == [rich.uris[0]]
    # The loser is the one recorded, pointing at the winner that replaced it.
    assert result.deduplicated_uris == {thin.uris[0]: rich.uris[0]}


@pytest.mark.asyncio
async def test_distinct_records_are_both_kept():
    result = MemoryUpdateResult()
    a = _traj("restart_server", BODY)
    b = _traj("batch_review_replies", UNRELATED_BODY)

    kept = await _updater()._drop_duplicate_add_only([a, b], _ctx(), result)

    assert [op.uris[0] for op in kept] == [a.uris[0], b.uris[0]]
    assert result.deduplicated_uris == {}


@pytest.mark.asyncio
async def test_an_op_duplicating_an_existing_sibling_is_skipped_and_points_at_it():
    sibling_uri = f"{TRAJ_DIR}/restart_server_{TIMESTAMP}.md"
    result = MemoryUpdateResult()
    incoming = _traj("restart_the_server", BODY)

    kept = await _updater({sibling_uri: BODY})._drop_duplicate_add_only(
        [incoming], _ctx(), result
    )

    assert kept == []
    assert result.deduplicated_uris == {incoming.uris[0]: sibling_uri}


@pytest.mark.asyncio
async def test_a_sibling_from_a_different_session_does_not_suppress():
    # The timestamp identifies the source session. Re-extracting session A must
    # not be silenced by an identical-looking record from session B.
    other_session = f"{TRAJ_DIR}/restart_server_20260101090000.md"
    result = MemoryUpdateResult()
    incoming = _traj("restart_the_server", BODY)

    kept = await _updater({other_session: BODY})._drop_duplicate_add_only(
        [incoming], _ctx(), result
    )

    assert [op.uris[0] for op in kept] == [incoming.uris[0]]
    assert result.deduplicated_uris == {}


@pytest.mark.asyncio
async def test_upsert_memory_types_pass_through_untouched():
    # experiences are upsert, not add_only: two similar experiences are a merge
    # concern for the experience policy, not something to silently discard here.
    result = MemoryUpdateResult()
    ops = [
        ResolvedOperation(
            memory_type="experiences",
            memory_fields={"experience_name": f"e{i}", "content": BODY},
            uris=[f"viking://user/u/memories/experiences/e{i}.md"],
            page_id=i,
        )
        for i in range(2)
    ]

    kept = await _updater()._drop_duplicate_add_only(ops, _ctx(), result)

    assert kept == ops
    assert result.deduplicated_uris == {}


@pytest.mark.asyncio
async def test_a_listing_failure_keeps_the_op_rather_than_dropping_it():
    updater = MemoryUpdater(registry=create_default_registry())
    updater._viking_fs = _fs({}, ls_error=RuntimeError("backend down"))
    result = MemoryUpdateResult()
    incoming = _traj("restart_server", BODY)

    kept = await updater._drop_duplicate_add_only([incoming], _ctx(), result)

    # Failing open matters: failing closed would discard real memories whenever
    # storage hiccuped, and the discard is unrecoverable.
    assert [op.uris[0] for op in kept] == [incoming.uris[0]]
    assert result.deduplicated_uris == {}


@pytest.mark.asyncio
async def test_an_incoming_line_number_artifact_still_matches_its_clean_sibling():
    # Sibling files were scrubbed before they were written. Comparing a raw
    # incoming field against them would let the very artifact this module
    # strips depress the similarity ratio and defeat the check.
    sibling_uri = f"{TRAJ_DIR}/restart_server_{TIMESTAMP}.md"
    numbered = "\n".join(
        f"{index}\t{line}" for index, line in enumerate(BODY.splitlines(), start=1)
    )
    result = MemoryUpdateResult()
    incoming = _traj("restart_the_server", numbered)

    kept = await _updater({sibling_uri: BODY})._drop_duplicate_add_only(
        [incoming], _ctx(), result
    )

    assert kept == []
    assert result.deduplicated_uris == {incoming.uris[0]: sibling_uri}


@pytest.mark.asyncio
async def test_dedupe_is_a_no_op_without_a_registry():
    updater = MemoryUpdater(registry=SimpleNamespace(get=lambda _memory_type: None))
    updater._viking_fs = _fs({})
    result = MemoryUpdateResult()
    ops = [_traj("a", BODY), _traj("b", BODY)]

    kept = await updater._drop_duplicate_add_only(ops, _ctx(), result)

    assert kept == ops
    assert result.deduplicated_uris == {}


def test_satisfied_uris_unions_written_edited_and_deduplicated():
    result = MemoryUpdateResult()
    result.add_written("viking://w.md")
    result.add_edited("viking://e.md")
    result.add_deduplicated("viking://d.md", "viking://existing.md")

    assert result.satisfied_uris() == {"viking://w.md", "viking://e.md", "viking://d.md"}
    # written_uris stays truthful: nothing was written for the deduplicated URI.
    assert result.written_uris == ["viking://w.md"]
    assert "Deduplicated: 1" in result.summary()
