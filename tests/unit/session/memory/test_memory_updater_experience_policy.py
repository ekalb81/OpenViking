from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import openviking.session.memory.memory_updater as memory_updater_module
from openviking.server.identity import RequestContext, Role
from openviking.session.memory.dataclass import (
    MemoryFile,
    ResolvedOperation,
    ResolvedOperations,
    StoredLink,
)
from openviking.session.memory.memory_type_registry import create_default_registry
from openviking.session.memory.memory_updater import MemoryUpdater
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking_cli.exceptions import NotFoundError
from openviking_cli.session.user_id import UserIdentifier

VALID_CONTENT = """## Situation
- A bounded implementation needs prior procedural guidance.

## Approach
- Inspect the current state before changing it.

## Reflect
- NEVER claim completion without verification.
"""


def _ctx(user_id: str = "u") -> RequestContext:
    return RequestContext(user=UserIdentifier("account", user_id), role=Role.USER)


class JournalFS:
    def __init__(self, files=None):
        self.files = dict(files or {})

    async def read_file(self, uri, **kwargs):
        del kwargs
        if uri not in self.files:
            raise NotFoundError(uri, "file")
        return self.files[uri]

    async def write_file(self, uri, content, **kwargs):
        del kwargs
        self.files[uri] = content

    async def rm(self, uri, **kwargs):
        del kwargs
        if uri not in self.files:
            raise NotFoundError(uri, "file")
        del self.files[uri]


@pytest.mark.asyncio
async def test_invalid_experience_rejects_the_batch_before_writes_or_deletes(monkeypatch):
    updater = MemoryUpdater(registry=SimpleNamespace(get=lambda _memory_type: None))
    updater._viking_fs = object()
    monkeypatch.setattr(updater, "_apply_upsert", pytest.fail)
    monkeypatch.setattr(updater, "_apply_delete", pytest.fail)

    invalid = ResolvedOperation(
        memory_type="experiences",
        memory_fields={
            "experience_name": "oversized_lesson",
            "content": "## Situation\n- one\n\n## Approach\n- " + ("x" * 3001) + "\n\n## Reflect\n- never",
        },
        uris=["viking://user/u/memories/experiences/oversized_lesson.md"],
        page_id=1,
    )
    delete_target = MemoryFile(uri="viking://user/u/memories/experiences/legacy.md")
    operations = ResolvedOperations(
        upsert_operations=[invalid],
        delete_file_contents=[delete_target],
        errors=[],
    )

    result = await updater.apply_operations(operations, ctx=_ctx())

    assert result.written_uris == []
    assert result.deleted_uris == []
    assert len(result.errors) >= 1


@pytest.mark.asyncio
async def test_unresolved_experience_rejects_resolved_siblings_before_any_side_effect(monkeypatch):
    updater = MemoryUpdater(registry=object())
    updater._viking_fs = object()
    monkeypatch.setattr(updater, "_apply_upsert", pytest.fail)
    monkeypatch.setattr(updater, "_apply_delete", pytest.fail)
    monkeypatch.setattr(updater, "_inherit_deleted_link_relations", pytest.fail)
    monkeypatch.setattr(updater, "_vectorize_memories", pytest.fail)
    monkeypatch.setattr(updater, "generate_overview", pytest.fail)

    resolved = ResolvedOperation(
        memory_type="experiences",
        memory_fields={"experience_name": "resolved", "content": VALID_CONTENT},
        uris=["viking://user/u/memories/experiences/resolved.md"],
        page_id=100,
    )
    unresolved = ResolvedOperation(
        memory_type="experiences",
        memory_fields={
            "experience_name": "unresolved",
            "content": VALID_CONTENT.replace("prior procedural guidance", "another lesson"),
        },
        uris=[],
        page_id=101,
    )
    operations = ResolvedOperations(
        upsert_operations=[resolved, unresolved],
        delete_file_contents=[
            MemoryFile(uri="viking://user/u/memories/experiences/legacy.md")
        ],
        errors=[],
    )

    result = await updater.apply_operations(operations, ctx=_ctx())

    assert result.written_uris == []
    assert result.deleted_uris == []
    assert any("Missing resolved" in str(error) for _, error in result.errors)


@pytest.mark.asyncio
async def test_valid_children_plus_unmapped_legacy_delete_has_zero_side_effects(monkeypatch):
    updater = MemoryUpdater(registry=object())
    updater._viking_fs = object()
    monkeypatch.setattr(updater, "_apply_upsert", pytest.fail)
    monkeypatch.setattr(updater, "_apply_delete", pytest.fail)
    monkeypatch.setattr(updater, "_inherit_deleted_link_relations", pytest.fail)
    monkeypatch.setattr(updater, "_vectorize_memories", pytest.fail)
    monkeypatch.setattr(updater, "generate_overview", pytest.fail)

    children = [
        ResolvedOperation(
            memory_type="experiences",
            memory_fields={
                "experience_name": name,
                "content": VALID_CONTENT.replace("prior procedural guidance", marker),
            },
            uris=[f"viking://user/u/memories/experiences/{name}.md"],
            page_id=page_id,
        )
        for name, marker, page_id in (
            ("first", "the first narrow lesson", 100),
            ("second", "the second narrow lesson", 101),
        )
    ]
    legacy_uri = "viking://user/u/memories/experiences/legacy.md"
    operations = ResolvedOperations(
        upsert_operations=children,
        delete_file_contents=[MemoryFile(uri=legacy_uri, memory_type="experiences")],
        errors=[],
    )

    result = await updater.apply_operations(operations, ctx=_ctx())

    assert result.written_uris == []
    assert result.deleted_uris == []
    assert any("experience deletion is disabled" in str(error) for _, error in result.errors)


@pytest.mark.asyncio
async def test_mapped_experience_delete_is_rejected_before_any_side_effect(monkeypatch):
    updater = MemoryUpdater(registry=object())
    updater._viking_fs = object()
    monkeypatch.setattr(updater, "_apply_upsert", pytest.fail)
    monkeypatch.setattr(updater, "_apply_delete", pytest.fail)

    replacement_uri = "viking://user/u/memories/experiences/replacement.md"
    legacy_uri = "viking://user/u/memories/experiences/legacy.md"
    operations = ResolvedOperations(
        upsert_operations=[
            ResolvedOperation(
                memory_type="experiences",
                memory_fields={"experience_name": "replacement", "content": VALID_CONTENT},
                uris=[replacement_uri],
                page_id=100,
            )
        ],
        delete_file_contents=[MemoryFile(uri=legacy_uri, memory_type="experiences")],
        errors=[],
        delete_replacements={legacy_uri: replacement_uri},
    )

    result = await updater.apply_operations(operations, ctx=_ctx())

    assert result.written_uris == []
    assert result.deleted_uris == []
    assert any("experience deletion is disabled" in str(error) for _, error in result.errors)
    assert any("replacement mappings are disabled" in str(error) for _, error in result.errors)


@pytest.mark.asyncio
async def test_orphan_experience_replacement_mapping_is_rejected_before_link_mutation(monkeypatch):
    updater = MemoryUpdater(registry=object())
    updater._viking_fs = object()
    monkeypatch.setattr(updater, "_apply_upsert", pytest.fail)
    monkeypatch.setattr(updater, "_inherit_deleted_link_relations", pytest.fail)

    replacement_uri = "viking://user/u/memories/experiences/replacement.md"
    legacy_uri = "viking://user/u/memories/experiences/legacy.md"
    operations = ResolvedOperations(
        upsert_operations=[
            ResolvedOperation(
                memory_type="experiences",
                memory_fields={"experience_name": "replacement", "content": VALID_CONTENT},
                uris=[replacement_uri],
            )
        ],
        delete_file_contents=[],
        errors=[],
        delete_replacements={legacy_uri: replacement_uri},
    )

    result = await updater.apply_operations(operations, ctx=_ctx())

    assert result.written_uris == []
    assert result.deleted_uris == []
    assert any("replacement mappings are disabled" in str(error) for _, error in result.errors)


@pytest.mark.asyncio
async def test_authoritative_readback_keeps_noncompliant_legacy_experience_read_only(monkeypatch):
    uri = "viking://user/u/memories/experiences/legacy.md"
    legacy = MemoryFile(
        uri=uri,
        content="legacy unstructured content with unique rules",
        memory_type="experiences",
    )
    fs = SimpleNamespace(read_file=AsyncMock(return_value=MemoryFileUtils.write(legacy)))
    updater = MemoryUpdater(registry=object())
    updater._viking_fs = fs
    monkeypatch.setattr(updater, "_apply_upsert", pytest.fail)
    monkeypatch.setattr(updater, "_apply_delete", pytest.fail)
    operations = ResolvedOperations(
        upsert_operations=[
            ResolvedOperation(
                memory_type="experiences",
                memory_fields={"experience_name": "legacy", "content": VALID_CONTENT},
                uris=[uri],
                page_id=100,
            )
        ],
        delete_file_contents=[],
        errors=[],
    )

    result = await updater.apply_operations(operations, ctx=_ctx())

    assert result.written_uris == []
    assert result.edited_uris == []
    assert result.deleted_uris == []
    assert any("predates the current admission policy" in str(error) for _, error in result.errors)


@pytest.mark.asyncio
async def test_non_experience_operation_cannot_overwrite_experience_uri(monkeypatch):
    updater = MemoryUpdater(registry=object())
    updater._viking_fs = object()
    monkeypatch.setattr(updater, "_apply_upsert", pytest.fail)
    uri = "viking://user/u/memories/experiences/collision.md"
    operations = ResolvedOperations(
        upsert_operations=[
            ResolvedOperation(
                memory_type="skills",
                memory_fields={"skill_name": "collision", "content": "unstructured"},
                uris=[uri],
            )
        ],
        delete_file_contents=[],
        errors=[],
    )

    result = await updater.apply_operations(operations, ctx=_ctx())

    assert result.written_uris == []
    assert result.edited_uris == []
    assert any("Only an experiences operation" in str(error) for _, error in result.errors)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored_file",
    [
        MemoryFile(
            uri="viking://user/u/memories/experiences/legacy.md",
            content=VALID_CONTENT,
            memory_type=None,
            extra_fields={"experience_name": "legacy"},
        ),
        MemoryFile(
            uri="viking://user/u/memories/experiences/legacy.md",
            content=VALID_CONTENT,
            memory_type="experiences",
            extra_fields={"experience_name": "different"},
        ),
        MemoryFile(
            uri="viking://user/u/memories/experiences/legacy.md",
            content="",
            memory_type=None,
            extra_fields={},
        ),
    ],
)
async def test_authoritative_readback_rejects_legacy_identity_before_write(
    monkeypatch, stored_file
):
    uri = "viking://user/u/memories/experiences/legacy.md"
    fs = SimpleNamespace(read_file=AsyncMock(return_value=MemoryFileUtils.write(stored_file)))
    updater = MemoryUpdater(registry=object())
    updater._viking_fs = fs
    monkeypatch.setattr(updater, "_apply_upsert", pytest.fail)
    operations = ResolvedOperations(
        upsert_operations=[
            ResolvedOperation(
                memory_type="experiences",
                memory_fields={"experience_name": "legacy", "content": VALID_CONTENT},
                uris=[uri],
            )
        ],
        delete_file_contents=[],
        errors=[],
    )

    result = await updater.apply_operations(operations, ctx=_ctx())

    assert result.written_uris == []
    assert result.edited_uris == []
    assert any("legacy-invalid and read-only" in str(error) for _, error in result.errors)


@pytest.mark.asyncio
async def test_apply_upsert_validates_final_materialized_file_before_write(monkeypatch):
    uri = "viking://user/u/memories/experiences/verification.md"
    existing = MemoryFile(
        uri=uri,
        content=VALID_CONTENT,
        memory_type="experiences",
        extra_fields={"experience_name": "verification", "supersedes": ""},
    )
    fs = SimpleNamespace(
        read_file=AsyncMock(return_value=MemoryFileUtils.write(existing)),
        write_file=AsyncMock(),
    )
    updater = MemoryUpdater(registry=create_default_registry())
    updater._viking_fs = fs
    real_validator = memory_updater_module.validate_stored_experience
    calls = 0

    def reject_final(memory_file, target_uri):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_validator(memory_file, target_uri)
        return [{"code": "forced_final_identity_failure", "message": "invalid final file"}]

    monkeypatch.setattr(memory_updater_module, "validate_stored_experience", reject_final)
    operation = ResolvedOperation(
        memory_type="experiences",
        memory_fields={"experience_name": "verification", "content": VALID_CONTENT},
        uris=[uri],
        old_memory_file_content=existing,
    )

    with pytest.raises(ValueError, match="final stored-file admission"):
        await updater._apply_upsert(operation, _ctx())

    assert calls == 2
    fs.write_file.assert_not_awaited()


def _configure_link_test_updater(monkeypatch, *, apply_side_effect):
    fs = SimpleNamespace(
        read_file=AsyncMock(side_effect=NotFoundError("missing", "file")),
        rm=AsyncMock(),
    )
    updater = MemoryUpdater(registry=SimpleNamespace(get=lambda _memory_type: None))
    updater._viking_fs = fs
    monkeypatch.setattr(updater, "_apply_upsert", AsyncMock(side_effect=apply_side_effect))
    monkeypatch.setattr(updater, "_inherit_deleted_link_relations", AsyncMock())
    monkeypatch.setattr(updater, "_apply_links_to_existing_files", AsyncMock())
    monkeypatch.setattr(updater, "_sync_resource_refs_for_result", AsyncMock())
    monkeypatch.setattr(updater, "_vectorize_memories", AsyncMock())
    monkeypatch.setattr(updater, "generate_overview", AsyncMock())
    return updater


@pytest.mark.asyncio
async def test_failed_experience_does_not_publish_link_to_existing_neighbor(monkeypatch):
    experience_uri = "viking://user/u/memories/experiences/failed.md"
    trajectory_uri = "viking://user/u/memories/trajectories/source.md"
    updater = _configure_link_test_updater(
        monkeypatch,
        apply_side_effect=RuntimeError("write failed"),
    )
    operations = ResolvedOperations(
        upsert_operations=[
            ResolvedOperation(
                memory_type="experiences",
                memory_fields={"experience_name": "failed", "content": VALID_CONTENT},
                uris=[experience_uri],
            )
        ],
        delete_file_contents=[],
        errors=[],
        resolved_links=[StoredLink(from_uri=experience_uri, to_uri=trajectory_uri)],
    )

    result = await updater.apply_operations(operations, ctx=_ctx())

    assert result.written_uris == []
    assert result.errors
    updater._apply_links_to_existing_files.assert_not_awaited()


@pytest.mark.asyncio
async def test_successful_earlier_upsert_does_not_embed_link_to_later_failed_upsert(monkeypatch):
    first_uri = "viking://user/u/memories/experiences/first.md"
    second_uri = "viking://user/u/memories/experiences/second.md"
    updater = _configure_link_test_updater(
        monkeypatch,
        apply_side_effect=[None, RuntimeError("second write failed")],
    )
    operations = ResolvedOperations(
        upsert_operations=[
            ResolvedOperation(
                memory_type="experiences",
                memory_fields={"experience_name": "first", "content": VALID_CONTENT},
                uris=[first_uri],
            ),
            ResolvedOperation(
                memory_type="experiences",
                memory_fields={
                    "experience_name": "second",
                    "content": VALID_CONTENT.replace("prior procedural", "another narrow"),
                },
                uris=[second_uri],
            ),
        ],
        delete_file_contents=[],
        errors=[],
        resolved_links=[StoredLink(from_uri=first_uri, to_uri=second_uri)],
    )

    result = await updater.apply_operations(operations, ctx=_ctx())

    assert result.written_uris == []
    assert any(uri == second_uri for uri, _ in result.errors)
    updater._apply_links_to_existing_files.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_declared_provenance_endpoint_rolls_back_successful_experience(monkeypatch):
    experience_uri = "viking://user/u/memories/experiences/verification.md"
    trajectory_uri = "viking://user/u/memories/trajectories/source.md"
    fs = JournalFS()
    updater = MemoryUpdater(
        registry=SimpleNamespace(get=lambda _memory_type: None),
        transaction_handle=object(),
    )
    updater._viking_fs = fs

    async def write_experience_then_fail_trajectory(
        operation, ctx, extract_context=None, lease_ref=None
    ):
        del ctx, extract_context, lease_ref
        uri = operation.uris[0]
        if operation.memory_type != "experiences":
            raise RuntimeError("injected trajectory write failure")
        memory_file = MemoryFile(
            uri=uri,
            content=operation.memory_fields["content"],
            memory_type="experiences",
            extra_fields={
                "experience_name": operation.memory_fields["experience_name"],
                "supersedes": "",
            },
        )
        await fs.write_file(uri, MemoryFileUtils.write(memory_file))

    monkeypatch.setattr(
        updater,
        "_apply_upsert",
        AsyncMock(side_effect=write_experience_then_fail_trajectory),
    )
    publish_links = AsyncMock()
    sync_refs = AsyncMock()
    vectorize = AsyncMock()
    overview = AsyncMock()
    monkeypatch.setattr(updater, "_apply_links_to_existing_files", publish_links)
    monkeypatch.setattr(updater, "_sync_resource_refs_for_result", sync_refs)
    monkeypatch.setattr(updater, "_vectorize_memories", vectorize)
    monkeypatch.setattr(updater, "generate_overview", overview)
    operations = ResolvedOperations(
        upsert_operations=[
            ResolvedOperation(
                memory_type="experiences",
                memory_fields={"experience_name": "verification", "content": VALID_CONTENT},
                uris=[experience_uri],
            ),
            ResolvedOperation(
                memory_type="trajectories",
                memory_fields={"trajectory_name": "source", "content": "trajectory"},
                uris=[trajectory_uri],
            ),
        ],
        delete_file_contents=[],
        errors=[],
        resolved_links=[
            StoredLink(
                from_uri=experience_uri,
                to_uri=trajectory_uri,
                link_type="derived_from",
            )
        ],
    )

    result = await updater.apply_operations(operations, _ctx())

    assert experience_uri not in fs.files
    assert result.written_uris == []
    assert result.edited_uris == []
    assert any(
        uri == experience_uri and "relation endpoint did not publish" in str(error)
        for uri, error in result.errors
    )
    publish_links.assert_not_awaited()
    sync_refs.assert_not_awaited()
    vectorize.assert_not_awaited()
    overview.assert_not_awaited()


@pytest.mark.asyncio
async def test_deleted_provenance_endpoint_rolls_back_experience_before_derived_publication(
    monkeypatch,
):
    experience_uri = "viking://user/u/memories/experiences/verification.md"
    trajectory_uri = "viking://user/u/memories/trajectories/source.md"
    trajectory_blob = MemoryFileUtils.write(
        MemoryFile(
            uri=trajectory_uri,
            content="trajectory",
            memory_type="trajectories",
            extra_fields={"trajectory_name": "source"},
        )
    )
    fs = JournalFS({trajectory_uri: trajectory_blob})
    updater = MemoryUpdater(
        registry=SimpleNamespace(get=lambda _memory_type: None),
        transaction_handle=object(),
    )
    updater._viking_fs = fs

    async def write_experience(operation, ctx, extract_context=None, lease_ref=None):
        del ctx, extract_context, lease_ref
        memory_file = MemoryFile(
            uri=experience_uri,
            content=operation.memory_fields["content"],
            memory_type="experiences",
            extra_fields={"experience_name": "verification", "supersedes": ""},
        )
        await fs.write_file(experience_uri, MemoryFileUtils.write(memory_file))

    monkeypatch.setattr(updater, "_apply_upsert", AsyncMock(side_effect=write_experience))
    write_links = AsyncMock()
    sync_refs = AsyncMock()
    vectorize = AsyncMock()
    overview = AsyncMock()
    monkeypatch.setattr(memory_updater_module, "write_stored_links", write_links)
    monkeypatch.setattr(updater, "_sync_resource_refs_for_result", sync_refs)
    monkeypatch.setattr(updater, "_vectorize_memories", vectorize)
    monkeypatch.setattr(updater, "generate_overview", overview)
    operations = ResolvedOperations(
        upsert_operations=[
            ResolvedOperation(
                memory_type="experiences",
                memory_fields={"experience_name": "verification", "content": VALID_CONTENT},
                uris=[experience_uri],
            )
        ],
        delete_file_contents=[MemoryFile(uri=trajectory_uri, memory_type="trajectories")],
        errors=[],
        resolved_links=[
            StoredLink(
                from_uri=experience_uri,
                to_uri=trajectory_uri,
                link_type="derived_from",
            )
        ],
    )

    result = await updater.apply_operations(operations, _ctx())

    assert experience_uri not in fs.files
    assert trajectory_uri not in fs.files
    assert result.written_uris == []
    assert result.edited_uris == []
    assert result.deleted_uris == [trajectory_uri]
    assert any(
        uri == experience_uri and "endpoint was deleted" in str(error)
        for uri, error in result.errors
    )
    write_links.assert_not_awaited()
    sync_refs.assert_not_awaited()
    vectorize.assert_not_awaited()
    overview.assert_not_awaited()


@pytest.mark.asyncio
async def test_experience_content_failure_deletes_every_new_attempt_before_refs(monkeypatch):
    first_uri = "viking://user/u/memories/experiences/first.md"
    second_uri = "viking://user/u/memories/experiences/second.md"
    fs = JournalFS()
    updater = MemoryUpdater(
        registry=SimpleNamespace(get=lambda _memory_type: None),
        transaction_handle=object(),
    )
    updater._viking_fs = fs

    async def write_then_fail_second(operation, ctx, extract_context=None, lease_ref=None):
        del ctx, extract_context, lease_ref
        uri = operation.uris[0]
        memory_file = MemoryFile(
            uri=uri,
            content=operation.memory_fields["content"],
            memory_type="experiences",
            extra_fields={
                "experience_name": operation.memory_fields["experience_name"],
                "supersedes": "",
            },
        )
        await fs.write_file(uri, MemoryFileUtils.write(memory_file))
        if uri == second_uri:
            raise RuntimeError("injected second experience content failure")

    monkeypatch.setattr(updater, "_apply_upsert", AsyncMock(side_effect=write_then_fail_second))
    sync_refs = AsyncMock()
    vectorize = AsyncMock()
    overview = AsyncMock()
    monkeypatch.setattr(updater, "_sync_resource_refs_for_result", sync_refs)
    monkeypatch.setattr(updater, "_vectorize_memories", vectorize)
    monkeypatch.setattr(updater, "generate_overview", overview)
    operations = ResolvedOperations(
        upsert_operations=[
            ResolvedOperation(
                memory_type="experiences",
                memory_fields={"experience_name": "first", "content": VALID_CONTENT},
                uris=[first_uri],
            ),
            ResolvedOperation(
                memory_type="experiences",
                memory_fields={
                    "experience_name": "second",
                    "content": VALID_CONTENT.replace(
                        "prior procedural guidance", "another bounded lesson"
                    ),
                },
                uris=[second_uri],
            ),
        ],
        delete_file_contents=[],
        errors=[],
    )

    result = await updater.apply_operations(operations, _ctx())

    assert first_uri not in fs.files
    assert second_uri not in fs.files
    assert result.written_uris == []
    assert result.index_pending_uris == []
    assert result.errors
    sync_refs.assert_not_awaited()
    vectorize.assert_not_awaited()
    overview.assert_not_awaited()


@pytest.mark.asyncio
async def test_experience_content_failure_restores_existing_blob_exactly(monkeypatch):
    uri = "viking://user/u/memories/experiences/verification.md"
    existing = MemoryFile(
        uri=uri,
        content=VALID_CONTENT,
        memory_type="experiences",
        extra_fields={"experience_name": "verification", "supersedes": "", "version": 4},
    )
    original_blob = MemoryFileUtils.write(existing)
    fs = JournalFS({uri: original_blob})
    updater = MemoryUpdater(
        registry=SimpleNamespace(get=lambda _memory_type: None),
        transaction_handle=object(),
    )
    updater._viking_fs = fs

    async def overwrite_then_fail(operation, ctx, extract_context=None, lease_ref=None):
        del operation, ctx, extract_context, lease_ref
        changed = MemoryFile(
            uri=uri,
            content=VALID_CONTENT.replace(
                "prior procedural guidance", "temporarily changed guidance"
            ),
            memory_type="experiences",
            extra_fields={"experience_name": "verification", "supersedes": "", "version": 5},
        )
        await fs.write_file(uri, MemoryFileUtils.write(changed))
        raise RuntimeError("injected existing experience write failure")

    monkeypatch.setattr(updater, "_apply_upsert", AsyncMock(side_effect=overwrite_then_fail))
    sync_refs = AsyncMock()
    vectorize = AsyncMock()
    overview = AsyncMock()
    monkeypatch.setattr(updater, "_sync_resource_refs_for_result", sync_refs)
    monkeypatch.setattr(updater, "_vectorize_memories", vectorize)
    monkeypatch.setattr(updater, "generate_overview", overview)
    operations = ResolvedOperations(
        upsert_operations=[
            ResolvedOperation(
                memory_type="experiences",
                memory_fields={"experience_name": "verification", "content": VALID_CONTENT},
                uris=[uri],
            )
        ],
        delete_file_contents=[],
        errors=[],
    )

    result = await updater.apply_operations(operations, _ctx())

    assert fs.files[uri] == original_blob
    assert result.written_uris == []
    assert result.edited_uris == []
    assert result.errors
    sync_refs.assert_not_awaited()
    vectorize.assert_not_awaited()
    overview.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_vector_backend_retains_canonical_experience_as_index_pending(monkeypatch):
    uri = "viking://user/u/memories/experiences/verification.md"
    fs = JournalFS()
    updater = MemoryUpdater(
        registry=create_default_registry(),
        vikingdb=None,
        transaction_handle=object(),
    )
    updater._viking_fs = fs
    sync_refs = AsyncMock()
    overview = AsyncMock()
    monkeypatch.setattr(updater, "_sync_resource_refs_for_result", sync_refs)
    monkeypatch.setattr(updater, "generate_overview", overview)
    operations = ResolvedOperations(
        upsert_operations=[
            ResolvedOperation(
                memory_type="experiences",
                memory_fields={"experience_name": "verification", "content": VALID_CONTENT},
                uris=[uri],
            )
        ],
        delete_file_contents=[],
        errors=[],
    )

    result = await updater.apply_operations(operations, _ctx())

    assert result.errors == []
    assert result.written_uris == [uri]
    assert result.index_pending_uris == [uri]
    assert uri in fs.files
    sync_refs.assert_awaited_once()
    overview.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("enqueue_failure", [False, RuntimeError("enqueue failed")])
async def test_rejected_vector_enqueue_retains_canonical_experience_as_pending(
    monkeypatch, enqueue_failure
):
    uri = "viking://user/u/memories/experiences/verification.md"
    fs = JournalFS()
    enqueue = (
        AsyncMock(side_effect=enqueue_failure)
        if isinstance(enqueue_failure, Exception)
        else AsyncMock(return_value=enqueue_failure)
    )
    updater = MemoryUpdater(
        registry=create_default_registry(),
        vikingdb=SimpleNamespace(enqueue_embedding_msg=enqueue),
        transaction_handle=object(),
    )
    updater._viking_fs = fs
    sync_refs = AsyncMock()
    overview = AsyncMock()
    monkeypatch.setattr(updater, "_sync_resource_refs_for_result", sync_refs)
    monkeypatch.setattr(updater, "generate_overview", overview)
    operations = ResolvedOperations(
        upsert_operations=[
            ResolvedOperation(
                memory_type="experiences",
                memory_fields={"experience_name": "verification", "content": VALID_CONTENT},
                uris=[uri],
            )
        ],
        delete_file_contents=[],
        errors=[],
    )

    result = await updater.apply_operations(operations, _ctx())

    assert result.errors == []
    assert result.written_uris == [uri]
    assert result.index_pending_uris == [uri]
    assert uri in fs.files
    enqueue.assert_awaited_once()
    sync_refs.assert_awaited_once()
    overview.assert_awaited_once()
