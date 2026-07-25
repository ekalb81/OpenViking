from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.session.memory.dataclass import (
    MemoryFile,
    ResolvedOperation,
    ResolvedOperations,
    StoredLink,
)
from openviking.session.memory.memory_type_registry import create_default_registry
from openviking.session.memory.memory_updater import MemoryUpdater, MemoryUpdateResult
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


def _ctx() -> RequestContext:
    return RequestContext(user=UserIdentifier("account", "u"), role=Role.USER)


class DictFS:
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


class FailSecondWriteOnceFS(DictFS):
    def __init__(self, files=None, *, fail_on_attempt=2):
        super().__init__(files)
        self.write_attempts = 0
        self.fail_on_attempt = fail_on_attempt

    async def write_file(self, uri, content, **kwargs):
        self.write_attempts += 1
        if self.write_attempts == self.fail_on_attempt:
            raise RuntimeError("injected endpoint write failure")
        await super().write_file(uri, content, **kwargs)


class CorruptSecondWriteOnceFS(DictFS):
    def __init__(self, files=None):
        super().__init__(files)
        self.write_attempts = 0

    async def write_file(self, uri, content, **kwargs):
        self.write_attempts += 1
        if self.write_attempts == 2:
            self.files[uri] = f"{content}\ncorrupted"
            return
        await super().write_file(uri, content, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(("existing_version", "expected_version"), [(None, 1), (1, 2)])
async def test_same_batch_links_do_not_double_bump_experience_version(
    monkeypatch, existing_version, expected_version
):
    experience_uri = "viking://user/u/memories/experiences/verification.md"
    trajectory_uri = "viking://user/u/memories/trajectories/source.md"
    files = {
        trajectory_uri: MemoryFileUtils.write(
            MemoryFile(
                uri=trajectory_uri,
                content="trajectory content",
                memory_type="trajectories",
                extra_fields={"trajectory_name": "source", "version": 1},
            )
        )
    }
    old_file = None
    if existing_version is not None:
        old_file = MemoryFile(
            uri=experience_uri,
            content=VALID_CONTENT,
            memory_type="experiences",
            extra_fields={
                "experience_name": "verification",
                "supersedes": "",
                "version": existing_version,
            },
        )
        files[experience_uri] = MemoryFileUtils.write(old_file)

    fs = DictFS(files)
    updater = MemoryUpdater(registry=create_default_registry())
    updater._viking_fs = fs
    monkeypatch.setattr(updater, "generate_overview", AsyncMock())
    operation = ResolvedOperation(
        old_memory_file_content=old_file,
        memory_type="experiences",
        memory_fields={"experience_name": "verification", "content": VALID_CONTENT},
        uris=[experience_uri],
    )
    operations = ResolvedOperations(
        upsert_operations=[operation],
        delete_file_contents=[],
        errors=[],
        resolved_links=[StoredLink(from_uri=experience_uri, to_uri=trajectory_uri)],
    )

    result = await updater.apply_operations(operations, _ctx())

    assert result.errors == []
    stored_experience = MemoryFileUtils.read(fs.files[experience_uri], uri=experience_uri)
    stored_trajectory = MemoryFileUtils.read(fs.files[trajectory_uri], uri=trajectory_uri)
    assert stored_experience.extra_fields["version"] == expected_version
    assert stored_trajectory.extra_fields["version"] == 2
    assert any(link["to_uri"] == trajectory_uri for link in stored_experience.links)
    assert any(link["from_uri"] == experience_uri for link in stored_trajectory.backlinks)
    assert trajectory_uri in result.edited_uris


@pytest.mark.asyncio
async def test_failed_replacement_write_preserves_non_experience_source(monkeypatch):
    source_uri = "viking://user/u/memories/skills/source/SKILL.md"
    replacement_uri = "viking://user/u/memories/skills/replacement/SKILL.md"
    updater = MemoryUpdater(registry=SimpleNamespace(get=lambda _memory_type: None))
    updater._viking_fs = object()
    monkeypatch.setattr(updater, "_apply_upsert", AsyncMock(side_effect=RuntimeError("failed")))
    monkeypatch.setattr(updater, "_apply_delete", AsyncMock())
    monkeypatch.setattr(updater, "_inherit_deleted_link_relations", AsyncMock())
    monkeypatch.setattr(updater, "_sync_resource_refs_for_result", AsyncMock())
    monkeypatch.setattr(updater, "_vectorize_memories", AsyncMock())
    monkeypatch.setattr(updater, "generate_overview", AsyncMock())
    operations = ResolvedOperations(
        upsert_operations=[
            ResolvedOperation(
                memory_type="skills",
                memory_fields={"skill_name": "replacement", "content": "replacement"},
                uris=[replacement_uri],
            )
        ],
        delete_file_contents=[MemoryFile(uri=source_uri, memory_type="skills")],
        errors=[],
        delete_replacements={source_uri: replacement_uri},
    )

    result = await updater.apply_operations(operations, _ctx())

    updater._apply_delete.assert_not_awaited()
    updater._inherit_deleted_link_relations.assert_not_awaited()
    assert any("replacement write failed" in str(error) for _, error in result.errors)


@pytest.mark.asyncio
async def test_inherited_replacement_links_are_written_to_both_endpoints_without_double_bump():
    source_uri = "viking://user/u/memories/skills/source/SKILL.md"
    replacement_uri = "viking://user/u/memories/skills/replacement/SKILL.md"
    neighbor_uri = "viking://user/u/memories/tools/neighbor.md"
    original_link = StoredLink(from_uri=source_uri, to_uri=neighbor_uri).model_dump()
    files = {
        source_uri: MemoryFileUtils.write(
            MemoryFile(uri=source_uri, content="source", links=[original_link], memory_type="skills")
        ),
        replacement_uri: MemoryFileUtils.write(
            MemoryFile(
                uri=replacement_uri,
                content="replacement",
                memory_type="skills",
                extra_fields={"version": 1},
            )
        ),
        neighbor_uri: MemoryFileUtils.write(
            MemoryFile(
                uri=neighbor_uri,
                content="neighbor",
                backlinks=[original_link],
                memory_type="tools",
                extra_fields={"version": 1},
            )
        ),
    }
    fs = DictFS(files)
    updater = MemoryUpdater(registry=object())
    updater._viking_fs = fs
    result = MemoryUpdateResult()
    result.add_written(replacement_uri)
    operations = SimpleNamespace(delete_replacements={source_uri: replacement_uri})

    await updater._inherit_deleted_link_relations(operations, result, _ctx())

    replacement = MemoryFileUtils.read(fs.files[replacement_uri], uri=replacement_uri)
    neighbor = MemoryFileUtils.read(fs.files[neighbor_uri], uri=neighbor_uri)
    assert any(
        link["from_uri"] == replacement_uri and link["to_uri"] == neighbor_uri
        for link in replacement.links
    )
    assert any(
        link["from_uri"] == replacement_uri and link["to_uri"] == neighbor_uri
        for link in neighbor.backlinks
    )
    assert replacement.extra_fields["version"] == 1
    assert neighbor.extra_fields["version"] == 2


@pytest.mark.asyncio
async def test_failed_inherited_link_batch_rolls_back_and_preserves_source(monkeypatch):
    source_uri = "viking://user/u/memories/skills/source/SKILL.md"
    replacement_uri = "viking://user/u/memories/skills/replacement/SKILL.md"
    neighbor_uri = "viking://user/u/memories/tools/neighbor.md"
    original_link = StoredLink(from_uri=source_uri, to_uri=neighbor_uri).model_dump()
    files = {
        source_uri: MemoryFileUtils.write(
            MemoryFile(uri=source_uri, content="source", links=[original_link], memory_type="skills")
        ),
        replacement_uri: MemoryFileUtils.write(
            MemoryFile(
                uri=replacement_uri,
                content="replacement",
                memory_type="skills",
                extra_fields={"version": 1},
            )
        ),
        neighbor_uri: MemoryFileUtils.write(
            MemoryFile(
                uri=neighbor_uri,
                content="neighbor",
                backlinks=[original_link],
                memory_type="tools",
                extra_fields={"version": 1},
            )
        ),
    }
    fs = FailSecondWriteOnceFS(files)
    original_files = dict(fs.files)
    updater = MemoryUpdater(
        registry=SimpleNamespace(get=lambda _memory_type: None),
        transaction_handle=object(),
    )
    updater._viking_fs = fs
    monkeypatch.setattr(updater, "_apply_upsert", AsyncMock())
    monkeypatch.setattr(updater, "_apply_delete", AsyncMock())
    monkeypatch.setattr(updater, "_sync_resource_refs_for_result", AsyncMock())
    monkeypatch.setattr(updater, "_vectorize_memories", AsyncMock())
    monkeypatch.setattr(updater, "generate_overview", AsyncMock())
    operations = ResolvedOperations(
        upsert_operations=[
            ResolvedOperation(
                memory_type="skills",
                memory_fields={"skill_name": "replacement", "content": "replacement"},
                uris=[replacement_uri],
            )
        ],
        delete_file_contents=[MemoryFile(uri=source_uri, memory_type="skills")],
        errors=[],
        resolved_links=[],
        delete_replacements={source_uri: replacement_uri},
    )

    result = await updater.apply_operations(operations, _ctx())

    updater._apply_delete.assert_not_awaited()
    assert operations.delete_replacements == {}
    assert fs.files == original_files
    assert any(uri == neighbor_uri for uri, _ in result.errors)
    assert any(
        uri == source_uri and "link inheritance failed" in str(error)
        for uri, error in result.errors
    )


@pytest.mark.asyncio
async def test_deferred_link_publication_rolls_back_first_endpoint_and_reports_failure():
    source_uri = "viking://user/u/memories/skills/source/SKILL.md"
    target_uri = "viking://user/u/memories/tools/target.md"
    files = {
        source_uri: MemoryFileUtils.write(
            MemoryFile(
                uri=source_uri,
                content="source",
                memory_type="skills",
                extra_fields={"version": 3},
            )
        ),
        target_uri: MemoryFileUtils.write(
            MemoryFile(
                uri=target_uri,
                content="target",
                memory_type="tools",
                extra_fields={"version": 7},
            )
        ),
    }
    fs = FailSecondWriteOnceFS(files)
    original_files = dict(fs.files)
    updater = MemoryUpdater(registry=object(), transaction_handle=object())
    updater._viking_fs = fs
    result = MemoryUpdateResult()

    await updater._apply_links_to_existing_files(
        [StoredLink(from_uri=source_uri, to_uri=target_uri)],
        result,
        _ctx(),
    )

    assert result.written_uris == []
    assert result.edited_uris == []
    assert len(result.errors) == 1
    error_uri, error = result.errors[0]
    assert error_uri == target_uri
    assert "attempted endpoint writes were rolled back" in str(error)
    assert fs.files == original_files
    stored_source = MemoryFileUtils.read(fs.files[source_uri], uri=source_uri)
    stored_target = MemoryFileUtils.read(fs.files[target_uri], uri=target_uri)
    assert stored_source.links == []
    assert stored_target.backlinks == []


@pytest.mark.asyncio
async def test_link_readback_mismatch_compensates_both_endpoints():
    source_uri = "viking://user/u/memories/skills/source/SKILL.md"
    target_uri = "viking://user/u/memories/tools/target.md"
    files = {
        source_uri: MemoryFileUtils.write(
            MemoryFile(uri=source_uri, content="source", memory_type="skills")
        ),
        target_uri: MemoryFileUtils.write(
            MemoryFile(uri=target_uri, content="target", memory_type="tools")
        ),
    }
    fs = CorruptSecondWriteOnceFS(files)
    original_files = dict(fs.files)
    updater = MemoryUpdater(registry=object(), transaction_handle=object())
    updater._viking_fs = fs
    result = MemoryUpdateResult()

    published = await updater._apply_links_to_existing_files(
        [StoredLink(from_uri=source_uri, to_uri=target_uri)], result, _ctx()
    )

    assert published is False
    assert result.errors
    assert fs.files == original_files


@pytest.mark.asyncio
@pytest.mark.parametrize("foreign", [False, True])
async def test_link_publication_rejects_legacy_or_foreign_experience_endpoint(foreign):
    source_uri = "viking://user/u/memories/skills/source/SKILL.md"
    experience_user = "other" if foreign else "u"
    experience_uri = (
        f"viking://user/{experience_user}/memories/experiences/verification.md"
    )
    experience_content = VALID_CONTENT if foreign else "legacy unstructured guidance"
    files = {
        source_uri: MemoryFileUtils.write(
            MemoryFile(uri=source_uri, content="source", memory_type="skills")
        ),
        experience_uri: MemoryFileUtils.write(
            MemoryFile(
                uri=experience_uri,
                content=experience_content,
                memory_type="experiences",
                extra_fields={"experience_name": "verification", "supersedes": ""},
            )
        ),
    }
    fs = DictFS(files)
    original_files = dict(fs.files)
    updater = MemoryUpdater(registry=object(), transaction_handle=object())
    updater._viking_fs = fs
    result = MemoryUpdateResult()

    published = await updater._apply_links_to_existing_files(
        [StoredLink(from_uri=source_uri, to_uri=experience_uri)],
        result,
        _ctx(),
    )

    assert published is False
    assert any(uri == experience_uri for uri, _ in result.errors)
    assert fs.files == original_files


@pytest.mark.asyncio
async def test_failed_experience_links_leave_recoverable_body_unpublished(monkeypatch):
    experience_uri = "viking://user/u/memories/experiences/verification.md"
    trajectory_uri = "viking://user/u/memories/trajectories/source.md"
    trajectory_content = MemoryFileUtils.write(
        MemoryFile(
            uri=trajectory_uri,
            content="trajectory content",
            memory_type="trajectories",
            extra_fields={"trajectory_name": "source", "version": 1},
        )
    )
    fs = FailSecondWriteOnceFS(
        {trajectory_uri: trajectory_content},
        fail_on_attempt=3,
    )
    updater = MemoryUpdater(registry=create_default_registry(), transaction_handle=object())
    updater._viking_fs = fs
    sync_refs = AsyncMock()
    vectorize = AsyncMock()
    generate_overview = AsyncMock()
    monkeypatch.setattr(updater, "_sync_resource_refs_for_result", sync_refs)
    monkeypatch.setattr(updater, "_vectorize_memories", vectorize)
    monkeypatch.setattr(updater, "generate_overview", generate_overview)
    operations = ResolvedOperations(
        upsert_operations=[
            ResolvedOperation(
                old_memory_file_content=None,
                memory_type="experiences",
                memory_fields={
                    "experience_name": "verification",
                    "content": VALID_CONTENT,
                },
                uris=[experience_uri],
            )
        ],
        delete_file_contents=[],
        errors=[],
        resolved_links=[StoredLink(from_uri=experience_uri, to_uri=trajectory_uri)],
    )

    result = await updater.apply_operations(operations, _ctx())

    assert result.written_uris == []
    assert result.errors
    assert experience_uri not in fs.files
    assert MemoryFileUtils.read(fs.files[trajectory_uri], uri=trajectory_uri).backlinks == []
    sync_refs.assert_not_awaited()
    vectorize.assert_not_awaited()
    generate_overview.assert_not_awaited()
