# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from openviking.message import Message, TextPart
from openviking.server.identity import RequestContext, Role
from openviking.session.compressor_v2 import SessionCompressorV2
from openviking.session.memory.dataclass import (
    MemoryField,
    MemoryFile,
    MemoryTypeSchema,
    ResolvedOperation,
    ResolvedOperations,
)
from openviking.session.memory.memory_type_registry import MemoryTypeRegistry
from openviking.session.memory.memory_updater import ExtractContext, MemoryUpdater
from openviking.session.memory.merge_op import FieldType, MergeOp
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking_cli.exceptions import NotFoundError
from openviking_cli.session.user_id import UserIdentifier


async def _run_experience_source_publication(*, fail_backlink_write: bool):
    compressor = SessionCompressorV2(vikingdb=None)
    ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.ROOT)
    messages = [Message(id="msg-test", role="user", parts=[TextPart("test")])]
    experience_uri = "viking://user/default/memories/experiences/bounded_publish.md"
    trajectory_uri = "viking://user/default/memories/trajectories/traj-1.md"

    class FakeVikingFS:
        agfs = None

        def __init__(self):
            self.files = {
                trajectory_uri: MemoryFileUtils.write(
                    MemoryFile(
                        uri=trajectory_uri,
                        content="traj content",
                        memory_type="trajectories",
                        extra_fields={"version": 7},
                    )
                ),
            }
            self.failed_backlink_write = False
            self.successful_writes = []

        async def read_file(self, uri: str, **kwargs):
            if uri not in self.files:
                raise NotFoundError(uri)
            return self.files[uri]

        async def write_file(self, uri: str, content: str, **kwargs):
            parsed = MemoryFileUtils.read(content, uri=uri)
            if (
                fail_backlink_write
                and uri == trajectory_uri
                and parsed.backlinks
                and not self.failed_backlink_write
            ):
                self.failed_backlink_write = True
                raise RuntimeError("injected trajectory backlink failure")
            self.files[uri] = content
            self.successful_writes.append((uri, content))

        async def rm(self, uri: str, **kwargs):
            del kwargs
            if uri not in self.files:
                raise NotFoundError(uri)
            del self.files[uri]

    schema = MemoryTypeSchema(
        memory_type="experiences",
        directory="viking://user/{{ user_space }}/memories/experiences",
        filename_template="{{ experience_name }}.md",
        operation_mode="upsert",
        stage="agent",
        fields=[
            MemoryField(
                name="experience_name",
                field_type=FieldType.STRING,
                merge_op=MergeOp.IMMUTABLE,
            ),
            MemoryField(
                name="content",
                field_type=FieldType.STRING,
                merge_op=MergeOp.REPLACE,
            ),
            MemoryField(
                name="supersedes",
                field_type=FieldType.STRING,
                merge_op=MergeOp.REPLACE,
            ),
        ],
    )
    registry = MemoryTypeRegistry(load_schemas=False)
    registry.register(schema)
    operations = ResolvedOperations(
        upsert_operations=[
            ResolvedOperation(
                old_memory_file_content=None,
                memory_fields={
                    "experience_name": "bounded_publish",
                    "content": (
                        "## Situation\n"
                        "- A publication needs two-sided provenance.\n\n"
                        "## Approach\n"
                        "- Publish both relation endpoints as one compensated batch.\n\n"
                        "## Reflect\n"
                        "- NEVER expose a one-sided provenance edge."
                    ),
                    "supersedes": "",
                },
                memory_type="experiences",
                uris=[experience_uri],
            )
        ],
        delete_file_contents=[],
        errors=[],
    )

    class DummyProvider:
        async def prepare_extraction_messages(self):
            pass

        def get_memory_schemas(self, _ctx):
            return [schema]

        def get_extract_context(self):
            return ExtractContext(messages)

        def _get_registry(self):
            return registry

    class DummyExtractLoop:
        def __init__(self, **kwargs):
            pass

        async def run(self):
            return operations, []

    viking_fs = FakeVikingFS()
    updater = MemoryUpdater(registry=registry)
    updater._viking_fs = viking_fs
    updater._sync_resource_refs_for_result = AsyncMock()
    updater._vectorize_memories = AsyncMock()
    updater.generate_overview = AsyncMock()

    class CapturingUpdater:
        result = None

        async def apply_operations(self, *args, **kwargs):
            self.result = await updater.apply_operations(*args, **kwargs)
            return self.result

    capturing_updater = CapturingUpdater()
    config = SimpleNamespace(vlm=SimpleNamespace(get_vlm_instance=lambda: object()))
    provider = DummyProvider()
    provider.trajectory_uri = trajectory_uri

    with (
        patch("openviking.session.compressor_v2.get_viking_fs", return_value=viking_fs),
        patch("openviking.session.compressor_v2.get_openviking_config", return_value=config),
        patch("openviking.session.compressor_v2.ExtractLoop", DummyExtractLoop),
        patch.object(compressor, "_get_or_create_updater", return_value=capturing_updater),
    ):
        phase_result = await compressor._run_extract_phase(
            provider=provider,
            messages=messages,
            ctx=ctx,
            strict_extract_errors=True,
            phase_label="experience(test)",
            allowed_memory_types={"experiences"},
        )

    return (
        phase_result,
        capturing_updater.result,
        updater,
        viking_fs,
        experience_uri,
        trajectory_uri,
    )


@pytest.mark.asyncio
async def test_experience_source_link_success_publishes_both_sides_once():
    phase_result, result, updater, viking_fs, experience_uri, trajectory_uri = (
        await _run_experience_source_publication(fail_backlink_write=False)
    )

    assert result.errors == []
    assert phase_result[0] == [experience_uri]
    assert phase_result[1] == [trajectory_uri]
    assert [(context.uri, context.category) for context in phase_result[2]] == [
        (experience_uri, "memory_write"),
        (trajectory_uri, "memory_edit"),
    ]
    experience_file = MemoryFileUtils.read(
        viking_fs.files[experience_uri], uri=experience_uri
    )
    trajectory_file = MemoryFileUtils.read(
        viking_fs.files[trajectory_uri], uri=trajectory_uri
    )
    assert [(link["from_uri"], link["to_uri"]) for link in experience_file.links] == [
        (experience_uri, trajectory_uri)
    ]
    assert [
        (link["from_uri"], link["to_uri"]) for link in trajectory_file.backlinks
    ] == [(experience_uri, trajectory_uri)]
    assert experience_file.extra_fields["version"] == 1
    assert trajectory_file.extra_fields["version"] == 8

    relation_write_uris = []
    for uri, content in viking_fs.successful_writes:
        parsed = MemoryFileUtils.read(content, uri=uri)
        if parsed.links or parsed.backlinks:
            relation_write_uris.append(uri)
    assert relation_write_uris.count(experience_uri) == 1
    assert relation_write_uris.count(trajectory_uri) == 1
    updater._sync_resource_refs_for_result.assert_awaited_once()
    updater._vectorize_memories.assert_awaited_once()
    updater.generate_overview.assert_awaited_once()


@pytest.mark.asyncio
async def test_experience_backlink_failure_rolls_back_and_suppresses_publication():
    phase_result, result, updater, viking_fs, experience_uri, trajectory_uri = (
        await _run_experience_source_publication(fail_backlink_write=True)
    )

    assert result.errors
    assert "trajectory backlink failure" in str(result.errors[0][1])
    assert phase_result[0] == []
    assert phase_result[1] == []
    assert phase_result[2] == []
    assert experience_uri not in viking_fs.files
    trajectory_file = MemoryFileUtils.read(
        viking_fs.files[trajectory_uri], uri=trajectory_uri
    )
    assert trajectory_file.backlinks == []
    updater._sync_resource_refs_for_result.assert_not_awaited()
    updater._vectorize_memories.assert_not_awaited()
    updater.generate_overview.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_experience_phase_does_not_guess_a_sole_existing_experience():
    compressor = SessionCompressorV2(vikingdb=None)
    ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.ROOT)
    messages = [Message(id="msg-test", role="user", parts=[TextPart("test")])]
    trajectory_uri = "viking://user/default/memories/trajectories/traj-1.md"
    viking_fs = SimpleNamespace(
        read_file=AsyncMock(
            return_value=MemoryFileUtils.write(
                MemoryFile(
                    uri=trajectory_uri,
                    content="traj content",
                    memory_type="trajectories",
                )
            )
        ),
        ls=AsyncMock(),
        write_file=AsyncMock(),
    )
    compressor._run_extract_phase = AsyncMock(
        side_effect=[
            ([trajectory_uri], [], [], {}, []),
            None,
        ]
    )
    config = SimpleNamespace(
        memory=SimpleNamespace(session_skill_extraction_enabled=False),
    )

    with (
        patch("openviking.session.compressor_v2.get_viking_fs", return_value=viking_fs),
        patch("openviking.session.compressor_v2.get_openviking_config", return_value=config),
    ):
        result = await compressor.extract_execution_memories(messages=messages, ctx=ctx)

    assert result == {"contexts": [], "session_skills": []}
    assert compressor._run_extract_phase.await_count == 2
    viking_fs.ls.assert_not_awaited()
    viking_fs.write_file.assert_not_awaited()
