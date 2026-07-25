from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from openviking.message import Message, TextPart
from openviking.server.identity import RequestContext, Role
from openviking.session.compressor_v2 import SessionCompressorV2
from openviking.session.memory.dataclass import MemoryFile, ResolvedOperation, ResolvedOperations
from openviking.session.memory.extract_loop import ExtractLoop
from openviking.session.memory.memory_updater import ExtractContext, MemoryUpdateResult
from openviking.session.memory.patch_merge_context_provider import PatchMergeContextProvider
from openviking_cli.session.user_id import UserIdentifier

VALID_CONTENT = """## Situation
- A bounded implementation needs prior procedural guidance.

## Approach
- Inspect the current state before changing it.

## Reflect
- NEVER claim completion without verification.
"""
EXPERIENCE_DIR = "viking://user/u/memories/experiences"


def _operation(name: str, supersedes: str) -> ResolvedOperation:
    return ResolvedOperation(
        memory_type="experiences",
        memory_fields={
            "experience_name": name,
            "content": VALID_CONTENT.replace("prior procedural guidance", name),
            "supersedes": supersedes,
        },
        uris=[f"{EXPERIENCE_DIR}/{name}.md"],
    )


def _operations(*upserts: ResolvedOperation) -> ResolvedOperations:
    return ResolvedOperations(
        upsert_operations=list(upserts),
        delete_file_contents=[],
        errors=[],
    )


@pytest.mark.asyncio
async def test_any_supersession_fails_before_source_read_or_delete():
    compressor = object.__new__(SessionCompressorV2)
    operation = _operation("renamed_lesson", "legacy")
    operations = _operations(operation)
    fs = SimpleNamespace(read_file=AsyncMock())
    provider = SimpleNamespace(_render_experience_dir=lambda ctx: EXPERIENCE_DIR)

    inheritance = await compressor._resolve_supersedes(operations, None, fs, provider)

    assert inheritance == {}
    assert operations.errors
    assert operations.delete_file_contents == []
    assert operations.delete_replacements == {}
    assert "supersedes" not in operation.memory_fields
    fs.read_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_supersedes_is_stripped_without_read_or_delete():
    compressor = object.__new__(SessionCompressorV2)
    operation = _operation("bounded_lesson", "")
    operations = _operations(operation)
    fs = SimpleNamespace(read_file=AsyncMock())
    provider = SimpleNamespace(_render_experience_dir=lambda ctx: EXPERIENCE_DIR)

    inheritance = await compressor._resolve_supersedes(operations, None, fs, provider)

    assert inheritance == {}
    assert operations.errors == []
    assert operations.delete_file_contents == []
    assert operations.delete_replacements == {}
    assert "supersedes" not in operation.memory_fields
    fs.read_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_experience_delete_intent_is_removed_and_fails_closed():
    compressor = object.__new__(SessionCompressorV2)
    legacy_uri = f"{EXPERIENCE_DIR}/legacy.md"
    legacy = MemoryFile(
        uri=legacy_uri,
        content=VALID_CONTENT,
        memory_type="experiences",
    )
    operations = ResolvedOperations(
        upsert_operations=[],
        delete_file_contents=[legacy],
        delete_replacements={legacy_uri: f"{EXPERIENCE_DIR}/replacement.md"},
        errors=[],
    )
    fs = SimpleNamespace(read_file=AsyncMock())
    provider = SimpleNamespace(_render_experience_dir=lambda ctx: EXPERIENCE_DIR)

    inheritance = await compressor._resolve_supersedes(operations, None, fs, provider)

    assert inheritance == {}
    assert operations.errors
    assert operations.delete_file_contents == []
    assert operations.delete_replacements == {}
    fs.read_file.assert_not_awaited()


@pytest.mark.parametrize(
    ("written", "edited", "errors", "expected_calls"),
    [
        ([f"{EXPERIENCE_DIR}/bounded_lesson.md"], [], [], 1),
        ([], [], [], 0),
        ([f"{EXPERIENCE_DIR}/bounded_lesson.md"], [], [("write", RuntimeError())], 0),
    ],
)
@pytest.mark.asyncio
async def test_post_apply_requires_error_free_written_or_edited_result(
    written, edited, errors, expected_calls
):
    compressor = SessionCompressorV2(vikingdb=None)
    user = UserIdentifier.the_default_user()
    ctx = RequestContext(user=user, role=Role.ROOT)
    messages = [Message(id="msg-test", role="user", parts=[TextPart("test")])]

    class FakeVikingFS:
        agfs = None

    class DummyProvider:
        async def prepare_extraction_messages(self):
            return None

        def get_extract_context(self):
            return ExtractContext(messages)

        def _get_registry(self):
            return object()

    class DummyExtractLoop:
        def __init__(self, **kwargs):
            pass

        async def run(self):
            return _operations(_operation("bounded_lesson", "")), []

    memory_result = MemoryUpdateResult()
    memory_result.written_uris = written
    memory_result.edited_uris = edited
    memory_result.errors = errors

    class DummyUpdater:
        async def apply_operations(self, operations, request_context, **kwargs):
            return memory_result

    config = SimpleNamespace(vlm=SimpleNamespace(get_vlm_instance=lambda: object()))
    post_apply = AsyncMock()
    with (
        patch("openviking.session.compressor_v2.get_viking_fs", return_value=FakeVikingFS()),
        patch("openviking.session.compressor_v2.get_openviking_config", return_value=config),
        patch("openviking.session.compressor_v2.ExtractLoop", DummyExtractLoop),
        patch.object(compressor, "_get_or_create_updater", return_value=DummyUpdater()),
    ):
        await compressor._run_extract_phase(
            provider=DummyProvider(),
            messages=messages,
            ctx=ctx,
            strict_extract_errors=True,
            phase_label="experience(test)",
            post_apply=post_apply,
        )

    assert post_apply.await_count == expected_calls


def test_experience_patch_merge_instruction_is_additive_only():
    instruction = PatchMergeContextProvider(
        memory_type="experiences",
        patches=[],
        output_language="en",
    ).instruction()

    assert "Keep `delete_ids` empty" in instruction
    assert "`supersedes` value to the empty string" in instruction
    assert "put it in delete_ids" not in instruction
    assert "Merge them by producing" not in instruction


def test_experience_repair_instructions_forbid_automatic_supersession():
    loop = object.__new__(ExtractLoop)

    patch_repair = loop._build_patch_repair_instruction([])
    policy_repair = loop._build_experience_repair_instruction([])

    for instruction in (patch_repair, policy_repair):
        assert (
            "supersedes empty" in instruction
            or "supersedes value to the empty string" in instruction
        )
        assert "delete_ids" in instruction
        assert "replacement_page_id" in instruction
