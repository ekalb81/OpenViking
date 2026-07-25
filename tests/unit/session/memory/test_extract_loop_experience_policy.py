import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from openviking.session.memory.dataclass import MemoryFile, WikiLink
from openviking.session.memory.extract_loop import ExtractLoop
from openviking.session.memory.merge_op import MergeOp
from openviking.session.memory.page_id_map import PageIdMap


class AttrDict(dict):
    __getattr__ = dict.get


VALID_CONTENT = """## Situation
- A bounded implementation needs prior procedural guidance.

## Approach
- Inspect the current state before changing it.

## Reflect
- NEVER claim completion without verification.
"""


def _experience_resolve_loop(
    *,
    page_id_map: PageIdMap,
    read_file_contents: dict[str, MemoryFile] | None = None,
) -> tuple[ExtractLoop, Mock]:
    schema = SimpleNamespace(
        memory_type="experiences",
        fields=[
            SimpleNamespace(name="experience_name", merge_op=MergeOp.IMMUTABLE),
            SimpleNamespace(name="content", merge_op=MergeOp.PATCH),
        ],
    )
    context_provider = Mock()
    context_provider.get_memory_schemas.return_value = [schema]
    context_provider.get_output_language.return_value = "en"
    context_provider.get_tools.return_value = []
    context_provider.get_extract_context.return_value = SimpleNamespace(page_id_map=page_id_map)
    context_provider.prefetch = AsyncMock(return_value=[])
    context_provider.read_file_contents = read_file_contents or {}
    context_provider.instruction.return_value = "Extract experience operations."

    isolation_handler = Mock()
    isolation_handler.get_read_scope.return_value = None
    isolation_handler.fill_identity_fields.side_effect = lambda item, role_scope=None: item
    isolation_handler.calculate_memory_uris.side_effect = lambda **kwargs: [
        "viking://user/alice/memories/experiences/"
        + kwargs["operation"].memory_fields["experience_name"]
        + ".md"
    ]

    loop = ExtractLoop(
        vlm=Mock(model="test-model"),
        viking_fs=Mock(),
        context_provider=context_provider,
        isolation_handler=isolation_handler,
    )
    loop._extract_context = SimpleNamespace(page_id_map=page_id_map)
    return loop, isolation_handler


@pytest.mark.asyncio
async def test_existing_experience_page_id_rejects_name_mismatch_before_coercion():
    existing_uri = "viking://user/alice/memories/experiences/verification.md"
    old_file = MemoryFile(
        uri=existing_uri,
        content=VALID_CONTENT,
        memory_type="experiences",
        extra_fields={"experience_name": "verification"},
    )
    page_id_map = PageIdMap()
    existing_page_id = page_id_map.get_page_id(existing_uri)
    loop, isolation_handler = _experience_resolve_loop(
        page_id_map=page_id_map,
        read_file_contents={existing_uri: old_file},
    )

    operations, _ = await loop.resolve_operations(
        AttrDict(
            experiences=[
                {
                    "experience_name": "different_experience",
                    "content": VALID_CONTENT,
                    "page_id": existing_page_id,
                }
            ]
        )
    )

    assert any("experience_page_id_name_mismatch" in error for error in operations.errors)
    operation = operations.upsert_operations[0]
    assert operation.uris == [existing_uri]
    assert operation.old_memory_file_content is old_file
    assert operation.memory_fields["experience_name"] == "different_experience"
    isolation_handler.calculate_memory_uris.assert_not_called()


@pytest.mark.asyncio
async def test_existing_experience_page_id_mismatch_stops_run_before_repair():
    existing_uri = "viking://user/alice/memories/experiences/verification.md"
    old_file = MemoryFile(
        uri=existing_uri,
        content=VALID_CONTENT,
        memory_type="experiences",
        extra_fields={"experience_name": "verification"},
    )
    page_id_map = PageIdMap()
    existing_page_id = page_id_map.get_page_id(existing_uri)
    loop, _ = _experience_resolve_loop(
        page_id_map=page_id_map,
        read_file_contents={existing_uri: old_file},
    )
    loop._call_llm = AsyncMock(
        return_value=(
            [],
            AttrDict(
                experiences=[
                    {
                        "experience_name": "different_experience",
                        "content": VALID_CONTENT,
                        "page_id": existing_page_id,
                    }
                ]
            ),
        )
    )
    loop._check_unread_existing_files = AsyncMock(return_value=[])
    loop.finalize_operations = AsyncMock()

    with (
        patch("openviking.session.memory.extract_loop.get_openviking_config") as mock_config,
        patch("openviking.session.memory.extract_loop.pretty_print_messages"),
        patch("openviking.session.memory.extract_loop.SchemaModelGenerator.generate_all_models"),
        patch(
            "openviking.session.memory.extract_loop.SchemaModelGenerator.create_structured_operations_model"
        ) as mock_create_model,
    ):
        mock_config.return_value = SimpleNamespace(memory=SimpleNamespace(link_enabled=False))
        mock_create_model.return_value = SimpleNamespace(model_json_schema=lambda: {})

        operations, _ = await loop.run()

    assert loop._call_llm.await_count == 1
    loop._check_unread_existing_files.assert_not_awaited()
    assert any("experience_page_id_name_mismatch" in error for error in operations.errors)
    assert operations.upsert_operations[0].memory_fields["experience_name"] == (
        "different_experience"
    )


@pytest.mark.asyncio
async def test_duplicate_new_experience_page_id_fails_closed_before_link_fanout():
    page_id_map = PageIdMap()
    loop, _ = _experience_resolve_loop(page_id_map=page_id_map)
    loop._link_enabled = True
    raw_link = WikiLink(f=100, t=101, match_text="shared evidence")

    operations, raw_links = await loop.resolve_operations(
        AttrDict(
            experiences=[
                {"experience_name": "first", "content": VALID_CONTENT, "page_id": 100},
                {"experience_name": "second", "content": VALID_CONTENT, "page_id": 100},
                {"experience_name": "target", "content": VALID_CONTENT, "page_id": 101},
            ],
            links=[raw_link],
        )
    )
    await loop.finalize_operations(operations, raw_links)

    assert any("duplicate_experience_page_id" in error for error in operations.errors)
    assert operations.resolved_links == []
    assert page_id_map.resolve(100) is None
    assert page_id_map.resolve(101) is None


@pytest.mark.asyncio
async def test_unknown_existing_experience_page_id_fails_closed():
    loop, _ = _experience_resolve_loop(page_id_map=PageIdMap())

    operations, _ = await loop.resolve_operations(
        AttrDict(
            experiences=[
                {
                    "experience_name": "verification",
                    "content": VALID_CONTENT,
                    "page_id": 7,
                }
            ]
        )
    )

    assert any(
        "unknown_existing_experience_page_id" in error for error in operations.errors
    )


@pytest.mark.asyncio
async def test_invalid_experience_gets_one_repair_iteration_before_acceptance():
    context_provider = Mock()
    context_provider.get_memory_schemas.return_value = [
        SimpleNamespace(memory_type="experiences", fields=[])
    ]
    context_provider.get_output_language.return_value = "en"
    context_provider.get_tools.return_value = []
    context_provider.get_extract_context.return_value = SimpleNamespace(page_id_map=PageIdMap())
    context_provider.prefetch = AsyncMock(return_value=[])
    context_provider.read_file_contents = {}
    context_provider.instruction.return_value = "Extract experience operations."

    isolation_handler = Mock()
    isolation_handler.get_read_scope.return_value = None
    isolation_handler.fill_identity_fields.side_effect = lambda item, role_scope=None: item
    isolation_handler.calculate_memory_uris.side_effect = lambda **kwargs: [
        "viking://user/alice/memories/experiences/"
        + kwargs["operation"].memory_fields["experience_name"]
        + ".md"
    ]

    loop = ExtractLoop(
        vlm=Mock(model="test-model"),
        viking_fs=Mock(),
        context_provider=context_provider,
        isolation_handler=isolation_handler,
        max_iterations=1,
    )
    proposals = [
            (
                [],
                AttrDict(
                    experiences=[
                        {
                            "experience_name": "verification",
                            "content": "unstructured and invalid",
                            "page_id": 100,
                        },
                        {
                            "experience_name": "bounded_sibling",
                            "content": VALID_CONTENT,
                            "page_id": 101,
                        }
                    ]
                ),
            ),
            (
                [],
                AttrDict(
                    experiences=[
                        {
                            "experience_name": "verification",
                            "content": VALID_CONTENT,
                            "page_id": 100,
                        },
                        {
                            "experience_name": "bounded_sibling",
                            "content": VALID_CONTENT,
                            "page_id": 101,
                        }
                    ]
                ),
            ),
    ]

    async def fake_call_llm(messages):
        response = proposals.pop(0)
        if response[1] is not None:
            loop._last_successful_llm_content = json.dumps(response[1])
        return response

    loop._call_llm = AsyncMock(side_effect=fake_call_llm)
    loop._check_unread_existing_files = AsyncMock(return_value=[])
    loop.finalize_operations = AsyncMock()

    captured_messages = []

    def capture_messages(messages):
        captured_messages[:] = list(messages)

    with (
        patch("openviking.session.memory.extract_loop.get_openviking_config") as mock_config,
        patch("openviking.session.memory.extract_loop.pretty_print_messages", capture_messages),
        patch("openviking.session.memory.extract_loop.SchemaModelGenerator.generate_all_models"),
        patch(
            "openviking.session.memory.extract_loop.SchemaModelGenerator.create_structured_operations_model"
        ) as mock_create_model,
    ):
        mock_config.return_value = SimpleNamespace(memory=SimpleNamespace(link_enabled=False))
        mock_create_model.return_value = SimpleNamespace(model_json_schema=lambda: {})

        operations, _ = await loop.run()

    assert loop._call_llm.await_count == 2
    assert operations.errors == []
    assert len(operations.upsert_operations) == 2
    assert operations.upsert_operations[0].memory_fields["content"] == VALID_CONTENT
    assert any(
        "deterministic atomic-memory policy" in message.get("content", "")
        for message in captured_messages
    )
    rejected_proposal = next(
        message["content"]
        for message in captured_messages
        if message.get("role") == "assistant"
    )
    assert "bounded_sibling" in rejected_proposal
    assert "unstructured and invalid" in rejected_proposal


@pytest.mark.asyncio
async def test_experience_repair_fails_closed_when_valid_sibling_is_dropped():
    context_provider = Mock()
    context_provider.get_memory_schemas.return_value = [
        SimpleNamespace(memory_type="experiences", fields=[])
    ]
    context_provider.get_output_language.return_value = "en"
    context_provider.get_tools.return_value = []
    context_provider.get_extract_context.return_value = SimpleNamespace(page_id_map=PageIdMap())
    context_provider.prefetch = AsyncMock(return_value=[])
    context_provider.read_file_contents = {}
    context_provider.instruction.return_value = "Extract experience operations."

    isolation_handler = Mock()
    isolation_handler.get_read_scope.return_value = None
    isolation_handler.fill_identity_fields.side_effect = lambda item, role_scope=None: item
    isolation_handler.calculate_memory_uris.side_effect = lambda **kwargs: [
        "viking://user/alice/memories/experiences/"
        + kwargs["operation"].memory_fields["experience_name"]
        + ".md"
    ]

    loop = ExtractLoop(
        vlm=Mock(model="test-model"),
        viking_fs=Mock(),
        context_provider=context_provider,
        isolation_handler=isolation_handler,
        max_iterations=1,
    )
    proposals = [
        (
            [],
            AttrDict(
                experiences=[
                    {
                        "experience_name": "verification",
                        "content": "unstructured and invalid",
                        "page_id": 100,
                    },
                    {
                        "experience_name": "bounded_sibling",
                        "content": VALID_CONTENT,
                        "page_id": 101,
                    },
                ]
            ),
        ),
        (
            [],
            AttrDict(
                experiences=[
                    {
                        "experience_name": "verification",
                        "content": VALID_CONTENT,
                        "page_id": 100,
                    }
                ]
            ),
        ),
    ]

    async def fake_call_llm(messages):
        response = proposals.pop(0)
        if response[1] is not None:
            loop._last_successful_llm_content = json.dumps(response[1])
        return response

    loop._call_llm = AsyncMock(side_effect=fake_call_llm)
    loop._check_unread_existing_files = AsyncMock(return_value=[])
    loop.finalize_operations = AsyncMock()

    with (
        patch("openviking.session.memory.extract_loop.get_openviking_config") as mock_config,
        patch("openviking.session.memory.extract_loop.pretty_print_messages"),
        patch("openviking.session.memory.extract_loop.SchemaModelGenerator.generate_all_models"),
        patch(
            "openviking.session.memory.extract_loop.SchemaModelGenerator.create_structured_operations_model"
        ) as mock_create_model,
    ):
        mock_config.return_value = SimpleNamespace(memory=SimpleNamespace(link_enabled=False))
        mock_create_model.return_value = SimpleNamespace(model_json_schema=lambda: {})

        operations, _ = await loop.run()

    assert loop._call_llm.await_count == 2
    assert len(operations.upsert_operations) == 1
    assert any(
        "omitted or altered structurally valid first-pass operations" in error
        and '"experience_name": "bounded_sibling"' in error
        and '"status": "omitted"' in error
        for error in operations.errors
    )


@pytest.mark.asyncio
async def test_call_llm_retains_the_complete_parsed_proposal_for_repair():
    proposal = '{"experiences": [{"experience_name": "bounded_sibling"}]}'
    parsed = object()
    loop = ExtractLoop(
        vlm=SimpleNamespace(get_completion_async=AsyncMock(return_value=proposal), model="test"),
        viking_fs=Mock(),
        context_provider=Mock(),
    )
    loop._tool_schemas = []
    loop._operations_model = object()
    loop._expected_fields = ["experiences"]

    with patch(
        "openviking.session.memory.extract_loop.parse_json_with_stability",
        return_value=(parsed, None),
    ):
        _, operations = await loop._call_llm([])

    assert operations is parsed
    assert loop._last_successful_llm_content == proposal
