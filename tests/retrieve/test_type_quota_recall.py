# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import asyncio
from types import SimpleNamespace

from openviking.retrieve import type_quota_recall
from openviking.retrieve.type_quota_recall import search_type_quota_recall
from openviking.server.identity import RequestContext, Role
from openviking_cli.session.user_id import UserIdentifier


class _FakeFindResult:
    def __init__(self, memories=None):
        self.memories = memories or []


async def test_independent_type_searches_start_concurrently():
    started_targets: list[str] = []
    all_started = asyncio.Event()

    async def fake_find(**kwargs):
        started_targets.append(kwargs["target_uri"])
        if len(started_targets) == 4:
            all_started.set()
        await all_started.wait()
        return _FakeFindResult()

    service = SimpleNamespace(
        search=SimpleNamespace(find=fake_find),
        fs=SimpleNamespace(),
    )
    ctx = RequestContext(
        user=UserIdentifier.the_default_user("test_user"),
        role=Role.USER,
        actor_peer_id="current",
    )

    result = await asyncio.wait_for(
        search_type_quota_recall(
            service=service,
            ctx=ctx,
            query="parallel recall",
            peer_scope="actor",
            quotas={
                "events": 1,
                "entities": 1,
                "preferences": 0,
                "experiences": 0,
            },
        ),
        timeout=1.0,
    )

    assert set(started_targets) == {
        "viking://user/test_user/memories/events",
        "viking://user/test_user/peers/current/memories/events",
        "viking://user/test_user/memories/entities",
        "viking://user/test_user/peers/current/memories/entities",
    }
    assert result.stats["searched"] == {
        "events": 0,
        "entities": 0,
        "preferences": 0,
        "experiences": 0,
    }


async def test_parallel_search_preserves_type_order():
    async def fake_find(**kwargs):
        target_uri = kwargs["target_uri"]
        memory_type = target_uri.rsplit("/", 1)[-1]
        if memory_type == "events":
            await asyncio.sleep(0.02)
        if "/peers/" in target_uri:
            return _FakeFindResult()
        return _FakeFindResult(
            [
                {
                    "uri": f"{target_uri}/{memory_type}.md",
                    "score": 0.9,
                    "abstract": f"{memory_type} abstract",
                }
            ]
        )

    async def fake_read(uri, **kwargs):
        del kwargs
        return f"content for {uri}"

    service = SimpleNamespace(
        search=SimpleNamespace(find=fake_find),
        fs=SimpleNamespace(read=fake_read),
    )
    ctx = RequestContext(
        user=UserIdentifier.the_default_user("test_user"),
        role=Role.USER,
        actor_peer_id="current",
    )

    result = await search_type_quota_recall(
        service=service,
        ctx=ctx,
        query="deterministic recall",
        peer_scope="actor",
        quotas={
            "events": 1,
            "entities": 1,
            "preferences": 0,
            "experiences": 0,
        },
    )

    assert [entry.type for entry in result.entries] == ["events", "entities"]
    assert [entry.rank for entry in result.entries] == [1, 1]


async def test_recall_hides_persisted_memory_fields_metadata():
    memory_uri = "viking://user/test_user/memories/events/example.md"
    raw_memory = """Visible memory body

<!-- MEMORY_FIELDS
{
  "event_name": "internal-event",
  "user_id": "test_user",
  "memory_type": "events"
}
-->"""

    async def fake_find(**kwargs):
        if kwargs["target_uri"].endswith("/events") and "/peers/" not in kwargs["target_uri"]:
            return _FakeFindResult([{"uri": memory_uri, "score": 0.9}])
        return _FakeFindResult()

    async def fake_read(uri, **kwargs):
        del kwargs
        assert uri == memory_uri
        return raw_memory

    service = SimpleNamespace(
        search=SimpleNamespace(find=fake_find),
        fs=SimpleNamespace(read=fake_read),
    )
    ctx = RequestContext(
        user=UserIdentifier.the_default_user("test_user"),
        role=Role.USER,
        actor_peer_id="current",
    )

    result = await search_type_quota_recall(
        service=service,
        ctx=ctx,
        query="visible memory",
        peer_scope="actor",
        quotas={"events": 1, "entities": 0, "preferences": 0, "experiences": 0},
        max_chars=10_000,
    )

    assert len(result.entries) == 1
    assert result.entries[0].content == "Visible memory body"
    assert "Visible memory body" in result.rendered
    assert "MEMORY_FIELDS" not in result.rendered
    assert "internal-event" not in result.rendered


async def test_experiences_survive_a_budget_filled_by_earlier_types():
    """Experiences render last in TYPE_ORDER against a shared max_chars budget.

    Real recalls return many mid-sized events plus entities, which cumulatively consume the budget
    before experiences are ever considered — so the distilled procedural memory was retrievable but
    never reachable at a realistic budget. One oversized item does NOT reproduce this: it fails the
    full-fragment check, degrades to a URI stub, and leaves room behind it.
    """

    async def fake_find(**kwargs):
        target_uri = kwargs["target_uri"]
        memory_type = target_uri.rsplit("/", 1)[-1]
        if "/peers/" in target_uri:
            return _FakeFindResult()
        return _FakeFindResult(
            [
                {
                    "uri": f"{target_uri}/{memory_type}-{index}.md",
                    "score": 0.9 - (index * 0.01),
                    "abstract": (
                        f"{memory_type} actionable abstract {index}: verify the real outcome first. "
                        + ("A" * 2400)
                    ),
                }
                for index in range(6)
            ]
        )

    async def fake_read(uri, **kwargs):
        del kwargs
        # Unique per uri: the renderer dedupes by content hash, so identical bodies collapse to one
        # entry and the budget interaction under test never happens.
        return f"{uri} " + ("C" * 3200)

    service = SimpleNamespace(
        search=SimpleNamespace(find=fake_find),
        fs=SimpleNamespace(read=fake_read),
    )
    ctx = RequestContext(
        user=UserIdentifier.the_default_user("test_user"),
        role=Role.USER,
        actor_peer_id="current",
    )

    result = await search_type_quota_recall(
        service=service,
        ctx=ctx,
        query="how should I approach this",
        peer_scope="actor",
        quotas={"events": 6, "entities": 2, "preferences": 0, "experiences": 2},
        max_chars=3000,
    )

    # Real experience bodies are multi-kilobyte and cannot fit in the 25% reserve at this budget.
    # Presence is not the bar: a starved experience can appear as a bare URI stub carrying no lesson.
    # Its compact abstract must survive as a usable summary when the full body does not fit.
    experience_entries = [entry for entry in result.entries if entry.type == "experiences"]
    assert any(entry.mode == "summary" for entry in experience_entries), (
        "experiences starved by earlier types; "
        f"modes={[entry.mode for entry in experience_entries]} "
        f"all={[(e.type, e.mode) for e in result.entries]}"
    )
    assert "experiences actionable abstract" in result.rendered
    assert "verify the real outcome first" in result.rendered


async def test_no_reserve_is_held_when_experiences_are_not_requested(monkeypatch):
    """The reserve must not shrink other types for callers that never ask for experiences.

    Compared against the same call with the reserve forced to zero, so the assertion tests the reserve
    itself rather than whatever the events budget ratio happens to allow.
    """

    async def fake_find(**kwargs):
        target_uri = kwargs["target_uri"]
        memory_type = target_uri.rsplit("/", 1)[-1]
        if "/peers/" in target_uri:
            return _FakeFindResult()
        return _FakeFindResult(
            [
                {
                    "uri": f"{target_uri}/{memory_type}-{index}.md",
                    "score": 0.9 - (index * 0.01),
                    "abstract": f"{memory_type} abstract {index}",
                }
                for index in range(4)
            ]
        )

    async def fake_read(uri, **kwargs):
        del kwargs
        return "C" * 400

    service = SimpleNamespace(
        search=SimpleNamespace(find=fake_find),
        fs=SimpleNamespace(read=fake_read),
    )
    ctx = RequestContext(
        user=UserIdentifier.the_default_user("test_user"),
        role=Role.USER,
        actor_peer_id="current",
    )

    async def run() -> list[str]:
        result = await search_type_quota_recall(
            service=service,
            ctx=ctx,
            query="unrelated",
            peer_scope="actor",
            quotas={"events": 4, "entities": 0, "preferences": 0, "experiences": 0},
            max_chars=3000,
        )
        return [f"{entry.uri}:{entry.mode}" for entry in result.entries]

    with_reserve_constant = await run()
    monkeypatch.setattr(type_quota_recall, "EXPERIENCES_BUDGET_RESERVE_RATIO", 0.0)
    without_reserve_constant = await run()

    assert with_reserve_constant == without_reserve_constant
