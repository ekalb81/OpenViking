# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Empty link keys must not be written into the MEMORY_FIELDS trailer."""

import json
import re

from openviking.session.memory.dataclass import MemoryFile, StoredLink
from openviking.session.memory.merge_op.link_merge import merge_links
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils

_TRAILER = re.compile(r"<!-- MEMORY_FIELDS\n(.*)\n-->", re.DOTALL)


def _trailer_of(rendered: str) -> dict:
    return json.loads(_TRAILER.search(rendered).group(1))


def _link(**overrides) -> dict:
    base = StoredLink(
        from_uri="viking://user/default/memories/cases/a.md",
        to_uri="viking://user/default/memories/trajectories/b.md",
        link_type="related_to",
        weight=1.0,
        created_at="2026-07-27T16:15:36.781720+00:00",
    ).model_dump()
    base.update(overrides)
    return base


def test_empty_match_text_and_description_are_omitted():
    rendered = MemoryFileUtils.write(
        MemoryFile(extra_fields={"case_name": "a", "links": [_link()]})
    )

    stored = _trailer_of(rendered)["links"][0]
    assert "match_text" not in stored
    assert "description" not in stored
    # Everything that carries information survives.
    assert stored["to_uri"].endswith("/b.md")
    assert stored["link_type"] == "related_to"
    assert stored["weight"] == 1.0


def test_populated_match_text_and_description_are_kept():
    rendered = MemoryFileUtils.write(
        MemoryFile(
            extra_fields={
                "case_name": "a",
                "links": [_link(match_text="turret", description="root cause")],
            }
        )
    )

    stored = _trailer_of(rendered)["links"][0]
    assert stored["match_text"] == "turret"
    assert stored["description"] == "root cause"


def test_weight_is_never_omitted_even_at_a_falsy_default():
    # StoredLink defaults weight to 0.5, so dropping an explicit 0.0 would silently
    # reload as a different link.
    rendered = MemoryFileUtils.write(
        MemoryFile(extra_fields={"case_name": "a", "links": [_link(weight=0.0)]})
    )

    assert _trailer_of(rendered)["links"][0]["weight"] == 0.0


def test_compacted_link_round_trips_without_duplicating_on_merge():
    """The regression this guards: a reloaded link must dedup against a fresh one."""
    fresh = _link()
    reloaded = _trailer_of(
        MemoryFileUtils.write(MemoryFile(extra_fields={"case_name": "a", "links": [fresh]}))
    )["links"][0]

    merged = merge_links([reloaded], [fresh])

    assert len(merged) == 1
