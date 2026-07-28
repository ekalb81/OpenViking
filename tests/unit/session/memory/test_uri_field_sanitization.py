# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Model-supplied URI field values must stay inside one path segment."""

import pytest

from openviking.session.memory.dataclass import MemoryTypeSchema
from openviking.session.memory.utils.uri import (
    _pattern_matches_uri,
    generate_uri,
    sanitize_uri_field_value,
)


class TestSanitizeUriFieldValue:
    def test_forward_slash_becomes_separator_free(self):
        assert sanitize_uri_field_value("WebFetch / WebSearch for GitHub") == (
            "WebFetch _ WebSearch for GitHub"
        )

    def test_backslash_is_neutralized(self):
        assert sanitize_uri_field_value(r"a\b") == "a_b"

    def test_parent_traversal_is_neutralized(self):
        assert sanitize_uri_field_value("../../etc/passwd") == "____etc_passwd"

    def test_trailing_space_and_dot_are_stripped(self):
        # Windows silently drops these, leaving a path that no longer matches the URI.
        assert sanitize_uri_field_value("WebFetch ") == "WebFetch"
        assert sanitize_uri_field_value("report.") == "report"

    def test_ordinary_name_is_untouched(self):
        assert sanitize_uri_field_value("openviking_injector_fix") == "openviking_injector_fix"

    def test_non_string_passes_through(self):
        assert sanitize_uri_field_value(7) == 7
        assert sanitize_uri_field_value(None) is None

    @pytest.mark.parametrize("value", ["   ", ".", ". .", "", " . . "])
    def test_value_that_sanitizes_away_does_not_vanish(self, value):
        # Rendering nothing would give the bare filename ".md": a dotfile that
        # tools/*.md does not match, and one path every such value collides on.
        assert sanitize_uri_field_value(value) == "_"


def _tools_like_schema() -> MemoryTypeSchema:
    return MemoryTypeSchema(
        memory_type="tools",
        directory="viking://user/{{ user_space }}/memories/tools",
        filename_template="{{ tool_name }}.md",
        fields=[],
    )


class TestGenerateUri:
    def test_slash_in_field_does_not_create_a_nested_path(self):
        uri = generate_uri(_tools_like_schema(), {"tool_name": "WebFetch / WebSearch for GitHub"})

        assert uri == (
            "viking://user/default/memories/tools/WebFetch _ WebSearch for GitHub.md"
        )
        # The bug this guards: the name used to add path segments of its own.
        assert uri.count("/", len("viking://")) == 4

    def test_field_that_sanitizes_away_still_yields_a_visible_filename(self):
        uri = generate_uri(_tools_like_schema(), {"tool_name": ". ."})

        assert uri == "viking://user/default/memories/tools/_.md"
        assert not uri.endswith("/.md")

    def test_traversal_in_field_cannot_escape_the_directory(self):
        uri = generate_uri(_tools_like_schema(), {"tool_name": "../../../secrets"})

        assert uri.startswith("viking://user/default/memories/tools/")
        assert ".." not in uri

    def test_sanitized_uri_still_matches_its_allow_pattern(self):
        # _pattern_matches_uri expands {{ variable }} to [^/]+, so a value that
        # spanned segments would fall outside the pattern that authorizes it.
        pattern = "viking://user/default/memories/tools/{{ tool_name }}.md"
        uri = generate_uri(_tools_like_schema(), {"tool_name": "WebFetch / WebSearch for GitHub"})

        assert _pattern_matches_uri(pattern, uri)

    def test_user_space_keeps_its_path_structure(self):
        # user_space is caller-supplied, not model output, and peer scopes rely on
        # it rendering as real path segments.
        uri = generate_uri(_tools_like_schema(), {"tool_name": "gh"}, user_space="u/peers/conv-42")

        assert uri == "viking://user/u/peers/conv-42/memories/tools/gh.md"
