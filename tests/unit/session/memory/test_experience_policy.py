from types import SimpleNamespace

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.session.memory.experience_policy import (
    EXPERIENCE_MAX_CONTENT_CHARS,
    validate_experience_content,
    validate_experience_operation_context,
    validate_experience_operations,
    validate_stored_experience,
)
from openviking_cli.session.user_id import UserIdentifier

VALID_CONTENT = """## Situation
- A bounded implementation needs prior procedural guidance.

## Approach
- Inspect the current state before changing it.
- Run the smallest discriminating verification.

## Reflect
- NEVER claim completion without the verification result.
"""


def _codes(content: str) -> set[str]:
    return {item["code"] for item in validate_experience_content(content)}


def test_valid_atomic_experience_passes():
    assert validate_experience_content(VALID_CONTENT) == []


def test_requires_exact_three_section_structure_and_bullets():
    assert "section_structure" in _codes(VALID_CONTENT.replace("## Reflect", "## Notes"))
    assert "non_section_content" in _codes("Introductory prose.\n\n" + VALID_CONTENT)
    assert "non_bullet_content" in _codes(
        VALID_CONTENT.replace("- Inspect the current state", "Inspect the current state")
    )
    assert "non_bullet_content" in _codes(
        VALID_CONTENT.replace("- Inspect the current state", "    - Inspect the current state")
    )
    assert "non_bullet_content" in _codes(
        VALID_CONTENT.replace("- Inspect the current state", "\u00a0- Inspect the current state")
    )


def test_enforces_section_body_and_bullet_limits():
    too_many_situations = VALID_CONTENT.replace(
        "- A bounded implementation needs prior procedural guidance.",
        "\n".join(f"- Situation {index}" for index in range(4)),
    )
    assert "too_many_bullets" in _codes(too_many_situations)

    long_bullet = VALID_CONTENT.replace(
        "- Inspect the current state before changing it.",
        "- " + ("x" * 401),
    )
    assert "bullet_too_long" in _codes(long_bullet)

    empty_bullet = VALID_CONTENT.replace(
        "- Inspect the current state before changing it.",
        "- ",
    )
    assert "empty_bullet" in _codes(empty_bullet)

    assert "content_too_long" in _codes(VALID_CONTENT + "\n- " + ("y" * 3000))


def test_size_limit_measures_the_stored_value_including_trailing_whitespace():
    padding = " " * (EXPERIENCE_MAX_CONTENT_CHARS - len(VALID_CONTENT))
    assert "content_too_long" not in _codes(VALID_CONTENT + padding)
    assert "content_too_long" in _codes(VALID_CONTENT + padding + " ")


def test_rejects_duplicate_rules_across_sections_and_accepts_multilingual_bullets():
    duplicate = VALID_CONTENT.replace(
        "- NEVER claim completion without the verification result.",
        "- Inspect the current state before changing it.",
    )
    assert "duplicate_rule" in _codes(duplicate)

    multilingual = VALID_CONTENT.replace(
        "A bounded implementation needs prior procedural guidance.",
        "任务需要先读取可复用的执行经验。",
    )
    assert validate_experience_content(multilingual) == []


def _operation(name: str, uri: str, *, supersedes: str = ""):
    return SimpleNamespace(
        memory_type="experiences",
        memory_fields={
            "experience_name": name,
            "content": VALID_CONTENT,
            "supersedes": supersedes,
        },
        uris=[uri],
        page_id=100,
    )


def _ctx(user_id: str = "u") -> RequestContext:
    return RequestContext(user=UserIdentifier("account", user_id), role=Role.USER)


def test_batch_rejects_duplicate_names_and_uris():
    first = _operation(
        "verification",
        "viking://user/u/memories/experiences/verification.md",
    )
    second = _operation(
        "verification",
        "viking://user/u/memories/experiences/verification.md",
    )

    codes = {error["code"] for error in validate_experience_operations([first, second])}

    assert codes >= {
        "duplicate_experience_name",
        "duplicate_experience_uri",
    }


def test_batch_rejects_all_automatic_supersession():
    operation = _operation(
        "verification",
        "viking://user/u/memories/experiences/verification.md",
        supersedes="legacy",
    )

    assert {error["code"] for error in validate_experience_operations([operation])} == {
        "automatic_supersession_disabled"
    }


def test_batch_requires_a_safe_nonempty_experience_name():
    blank = _operation("   ", "viking://user/u/memories/experiences/blank.md")
    nested = _operation("nested/name", "viking://user/u/memories/experiences/nested/name.md")
    hidden = _operation(".overview", "viking://user/u/memories/experiences/.overview.md")
    reserved = _operation("CON", "viking://user/u/memories/experiences/CON.md")
    overlong_name = "x" * 253
    overlong = _operation(
        overlong_name,
        f"viking://user/u/memories/experiences/{overlong_name}.md",
    )

    blank_codes = {error["code"] for error in validate_experience_operations([blank])}
    nested_codes = {error["code"] for error in validate_experience_operations([nested])}
    hidden_codes = {error["code"] for error in validate_experience_operations([hidden])}
    reserved_codes = {error["code"] for error in validate_experience_operations([reserved])}
    overlong_codes = {error["code"] for error in validate_experience_operations([overlong])}

    assert "experience_name_required" in blank_codes
    assert "unsafe_experience_name" in nested_codes
    assert "unsafe_experience_name" in hidden_codes
    assert "unsafe_experience_name" in reserved_codes
    assert "unsafe_experience_name" in overlong_codes


def test_batch_rejects_unicode_casefold_aliases():
    lowercase_accented = _operation(
        "échec",
        "viking://user/u/memories/experiences/échec.md",
    )
    uppercase_accented = _operation(
        "Échec",
        "viking://user/u/memories/experiences/Échec.md",
    )
    fold_expansion = _operation(
        "straße",
        "viking://user/u/memories/experiences/straße.md",
    )

    assert validate_experience_operations([lowercase_accented]) == []
    assert "unsafe_experience_name" in {
        error["code"] for error in validate_experience_operations([uppercase_accented])
    }
    assert "unsafe_experience_name" in {
        error["code"] for error in validate_experience_operations([fold_expansion])
    }


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "verification\u200b",
        "safe\u202eexe",
        "surrogate\ud800",
    ],
)
def test_batch_rejects_unicode_control_and_surrogate_names(unsafe_name):
    operation = _operation(
        unsafe_name,
        f"viking://user/u/memories/experiences/{unsafe_name}.md",
    )

    assert "unsafe_experience_name" in {
        error["code"] for error in validate_experience_operations([operation])
    }


@pytest.mark.parametrize(
    "reserved_name",
    ["conin$", "conout$", "com¹", "com²", "com³", "lpt¹", "lpt²", "lpt³"],
)
def test_batch_rejects_extended_windows_device_names(reserved_name):
    operation = _operation(
        reserved_name,
        f"viking://user/u/memories/experiences/{reserved_name}.md",
    )

    assert "unsafe_experience_name" in {
        error["code"] for error in validate_experience_operations([operation])
    }


@pytest.mark.parametrize(
    "unsafe_name",
    ["foo bar", "foo\u00a0bar", "foo\u2028bar", "con .alias"],
)
def test_batch_rejects_internal_filename_whitespace(unsafe_name):
    operation = _operation(
        unsafe_name,
        f"viking://user/u/memories/experiences/{unsafe_name}.md",
    )

    codes = {error["code"] for error in validate_experience_operations([operation])}
    assert "unsafe_experience_name" in codes
    assert "experience_uri_scope" in codes


def test_batch_binds_experience_type_uri_directory_and_name():
    wrong_type_path = _operation(
        "verification",
        "viking://user/u/memories/skills/verification.md",
    )
    wrong_name = _operation(
        "verification",
        "viking://user/u/memories/experiences/different.md",
    )

    type_codes = {
        error["code"] for error in validate_experience_operations([wrong_type_path])
    }
    name_codes = {error["code"] for error in validate_experience_operations([wrong_name])}

    assert "experience_uri_scope" in type_codes
    assert "experience_uri_name_mismatch" in name_codes


def test_batch_accepts_supported_short_user_uri():
    operation = _operation(
        "verify_result",
        "viking://user/memories/experiences/verify_result.md",
    )

    assert validate_experience_operations([operation]) == []
    assert operation.memory_fields["memory_type"] == "experiences"


def test_batch_rejects_mismatched_persisted_payload_type():
    operation = _operation(
        "verification",
        "viking://user/u/memories/experiences/verification.md",
    )
    operation.memory_fields["memory_type"] = "skills"

    assert {error["code"] for error in validate_experience_operations([operation])} == {
        "experience_payload_type_mismatch"
    }


def test_batch_rejects_encoded_or_case_variant_filename_aliases():
    encoded = _operation(
        "verify result",
        "viking://user/u/memories/experiences/verify%20result.md",
    )
    case_variant = _operation(
        "verification",
        "viking://user/u/memories/experiences/Verification.md",
    )

    encoded_codes = {error["code"] for error in validate_experience_operations([encoded])}
    case_codes = {error["code"] for error in validate_experience_operations([case_variant])}

    assert "experience_uri_name_mismatch" in encoded_codes
    assert "experience_uri_name_mismatch" in case_codes


def test_batch_rejects_raw_uri_controls_that_urlsplit_would_strip():
    operation = _operation(
        "verification",
        "viking://user/u/memories/experiences/verification.md\n",
    )

    assert "experience_uri_scope" in {
        error["code"] for error in validate_experience_operations([operation])
    }


def test_context_binding_accepts_current_user_forms_and_rejects_aliases():
    canonical = _operation(
        "verification",
        "viking://user/u/memories/experiences/verification.md",
    )
    shorthand = _operation(
        "verification",
        "viking://user/memories/experiences/verification.md",
    )
    other_user = _operation(
        "verification",
        "viking://user/other/memories/experiences/verification.md",
    )
    reserved_root_alias = _operation(
        "verification",
        "viking://user/memories/memories/experiences/verification.md",
    )

    assert validate_experience_operation_context([canonical, shorthand], _ctx()) == []
    assert {
        error["code"]
        for error in validate_experience_operation_context(
            [other_user, reserved_root_alias], _ctx()
        )
    } == {"experience_uri_context_mismatch"}


def test_context_binding_handles_user_id_equal_to_reserved_root_segment():
    operation = _operation(
        "verification",
        "viking://user/memories/memories/experiences/verification.md",
    )

    assert validate_experience_operation_context([operation], _ctx("memories")) == []


def test_non_experience_operation_cannot_target_experience_store():
    operation = SimpleNamespace(
        memory_type="skills",
        memory_fields={"skill_name": "verification", "content": "unstructured"},
        uris=["viking://user/u/memories/experiences/verification.md"],
        page_id=100,
    )

    assert {error["code"] for error in validate_experience_operations([operation])} == {
        "experience_uri_type_mismatch"
    }


def test_non_experience_operation_cannot_bypass_type_binding_with_duplicate_slashes():
    operation = SimpleNamespace(
        memory_type="skills",
        memory_fields={"skill_name": "bypass", "content": "unstructured"},
        uris=["viking://user/u/memories//experiences/bypass.md"],
        page_id=100,
    )

    assert {error["code"] for error in validate_experience_operations([operation])} == {
        "experience_uri_type_mismatch"
    }


def test_legacy_noncompliant_experience_is_read_only():
    operation = _operation(
        "verification",
        "viking://user/u/memories/experiences/verification.md",
    )
    operation.old_memory_file_content = SimpleNamespace(
        uri=operation.uris[0],
        content="legacy unstructured body",
    )

    codes = {error["code"] for error in validate_experience_operations([operation])}

    assert "legacy_experience_read_only" in codes


def test_stored_experience_requires_exact_type_name_and_uri_identity():
    missing_type = SimpleNamespace(
        uri="viking://user/u/memories/experiences/verification.md",
        content=VALID_CONTENT,
        memory_type=None,
        extra_fields={"experience_name": "verification"},
    )
    mismatched_name = SimpleNamespace(
        uri="viking://user/u/memories/experiences/verification.md",
        content=VALID_CONTENT,
        memory_type="experiences",
        extra_fields={"experience_name": "different"},
    )

    assert "experience_payload_type_mismatch" in {
        error["code"]
        for error in validate_stored_experience(missing_type, missing_type.uri)
    }
    assert "experience_uri_name_mismatch" in {
        error["code"]
        for error in validate_stored_experience(mismatched_name, mismatched_name.uri)
    }
