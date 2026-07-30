"""The optimizer may remove an experience only while it satisfies the policy.

Experience deletes and supersessions were disabled wholesale because legacy
sources predating the experience policy could not survive being folded away.
The guard is now per file, so the interesting cases are the two edges: a
conforming file must be removable, and a non-conforming one must not be --
regardless of what the model asked for.
"""

import pytest

from openviking.session.memory.dataclass import MemoryFile
from openviking.session.train.components.policy_optimizer import (
    _experience_removal_allowed,
)

URI = "viking://user/default/memories/experiences/sample.md"

CONFORMING = """## Situation
- A stored experience needs to satisfy the bounded-experience policy.

## Approach
- Keep each section within its bullet cap and every bullet under the limit.

## Reflect
- NEVER let an unbounded body reach the store.
"""

# Four Situation bullets against a cap of three.
TOO_MANY_BULLETS = """## Situation
- One.
- Two.
- Three.
- Four.

## Approach
- Do the thing.

## Reflect
- NEVER skip the thing.
"""


def _file(content: str, uri: str = URI) -> MemoryFile:
    return MemoryFile(
        uri=uri,
        content=content,
        memory_type="experiences",
        extra_fields={"memory_type": "experiences", "experience_name": "sample"},
    )


def test_conforming_experience_is_removable():
    assert _experience_removal_allowed(_file(CONFORMING), URI) is True


def test_non_conforming_experience_is_not_removable():
    assert _experience_removal_allowed(_file(TOO_MANY_BULLETS), URI) is False


def test_oversized_body_is_not_removable():
    body = "## Situation\n- x\n\n## Approach\n- %s\n\n## Reflect\n- y\n" % ("a" * 4000)
    assert _experience_removal_allowed(_file(body), URI) is False


def test_missing_file_is_not_removable():
    assert _experience_removal_allowed(None, URI) is False


def test_missing_uri_is_not_removable():
    assert _experience_removal_allowed(_file(CONFORMING), "") is False


def test_uri_mismatch_is_not_removable():
    # Identity must match the file being validated; a mismatch means we cannot
    # be sure which memory we would be deleting.
    other = "viking://user/default/memories/experiences/other.md"
    assert _experience_removal_allowed(_file(CONFORMING), other) is False


def test_unvalidatable_file_is_not_removable(monkeypatch):
    """Over-blocking forgoes a cleanup; under-blocking destroys a memory."""
    import openviking.session.train.components.policy_optimizer as mod

    def _boom(*_a, **_k):
        raise RuntimeError("validator exploded")

    monkeypatch.setattr(mod, "validate_stored_experience", _boom)
    assert _experience_removal_allowed(_file(CONFORMING), URI) is False


@pytest.mark.parametrize("content", ["", "   ", "no sections at all"])
def test_malformed_bodies_are_not_removable(content):
    assert _experience_removal_allowed(_file(content), URI) is False


# --- behaviour: supersession is re-enabled, but only for conforming files -----

from types import SimpleNamespace  # noqa: E402

from openviking.session.train import Experience, ExperienceSet, PatchSemanticGradient  # noqa: E402
from openviking.session.train.components.policy_optimizer import (  # noqa: E402
    _operations_to_plan_items,
)

OLD_URI = "viking://user/u/memories/experiences/old_narrow.md"
NEW_URI = "viking://user/u/memories/experiences/broad_replacement.md"


def _exp_file(name, uri, content):
    return MemoryFile(
        uri=uri,
        content=content,
        memory_type="experiences",
        extra_fields={
            "memory_type": "experiences",
            "experience_name": name,
            "status": "production",
            "version": 1,
        },
    )


def _supersede_gradient():
    after = _exp_file("broad_replacement", NEW_URI, CONFORMING)
    after.extra_fields["supersedes"] = "old_narrow"
    return PatchSemanticGradient(
        before_file=None,
        after_file=after,
        base_version=1,
        rationale="broader experience subsumes the narrow one",
        links=[],
        confidence=0.9,
        metadata={},
    )


def _policy_set(old_content):
    return ExperienceSet(
        root_uri="viking://user/u/memories/experiences",
        policies=[
            Experience(
                name="old_narrow",
                uri=OLD_URI,
                version=1,
                status="production",
                content=old_content,
            )
        ],
    )


def _operations():
    return SimpleNamespace(
        upsert_operations=[],
        delete_file_contents=[],
        errors=[],
        model_fields={"experiences": None, "links": None, "delete_ids": None},
    )


def _plan_items(old_content):
    return _operations_to_plan_items(
        operations=_operations(),
        gradients=[_supersede_gradient()],
        policy_set=_policy_set(old_content),
        memory_type="experiences",
    )


def test_conforming_experience_can_be_superseded():
    """The whole point of re-enabling removal: this used to always return []."""
    items = _plan_items(CONFORMING)
    deletes = [i for i in items if i.kind == "delete"]
    assert len(deletes) == 1, "a conforming superseded experience must be removable"
    assert deletes[0].target_uri == OLD_URI


def test_non_conforming_experience_is_never_superseded():
    items = _plan_items(TOO_MANY_BULLETS)
    assert [i for i in items if i.kind == "delete"] == [], (
        "a legacy-invalid experience must stay untouchable"
    )


def test_non_experience_types_are_unaffected_by_the_gate():
    items = _operations_to_plan_items(
        operations=_operations(),
        gradients=[_supersede_gradient()],
        policy_set=_policy_set(TOO_MANY_BULLETS),
        memory_type="trajectories",
    )
    # The gate is experience-only; trajectories keep their prior behaviour even
    # though this body would fail the experience policy.
    assert len([i for i in items if i.kind == "delete"]) == 1
