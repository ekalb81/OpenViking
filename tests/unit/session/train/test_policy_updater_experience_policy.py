from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.session.memory.dataclass import MemoryFile, ResolvedOperation, ResolvedOperations
from openviking.session.train.components.policy_optimizer import (
    PatchMergePolicyOptimizer,
    PatchMergePolicyOptimizerContext,
    _operations_to_plan_items,
)
from openviking.session.train.components.policy_updater import (
    DryRunPolicyUpdater,
    MemoryFilePolicyUpdater,
    _apply_items_to_snapshot,
    _plan_to_resolved_operations,
)
from openviking.session.train.domain import Policy, PolicyPlanItem, PolicySet, PolicyUpdatePlan
from openviking.session.train.gradients import PatchSemanticGradient

VALID_CONTENT = """## Situation
- A bounded implementation needs prior procedural guidance.

## Approach
- Inspect the current state before changing it.

## Reflect
- NEVER claim completion without verification.
"""
ROOT = "viking://user/u/memories/experiences"
LEGACY_URI = f"{ROOT}/legacy.md"


def _policy_set() -> PolicySet:
    return PolicySet(
        root_uri=ROOT,
        policies=[
            Policy(
                name="legacy",
                uri=LEGACY_URI,
                version=1,
                status="production",
                content="legacy unstructured content",
            )
        ],
    )


def _upsert(name: str, *, memory_type: str = "experiences") -> PolicyPlanItem:
    return PolicyPlanItem(
        kind="upsert",
        memory_type=memory_type,
        target_name=name,
        target_uri=f"viking://user/u/memories/{memory_type}/{name}.md",
        before_content=None,
        after_content=VALID_CONTENT.replace("prior procedural guidance", name),
    )


def _superseded_delete(*targets: str) -> PolicyPlanItem:
    return PolicyPlanItem(
        kind="delete",
        memory_type="experiences",
        target_name="legacy",
        target_uri=LEGACY_URI,
        before_content="legacy unstructured content",
        after_content=None,
        metadata={"superseded_by": list(targets)},
    )


def _resolve(plan: PolicyUpdatePlan):
    policy_set = _policy_set()
    updated = _apply_items_to_snapshot(plan.items, policy_set)
    return _plan_to_resolved_operations(
        plan=plan,
        policy_set=policy_set,
        updated_policy_set=updated,
    )


def test_training_plan_rejects_mapped_experience_delete_with_multiple_replacements():
    first = _upsert("first")
    second = _upsert("second")
    plan = PolicyUpdatePlan(
        items=[first, second, _superseded_delete(first.target_uri, second.target_uri)]
    )

    operations, errors = _resolve(plan)

    assert errors == ["automatic experience delete is disabled: legacy"]
    assert operations.delete_file_contents == []
    assert operations.delete_replacements == {}


def test_training_plan_rejects_mapped_experience_delete_with_one_replacement():
    replacement = _upsert("replacement")
    plan = PolicyUpdatePlan(items=[replacement, _superseded_delete(replacement.target_uri)])

    operations, errors = _resolve(plan)

    assert errors == ["automatic experience delete is disabled: legacy"]
    assert operations.delete_file_contents == []
    assert operations.delete_replacements == {}


def test_training_plan_rejects_conflicting_delete_items_for_one_legacy_experience():
    first = _upsert("first")
    second = _upsert("second")
    plan = PolicyUpdatePlan(
        items=[
            first,
            second,
            _superseded_delete(first.target_uri),
            _superseded_delete(second.target_uri),
        ]
    )

    operations, errors = _resolve(plan)

    assert errors == [
        "automatic experience delete is disabled: legacy",
        "automatic experience delete is disabled: legacy",
    ]
    assert operations.delete_file_contents == []
    assert operations.delete_replacements == {}


def test_training_plan_rejects_cross_type_supersession_target():
    skill = _upsert("replacement", memory_type="skills")
    plan = PolicyUpdatePlan(items=[skill, _superseded_delete(skill.target_uri)])

    operations, errors = _resolve(plan)

    assert errors == ["automatic experience delete is disabled: legacy"]
    assert operations.delete_file_contents == []
    assert operations.delete_replacements == {}


def test_training_plan_rejects_unmapped_experience_delete():
    plan = PolicyUpdatePlan(items=[_superseded_delete()])

    operations, errors = _resolve(plan)

    assert errors == ["automatic experience delete is disabled: legacy"]
    assert operations.delete_file_contents == []
    assert operations.delete_replacements == {}


def test_training_plan_preserves_non_experience_delete_without_new_replacement_semantics():
    replacement = _upsert("replacement", memory_type="skills")
    legacy_uri = "viking://user/u/memories/skills/legacy/SKILL.md"
    delete = PolicyPlanItem(
        kind="delete",
        memory_type="skills",
        target_name="legacy",
        target_uri=legacy_uri,
        before_content=None,
        after_content=None,
        metadata={"superseded_by": [replacement.target_uri]},
    )

    operations, errors = _resolve(PolicyUpdatePlan(items=[replacement, delete]))

    assert errors == []
    assert [memory.uri for memory in operations.delete_file_contents] == [legacy_uri]
    assert operations.delete_replacements == {}


def test_optimizer_does_not_synthesize_experience_delete_from_resolved_deletes():
    operations = ResolvedOperations(
        upsert_operations=[],
        delete_file_contents=[
            MemoryFile(
                uri=LEGACY_URI,
                content="legacy unstructured content",
                memory_type="experiences",
                extra_fields={"experience_name": "legacy"},
            )
        ],
        errors=[],
    )

    items = _operations_to_plan_items(
        operations=operations,
        gradients=[],
        policy_set=_policy_set(),
        memory_type="experiences",
    )

    assert items == []


def test_optimizer_rejects_experience_items_when_resolved_operations_have_errors():
    replacement_uri = f"{ROOT}/replacement.md"
    operations = ResolvedOperations(
        upsert_operations=[
            ResolvedOperation(
                memory_type="experiences",
                memory_fields={
                    "experience_name": "replacement",
                    "content": VALID_CONTENT.replace("prior procedural guidance", "replacement"),
                },
                uris=[replacement_uri],
            )
        ],
        delete_file_contents=[],
        errors=["experience supersession validation failed after repair"],
    )

    items = _operations_to_plan_items(
        operations=operations,
        gradients=[],
        policy_set=_policy_set(),
        memory_type="experiences",
    )

    assert items == []


@pytest.mark.asyncio
async def test_optimizer_merge_errors_fail_closed_through_policy_updaters(monkeypatch):
    policy_set = _policy_set()
    merge_error = "experience supersession validation failed after repair"
    replacement_uri = f"{ROOT}/replacement.md"
    operations = ResolvedOperations(
        upsert_operations=[
            ResolvedOperation(
                memory_type="experiences",
                memory_fields={
                    "experience_name": "replacement",
                    "content": VALID_CONTENT.replace(
                        "prior procedural guidance", "replacement"
                    ),
                },
                uris=[replacement_uri],
            )
        ],
        delete_file_contents=[],
        errors=[merge_error],
    )

    async def run_merge(*args, **kwargs):
        return operations

    monkeypatch.setattr(PatchMergePolicyOptimizer, "_run_merge_extract_loop", run_merge)
    gradient = PatchSemanticGradient(
        before_file=None,
        after_file=MemoryFile(
            uri=replacement_uri,
            content=VALID_CONTENT,
            memory_type="experiences",
            extra_fields={"experience_name": "replacement"},
        ),
        base_version=None,
        rationale="replace legacy",
        links=[],
        confidence=0.9,
    )
    plan = await PatchMergePolicyOptimizer().plan(
        [gradient],
        policy_set,
        PatchMergePolicyOptimizerContext(request_context=SimpleNamespace()),
    )

    assert plan.items == []
    assert plan.metadata["merge_errors"] == [merge_error]

    dry_run_result = await DryRunPolicyUpdater().apply(plan, policy_set)
    write_result = await MemoryFilePolicyUpdater(viking_fs=object()).apply(plan, policy_set)

    for result in (dry_run_result, write_result):
        assert result.errors == [merge_error]
        assert result.updated_policy_set is policy_set
        assert result.written_uris == []
        assert result.deleted_uris == []


def test_optimizer_does_not_synthesize_experience_delete_from_supersession_hint():
    replacement_uri = f"{ROOT}/replacement.md"
    replacement_file = MemoryFile(
        uri=replacement_uri,
        content=VALID_CONTENT.replace("prior procedural guidance", "replacement"),
        memory_type="experiences",
        extra_fields={"experience_name": "replacement"},
    )
    gradient = PatchSemanticGradient(
        before_file=None,
        after_file=replacement_file,
        base_version=None,
        rationale="replace legacy",
        links=[],
        confidence=0.9,
        metadata={"supersedes": "legacy"},
    )
    operations = ResolvedOperations(
        upsert_operations=[
            ResolvedOperation(
                memory_type="experiences",
                memory_fields={
                    "experience_name": "replacement",
                    "content": replacement_file.content,
                },
                uris=[replacement_uri],
            )
        ],
        delete_file_contents=[],
        errors=[],
    )

    items = _operations_to_plan_items(
        operations=operations,
        gradients=[gradient],
        policy_set=_policy_set(),
        memory_type="experiences",
    )

    assert [(item.kind, item.target_name) for item in items] == [("upsert", "replacement")]


def test_optimizer_preserves_non_experience_delete_synthesis():
    legacy_uri = "viking://user/u/memories/skills/legacy/SKILL.md"
    operations = SimpleNamespace(
        upsert_operations=[],
        delete_file_contents=[
            MemoryFile(
                uri=legacy_uri,
                content="legacy skill",
                memory_type="skills",
                extra_fields={"skill_name": "legacy"},
            )
        ],
        delete_replacements={},
    )

    items = _operations_to_plan_items(
        operations=operations,
        gradients=[],
        policy_set=PolicySet(root_uri="viking://user/u/memories/skills", policies=[]),
        memory_type="skills",
    )

    assert [(item.kind, item.target_name, item.target_uri) for item in items] == [
        ("delete", "legacy", legacy_uri)
    ]


@pytest.mark.asyncio
async def test_dry_run_rejects_the_same_invalid_experience_body_as_real_admission():
    policy_set = _policy_set()
    invalid = PolicyPlanItem(
        kind="upsert",
        memory_type="experiences",
        target_name="legacy",
        target_uri=LEGACY_URI,
        before_content="legacy unstructured content",
        after_content="still unstructured",
    )

    result = await DryRunPolicyUpdater().apply(PolicyUpdatePlan(items=[invalid]), policy_set)

    assert result.errors
    assert result.updated_policy_set is policy_set
    assert result.metadata["validation_error_count"] >= 1


@pytest.mark.asyncio
async def test_dry_run_preserves_string_merge_error_without_splitting_it():
    policy_set = _policy_set()
    merge_error = "merge failed"
    plan = PolicyUpdatePlan(items=[], metadata={"merge_errors": merge_error})

    result = await DryRunPolicyUpdater().apply(plan, policy_set)

    assert result.errors == [merge_error]
    assert result.updated_policy_set is policy_set


@pytest.mark.asyncio
async def test_policy_apply_metadata_propagates_pending_experience_index(monkeypatch):
    policy_set = _policy_set()
    item = _upsert("replacement")
    pending_uri = item.target_uri
    memory_result = SimpleNamespace(
        written_uris=[pending_uri],
        edited_uris=[],
        deleted_uris=[],
        index_pending_uris=[pending_uri],
        errors=[],
    )
    apply_operations = AsyncMock(return_value=memory_result)
    monkeypatch.setattr(
        "openviking.session.train.components.policy_updater.MemoryUpdater.apply_operations",
        apply_operations,
    )

    result = await MemoryFilePolicyUpdater(viking_fs=object()).apply(
        PolicyUpdatePlan(items=[item]),
        policy_set,
        context=SimpleNamespace(),
    )

    assert result.errors == []
    assert result.metadata["index_pending_uris"] == [pending_uri]
    apply_operations.assert_awaited_once()
