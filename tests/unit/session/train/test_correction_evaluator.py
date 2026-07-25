# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""The correction evaluator's safety rests entirely on verification, so that is what is tested hardest.

An LLM asked "what did the user correct?" will find something even when the user corrected nothing. The
design accepts that and does not trust the answer: every reported correction must cite a turn and quote the
user verbatim, and the quote is re-checked against the real transcript. These tests are the proof that a
fabricated report cannot become a stored lesson.
"""

from __future__ import annotations

from openviking.message import Message, TextPart
from openviking.session.train.components.correction_evaluator import (
    CORRECTION_CRITERION,
    MIN_QUOTE_CHARS,
    NO_CORRECTION_CRITERION,
    SessionCorrectionEvaluator,
    evaluation_from_corrections,
    verify_corrections,
)
from openviking.session.train.domain import Case, Rollout, Rubric


def _msg(role: str, text: str, index: int) -> Message:
    return Message(id=f"msg-{index}", role=role, parts=[TextPart(text=text)])


# A transcript modelled on a real exchange: the assistant asserts a wrong gap, the user corrects it.
TRANSCRIPT = [
    _msg("user", "Why did trajectories stop being written?", 0),
    _msg("assistant", "Acquisition stayed invisible for a month because v3 dropped the method.", 1),
    _msg("user", "Are you okay, the 23rd was 2 days ago", 2),
    _msg("assistant", "You are right, it was two days, not a month.", 3),
]


def _report(**overrides):
    base = {
        "corrected_turn": 1,
        "correction_turn": 2,
        "user_quote": "the 23rd was 2 days ago",
        "what_was_wrong": "Claimed the gap was a month.",
        "what_is_correct": "The gap was two days.",
    }
    base.update(overrides)
    return {"corrections": [base]}


def test_a_real_correction_verifies():
    events = verify_corrections(_report(), TRANSCRIPT)

    assert len(events) == 1
    event = events[0]
    assert event.corrected_turn == 1
    assert event.correction_turn == 2
    evidence = event.as_evidence()
    assert any("[2][user]" in line for line in evidence)
    assert any("[1][assistant]" in line for line in evidence)


def test_a_fabricated_quote_is_discarded():
    # The central guarantee. The model may report a correction it read; it cannot invent words the user
    # never said, because the quote is checked against the transcript.
    events = verify_corrections(
        _report(user_quote="you should always double-check your arithmetic"),
        TRANSCRIPT,
    )

    assert events == []


def test_quote_must_come_from_the_cited_turn_not_merely_exist_somewhere():
    # "the 23rd was 2 days ago" is genuinely in the transcript, but at turn 2, not turn 0. Accepting it
    # against the wrong turn would let a real phrase justify a fabricated pairing.
    events = verify_corrections(
        _report(correction_turn=0, corrected_turn=0),
        TRANSCRIPT,
    )

    assert events == []


def test_paraphrase_is_discarded():
    # Paraphrase is where a fabricated lesson would hide: close enough to look cited, different enough to
    # mean something the user did not say.
    events = verify_corrections(
        _report(user_quote="the twenty-third was two days ago"),
        TRANSCRIPT,
    )

    assert events == []


def test_correction_cannot_precede_the_turn_it_corrects():
    events = verify_corrections(
        _report(corrected_turn=3, correction_turn=2),
        TRANSCRIPT,
    )

    assert events == []


def test_roles_must_match_a_user_correcting_an_assistant():
    # Citing an assistant turn as the corrector, or a user turn as the corrected one, is incoherent.
    assert verify_corrections(_report(correction_turn=3, corrected_turn=1), TRANSCRIPT) == []
    assert verify_corrections(_report(corrected_turn=0, correction_turn=2), TRANSCRIPT) == []


def test_trivial_quotes_are_rejected():
    # Short spans match too much of any transcript to be evidence: "no" appears everywhere.
    short = "no" * 2
    assert len(short) < MIN_QUOTE_CHARS
    transcript = [
        _msg("assistant", "I will flip the flag.", 0),
        _msg("user", f"{short} really", 1),
    ]

    assert verify_corrections(_report(corrected_turn=0, correction_turn=1, user_quote=short), transcript) == []


def test_out_of_range_and_malformed_indices_are_discarded():
    assert verify_corrections(_report(corrected_turn=99), TRANSCRIPT) == []
    assert verify_corrections(_report(correction_turn=-1), TRANSCRIPT) == []
    assert verify_corrections(_report(corrected_turn="one"), TRANSCRIPT) == []


def test_missing_lesson_fields_are_discarded():
    assert verify_corrections(_report(what_was_wrong=""), TRANSCRIPT) == []
    assert verify_corrections(_report(what_is_correct="   "), TRANSCRIPT) == []


def test_malformed_payloads_yield_nothing():
    for payload in (None, [], {}, {"corrections": "nope"}, {"corrections": [None, 7, "x"]}):
        assert verify_corrections(payload, TRANSCRIPT) == []


def test_duplicate_reports_collapse():
    payload = _report()
    payload["corrections"].append(dict(payload["corrections"][0]))

    assert len(verify_corrections(payload, TRANSCRIPT)) == 1


def test_no_corrections_is_a_first_class_answer():
    # The protocol returns a non-optional evaluation, so "nothing found" must be expressible without
    # inventing a correction and without claiming success.
    evaluation = evaluation_from_corrections([])

    assert evaluation.passed is False
    assert evaluation.metadata["verified_correction_count"] == 0
    assert len(evaluation.criterion_results) == 1
    assert evaluation.criterion_results[0].criterion_name == NO_CORRECTION_CRITERION
    assert evaluation.criterion_results[0].evidence == []


def test_verified_corrections_carry_citations_into_the_evaluation():
    events = verify_corrections(_report(), TRANSCRIPT)

    evaluation = evaluation_from_corrections(events, reported_count=1)

    assert evaluation.metadata["verified_correction_count"] == 1
    result = evaluation.criterion_results[0]
    assert result.criterion_name.startswith(CORRECTION_CRITERION)
    assert result.evidence, "a correction without evidence is exactly what this design forbids"


async def test_evaluator_discards_a_wholly_fabricated_report():
    # End to end: a model that invents a correction produces no correction-grounded evaluation, and the
    # count of what it claimed is retained so the discard is visible rather than silent.
    class FabricatingVlm:
        async def get_completion_async(self, prompt: str, thinking=None):
            del prompt, thinking
            return (
                '{"corrections": [{"corrected_turn": 1, "correction_turn": 2, '
                '"user_quote": "you must never trust a compressor", '
                '"what_was_wrong": "invented", "what_is_correct": "also invented"}]}'
            )

    evaluator = SessionCorrectionEvaluator(vlm=FabricatingVlm())
    rollout = Rollout(
        case=Case(name="c", task_signature="t", input={}, rubric=Rubric("r", "d", [])),
        messages=TRANSCRIPT,
        policy_snapshot_id="snap",
    )

    evaluation = await evaluator.evaluate(rollout)

    assert evaluation.metadata["verified_correction_count"] == 0
    assert evaluation.metadata["reported_correction_count"] == 1
    assert evaluation.criterion_results[0].criterion_name == NO_CORRECTION_CRITERION


async def test_evaluator_accepts_a_grounded_report():
    class HonestVlm:
        async def get_completion_async(self, prompt: str, thinking=None):
            del thinking
            assert "[2][user]" in prompt, "the prompt must carry the indices the model is asked to cite"
            return (
                '```json\n{"corrections": [{"corrected_turn": 1, "correction_turn": 2, '
                '"user_quote": "the 23rd was 2 days ago", '
                '"what_was_wrong": "Claimed a month.", "what_is_correct": "It was two days."}]}\n```'
            )

    evaluator = SessionCorrectionEvaluator(vlm=HonestVlm())
    rollout = Rollout(
        case=Case(name="c", task_signature="t", input={}, rubric=Rubric("r", "d", [])),
        messages=TRANSCRIPT,
        policy_snapshot_id="snap",
    )

    evaluation = await evaluator.evaluate(rollout)

    assert evaluation.metadata["verified_correction_count"] == 1
    assert "[2][user] the 23rd was 2 days ago" in evaluation.criterion_results[0].evidence[0]


async def test_provider_failure_does_not_fabricate_a_pass():
    class BrokenVlm:
        async def get_completion_async(self, prompt: str, thinking=None):
            del prompt, thinking
            raise RuntimeError("provider down")

    evaluator = SessionCorrectionEvaluator(vlm=BrokenVlm())
    rollout = Rollout(
        case=Case(name="c", task_signature="t", input={}, rubric=Rubric("r", "d", [])),
        messages=TRANSCRIPT,
        policy_snapshot_id="snap",
    )

    evaluation = await evaluator.evaluate(rollout)

    assert evaluation.passed is False
    assert evaluation.metadata["verified_correction_count"] == 0


def test_production_compressor_wires_a_real_evaluator(monkeypatch):
    """Without an evaluator the analysis falls back to a fabricated passed=True.

    ``_evaluation_from_trajectories`` reports success whenever a trajectory file was written, and
    ``gradient_estimator._confidence`` adds +0.2 for it — so an unwired analyzer silently inflates the
    confidence of every distilled experience. This pins the wiring rather than the default.
    """
    from openviking.session import compressor_v3

    # The analyzer is built against a live VikingFS; a unit test has no business standing one up.
    monkeypatch.setattr(compressor_v3, "get_viking_fs", lambda: None)

    compressor = compressor_v3.SessionCompressorV3(vikingdb=None, rollout_analyzer=None)

    evaluator = compressor.rollout_analyzer.evaluator
    assert evaluator is not None, "analyzer would fall back to the fabricated passed=True evaluation"
    assert isinstance(evaluator, SessionCorrectionEvaluator)
