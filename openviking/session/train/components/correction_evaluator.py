# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Structured correction signal for rollout evaluation.

Experience distillation is only as trustworthy as its evidence that something went wrong. Without an
evaluator, ``TrajectoryRolloutAnalyzer`` falls back to ``_evaluation_from_trajectories``, which reports
``passed=True`` whenever any trajectory file was written — a success claim resting on nothing but the
existence of a file. Distilling "lessons" against that signal is how a memory store fills with
confidently-stated things nobody ever established.

This evaluator supplies the missing evidence: it finds the points where the **user corrected the
assistant**, and reports each one with a citation.

Why an LLM, and why that is safe here
-------------------------------------
The pipeline already trusts an LLM to read the whole conversation and *write the lesson*. Asking one to
notice "the 23rd was 2 days ago" is strictly less demanding than that — reading comprehension rather than
judgement — so refusing the LLM here would be inconsistent, and a hand-built lexical detector would be
worse: brittle, low-recall, and still unable to say what the correct answer was.

The real risk is not missing a correction. It is **inventing one** when the user never corrected anything,
because a model asked "what did the user correct?" will find something. So the model's answer is not
trusted: every reported correction must cite a turn index *and quote the user verbatim*, and
:func:`_verify_correction` re-reads the actual transcript and discards anything that does not check out. A
model can report a correction it read; it cannot fabricate a quote that is not in the messages.

That "cite an index, then resolve it against the real messages" pattern is not new here — ``events.yaml``
already has an LLM emit a ``ranges`` field that ``ExtractContext.read_message_ranges`` resolves. This adds
the quote so the citation is checkable rather than merely well-formed.

Finding nothing is a first-class answer. ``RolloutEvaluator.evaluate`` returns a non-optional
``RubricEvaluation``, so there is structural pressure to always produce *something*; a run with no
corrections returns :data:`NO_CORRECTION_CRITERION` with ``passed=False`` and an explicit "none found"
note, never an invented one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from openviking.message import Message
from openviking.session.memory.utils.json_parser import extract_json_content
from openviking.session.train.domain import CriterionResult, Rollout, RubricEvaluation
from openviking.telemetry import tracer
from openviking_cli.utils import get_logger
from openviking_cli.utils.config import get_openviking_config
from openviking_cli.utils.llm import parse_json_from_response

logger = get_logger(__name__)

CORRECTION_CRITERION = "user_correction"
NO_CORRECTION_CRITERION = "no_user_correction_found"

# A quote shorter than this matches too much of any transcript to be evidence of anything: "no", "wrong"
# and "yes" all appear constantly, so requiring a substantive span is what makes verification meaningful.
MIN_QUOTE_CHARS = 12

_WHITESPACE_RE = re.compile(r"\s+")

_PROMPT_TEMPLATE = """\
You are auditing a conversation between a user and an AI assistant to find places where the USER
CORRECTED THE ASSISTANT.

A correction is a user turn that tells the assistant something it did or claimed was wrong: a wrong fact,
a wrong diagnosis, a wrong action, a misread instruction, or an approach the user rejects.

These are NOT corrections:
- A new request, a follow-up, or a change of direction.
- A clarifying question from the user.
- The user supplying information the assistant asked for.
- The assistant correcting itself with no user prompting.

Each message below is prefixed with its index as [N][role][speaker].

Return JSON only, matching this shape:

{{
  "corrections": [
    {{
      "corrected_turn": <index of the ASSISTANT message that was wrong>,
      "correction_turn": <index of the USER message that corrected it>,
      "user_quote": "<a verbatim span copied exactly from that user message, at least {min_quote} characters>",
      "what_was_wrong": "<what the assistant got wrong, in one sentence>",
      "what_is_correct": "<what the user established was actually true or required, in one sentence>"
    }}
  ]
}}

Rules:
- "user_quote" MUST be copied character-for-character from the user message at "correction_turn". It is
  checked against the transcript and the entry is discarded if it does not match. Do not paraphrase,
  summarise, translate, or fix typos.
- "corrected_turn" MUST be an assistant message that appears BEFORE "correction_turn".
- Report only corrections you can point at. If the user never corrected the assistant, return
  {{"corrections": []}}. An empty list is a correct and expected answer; do not manufacture an entry.
- Do not infer a lesson beyond what the user actually said in the quoted span.

Conversation:

{conversation}
"""


@dataclass(slots=True)
class CorrectionEvent:
    """One user correction, after verification against the transcript."""

    corrected_turn: int
    correction_turn: int
    user_quote: str
    what_was_wrong: str
    what_is_correct: str

    def as_evidence(self) -> list[str]:
        return [
            f"[{self.correction_turn}][user] {self.user_quote}",
            f"[{self.corrected_turn}][assistant] corrected: {self.what_was_wrong}",
        ]


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text or "").strip().lower()


def _message_text(message: Message) -> str:
    parts = []
    for part in getattr(message, "parts", None) or []:
        text = getattr(part, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
    return "\n".join(parts)


def _render_conversation(messages: list[Message]) -> str:
    """Render messages with the same [idx][role][speaker] header the extract prompts use."""
    lines = []
    for idx, message in enumerate(messages):
        body = _message_text(message)
        if not body.strip():
            continue
        speaker = getattr(message, "peer_id", None) or message.role
        lines.append(f"[{idx}][{message.role}][{speaker}]: {body}")
    return "\n\n".join(lines)


def _verify_correction(raw: Any, messages: list[Message]) -> CorrectionEvent | None:
    """Re-check one reported correction against the transcript.

    This is the whole safety argument: the model proposes, and this disposes. Every failure path here is a
    way a fabricated or confused entry gets dropped rather than becoming a stored lesson.
    """
    if not isinstance(raw, dict):
        return None

    try:
        corrected_turn = int(raw.get("corrected_turn"))
        correction_turn = int(raw.get("correction_turn"))
    except (TypeError, ValueError):
        return None

    if not (0 <= corrected_turn < len(messages)) or not (0 <= correction_turn < len(messages)):
        return None
    # A correction cannot precede the thing it corrects.
    if corrected_turn >= correction_turn:
        return None
    if messages[correction_turn].role != "user":
        return None
    if messages[corrected_turn].role != "assistant":
        return None

    quote = str(raw.get("user_quote") or "")
    if len(quote.strip()) < MIN_QUOTE_CHARS:
        return None
    # The load-bearing check: the user really said this, in that turn.
    if _normalize(quote) not in _normalize(_message_text(messages[correction_turn])):
        return None

    what_was_wrong = str(raw.get("what_was_wrong") or "").strip()
    what_is_correct = str(raw.get("what_is_correct") or "").strip()
    if not what_was_wrong or not what_is_correct:
        return None

    return CorrectionEvent(
        corrected_turn=corrected_turn,
        correction_turn=correction_turn,
        user_quote=quote.strip(),
        what_was_wrong=what_was_wrong,
        what_is_correct=what_is_correct,
    )


def verify_corrections(payload: Any, messages: list[Message]) -> list[CorrectionEvent]:
    """Verify every reported correction, dropping the ones that do not hold up."""
    if not isinstance(payload, dict):
        return []
    raw_corrections = payload.get("corrections")
    if not isinstance(raw_corrections, list):
        return []

    verified: list[CorrectionEvent] = []
    seen: set[tuple[int, int]] = set()
    for raw in raw_corrections:
        event = _verify_correction(raw, messages)
        if event is None:
            logger.debug("Discarded unverifiable correction report: %r", raw)
            continue
        key = (event.corrected_turn, event.correction_turn)
        if key in seen:
            continue
        seen.add(key)
        verified.append(event)
    return verified


def evaluation_from_corrections(
    corrections: list[CorrectionEvent],
    *,
    reported_count: int = 0,
) -> RubricEvaluation:
    """Build the rubric evaluation.

    ``passed`` answers "did this session run without the user having to correct the assistant", so a
    verified correction means ``passed=False``. Note that downstream this is a confidence weight
    (``gradient_estimator._confidence``), not a hard gate — it lowers the confidence attached to what is
    distilled rather than blocking it.
    """
    if not corrections:
        return RubricEvaluation(
            passed=False,
            score=0.0,
            criterion_results=[
                CriterionResult(
                    criterion_name=NO_CORRECTION_CRITERION,
                    passed=False,
                    score=0.0,
                    feedback=["No verified user correction found in this session."],
                    evidence=[],
                    metadata={"reported_unverified": reported_count},
                )
            ],
            feedback=["No verified user correction found; no correction-grounded lesson is available."],
            metadata={"verified_correction_count": 0, "reported_correction_count": reported_count},
        )

    results = [
        CriterionResult(
            criterion_name=f"{CORRECTION_CRITERION}_{index}",
            passed=False,
            score=0.0,
            feedback=[event.what_is_correct],
            evidence=event.as_evidence(),
            metadata={
                "corrected_turn": event.corrected_turn,
                "correction_turn": event.correction_turn,
            },
        )
        for index, event in enumerate(corrections)
    ]
    return RubricEvaluation(
        passed=False,
        score=0.0,
        criterion_results=results,
        feedback=[event.what_was_wrong for event in corrections],
        metadata={
            "verified_correction_count": len(corrections),
            "reported_correction_count": reported_count,
        },
    )


@dataclass(slots=True)
class SessionCorrectionEvaluator:
    """A :class:`RolloutEvaluator` that grounds distillation in verified user corrections."""

    vlm: Any = None
    # Thinking mode, matching the extraction phases. Deciding whether a turn corrected an earlier one — and
    # which earlier one — is a reasoning task, and a wrong pairing is exactly what the verifier then has to
    # throw away, so it is worth the tokens to get the pairing right the first time.
    #
    # Safe here specifically because this evaluator sends no tools. Alibaba rejects thinking mode combined
    # with a `required`/object tool_choice ("The tool_choice parameter does not support being set to
    # required or object in thinking mode"), which is what intermittently fails the tool-using extraction
    # phases; a plain prompt cannot hit that.
    thinking: bool | None = True

    @tracer("train.correction_evaluator.evaluate", ignore_result=True, ignore_args=True)
    async def evaluate(self, rollout: Rollout, context: Any = None) -> RubricEvaluation:
        del context  # The transcript is the whole input; no external context is consulted.
        messages = list(getattr(rollout, "messages", None) or [])
        if not messages:
            return evaluation_from_corrections([])

        conversation = _render_conversation(messages)
        if not conversation.strip():
            return evaluation_from_corrections([])

        vlm = self.vlm or get_openviking_config().vlm.get_vlm_instance()
        prompt = _PROMPT_TEMPLATE.format(conversation=conversation, min_quote=MIN_QUOTE_CHARS)

        try:
            response = await vlm.get_completion_async(prompt=prompt, thinking=self.thinking)
        except Exception:
            # A provider failure must not fabricate a pass. Degrade to "no verified correction", which
            # simply means nothing correction-grounded is available from this rollout.
            logger.exception("Correction evaluation call failed; reporting no verified correction")
            return evaluation_from_corrections([])

        payload = _parse_payload(response)
        reported = payload.get("corrections") if isinstance(payload, dict) else None
        reported_count = len(reported) if isinstance(reported, list) else 0
        corrections = verify_corrections(payload, messages)

        if reported_count and not corrections:
            logger.info(
                "Correction evaluator discarded all %d reported corrections as unverifiable",
                reported_count,
            )
        return evaluation_from_corrections(corrections, reported_count=reported_count)


def _parse_payload(response: Any) -> Any:
    text = response if isinstance(response, str) else _response_text(response)
    if not text:
        return {}
    try:
        return parse_json_from_response(text)
    except Exception:
        pass
    try:
        import json

        return json.loads(extract_json_content(text))
    except Exception:
        logger.warning("Correction evaluator returned unparseable output")
        return {}


def _response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    content = getattr(response, "content", None)
    return content if isinstance(content, str) else ""
