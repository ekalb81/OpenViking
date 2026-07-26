"""Reporting on how much tool-output hydration adds to the extraction input.

A session commits at commit_token_threshold (60k), but that is measured on the
truncated transcript. Hydration then restores every externalized tool output to
full size with no budget, so the prompt extraction receives can be far larger
than the threshold implies - one 190k-token extraction prompt was recorded on
2026-07-26. Traces rotate roughly every 13 hours at current volume, so the
frequency of that tail is unmeasurable today.

These tests pin the reporting, deliberately not a cap. The decision to bound
hydration should be made from recorded evidence, and a limit picked without it
risks starving extraction of the tool output it exists to read.
"""

import pytest

from openviking.message import Message, ToolPart


class _CapturedWarnings:
    """Collect records straight off the module logger.

    caplog relies on propagation reaching the root logger, which this project's
    logging configuration does not reliably allow. Attaching directly removes
    that dependency and makes the assertion about the code, not the harness.
    """

    def __init__(self):
        self.records = []

    def __enter__(self):
        import logging

        from openviking.session import session as session_module

        self._logger = session_module.logger
        self._handler = logging.Handler()
        self._handler.setLevel(logging.WARNING)
        self._handler.emit = self.records.append
        self._logger.addHandler(self._handler)
        return self

    def __exit__(self, *exc):
        self._logger.removeHandler(self._handler)
        return False

    @property
    def messages(self):
        return [r.getMessage() for r in self.records if r.levelno >= 30]


class _FakeStore:
    """Returns a payload of a requested size, standing in for the result store."""

    def __init__(self, payload: str):
        self.payload = payload

    async def read(self, tool_result_id, offset=0, limit=-1, include_metadata=False):
        del tool_result_id, offset, limit, include_metadata
        return {"content": self.payload}


def _tool_message(msg_id: str, preview: str) -> Message:
    part = ToolPart(
        tool_id=f"t-{msg_id}",
        tool_name="read_file",
        tool_input={},
        tool_output=preview,
    )
    part.tool_output_ref = "viking://temp/tool_results/abc"
    part.tool_output_truncated = True
    return Message(id=msg_id, role="assistant", parts=[part])


def _session_with_store(store):
    from openviking.session.session import Session

    session = Session.__new__(Session)
    session.session_id = "cx-test"
    session._tool_result_store = lambda: store
    return session


@pytest.mark.asyncio
async def test_a_large_hydration_is_reported():
    # 400k restored from a 100-char preview: the growth is what the commit
    # threshold never saw.
    session = _session_with_store(_FakeStore("x" * 400_000))
    messages = [_tool_message("m1", "x" * 100)]

    with _CapturedWarnings() as cap:
        await session._hydrate_tool_outputs_for_extraction(messages)

    assert cap.messages, "a hydration this large must leave evidence"
    text = cap.messages[0]
    assert "cx-test" in text
    assert "unbounded" in text


@pytest.mark.asyncio
async def test_ordinary_hydration_is_not_reported_as_a_warning():
    # The report must stay rare enough to mean something. A modest restore is
    # the normal case and should not raise a warning.
    session = _session_with_store(_FakeStore("x" * 5_000))
    messages = [_tool_message("m1", "x" * 100)]

    with _CapturedWarnings() as cap:
        await session._hydrate_tool_outputs_for_extraction(messages)

    assert not cap.messages


@pytest.mark.asyncio
async def test_growth_is_measured_not_absolute_size():
    """A big output that was never truncated adds nothing and must not report.

    Counting absolute size would fire on content the commit threshold already
    accounted for, making the signal useless for the question being asked.
    """
    payload = "x" * 400_000
    session = _session_with_store(_FakeStore(payload))
    messages = [_tool_message("m1", payload)]  # preview already the full size

    with _CapturedWarnings() as cap:
        await session._hydrate_tool_outputs_for_extraction(messages)

    assert not cap.messages


@pytest.mark.asyncio
async def test_growth_accumulates_across_several_outputs():
    # The tail may be one huge output or many mid-sized ones; the report has to
    # catch both, so it sums.
    session = _session_with_store(_FakeStore("x" * 30_000))
    messages = [_tool_message(f"m{i}", "") for i in range(4)]

    with _CapturedWarnings() as cap:
        await session._hydrate_tool_outputs_for_extraction(messages)

    assert cap.messages, "4 x 30k should cross the reporting line even though none alone does"
    assert "4 tool output(s)" in cap.messages[0]


@pytest.mark.asyncio
async def test_hydration_still_returns_the_full_content():
    """Reporting must not become a cap by accident."""
    payload = "x" * 400_000
    session = _session_with_store(_FakeStore(payload))
    messages = [_tool_message("m1", "x" * 100)]

    result = await session._hydrate_tool_outputs_for_extraction(messages)

    part = result[0].parts[0]
    assert len(part.tool_output) == len(payload), "hydration must remain unbounded for now"
