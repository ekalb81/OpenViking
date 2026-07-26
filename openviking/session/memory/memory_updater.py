# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
Memory updater - applies MemoryOperations directly.

This is the system executor that applies LLM's final output (MemoryOperations)
to the storage system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from openviking.session.memory.memory_isolation_handler import MemoryIsolationHandler

from openviking.core.namespace import canonical_user_root, canonicalize_uri
from openviking.message import Message
from openviking.message.part import TextPart
from openviking.server.identity import RequestContext
from openviking.session.memory.dataclass import (
    MemoryFile,
    ResolvedOperation,
    ResolvedOperations,
    StoredLink,
)
from openviking.session.memory.experience_policy import (
    uri_targets_experience_store,
    validate_experience_operation_context,
    validate_experience_operations,
    validate_stored_experience,
)
from openviking.session.memory.memory_type_registry import MemoryTypeRegistry
from openviking.session.memory.merge_op import MergeOpFactory
from openviking.session.memory.page_id_map import PageIdMap
from openviking.session.memory.utils.memory_file_utils import (
    MemoryFileUtils,
    bump_memory_version,
    next_memory_version,
)
from openviking.session.memory.utils.resource_refs import (
    RESOURCE_REF_SOURCE_SESSION_COMMIT,
    sync_memory_resource_refs,
)
from openviking.session.memory.utils.template_utils import TemplateUtils
from openviking.session.memory.utils.uri import render_template
from openviking.storage.viking_fs import get_viking_fs
from openviking.telemetry import tracer
from openviking.telemetry.request_wait_tracker import get_request_wait_tracker
from openviking.telemetry.tracer import get_trace_id
from openviking.utils.time_utils import parse_iso_datetime
from openviking_cli.exceptions import NotFoundError
from openviking_cli.utils import VikingURI, get_logger

logger = get_logger(__name__)

_MEMORY_ABSTRACT_MAX_BYTES = 50_000
_EXTRACTION_CHUNK_MIN_CHARS = 100

# --- Ingestion hygiene ---
# Models occasionally echo the extraction prompt back into field values, and
# tool output pasted into a field can carry whole-document line-number prefixes.
# Both survive into the stored memory and its embedding unless scrubbed here.
#
# An earlier revision instead capped the raw chatlog that event memories used to
# embed (observed at up to 470x the summary size). The events template no longer
# renders a chatlog at all, so the cap and its noise regexes were removed rather
# than left unreachable.
#
# Fingerprints of extraction-prompt text that models occasionally echo back
# into field values (observed: trajectories retrieval_anchor "Rules:" block and
# its format-spec placeholder line).
_TEMPLATE_ECHO_FINGERPRINTS = (
    "Keep it shorter and more retrieval-focused than content",
    "positive applies-when language",
    "Do not copy the opening request when the reusable lesson",
)
_TEMPLATE_PLACEHOLDER_FINGERPRINTS = (
    "operation, check, handoff, or response this record can guide",
    "verified reusable condition; Capability:",
)
_RULES_BLOCK_RE = re.compile(r"\n\s*Rules:\s*\n.*\Z", re.DOTALL)
_LINE_NUMBER_PREFIX_RE = re.compile(r"^\s*(\d+)\t ?(.*)$")
# Similarity threshold above which two add_only memories from the same batch or
# the same source session are treated as duplicates.
_ADD_ONLY_DEDUPE_RATIO = 0.85
_ADD_ONLY_DEDUPE_MAX_SIBLINGS = 24


# A rendered memory file is scaffolding (frontmatter, the fields comment, and
# headings) wrapped around a body. Strip the scaffolding to ask whether the body
# said anything.
_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}[^\n]*$", re.MULTILINE)


def _rendered_body_is_empty(rendered: str) -> bool:
    """True when a rendered memory file carries nothing beneath its scaffolding.

    A memory whose body is only headings is worse than no memory at all: it
    occupies a slot, is returned by recall, and says nothing when it gets there.
    """
    if not rendered:
        return True
    body = _FRONTMATTER_RE.sub("", rendered)
    body = _HTML_COMMENT_RE.sub("", body)
    body = _MARKDOWN_HEADING_RE.sub("", body)
    return not body.strip()


def _scrub_template_echo(value: str) -> str:
    """Remove extraction-prompt text the model echoed into a field value."""
    if not isinstance(value, str) or not value:
        return value
    match = _RULES_BLOCK_RE.search(value)
    if match and any(fp in match.group(0) for fp in _TEMPLATE_ECHO_FINGERPRINTS):
        value = value[: match.start()].rstrip()
    if any(fp in value for fp in _TEMPLATE_PLACEHOLDER_FINGERPRINTS):
        lines = [
            line
            for line in value.splitlines()
            if not any(fp in line for fp in _TEMPLATE_PLACEHOLDER_FINGERPRINTS)
        ]
        value = "\n".join(lines).strip()
    return value


def _strip_line_number_artifact(value: str) -> str:
    """Remove one or more whole-document ``line_number<TAB>`` layers."""
    if not isinstance(value, str) or not value:
        return value
    cleaned_value = value
    while True:
        lines = cleaned_value.splitlines()
        populated = [(index, line) for index, line in enumerate(lines) if line.strip()]
        if len(populated) < 3:
            return cleaned_value
        matches = [_LINE_NUMBER_PREFIX_RE.match(line) for _, line in populated]
        if any(match is None for match in matches):
            return cleaned_value
        numbers = [int(match.group(1)) for match in matches if match is not None]
        # Deliberately uneven: pairs each number with its successor, so the last
        # number has no partner and strict=True would raise.
        if any(
            current != previous + 1
            for previous, current in zip(numbers, numbers[1:], strict=False)
        ):
            return cleaned_value
        cleaned = list(lines)
        for (index, _), match in zip(populated, matches, strict=True):
            cleaned[index] = match.group(2)
        next_value = "\n".join(cleaned)
        if next_value == cleaned_value:
            return cleaned_value
        cleaned_value = next_value


_EXTRACTION_CHUNK_BOUNDARY_RE = re.compile(r"(\n+|[。！？；!?;]+|(?<!\d)\.(?!\d))")
_RESOURCE_ADDITION_FIELD_RE = re.compile(
    r"^(Resource URI|Source name|Added at|Resource abstract|User reason):\s*(.*)$",
    re.MULTILINE,
)
_RESOURCE_URI_MARKER_RE = re.compile(
    r"[，,；;：:\s]*(?:资源\s*URI\s*为|资源\s*URI|Resource\s+URI)\s*[:：为]?\s*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ChunkMeta:
    """Metadata for a derived extraction chunk message."""

    source_message_id: str
    chunk_index: int
    chunk_count: int


class _LinkPublicationError(RuntimeError):
    """A link batch could not be published as one consistent graph update."""

    def __init__(
        self,
        failed_uri: str,
        phase: str,
        cause: Exception,
        rollback_failures: Optional[List[Tuple[str, Exception]]] = None,
    ) -> None:
        self.failed_uri = failed_uri
        self.phase = phase
        self.rollback_failures = list(rollback_failures or [])
        message = f"Failed to {phase} link endpoint {failed_uri}: {cause}"
        if phase == "write":
            if self.rollback_failures:
                rollback_uris = ", ".join(uri for uri, _ in self.rollback_failures)
                message += f"; rollback also failed for: {rollback_uris}"
            else:
                message += "; attempted endpoint writes were rolled back"
        super().__init__(message)


@dataclass(frozen=True)
class _FileSnapshot:
    """Exact pre-publication state for one canonical filesystem object."""

    existed: bool
    content: Any = None


async def write_stored_links(
    links: List[StoredLink],
    ctx: RequestContext,
    viking_fs: Any,
    skip_uris: Optional[set] = None,
    preserve_version_uris: Optional[set] = None,
    lock_handle: Any = None,
) -> List[str]:
    """Write StoredLinks to their endpoint files' links/backlinks fields.

    For each link: from_uri's ``links`` receives the forward link;
    to_uri's ``backlinks`` receives the reverse reference.
    Files listed in skip_uris are skipped (caller handles them in the same write).
    When lock_handle is provided, all endpoint rewrites reuse that transaction.
    All endpoints are read and rendered before the first write. If a later
    endpoint write fails, every attempted endpoint is restored from its
    pre-publication content using the same lock handle. This compensation is
    best-effort rather than a true cross-file transaction: a rollback write can
    itself fail, and callers without a lock remain exposed to concurrent writes.

    Returns all endpoint URIs only after the full batch succeeds. Raises
    ``_LinkPublicationError`` on preflight or publication failure.
    """
    from openviking.session.memory.merge_op.link_merge import merge_links

    skip = skip_uris or set()
    preserve_versions = preserve_version_uris or set()
    lock_kwargs = {"lock_handle": lock_handle} if lock_handle is not None else {}
    endpoint_content: Dict[str, Any] = {}
    experience_endpoints = list(
        dict.fromkeys(
            uri
            for link in links
            for uri in (link.from_uri, link.to_uri)
            if uri_targets_experience_store(uri)
        )
    )
    for uri in experience_endpoints:
        try:
            canonical_uri = canonicalize_uri(uri, ctx)
            expected_root = f"{canonical_user_root(ctx)}/memories/experiences"
            if not canonical_uri.startswith(f"{expected_root}/"):
                raise ValueError(
                    "Experience link endpoint must belong to the authenticated user's "
                    "canonical experience store"
                )
            content = await viking_fs.read_file(uri, ctx=ctx)
            if not content:
                raise ValueError("experience endpoint content is empty")
            memory_file = MemoryFileUtils.read(content, uri=uri)
            stored_errors = validate_stored_experience(memory_file, uri)
            if stored_errors:
                raise ValueError(
                    "Experience link endpoint is legacy-invalid and read-only: "
                    + repr([error["code"] for error in stored_errors])
                )
            endpoint_content[uri] = content
        except Exception as e:
            raise _LinkPublicationError(uri, "prepare", e) from e

    file_links: Dict[str, Dict[str, List[StoredLink]]] = {}
    for link in links:
        if link.from_uri not in skip:
            file_links.setdefault(link.from_uri, {"links": [], "backlinks": []})
            file_links[link.from_uri]["links"].append(link)
        if link.to_uri not in skip:
            file_links.setdefault(link.to_uri, {"links": [], "backlinks": []})
            file_links[link.to_uri]["backlinks"].append(link)

    prepared_writes: List[Tuple[str, Any, str]] = []
    for uri, link_groups in file_links.items():
        try:
            content = endpoint_content.get(uri)
            if content is None:
                content = await viking_fs.read_file(uri, ctx=ctx)
            if not content:
                raise ValueError("endpoint content is empty")
            mf = MemoryFileUtils.read(content, uri=uri)
            if link_groups["links"]:
                mf.links = merge_links(mf.links, [l.model_dump() for l in link_groups["links"]])
            if link_groups["backlinks"]:
                mf.backlinks = merge_links(
                    mf.backlinks, [l.model_dump() for l in link_groups["backlinks"]]
                )
            current_trace_id = get_trace_id()
            if current_trace_id:
                mf.extra_fields["last_update_trace_id"] = current_trace_id
            if uri not in preserve_versions:
                bump_memory_version(mf)
            prepared_writes.append((uri, content, MemoryFileUtils.write(mf)))
        except Exception as e:
            raise _LinkPublicationError(uri, "prepare", e) from e

    attempted_writes: List[Tuple[str, Any]] = []
    for uri, original_content, updated_content in prepared_writes:
        attempted_writes.append((uri, original_content))
        try:
            await viking_fs.write_file(
                uri,
                updated_content,
                ctx=ctx,
                **lock_kwargs,
            )
            readback = await viking_fs.read_file(uri, ctx=ctx)
            if readback != updated_content:
                raise RuntimeError("endpoint readback did not match the published content")
        except Exception as e:
            rollback_failures: List[Tuple[str, Exception]] = []
            for rollback_uri, rollback_content in reversed(attempted_writes):
                try:
                    await viking_fs.write_file(
                        rollback_uri,
                        rollback_content,
                        ctx=ctx,
                        **lock_kwargs,
                    )
                    rollback_readback = await viking_fs.read_file(rollback_uri, ctx=ctx)
                    if rollback_readback != rollback_content:
                        raise RuntimeError(
                            "rollback readback did not match the pre-publication content"
                        )
                except Exception as rollback_error:
                    rollback_failures.append((rollback_uri, rollback_error))
                    tracer.error(
                        f"Failed to roll back link publication for {rollback_uri}: "
                        f"{rollback_error}"
                    )
            raise _LinkPublicationError(uri, "write", e, rollback_failures) from e
    return [uri for uri, _, _ in prepared_writes]



def _remap_link_dict(link: Dict[str, Any], uri_remap: Dict[str, str]) -> Dict[str, Any]:
    remapped = dict(link or {})
    if remapped.get("from_uri") in uri_remap:
        remapped["from_uri"] = uri_remap[remapped["from_uri"]]
    if remapped.get("to_uri") in uri_remap:
        remapped["to_uri"] = uri_remap[remapped["to_uri"]]
    return remapped


def remap_stored_links(links: List[StoredLink], uri_remap: Dict[str, str]) -> List[StoredLink]:
    if not links or not uri_remap:
        return list(links or [])
    remapped_links: List[StoredLink] = []
    for link in links:
        from_uri = uri_remap.get(link.from_uri, link.from_uri)
        to_uri = uri_remap.get(link.to_uri, link.to_uri)
        if from_uri == to_uri:
            continue
        remapped_links.append(link.model_copy(update={"from_uri": from_uri, "to_uri": to_uri}))
    return remapped_links

def _operation_trace_id(op: ResolvedOperation) -> str | None:
    source = getattr(op, "source", None)
    trace_id = getattr(source, "trace_id", None) if source else None
    if trace_id:
        return str(trace_id)
    fields = dict(getattr(op, "memory_fields", {}) or {})
    field_value = fields.get("last_update_trace_id") or fields.get("trace_id")
    if field_value:
        return str(field_value)
    current_trace_id = get_trace_id()
    return current_trace_id or None


class ExtractContext:
    """Extract context for template rendering."""

    def __init__(
        self,
        messages: List[Message],
        chunk_meta: Optional[Dict[int, ChunkMeta]] = None,
        *,
        split_long_text_messages: bool = True,
    ):
        if chunk_meta is None:
            if split_long_text_messages:
                self.messages, self.chunk_meta = self._build_extraction_messages(messages)
            else:
                self.messages, self.chunk_meta = list(messages or []), {}
        else:
            self.messages = messages
            self.chunk_meta = chunk_meta
        self.page_id_map = PageIdMap()

    @classmethod
    def _build_extraction_messages(
        cls, messages: List[Message]
    ) -> Tuple[List[Message], Dict[int, ChunkMeta]]:
        """Build messages used by memory extraction.

        Long text-only messages are split into derived chunks so event `ranges`
        can point to a narrower source span without relying on brittle text
        matching. The original session messages are not modified.
        """
        extraction_messages: List[Message] = []
        chunk_meta: Dict[int, ChunkMeta] = {}
        for message in messages:
            for extraction_message, meta in cls._split_message_for_extraction(message):
                extraction_messages.append(extraction_message)
                if meta is not None:
                    chunk_meta[id(extraction_message)] = meta
        return extraction_messages, chunk_meta

    @classmethod
    def _split_message_for_extraction(
        cls, message: Message
    ) -> List[Tuple[Message, Optional[ChunkMeta]]]:
        parts = getattr(message, "parts", [])
        if not parts or not all(isinstance(part, TextPart) for part in parts):
            return [(message, None)]

        text = "".join(part.text for part in parts)
        chunks = cls._split_text_for_extraction(text)
        if len(chunks) <= 1:
            return [(message, None)]

        chunk_messages = []
        for idx, chunk in enumerate(chunks):
            chunk_message = Message(
                id=f"{message.id}#chunk_{idx}",
                role=message.role,
                peer_id=getattr(message, "peer_id", None),
                parts=[TextPart(chunk)],
                created_at=message.created_at,
            )
            chunk_messages.append(
                (
                    chunk_message,
                    ChunkMeta(
                        source_message_id=message.id,
                        chunk_index=idx,
                        chunk_count=len(chunks),
                    ),
                )
            )
        return chunk_messages

    @classmethod
    def _split_text_for_extraction(cls, text: str) -> List[str]:
        return cls._pack_text_units(cls._split_text_units(text)) or [text]

    @staticmethod
    def _pack_text_units(units: List[str]) -> List[str]:
        chunks: List[str] = []
        current = ""
        for unit in units:
            current += unit
            if len(current) < _EXTRACTION_CHUNK_MIN_CHARS:
                continue
            chunks.append(current)
            current = ""

        if current:
            if chunks:
                chunks[-1] += current
            else:
                chunks.append(current)
        return chunks

    @staticmethod
    def _split_text_units(text: str) -> List[str]:
        pieces = _EXTRACTION_CHUNK_BOUNDARY_RE.split(text)
        units: List[str] = []
        current = ""
        for piece in pieces:
            if not piece:
                continue
            current += piece
            if _EXTRACTION_CHUNK_BOUNDARY_RE.fullmatch(piece):
                units.append(current)
                current = ""
        if current:
            units.append(current)
        return units or [text]

    def get_first_message_time_from_ranges(self, ranges_str: str) -> str | None:
        """根据 ranges 字符串获取第一条消息的时间（YAML 日期格式）"""
        if not ranges_str:
            return None
        msg_range = self.read_message_ranges(ranges_str)
        return msg_range._first_message_time()

    def get_first_message_time_with_weekday_from_ranges(self, ranges_str: str) -> str | None:
        """根据 ranges 字符串获取第一条消息的时间，带周几"""
        if not ranges_str:
            return None
        msg_range = self.read_message_ranges(ranges_str)
        return msg_range._first_message_time_with_weekday()

    def get_year(self, ranges_str: str) -> str:
        """根据 ranges 字符串获取第一条消息的年份，fallback 到当前年份"""
        from datetime import datetime
        if not ranges_str:
            return str(datetime.now().year)
        msg_range = self.read_message_ranges(ranges_str)
        first_time = msg_range._first_message_time()
        if first_time:
            return first_time.split("-")[0]
        return str(datetime.now().year)

    def get_month(self, ranges_str: str) -> str:
        """根据 ranges 字符串获取第一条消息的月份，fallback 到当前月份"""
        from datetime import datetime
        if not ranges_str:
            return f"{datetime.now().month:02d}"
        msg_range = self.read_message_ranges(ranges_str)
        first_time = msg_range._first_message_time()
        if first_time:
            return first_time.split("-")[1]
        return f"{datetime.now().month:02d}"

    def get_day(self, ranges_str: str) -> str:
        """根据 ranges 字符串获取第一条消息的日期，fallback 到当前日期"""
        from datetime import datetime
        if not ranges_str:
            return f"{datetime.now().day:02d}"
        msg_range = self.read_message_ranges(ranges_str)
        first_time = msg_range._first_message_time()
        if first_time:
            return first_time.split("-")[2]
        return f"{datetime.now().day:02d}"

    def get_timestamp_from_ranges(self, ranges_str: str) -> str:
        """根据 ranges 获取第一条消息的紧凑时间戳（YYYYMMDDHHMMSS），用于文件名去重。

        Fallback 到 datetime.now() 以保证总是返回非空字符串。
        """
        from datetime import datetime

        msg_range = self.read_message_ranges(ranges_str) if ranges_str else None
        if msg_range:
            for elem in msg_range.elements:
                if isinstance(elem, str):
                    continue
                created_at = getattr(elem, "created_at", None)
                if created_at:
                    try:
                        return datetime.fromisoformat(created_at).strftime("%Y%m%d%H%M%S")
                    except (ValueError, TypeError):
                        continue
        return datetime.now().strftime("%Y%m%d%H%M%S")

    def get_session_timestamp(self) -> str:
        """取对话第一条消息的时间戳（YYYYMMDDHHMMSS），用于文件名唯一化。

        Fallback 到 datetime.now() 以保证总是返回非空字符串。
        """
        from datetime import datetime

        for msg in self.messages:
            created_at = getattr(msg, "created_at", None)
            if created_at:
                try:
                    return datetime.fromisoformat(created_at).strftime("%Y%m%d%H%M%S")
                except (ValueError, TypeError):
                    continue
        return datetime.now().strftime("%Y%m%d%H%M%S")

    def get_event_content(
        self, ranges_str: str, summary: str | None, ratio_threshold: float = 0.2
    ) -> str:
        """根据原始消息与 summary 的字符数比例，决定返回原始消息还是摘要。"""
        if not ranges_str:
            return summary or ""
        msg_range = self.read_message_ranges(ranges_str)
        original = msg_range.pretty_print()
        if not summary or not summary.strip():
            return original or ""
        if not original:
            return summary
        if len(summary) / len(original) >= ratio_threshold:
            return original
        return summary

    def get_resource_event_content(self, ranges_str: str, summary: str) -> str:
        """Return a user-readable event body for add-resource derived events."""
        if not ranges_str:
            return ""
        additions = self._resource_additions_from_ranges(ranges_str)
        if not additions:
            return ""
        addition = additions[0]
        resource_uri = addition.get("Resource URI", "")
        if not resource_uri:
            return ""
        return self._link_resource_summary(summary or "", resource_uri, addition).strip()

    def _resource_additions_from_ranges(self, ranges_str: str) -> List[Dict[str, str]]:
        msg_range = self.read_message_ranges(ranges_str)
        additions: List[Dict[str, str]] = []
        for msg_group in msg_range.elements:
            for msg in msg_group:
                text = self._message_text(msg)
                if "## Resource Addition" not in text:
                    continue
                fields = {
                    match.group(1): match.group(2).strip()
                    for match in _RESOURCE_ADDITION_FIELD_RE.finditer(text)
                }
                if fields.get("Resource URI"):
                    additions.append(fields)
        return additions

    @staticmethod
    def _message_text(message: Message) -> str:
        parts = getattr(message, "parts", [])
        texts = [part.text for part in parts if isinstance(part, TextPart) and part.text]
        if texts:
            return "\n".join(texts)
        return message.content or ""

    @classmethod
    def _link_resource_summary(
        cls,
        summary: str,
        resource_uri: str,
        addition: Dict[str, str],
    ) -> str:
        text = (summary or "").strip()
        if not text:
            return cls._resource_addition_fallback_sentence(resource_uri, addition)
        if f"]({resource_uri})" in text:
            return text
        if resource_uri in text:
            return cls._replace_bare_resource_uri(text, resource_uri, addition)
        label = cls._resource_label_from_addition(addition)
        return cls._finish_sentence(f"{text.rstrip('。.!')}，关联资源为[{label}]({resource_uri})")

    @classmethod
    def _replace_bare_resource_uri(
        cls,
        text: str,
        resource_uri: str,
        addition: Dict[str, str],
    ) -> str:
        uri_start = text.find(resource_uri)
        if uri_start < 0:
            return text
        prefix = text[:uri_start]
        suffix = text[uri_start + len(resource_uri) :]
        marker = _RESOURCE_URI_MARKER_RE.search(prefix)
        if marker:
            visible_prefix = prefix[: marker.start()].rstrip("，,；;：: ")
            label = cls._resource_clause_from_summary_prefix(visible_prefix)
            if not label:
                label = cls._resource_label_from_addition(addition)
            if label and visible_prefix.endswith(label):
                visible_prefix = visible_prefix[: -len(label)] + f"[{label}]({resource_uri})"
            else:
                visible_prefix = f"{visible_prefix}[{label}]({resource_uri})"
            return cls._finish_sentence(visible_prefix)

        label = cls._resource_label_from_addition(addition)
        return cls._finish_sentence(f"{prefix.rstrip()}[{label}]({resource_uri}){suffix.strip()}")

    @staticmethod
    def _resource_clause_from_summary_prefix(prefix: str) -> str:
        text = prefix.strip("，,；;：: ")
        tail = re.split(r"[，,；;。.!?？]", text)[-1].strip()
        return tail if 0 < len(tail) <= 120 else ""

    @classmethod
    def _resource_label_from_addition(cls, addition: Dict[str, str]) -> str:
        reason = addition.get("User reason", "").strip()
        for prefix in ("这是一张", "这是一个", "该资源是", "这个是", "这是"):
            if reason.startswith(prefix):
                reason = reason[len(prefix) :].strip()
                break
        reason = reason.strip("。.!！ ")
        if reason:
            return reason[:80]
        source_name = addition.get("Source name", "").strip()
        return source_name or "相关资源"

    @classmethod
    def _resource_addition_fallback_sentence(
        cls,
        resource_uri: str,
        addition: Dict[str, str],
    ) -> str:
        label = cls._resource_label_from_addition(addition)
        return f"用户保存了[{label}]({resource_uri})。"

    @staticmethod
    def _finish_sentence(text: str) -> str:
        text = text.strip("，,；;：: ")
        if text.endswith(("。", ".", "！", "!", "？", "?")):
            return text
        return text + "。"

    def read_message_ranges(self, ranges_str: str) -> "MessageRange":
        """Parse ranges string like "0-10,50-60" or "7,9,11,13" and return combined MessageRange.

        If there's a gap between ranges (e.g., 0-10 and 50-60), add "..." as separator.
        Supports:
        - "0-10,50-60" - ranges
        - "7,9,11,13" - single indices
        - "0-10,15,20-25" - mixed
        """
        if not ranges_str:
            return MessageRange([])

        # 解析所有范围/索引
        ranges = []
        for part in ranges_str.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = part.split("-")
                ranges.append((int(start), int(end)))
            else:
                # 单个索引转为相同起止范围
                idx = int(part)
                ranges.append((idx, idx))

        if not ranges:
            return MessageRange([])

        # 按 start 排序
        ranges.sort(key=lambda x: x[0])

        # 合并连续/重叠的范围
        merged = [ranges[0]]
        for start, end in ranges[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end + 1:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

        # elements 是 List[List[Message]] - 每段连续消息是一个列表
        elements: List[List[Message]] = []
        for start, end in merged:
            # 兼容 LLM 提取的 range 越界情况
            if start < 0:
                start = 0
            if end >= len(self.messages):
                end = len(self.messages) - 1
            if start > end:
                continue
            range_msgs = self.messages[start : end + 1]
            elements.append(range_msgs)

        return MessageRange(elements, chunk_meta=self.chunk_meta)


class MessageRange:
    """Represents a range of messages for formatting."""

    def __init__(
        self,
        elements: List[List[Message]],
        chunk_meta: Optional[Dict[int, ChunkMeta]] = None,
    ):
        self.elements = elements
        self.chunk_meta = chunk_meta or {}

    def pretty_print(self) -> str:
        """Pretty print the message range with '...' separator between non-contiguous ranges."""
        result = []
        for i, msg_group in enumerate(self.elements):
            result.extend(self._format_contiguous_group(msg_group))
            if i < len(self.elements) - 1:
                result.append("...")
        return "\n".join(result)

    def _format_contiguous_group(self, msg_group: List[Message]) -> List[str]:
        formatted = []
        current_messages: List[Message] = []

        def flush_current() -> None:
            nonlocal current_messages
            if not current_messages:
                return
            content = self._format_merged_content(current_messages)
            if content.strip():
                formatted.append(f"**{self._speaker_for(current_messages[0])}**: {content}")
            current_messages = []

        for msg in msg_group:
            if current_messages and not self._can_merge_messages(current_messages[-1], msg):
                flush_current()
            current_messages.append(msg)

        flush_current()
        return formatted

    @staticmethod
    def _speaker_for(message: Message) -> str:
        return getattr(message, "peer_id", None) or message.role

    def _can_merge_messages(self, previous: Message, current: Message) -> bool:
        previous_meta = self._chunk_meta_for(previous)
        current_meta = self._chunk_meta_for(current)
        if previous_meta is None or current_meta is None:
            return False
        if self._speaker_for(previous) != self._speaker_for(current):
            return False
        return (
            previous_meta.source_message_id == current_meta.source_message_id
            and current_meta.chunk_index == previous_meta.chunk_index + 1
        )

    def _format_merged_content(self, messages: List[Message]) -> str:
        content = "".join((self._message_content(msg) or "") for msg in messages)
        if not messages or not self._contains_chunk_message(messages):
            return content

        first_chunk = self._chunk_meta_for(messages[0])
        if first_chunk is not None and first_chunk.chunk_index > 0:
            content = "..." + content.lstrip()
        last_chunk = self._chunk_meta_for(messages[-1])
        if last_chunk is not None and last_chunk.chunk_index < last_chunk.chunk_count - 1:
            content = content.rstrip() + "..."
        return content

    def _message_content(self, message: Message) -> str:
        texts: List[str] = []
        for part in getattr(message, "parts", []) or []:
            if isinstance(part, TextPart):
                texts.append(part.text or "")
        if texts:
            return "".join(texts)
        return getattr(message, "content", "") or ""

    def _contains_chunk_message(self, messages: List[Message]) -> bool:
        return any(self._chunk_meta_for(msg) is not None for msg in messages)

    def _chunk_meta_for(self, message: Message) -> Optional[ChunkMeta]:
        return self.chunk_meta.get(id(message))

    def _first_message_time(self) -> str | None:
        """获取第一条消息的时间（内部方法）"""
        for msg_group in self.elements:
            for msg in msg_group:
                if hasattr(msg, "created_at") and msg.created_at:
                    dt = parse_iso_datetime(msg.created_at)
                    return dt.strftime("%Y-%m-%d")
        return None

    def _first_message_time_with_weekday(self) -> str | None:
        """获取第一条消息的时间，带周几"""
        weekday_en = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        for msg_group in self.elements:
            for msg in msg_group:
                if hasattr(msg, "created_at") and msg.created_at:
                    dt = parse_iso_datetime(msg.created_at)
                    weekday = weekday_en[dt.weekday()]
                    return f"{dt.strftime('%Y-%m-%d')} ({weekday})"
        return None


class MemoryUpdateResult:
    """Result of memory update operation."""

    def __init__(self):
        self.written_uris: List[str] = []
        self.edited_uris: List[str] = []
        self.deleted_uris: List[str] = []
        self.index_pending_uris: List[str] = []
        self.errors: List[Tuple[str, Exception]] = []
        # Declared URI -> the already-stored URI that made writing it redundant.
        # Deliberately not folded into written_uris: nothing was written, and a
        # caller asking "what did this produce" must not be told otherwise. But
        # a suppressed duplicate is a satisfied request, not a failed one, so
        # callers that gate on "did this URI land" consult this too.
        self.deduplicated_uris: Dict[str, str] = {}

    def add_written(self, uri: str) -> None:
        self.written_uris.append(uri)

    def add_deduplicated(self, uri: str, existing_uri: str) -> None:
        self.deduplicated_uris[uri] = existing_uri

    def satisfied_uris(self) -> set:
        """URIs whose content is present in the store after this update.

        Written, edited, and deduplicated alike — a duplicate that was skipped
        is still backed by a real file, so anything checking for a resolvable
        endpoint should treat it as present.
        """
        return set(self.written_uris) | set(self.edited_uris) | set(self.deduplicated_uris)

    def add_edited(self, uri: str) -> None:
        self.edited_uris.append(uri)

    def add_deleted(self, uri: str) -> None:
        self.deleted_uris.append(uri)

    def add_index_pending(self, uri: str) -> None:
        if uri not in self.index_pending_uris:
            self.index_pending_uris.append(uri)

    def add_error(self, uri: str, error: Exception) -> None:
        self.errors.append((uri, error))

    def summary(self) -> str:
        return (
            f"Written: {len(self.written_uris)}, "
            f"Edited: {len(self.edited_uris)}, "
            f"Deleted: {len(self.deleted_uris)}, "
            f"Deduplicated: {len(self.deduplicated_uris)}, "
            f"Index pending: {len(self.index_pending_uris)}, "
            f"Errors: {len(self.errors)}"
        )


def _same_batch_delete_conflict_key(uri: str) -> str:
    """Return a conservative key for detecting same-batch upsert/delete URI conflicts.

    Some local filesystems are case-insensitive.  Treat case-only URI variants as
    conflicting inside one apply batch so a loser delete cannot remove a winner
    upsert before vectorization.
    """

    return str(uri or "").rstrip("/").casefold()


class MemoryUpdater:
    """
    Applies MemoryOperations to storage.

    This is the system executor that directly applies the LLM's final output.
    No function calls are used for write/edit/delete - these are executed directly.
    """

    def __init__(
        self, registry: Optional[MemoryTypeRegistry] = None, vikingdb=None, transaction_handle=None
    ):
        self._viking_fs = None
        self._registry = registry
        self._vikingdb = vikingdb
        self._transaction_handle = transaction_handle

    def _get_viking_fs(self):
        """Get or create VikingFS instance."""
        if self._viking_fs is None:
            self._viking_fs = get_viking_fs()
        return self._viking_fs

    @classmethod
    async def refresh_schema_overview(
        cls,
        *,
        viking_fs: Any,
        directory_uri: str,
        ctx: RequestContext,
    ) -> None:
        memory_type = cls.memory_type_from_uri(directory_uri)
        if not memory_type:
            return
        try:
            from openviking.session.memory.memory_type_registry import create_default_registry

            updater = cls(registry=create_default_registry())
            updater._viking_fs = viking_fs
            await updater.generate_overview(memory_type, directory_uri, ctx)
        except Exception:
            logger.warning(
                "Failed to refresh memory overview for %s",
                directory_uri,
                exc_info=True,
            )

    @classmethod
    async def refresh_file_embedding(
        cls,
        *,
        viking_fs: Any,
        vikingdb: Any,
        uri: str,
        memory_type: Optional[str],
        ctx: RequestContext,
    ) -> bool:
        if not vikingdb or not bool(getattr(vikingdb, "has_queue_manager", False)):
            return False
        try:
            from openviking.session.memory.memory_type_registry import create_default_registry

            result = MemoryUpdateResult()
            result.add_written(uri)
            updater = cls(registry=create_default_registry(), vikingdb=vikingdb)
            updater._viking_fs = viking_fs
            attempted = await updater._vectorize_memories(
                result,
                ctx,
                uri_memory_type_map={uri: memory_type} if memory_type else {},
            )
            return attempted > 0
        except Exception:
            logger.warning("Failed to refresh memory embedding for %s", uri, exc_info=True)
            return False

    @staticmethod
    def memory_type_from_uri(uri: str) -> Optional[str]:
        parts = [part for part in VikingURI(uri).full_path.split("/") if part]
        try:
            memories_idx = parts.index("memories")
        except ValueError:
            return None
        if len(parts) <= memories_idx + 1:
            return None
        return parts[memories_idx + 1]

    @tracer()
    async def apply_operations(
        self,
        operations: ResolvedOperations,
        ctx: RequestContext,
        extract_context: ExtractContext = None,
        isolation_handler: MemoryIsolationHandler = None,
    ) -> MemoryUpdateResult:
        result = MemoryUpdateResult()
        viking_fs = self._get_viking_fs()

        if not viking_fs:
            tracer.error("VikingFS not available, skipping memory operations")
            return result

        # Use provided registry or fall back to self._registry

        if not self._registry:
            raise ValueError("MemoryTypeRegistry is required for URI resolution")

        # Resolve all URIs first (pass extract_context for template rendering)
        tracer.info(f"[MemoryUpdater] applying operations, isolation_handler={isolation_handler}")

        if operations.has_errors():
            for error in operations.errors:
                result.add_error("unknown", ValueError(error))
            return result

        applicable_upserts: List[ResolvedOperation] = []
        has_unresolved_upserts = False
        for resolved_op in operations.upsert_operations:
            if resolved_op.uris:
                applicable_upserts.append(resolved_op)
                continue
            has_unresolved_upserts = True
            error_target = f"{resolved_op.memory_type}(page_id={resolved_op.page_id})"
            resolution_error = ValueError("Missing resolved URI")
            result.add_error(error_target, resolution_error)
            tracer.error(
                f"Skipping unresolved memory operation: {error_target}: {resolution_error}"
            )

        # Automatic extraction and training are fail-closed for experience admission. Validate even
        # unresolved operations, then return before links, writes, deletes, vectors, or overviews so a
        # malformed or unresolved split cannot partially publish. Privileged direct-storage APIs remain
        # available for separately verified legacy migration and administrative repair.
        experience_errors = validate_experience_operations(
            operations.upsert_operations,
            validate_existing=False,
        )
        experience_errors.extend(
            validate_experience_operation_context(operations.upsert_operations, ctx)
        )
        unresolved_experiences = [
            op
            for op in operations.upsert_operations
            if op.memory_type == "experiences" and not op.uris
        ]
        experience_delete_errors: list[tuple[str, str]] = []
        for file_content in operations.delete_file_contents:
            delete_uri = str(file_content.uri or "")
            declared_type = str(file_content.memory_type or "")
            if declared_type == "experiences" or uri_targets_experience_store(delete_uri):
                experience_delete_errors.append(
                    (
                        delete_uri or "experiences",
                        "Automatic experience deletion is disabled; retain the source for a "
                        "separately verified administrative migration.",
                    )
                )
        for deleted_uri, replacement_uri in dict(
            getattr(operations, "delete_replacements", {}) or {}
        ).items():
            if uri_targets_experience_store(deleted_uri) or uri_targets_experience_store(
                replacement_uri
            ):
                experience_delete_errors.append(
                    (
                        str(deleted_uri),
                        "Automatic experience replacement mappings are disabled; use a separately "
                        "verified administrative migration.",
                    )
                )

        if experience_errors or unresolved_experiences or experience_delete_errors:
            for error in experience_errors:
                target = (error.get("uris") or ["experiences"])[0]
                result.add_error(target, ValueError(error["message"]))
            for operation in unresolved_experiences:
                target = f"experiences(page_id={operation.page_id})"
                if not any(error_target == target for error_target, _ in result.errors):
                    result.add_error(target, ValueError("Missing resolved experience URI"))
            for target, message in experience_delete_errors:
                result.add_error(target, ValueError(message))
            tracer.error(f"Rejected experience operations before apply: {experience_errors}")
            return result

        experience_read_errors: list[tuple[str, str]] = []
        experience_snapshots: Dict[str, _FileSnapshot] = {}
        for operation in operations.upsert_operations:
            if operation.memory_type != "experiences":
                continue
            uri = operation.uris[0]
            supplied_old = operation.old_memory_file_content
            try:
                raw_current = await viking_fs.read_file(uri, ctx=ctx)
            except NotFoundError:
                experience_snapshots[uri] = _FileSnapshot(existed=False)
                if supplied_old is not None:
                    experience_read_errors.append(
                        (uri, "Existing experience disappeared during admission preflight.")
                    )
                continue
            except Exception as exc:
                experience_read_errors.append(
                    (uri, f"Could not verify existing experience before update: {exc}")
                )
                continue
            experience_snapshots[uri] = _FileSnapshot(existed=True, content=raw_current)
            try:
                current_file = MemoryFileUtils.read(raw_current, uri=uri)
                operation.old_memory_file_content = current_file
            except Exception as exc:
                experience_read_errors.append(
                    (uri, f"Could not parse existing experience before update: {exc}")
                )
                continue
            stored_errors = validate_stored_experience(current_file, uri)
            if stored_errors:
                experience_read_errors.append(
                    (
                        uri,
                        "Existing experience is legacy-invalid and read-only: "
                        + repr([error["code"] for error in stored_errors]),
                    )
                )

        experience_errors = validate_experience_operations(operations.upsert_operations)
        if experience_read_errors or experience_errors:
            for target, message in experience_read_errors:
                result.add_error(target, ValueError(message))
            for error in experience_errors:
                target = (error.get("uris") or ["experiences"])[0]
                result.add_error(target, ValueError(error["message"]))
            tracer.error(
                "Rejected experience update after authoritative legacy readback: "
                f"{experience_errors}"
            )
            return result

        # Drop near-duplicate add_only operations (same batch or re-extraction
        # of the same source session) before applying anything.
        applicable_upserts = await self._drop_duplicate_add_only(applicable_upserts, ctx, result)

        # Apply unified operations - _apply_edit returns True if edited, False if written
        attempted_experience_uris: List[str] = []
        experience_apply_failed = False
        for resolved_op in applicable_upserts:
            if resolved_op.memory_type == "experiences":
                attempted_experience_uris.extend(resolved_op.uris)
            try:
                await self._apply_upsert(
                    resolved_op,
                    ctx,
                    extract_context=extract_context,
                )
                # Add all uris to result (uris is List[str])
                if resolved_op.is_edit():
                    for uri in resolved_op.uris:
                        result.add_edited(uri)
                else:
                    for uri in resolved_op.uris:
                        result.add_written(uri)
            except Exception as e:
                if resolved_op.memory_type == "experiences":
                    experience_apply_failed = True
                tracer.error(
                    f"Failed to apply operation: op_type={type(resolved_op).__name__}, uris={resolved_op.uris}",
                    e,
                )
                for uri in resolved_op.uris:
                    result.add_error(uri, e)
                if resolved_op.memory_type == "experiences":
                    break

        if experience_apply_failed:
            await self._rollback_experience_files(
                experience_snapshots,
                attempted_experience_uris,
                result,
                ctx,
                reason="one or more experience content writes failed",
            )
            return result

        declared_upsert_uris = {
            uri for operation in operations.upsert_operations for uri in operation.uris
        }
        # Includes URIs suppressed as duplicates: their content is in the store
        # under an equivalent URI, so treating them as unlanded would refuse
        # their links and, when the counterpart is an experience, roll back the
        # whole experience batch over a write that was correctly skipped.
        successful_upsert_uris = result.satisfied_uris()
        failed_replacement_deletes: set[str] = set()
        safe_delete_replacements: dict[str, str] = {}
        for deleted_uri, replacement_uri in dict(
            getattr(operations, "delete_replacements", {}) or {}
        ).items():
            if (
                replacement_uri in declared_upsert_uris
                and replacement_uri not in successful_upsert_uris
            ):
                failed_replacement_deletes.add(deleted_uri)
                result.add_error(
                    deleted_uri,
                    ValueError(f"Skipped delete because replacement write failed: {replacement_uri}"),
                )
                continue
            safe_delete_replacements[deleted_uri] = replacement_uri
        operations.delete_replacements = safe_delete_replacements

        original_resolved_links = list(getattr(operations, "resolved_links", []) or [])
        remapped_resolved_links = remap_stored_links(
            original_resolved_links,
            safe_delete_replacements,
        )
        attempted_experience_set = set(attempted_experience_uris)
        admitted_resolved_links: List[StoredLink] = []
        dropped_required_experience_links: List[StoredLink] = []
        for link in remapped_resolved_links:
            endpoints_published = all(
                endpoint not in declared_upsert_uris or endpoint in successful_upsert_uris
                for endpoint in (link.from_uri, link.to_uri)
            )
            if endpoints_published:
                admitted_resolved_links.append(link)
            elif any(
                endpoint in attempted_experience_set
                for endpoint in (link.from_uri, link.to_uri)
            ):
                dropped_required_experience_links.append(link)
        operations.resolved_links = admitted_resolved_links
        if dropped_required_experience_links:
            for link in dropped_required_experience_links:
                experience_uri = next(
                    endpoint
                    for endpoint in (link.from_uri, link.to_uri)
                    if endpoint in attempted_experience_set
                )
                result.add_error(
                    experience_uri,
                    ValueError(
                        "Required experience relation endpoint did not publish: "
                        f"{link.from_uri} -> {link.to_uri}"
                    ),
                )
            await self._rollback_experience_files(
                experience_snapshots,
                attempted_experience_uris,
                result,
                ctx,
                reason="required experience relation endpoint did not publish",
            )
            return result
        if safe_delete_replacements:
            inheritance_succeeded = await self._inherit_deleted_link_relations(
                operations, result, ctx
            )
            if not inheritance_succeeded:
                for deleted_uri in safe_delete_replacements:
                    failed_replacement_deletes.add(deleted_uri)
                    result.add_error(
                        deleted_uri,
                        ValueError(
                            "Skipped delete because replacement link inheritance failed"
                        ),
                    )
                operations.delete_replacements = {}
                operations.resolved_links = [
                    link
                    for link in original_resolved_links
                    if link.from_uri not in safe_delete_replacements
                    and link.to_uri not in safe_delete_replacements
                ]

        def relation_identity(link: StoredLink) -> tuple[str, str, str, Optional[str]]:
            return (link.from_uri, link.to_uri, link.link_type, link.match_text)

        required_experience_relations = {
            relation_identity(link)
            for link in remapped_resolved_links
            if any(
                endpoint in attempted_experience_set
                for endpoint in (link.from_uri, link.to_uri)
            )
        }
        admitted_relation_identities = {
            relation_identity(link) for link in operations.resolved_links
        }
        missing_experience_relations = (
            required_experience_relations - admitted_relation_identities
        )
        if missing_experience_relations:
            result.add_error(
                next(iter(attempted_experience_set)),
                ValueError(
                    "Required experience relation was lost during replacement inheritance: "
                    + repr(sorted(missing_experience_relations, key=repr))
                ),
            )
            await self._rollback_experience_files(
                experience_snapshots,
                attempted_experience_uris,
                result,
                ctx,
                reason="required experience relation was lost during replacement inheritance",
            )
            return result

        # Apply delete operations (delete_file_contents is List[MemoryFile])
        # Skip deletes whose URI was just written in the same batch — this happens when the
        # LLM issues a Replace with the same experience_name (delete old + create same-name new),
        # which is semantically an Update. Executing the delete would remove the just-written file.
        upserted_uris = set(result.written_uris + result.edited_uris)
        upserted_uri_keys = {_same_batch_delete_conflict_key(uri) for uri in upserted_uris}
        for file_content in operations.delete_file_contents:
            delete_uri = file_content.uri
            if delete_uri in failed_replacement_deletes:
                tracer.error(
                    f"Skipping delete for {delete_uri}: replacement was not safely published"
                )
                continue
            if has_unresolved_upserts:
                delete_error = ValueError(
                    "Skipped delete because batch contains unresolved upsert URIs"
                )
                result.add_error(delete_uri, delete_error)
                tracer.error(f"Skipping delete for {delete_uri}: {delete_error}")
                continue
            if delete_uri in upserted_uris:
                tracer.info(
                    f"[apply_operations] skipping delete for {delete_uri}: "
                    "URI was upserted in the same batch (Replace-with-same-name treated as Update)"
                )
                continue
            if _same_batch_delete_conflict_key(delete_uri) in upserted_uri_keys:
                tracer.info(
                    f"[apply_operations] skipping delete for {delete_uri}: "
                    "URI case-conflicts with an upserted URI in the same batch"
                )
                continue
            try:
                await self._apply_delete(delete_uri, ctx)
                result.add_deleted(delete_uri)
            except Exception as e:
                tracer.error(f"Failed to delete memory {delete_uri}", e)
                result.add_error(delete_uri, e)

        # Publish relations only after every declared endpoint is known to have succeeded. This
        # prevents an earlier successful write or an existing neighbor from gaining a dangling link
        # when a later upsert fails.
        experience_link_batch = any(
            uri_targets_experience_store(endpoint)
            for link in operations.resolved_links
            for endpoint in (link.from_uri, link.to_uri)
        )
        links_published = True
        if operations.resolved_links:
            links_published = await self._apply_links_to_existing_files(
                operations.resolved_links,
                result,
                ctx,
                deleted_uris=set(result.deleted_uris),
                include_upserted=True,
            )

        if attempted_experience_uris and experience_link_batch and not links_published:
            tracer.error(
                "Experience batch remains unpublished because relation publication failed"
            )
            await self._rollback_experience_files(
                experience_snapshots,
                attempted_experience_uris,
                result,
                ctx,
                reason="experience relation publication failed",
            )
            return result

        await self._sync_resource_refs_for_result(result, ctx)

        # Vectorize written and edited memories
        uri_memory_type_map = {}
        for op in operations.upsert_operations:
            for uri in op.uris:
                uri_memory_type_map[uri] = op.memory_type
        await self._vectorize_memories(
            result,
            ctx,
            extract_context=extract_context,
            uri_memory_type_map=uri_memory_type_map,
        )

        tracer.info(f"Memory operations applied: {result.summary()}")

        # Collect directories that need overview generation
        # uri is now a string, so extract directory using os.path
        dirs = {}
        for operation in operations.upsert_operations:
            for uri_str in operation.uris:
                dir_path = "/".join(uri_str.split("/")[:-1])
                dirs[dir_path] = operation.memory_type
        for file_content in operations.delete_file_contents:
            dir_path = "/".join(file_content.uri.split("/")[:-1])
            dirs[dir_path] = (
                file_content.extra_fields.get("memory_type")
                or file_content.memory_type
                or "unknown"
            )

        for dir, memory_type in dirs.items():
            await self.generate_overview(memory_type, dir, ctx, extract_context)

        return result

    async def _rollback_experience_files(
        self,
        snapshots: Dict[str, _FileSnapshot],
        attempted_uris: List[str],
        result: MemoryUpdateResult,
        ctx: RequestContext,
        *,
        reason: str,
    ) -> None:
        """Restore exact experience blobs after canonical publication fails.

        The caller holds the same transaction lock used by the writes. This is
        compensating I/O, not a storage transaction, so every restoration is
        read back and any rollback failure remains explicit in ``result``.
        """
        viking_fs = self._get_viking_fs()
        rollback_uris = list(dict.fromkeys(attempted_uris))
        rollback_set = set(rollback_uris)
        result.written_uris = [uri for uri in result.written_uris if uri not in rollback_set]
        result.edited_uris = [uri for uri in result.edited_uris if uri not in rollback_set]
        result.index_pending_uris = [
            uri for uri in result.index_pending_uris if uri not in rollback_set
        ]

        for uri in reversed(rollback_uris):
            snapshot = snapshots.get(uri)
            result.add_error(uri, RuntimeError(f"Experience publication rolled back: {reason}"))
            if snapshot is None:
                result.add_error(uri, RuntimeError("Experience rollback snapshot is missing"))
                continue
            try:
                if snapshot.existed:
                    await viking_fs.write_file(
                        uri,
                        snapshot.content,
                        ctx=ctx,
                        lock_handle=self._transaction_handle,
                    )
                    readback = await viking_fs.read_file(uri, ctx=ctx)
                    if readback != snapshot.content:
                        raise RuntimeError(
                            "Experience rollback readback did not match the original blob"
                        )
                else:
                    try:
                        await viking_fs.rm(
                            uri,
                            recursive=False,
                            ctx=ctx,
                            lock_handle=self._transaction_handle,
                        )
                    except NotFoundError:
                        pass
                    try:
                        remaining = await viking_fs.read_file(uri, ctx=ctx)
                    except (NotFoundError, FileNotFoundError):
                        remaining = None
                    if remaining is not None:
                        raise RuntimeError("New experience still exists after rollback delete")
            except Exception as exc:
                tracer.error(f"Failed to roll back experience publication for {uri}: {exc}")
                result.add_error(uri, RuntimeError(f"Experience rollback failed: {exc}"))

    async def _sync_resource_refs_for_result(
        self,
        result: MemoryUpdateResult,
        ctx: RequestContext,
    ) -> None:
        """Synchronize resource refs for memory files touched by session extraction."""
        viking_fs = self._get_viking_fs()
        deleted_uris = set(result.deleted_uris)
        for uri in dict.fromkeys(result.written_uris + result.edited_uris):
            if (
                uri in deleted_uris
                or uri.endswith("/.overview.md")
                or uri.endswith("/.abstract.md")
            ):
                continue
            try:
                raw = await viking_fs.read_file(uri, ctx=ctx)
                mf = MemoryFileUtils.read(raw, uri=uri)
                changed = sync_memory_resource_refs(
                    mf,
                    source=RESOURCE_REF_SOURCE_SESSION_COMMIT,
                )
                if changed:
                    await viking_fs.write_file(
                        uri,
                        MemoryFileUtils.write(mf),
                        ctx=ctx,
                        lock_handle=self._transaction_handle,
                    )
            except Exception as exc:
                logger.warning("Failed to sync resource refs for %s: %s", uri, exc)

    async def _apply_upsert(
        self, resolved_op: ResolvedOperation, ctx: RequestContext, extract_context: Any = None
    ):
        """Apply upsert operation from a flat model."""
        viking_fs = self._get_viking_fs()

        memory_type = resolved_op.memory_type
        schema = self._registry.get(memory_type)
        # Scrub known extraction artifacts from LLM-produced string fields
        # before any of them reach metadata or merge_op processing.
        for field_name, field_value in list(resolved_op.memory_fields.items()):
            if isinstance(field_value, str):
                cleaned_value = _scrub_template_echo(field_value)
                if field_name == "content":
                    cleaned_value = _strip_line_number_artifact(cleaned_value)
                resolved_op.memory_fields[field_name] = cleaned_value
        if memory_type == "experiences":
            final_experience_errors = validate_experience_operations([resolved_op])
            if final_experience_errors:
                raise ValueError(
                    "Experience content failed final admission immediately before persistence: "
                    + repr(final_experience_errors)
                )
        # Process each URI independently
        for uri in resolved_op.uris:
            # Always read from disk first to get the latest content,
            # so consecutive patches to the same URI see each other's changes.
            old_content: Optional[MemoryFile] = None
            try:
                content = await viking_fs.read_file(uri, ctx=ctx)
                if memory_type == "experiences" or content:
                    old_content = MemoryFileUtils.read(content, uri=uri)
                    if memory_type == "experiences":
                        current_experience_errors = validate_stored_experience(old_content, uri)
                        if current_experience_errors:
                            raise ValueError(
                                "Existing experience changed to a legacy-invalid state during "
                                "persistence: "
                                + repr(current_experience_errors)
                            )
            except NotFoundError:
                if memory_type == "experiences" and resolved_op.old_memory_file_content is not None:
                    raise ValueError("Existing experience disappeared before persistence.")
            except Exception:
                if memory_type == "experiences":
                    raise
                # File doesn't exist yet, that's okay
                pass
            # Fall back to pre-fetched content if disk read failed
            if old_content is None:
                old_content = resolved_op.old_memory_file_content

            metadata: Dict[str, Any] = dict(resolved_op.memory_fields)
            source = getattr(resolved_op, "source", None)
            source_extraction_id = getattr(source, "extraction_id", None) if source else None
            if source_extraction_id:
                metadata["source_extraction_id"] = str(source_extraction_id)
            source_trace_id = _operation_trace_id(resolved_op)
            if source_trace_id:
                metadata["last_update_trace_id"] = source_trace_id
            # Process fields defined in schema (apply merge_op)
            for field in schema.fields:
                if field.name in resolved_op.memory_fields:
                    patch_value = resolved_op.memory_fields[field.name]
                    # Get current value for this URI
                    if old_content is None:
                        current_value = None
                    else:
                        if field.name == "content":
                            current_value = old_content.plain_content()
                        else:
                            current_value = old_content.extra_fields.get(field.name)
                    # Use merge_op to process field value
                    merge_op = MergeOpFactory.from_field(field)
                    try:
                        new_value = merge_op.apply(current_value, patch_value)
                    except Exception as e:
                        tracer.info(
                            f"[memory_updater] Skipping field update after merge_op failure: uri={uri}, field={field.name}, error={e}"
                        )
                        if current_value is None:
                            metadata.pop(field.name, None)
                        else:
                            metadata[field.name] = current_value
                        continue
                    metadata[field.name] = new_value

            # Preserve system-managed metadata from the old file that is not
            # covered by the schema. These fields are written by the system,
            # never by the LLM, so they would be silently dropped on every
            # Update without this copy.
            if old_content and old_content.extra_fields:
                schema_field_names = {f.name for f in schema.fields} | {"content", "memory_type"}
                for key, val in old_content.extra_fields.items():
                    if key not in schema_field_names and key not in metadata and val is not None:
                        metadata[key] = val

            metadata["version"] = next_memory_version(old_content)

            # Handle links/backlinks fields: merge with existing
            incoming_links_by_uri = getattr(resolved_op, "_incoming_links_by_uri", {})
            incoming_backlinks_by_uri = getattr(resolved_op, "_incoming_backlinks_by_uri", {})
            incoming_links = incoming_links_by_uri.get(uri, [])
            incoming_backlinks = incoming_backlinks_by_uri.get(uri, [])
            has_existing_links = old_content is not None
            if (
                incoming_links
                or incoming_backlinks
                or (has_existing_links and old_content.links)
                or (has_existing_links and old_content.backlinks)
            ):
                from openviking.session.memory.merge_op.link_merge import merge_links

                # Merge links
                existing_links = old_content.links if has_existing_links else []
                if incoming_links:
                    merged_links = merge_links(
                        existing_links,
                        [link.model_dump() for link in incoming_links],
                    )
                    metadata["links"] = merged_links
                elif existing_links:
                    metadata["links"] = existing_links

                # Merge backlinks
                existing_backlinks = old_content.backlinks if has_existing_links else []
                if incoming_backlinks:
                    merged_backlinks = merge_links(
                        existing_backlinks,
                        [link.model_dump() for link in incoming_backlinks],
                    )
                    metadata["backlinks"] = merged_backlinks
                elif existing_backlinks:
                    metadata["backlinks"] = existing_backlinks

            mf = MemoryFile.from_parsed(uri=uri, parsed=metadata)
            if memory_type == "experiences":
                stored_experience_errors = validate_stored_experience(mf, uri)
                if stored_experience_errors:
                    raise ValueError(
                        "Experience failed final stored-file admission immediately before "
                        "persistence: "
                        + repr(stored_experience_errors)
                    )
            new_full_content = MemoryFileUtils.write(
                mf,
                content_template=schema.content_template,
                extract_context=extract_context,
            )
            # Templates render their headings unconditionally, so a field the
            # model left blank still produces a well-formed file with nothing
            # in it. Events are the live case: their template used to fall back
            # to the raw chatlog when the summary was empty, and removing that
            # fallback left the prompt's "REQUIRED" instruction as the only
            # control. Refuse the write instead of storing an empty memory.
            if _rendered_body_is_empty(new_full_content):
                raise ValueError(
                    f"Refusing to write {memory_type} memory with an empty body "
                    f"(all fields blank after rendering): {uri}"
                )
            await viking_fs.write_file(
                uri,
                new_full_content,
                ctx=ctx,
                lock_handle=self._transaction_handle,
            )

    def _distribute_links_to_operations(self, operations: ResolvedOperations) -> None:
        """Distribute resolved_links to corresponding upsert operations by URI.

        Links go into from_uri's "links" field; backlinks go into to_uri's "backlinks" field.
        """
        # Collect all URIs that will be upserted
        upserted_uris = set()
        for op in operations.upsert_operations:
            op._incoming_links_by_uri = {uri: [] for uri in op.uris}
            op._incoming_backlinks_by_uri = {uri: [] for uri in op.uris}
            for uri in op.uris:
                upserted_uris.add(uri)

        # Attach links to their corresponding upsert operations
        for link in operations.resolved_links:
            # Forward link -> stored in from_uri's "links"
            if link.from_uri in upserted_uris:
                for op in operations.upsert_operations:
                    if link.from_uri in op.uris:
                        op._incoming_links_by_uri[link.from_uri].append(link)
                        break
            # Backlink -> stored in to_uri's "backlinks"
            if link.to_uri in upserted_uris:
                for op in operations.upsert_operations:
                    if link.to_uri in op.uris:
                        op._incoming_backlinks_by_uri[link.to_uri].append(link)
                        break

    @staticmethod
    def _dedupe_text_from_fields(fields: Dict[str, Any]) -> str:
        text = str((fields or {}).get("content") or (fields or {}).get("summary") or "")
        # Apply the same scrubbing _apply_upsert will apply before this text is
        # written. Sibling files were scrubbed before they landed, so comparing
        # an unscrubbed incoming field against them lets an echoed template
        # block or a line-number prefix push the ratio below the threshold —
        # defeating the check in exactly the cases it was written for.
        text = _strip_line_number_artifact(_scrub_template_echo(text))
        return re.sub(r"\s+", " ", text).strip().lower()[:4000]

    @staticmethod
    def _dedupe_text_from_file(raw: Any) -> str:
        text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
        text = re.sub(r"<!--\s*MEMORY_FIELDS.*?-->", "", text, flags=re.DOTALL)
        text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.DOTALL)
        # Events embed the chatlog below the "ChatLog:" heading; compare only
        # the distilled part, mirroring _dedupe_text_from_fields using summary.
        text = re.split(r"\n#[^\n]*ChatLog:\n", text)[0]
        text = text.replace("# Summary", " ")
        return re.sub(r"\s+", " ", text).strip().lower()[:4000]

    async def _drop_duplicate_add_only(
        self,
        ops: List[ResolvedOperation],
        ctx: RequestContext,
        result: Optional["MemoryUpdateResult"] = None,
    ) -> List[ResolvedOperation]:
        """Suppress near-duplicate add_only operations.

        Catches two observed failure modes: one extraction emitting several
        same-content records under different names, and a re-run (backfill)
        re-extracting a session whose records already exist on disk.

        Every suppressed operation is recorded on ``result`` against the URI
        that made it redundant. Callers downstream gate on whether a declared
        URI landed; without that record a suppressed duplicate is
        indistinguishable from a failed write.
        """
        from difflib import SequenceMatcher

        kept: List[ResolvedOperation] = []
        kept_add_only: List[Tuple[int, str, str]] = []  # (kept index, memory_type, text)

        def _record(op: ResolvedOperation, existing_uri: str) -> None:
            if result is None:
                return
            for uri in op.uris:
                result.add_deduplicated(uri, existing_uri)

        for op in ops:
            schema = self._registry.get(op.memory_type) if self._registry else None
            if not schema or getattr(schema, "operation_mode", None) != "add_only":
                kept.append(op)
                continue
            # Compare on scrubbed text. Sibling files on disk were scrubbed
            # before they were written, so comparing a raw incoming field
            # against them would let the very artifacts this module strips
            # depress the ratio and defeat the check.
            text = self._dedupe_text_from_fields(op.memory_fields)
            if not text:
                kept.append(op)
                continue
            duplicate = False
            for slot, (kept_idx, kept_type, kept_text) in enumerate(kept_add_only):
                if kept_type != op.memory_type:
                    continue
                if SequenceMatcher(None, text, kept_text).ratio() < _ADD_ONLY_DEDUPE_RATIO:
                    continue
                duplicate = True
                if len(text) > len(kept_text):
                    loser = kept[kept_idx]
                    tracer.info(
                        f"[memory_updater] same-batch near-duplicate {op.memory_type}: "
                        f"replacing {loser.uris} with richer {op.uris}"
                    )
                    kept[kept_idx] = op
                    kept_add_only[slot] = (kept_idx, kept_type, text)
                    # The richer op takes the slot, so it is the loser that was
                    # suppressed and the winner's URI that now carries it.
                    _record(loser, op.uris[0] if op.uris else "")
                else:
                    tracer.info(
                        f"[memory_updater] dropping same-batch near-duplicate "
                        f"{op.memory_type} op: uris={op.uris}"
                    )
                    _record(op, kept[kept_idx].uris[0] if kept[kept_idx].uris else "")
                break
            if duplicate:
                continue
            existing_uri = await self._duplicates_existing_sibling(op, text, ctx)
            if existing_uri:
                tracer.info(
                    f"[memory_updater] skipping add_only op duplicating existing "
                    f"memory {existing_uri}: uris={op.uris}"
                )
                _record(op, existing_uri)
                continue
            kept_add_only.append((len(kept), op.memory_type, text))
            kept.append(op)
        return kept

    async def _duplicates_existing_sibling(
        self, op: ResolvedOperation, text: str, ctx: RequestContext
    ) -> Optional[str]:
        """Return the URI of an existing sibling this add_only op duplicates."""
        from difflib import SequenceMatcher

        viking_fs = self._get_viking_fs()
        if not viking_fs or not op.uris:
            return None
        uri = op.uris[0]
        parent, _, name = uri.rpartition("/")
        if not parent:
            return None
        # Timestamped add_only names (trajectories) only need comparing against
        # siblings from the same source session.
        ts_match = re.search(r"_(\d{14})\.md$", name)
        try:
            entries = await viking_fs.ls(parent, ctx=ctx)
        except Exception as exc:
            # Silence here would turn the whole check into a permanent no-op
            # with no signal that it had stopped working.
            logger.warning(f"[memory_updater] add_only dedupe could not list {parent}: {exc}")
            return None
        candidates: List[str] = []
        for entry in entries or []:
            entry_name = entry.get("name", "")
            if entry.get("isDir") or not entry_name.endswith(".md") or entry_name.startswith("."):
                continue
            if entry_name == name:
                continue
            if ts_match and ts_match.group(1) not in entry_name:
                continue
            candidates.append(entry.get("uri") or f"{parent}/{entry_name}")
        # ls order is backend-defined. Untimestamped add_only types (events)
        # can exceed the cap in one directory, so sort before truncating or
        # which duplicates get caught varies run to run.
        candidates.sort()
        truncated = len(candidates) - _ADD_ONLY_DEDUPE_MAX_SIBLINGS
        if truncated > 0:
            tracer.info(
                f"[memory_updater] add_only dedupe comparing "
                f"{_ADD_ONLY_DEDUPE_MAX_SIBLINGS} of {len(candidates)} siblings in "
                f"{parent}; {truncated} not compared"
            )
        for sibling_uri in candidates[:_ADD_ONLY_DEDUPE_MAX_SIBLINGS]:
            try:
                raw = await viking_fs.read_file(sibling_uri, ctx=ctx)
            except Exception as exc:
                logger.warning(
                    f"[memory_updater] add_only dedupe could not read {sibling_uri}: {exc}"
                )
                continue
            if not raw:
                continue
            sibling_text = self._dedupe_text_from_file(raw)
            if not sibling_text:
                continue
            if SequenceMatcher(None, text, sibling_text).ratio() >= _ADD_ONLY_DEDUPE_RATIO:
                return sibling_uri
        return None

    async def _apply_links_to_existing_files(
        self,
        resolved_links: List[StoredLink],
        result: MemoryUpdateResult,
        ctx: RequestContext,
        deleted_uris: Optional[set[str]] = None,
        include_upserted: bool = False,
    ) -> bool:
        """Apply links to admitted endpoint files after content writes have succeeded."""
        viking_fs = self._get_viking_fs()
        if not viking_fs:
            return False
        from openviking.core.namespace import context_type_for_uri

        upserted_uris = set(result.written_uris + result.edited_uris)
        deleted = deleted_uris or set()
        deleted_links = [
            link
            for link in resolved_links
            if link.from_uri in deleted or link.to_uri in deleted
        ]
        dropped_experience_links = [
            link
            for link in deleted_links
            if uri_targets_experience_store(link.from_uri)
            or uri_targets_experience_store(link.to_uri)
        ]
        if dropped_experience_links:
            first = dropped_experience_links[0]
            experience_uri = (
                first.from_uri
                if uri_targets_experience_store(first.from_uri)
                else first.to_uri
            )
            result.add_error(
                experience_uri,
                ValueError(
                    "Required experience relation endpoint was deleted before link publication: "
                    f"{first.from_uri} -> {first.to_uri}"
                ),
            )
            return False
        admitted_links = [
            link
            for link in resolved_links
            if link.from_uri not in deleted and link.to_uri not in deleted
        ]
        if not admitted_links:
            return True
        non_memory_endpoints = {
            uri
            for link in admitted_links
            for uri in (link.from_uri, link.to_uri)
            if context_type_for_uri(uri) != "memory"
        }
        skip = (
            set() if include_upserted else upserted_uris
        ) | non_memory_endpoints
        try:
            updated_uris = await write_stored_links(
                admitted_links,
                ctx,
                viking_fs,
                skip_uris=skip,
                preserve_version_uris=upserted_uris if include_upserted else set(),
                lock_handle=self._transaction_handle,
            )
        except _LinkPublicationError as error:
            tracer.error(f"Deferred link publication failed: {error}")
            result.add_error(error.failed_uri, error)
            return False
        for uri in updated_uris:
            if uri not in upserted_uris and uri not in result.edited_uris:
                result.add_edited(uri)
        return True


    async def _inherit_deleted_link_relations(
        self,
        operations: ResolvedOperations,
        result: MemoryUpdateResult,
        ctx: RequestContext,
    ) -> bool:
        uri_remap = dict(getattr(operations, "delete_replacements", {}) or {})
        if not uri_remap:
            return True
        viking_fs = self._get_viking_fs()
        if not viking_fs:
            result.add_error(
                next(iter(uri_remap)),
                RuntimeError("Cannot inherit replacement links without VikingFS"),
            )
            return False

        inherited_links: List[StoredLink] = []
        for deleted_uri, replacement_uri in uri_remap.items():
            if not deleted_uri or not replacement_uri or deleted_uri == replacement_uri:
                continue
            try:
                content = await viking_fs.read_file(deleted_uri, ctx=ctx)
            except Exception as e:
                tracer.error(f"Failed to read deleted memory links for replacement {deleted_uri}: {e}")
                result.add_error(deleted_uri, e)
                return False
            if not content:
                error = ValueError("Deleted memory is empty during replacement link inheritance")
                result.add_error(deleted_uri, error)
                return False
            try:
                deleted_file = MemoryFileUtils.read(content, uri=deleted_uri)
            except Exception as e:
                tracer.error(
                    f"Failed to parse deleted memory links for replacement {deleted_uri}: {e}"
                )
                result.add_error(deleted_uri, e)
                return False
            try:
                for link in list(deleted_file.links or []):
                    remapped = _remap_link_dict(link, uri_remap)
                    if remapped.get("from_uri") == remapped.get("to_uri"):
                        continue
                    inherited_links.append(StoredLink(**remapped))
                for link in list(deleted_file.backlinks or []):
                    remapped = _remap_link_dict(link, uri_remap)
                    if remapped.get("from_uri") == remapped.get("to_uri"):
                        continue
                    inherited_links.append(StoredLink(**remapped))
            except Exception as e:
                tracer.error(
                    f"Failed to validate inherited links for replacement {deleted_uri}: {e}"
                )
                result.add_error(deleted_uri, e)
                return False

        if not inherited_links:
            return True
        written_or_edited = set(result.written_uris + result.edited_uris)
        try:
            updated_uris = await write_stored_links(
                inherited_links,
                ctx,
                viking_fs,
                skip_uris=set(uri_remap),
                preserve_version_uris=written_or_edited,
                lock_handle=self._transaction_handle,
            )
        except _LinkPublicationError as error:
            tracer.error(f"Failed to inherit replacement links: {error}")
            result.add_error(error.failed_uri, error)
            return False
        for uri in updated_uris:
            if uri not in written_or_edited:
                result.add_edited(uri)
        return True

    async def _apply_delete(self, uri: str, ctx: RequestContext) -> None:
        """Apply delete operation (uri is already a string)."""
        viking_fs = self._get_viking_fs()

        # Delete from VikingFS
        # VikingFS automatically handles vector index cleanup
        # Pass transaction_handle so rm() reuses the compressor's tree lock
        # instead of trying to acquire a new lock (which would conflict).
        try:
            await viking_fs.rm(uri, recursive=False, ctx=ctx, lock_handle=self._transaction_handle)
        except NotFoundError:
            tracer.error(f"Memory not found for delete: {uri}")
            # Idempotent - deleting non-existent file succeeds

    async def _vectorize_memories(
        self,
        result: MemoryUpdateResult,
        ctx: RequestContext,
        extract_context: Any = None,
        uri_memory_type_map: Dict[str, str] = None,
    ) -> int:
        """Vectorize written and edited memory files.

        Args:
            result: MemoryUpdateResult with written_uris and edited_uris
            ctx: Request context
            extract_context: Extract context for embedding template rendering
            uri_memory_type_map: Mapping from URI to memory_type
        """
        uri_memory_type_map = uri_memory_type_map or {}
        deleted_set = set(result.deleted_uris)

        def is_experience_uri(uri: str) -> bool:
            memory_type = uri_memory_type_map.get(uri) or self.memory_type_from_uri(uri)
            return memory_type == "experiences"

        if not self._vikingdb:
            for uri in dict.fromkeys(result.written_uris + result.edited_uris):
                if uri not in deleted_set and is_experience_uri(uri):
                    result.add_index_pending(uri)
            logger.debug("VikingDB not available, skipping vectorization")
            return 0

        viking_fs = self._get_viking_fs()
        request_wait_tracker = get_request_wait_tracker()
        attempted_count = 0

        # Collect all URIs to vectorize (skip .overview.md and .abstract.md - they are handled separately)
        # Also skip URIs that were deleted in the same batch
        uris_to_vectorize = []
        for uri in result.written_uris + result.edited_uris:
            if uri in deleted_set:
                continue
            if not uri.endswith("/.overview.md") and not uri.endswith("/.abstract.md"):
                uris_to_vectorize.append(uri)

        if not uris_to_vectorize:
            logger.debug("No memory files to vectorize")
            return 0

        for uri in uris_to_vectorize:
            try:
                # Read the memory file to get content
                content = await viking_fs.read_file(uri, ctx=ctx) or ""

                mf = MemoryFileUtils.read(content, uri=uri)
                from openviking.session.memory.utils.link_renderer import LinkRenderer

                abstract = LinkRenderer.strip_all_links(mf.content or "")
                abstract = self._truncate_memory_abstract(abstract)
                embedding_text = abstract

                memory_type = uri_memory_type_map.get(uri) or self.memory_type_from_uri(uri)
                if memory_type and self._registry:
                    schema = self._registry.get(memory_type)
                    if schema and schema.embedding_template:
                        template_vars = dict(mf.extra_fields)
                        template_vars["content"] = abstract
                        missing_vars = TemplateUtils.find_missing_variables(
                            schema.embedding_template,
                            template_vars,
                        )
                        if missing_vars:
                            logger.warning(
                                f"Missing embedding template variables for {uri}, falling back to plain content: {sorted(missing_vars)}"
                            )
                        else:
                            try:
                                embedding_text = render_template(
                                    schema.embedding_template,
                                    template_vars,
                                    extract_context=extract_context,
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to render embedding template for {uri}, falling back to plain content: {e}"
                                )

                # Get parent URI
                from openviking_cli.utils.uri import VikingURI

                parent_uri = VikingURI(uri).parent.uri

                # Create Context for vectorization
                from openviking.core.context import Context, ContextLevel, Vectorize
                from openviking.storage.queuefs.embedding_msg_converter import EmbeddingMsgConverter

                memory_context = Context(
                    uri=uri,
                    parent_uri=parent_uri,
                    is_leaf=True,
                    abstract=abstract,
                    context_type="memory",
                    level=ContextLevel.DETAIL,
                    user=ctx.user,
                    account_id=ctx.account_id,
                )
                memory_context.set_vectorize(Vectorize(text=embedding_text))

                # Convert to embedding msg and enqueue
                embedding_msg = EmbeddingMsgConverter.from_context(memory_context)
                if embedding_msg:
                    if embedding_msg.telemetry_id:
                        request_wait_tracker.register_embedding_root(
                            embedding_msg.telemetry_id, embedding_msg.id
                        )
                    attempted_count += 1
                    try:
                        enqueued = await self._vikingdb.enqueue_embedding_msg(embedding_msg)
                    except Exception as e:
                        if embedding_msg.telemetry_id:
                            request_wait_tracker.mark_embedding_failed(
                                embedding_msg.telemetry_id,
                                embedding_msg.id,
                                str(e),
                            )
                        raise
                    if not enqueued and embedding_msg.telemetry_id:
                        request_wait_tracker.mark_embedding_failed(
                            embedding_msg.telemetry_id,
                            embedding_msg.id,
                            "embedding enqueue returned false",
                        )
                    if not enqueued:
                        if is_experience_uri(uri):
                            result.add_index_pending(uri)
                    else:
                        logger.debug(f"Enqueued memory for vectorization: {uri}")
                elif is_experience_uri(uri):
                    result.add_index_pending(uri)

            except Exception as e:
                tracer.error(f"Failed to vectorize memory {uri}: {e}")
                if is_experience_uri(uri):
                    result.add_index_pending(uri)
        return attempted_count

    @staticmethod
    def _truncate_memory_abstract(abstract: str) -> str:
        """Cap memory vector-store abstract fields below backend byte limits."""
        encoded = (abstract or "").encode("utf-8")
        if len(encoded) <= _MEMORY_ABSTRACT_MAX_BYTES:
            return abstract or ""
        return encoded[:_MEMORY_ABSTRACT_MAX_BYTES].decode("utf-8", errors="ignore")

    async def generate_overview(
        self,
        memory_type: str,
        directory: str,
        ctx: RequestContext,
        extract_context: Any = None,
    ) -> None:
        """
        Generate .overview.md file for a directory based on overview_template.

        Args:
            memory_type: Memory type name (e.g., 'events')
            directory: Directory path containing memory files
            ctx: Request context
        """
        from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils

        # Get the schema for this memory type
        registry = self._registry
        schema = registry.get(memory_type)

        if not schema or not schema.overview_template:
            logger.debug(f"No overview_template for memory type: {memory_type}")
            return

        viking_fs = self._get_viking_fs()

        # List direct .md files in the directory (excluding .overview.md and .abstract.md)
        try:
            # Use ls to list direct children
            entries = await viking_fs.ls(directory, show_all_hidden=True, ctx=ctx)

            # Extract file paths from ls entries
            md_files = []
            base_uri = directory.rstrip("/")
            for entry in entries:
                name = entry.get("name", "")
                if (
                    name.endswith(".md")
                    and not name.endswith(".overview.md")
                    and not name.endswith(".abstract.md")
                ):
                    md_files.append(f"{base_uri}/{name}")

        except (NotFoundError, FileNotFoundError):
            logger.debug("Skip overview generation for deleted directory: %s", directory)
            return
        except Exception as e:
            tracer.error(f"Failed to list files in {directory}: {e}")
            return

        # If no memory files, delete the .overview.md and the directory if empty
        if not md_files:
            overview_path = f"{directory.rstrip('/')}/.overview.md"
            can_delete_directory = all(
                entry.get("name", "") in {"", ".overview.md"} for entry in entries
            )
            try:
                await viking_fs.rm(
                    overview_path,
                    recursive=False,
                    ctx=ctx,
                    lock_handle=self._transaction_handle,
                )
            except Exception:
                pass
            # Try to delete empty directory
            if can_delete_directory:
                try:
                    await viking_fs.rm(
                        directory,
                        recursive=True,
                        ctx=ctx,
                        lock_handle=self._transaction_handle,
                    )
                except Exception:
                    pass
            return

        # Parse each file and collect items
        items = []
        for file_path in md_files:
            try:
                content = await viking_fs.read_file(file_path, ctx=ctx)
                mf = MemoryFileUtils.read(content, uri=file_path)

                # Extract filename from path
                filename = file_path.split("/")[-1]
                metadata = mf.to_metadata()

                items.append(
                    {
                        "file_name": filename,
                        "file_content": metadata,
                    }
                )
            except Exception as e:
                tracer.error(f"Failed to parse {file_path}: {e}")
                continue

        if not items:
            logger.debug(f"No valid memory files parsed in {directory}")
            return

        overview_context = {
            "memory_type": memory_type,
            "directory_name": directory.rstrip("/").split("/")[-1],
            "items": items,
        }

        # Render the template
        try:
            rendered = render_template(
                schema.overview_template,
                overview_context,
                extract_context=extract_context,
            )
        except Exception as e:
            tracer.error(f"Failed to render overview template for {memory_type}: {e}")
            return

        # Write .overview.md to the directory
        overview_path = f"{directory.rstrip('/')}/.overview.md"
        try:
            await viking_fs.write_file(
                overview_path,
                rendered,
                ctx=ctx,
                lock_handle=self._transaction_handle,
            )
        except Exception as e:
            tracer.error(f"Failed to write overview {overview_path}: {e}")
