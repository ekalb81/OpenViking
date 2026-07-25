# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Deterministic admission policy for reusable experience memories."""

from __future__ import annotations

import re
import unicodedata
from types import SimpleNamespace
from typing import Any, Iterable
from urllib.parse import urlsplit

from openviking.core.namespace import canonical_user_root, canonicalize_uri

EXPERIENCE_MEMORY_TYPE = "experiences"
EXPERIENCE_MAX_CONTENT_CHARS = 3000
EXPERIENCE_MAX_BULLET_CHARS = 400
EXPERIENCE_MAX_FILENAME_BYTES = 255
EXPERIENCE_SECTION_LIMITS = {
    "Situation": 3,
    "Approach": 8,
    "Reflect": 8,
}
_EXPECTED_HEADINGS = tuple(EXPERIENCE_SECTION_LIMITS)
_HEADING_RE = re.compile(r"^## (Situation|Approach|Reflect)\s*$", re.MULTILINE)
_UNSAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*%\x00-\x1f]')
_WINDOWS_RESERVED_STEMS = {
    "aux",
    "con",
    "conin$",
    "conout$",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
    *(f"com{index}" for index in "¹²³"),
    *(f"lpt{index}" for index in "¹²³"),
}


def _violation(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, **details}


def _identity_key(value: str) -> str:
    """Return a collision key while preserving the raw persisted URI identity."""
    return unicodedata.normalize("NFC", value).casefold()


def uri_targets_experience_store(uri: Any) -> bool:
    """Return whether a URI path names the experience store in either supported shape."""
    try:
        parsed = urlsplit(str(uri or ""))
    except ValueError:
        return False
    parts = [part for part in parsed.path.split("/") if part]
    lowered = [part.casefold() for part in parts]
    return any(
        lowered[index : index + 2] == ["memories", "experiences"]
        for index in range(max(0, len(lowered) - 1))
    )


def _safe_experience_name(name: str) -> bool:
    if (
        not name
        or name != name.strip()
        or name != unicodedata.normalize("NFC", name)
        or name.startswith(".")
        or name.endswith((".", " "))
        or _UNSAFE_FILENAME_RE.search(name)
        or name != name.casefold()
        or any(character.isspace() for character in name)
        or any(unicodedata.category(character).startswith("C") for character in name)
        or len(f"{name}.md".encode("utf-8")) > EXPERIENCE_MAX_FILENAME_BYTES
    ):
        return False
    return name.split(".", 1)[0].casefold() not in _WINDOWS_RESERVED_STEMS


def _experience_uri_filename(uri: str) -> tuple[str | None, str | None]:
    """Return the exact direct-child filename or a stable validation reason."""
    if (
        not uri
        or uri != uri.strip()
        or uri != unicodedata.normalize("NFC", uri)
        or any(character.isspace() for character in uri)
        or any(ord(character) < 32 or ord(character) == 127 for character in uri)
    ):
        return None, "Experience URI must not contain whitespace, controls, or Unicode aliases."
    try:
        parsed = urlsplit(uri)
    except ValueError:
        return None, "Experience URI must be a well-formed viking://user URI."
    if (
        parsed.scheme != "viking"
        or parsed.netloc != "user"
        or parsed.query
        or parsed.fragment
        or "\\" in parsed.path
        or parsed.path != unicodedata.normalize("NFC", parsed.path)
    ):
        return None, "Experience URI must use viking://user without query or fragment data."

    parts = parsed.path.split("/")
    short_shape = (
        len(parts) == 4
        and parts[0] == ""
        and parts[1] == "memories"
        and parts[2] == "experiences"
        and bool(parts[3])
    )
    canonical_shape = (
        len(parts) == 5
        and parts[0] == ""
        and bool(parts[1])
        and parts[2] == "memories"
        and parts[3] == "experiences"
        and bool(parts[4])
    )
    if not short_shape and not canonical_shape:
        return None, "Experience URI must be a direct child of /memories/experiences/."
    return parts[-1], None


def validate_experience_content(content: Any) -> list[dict[str, Any]]:
    """Return structured violations for one experience body.

    This validates the mechanically provable part of atomicity. Whether two bullets describe the
    same semantic user intent remains an extraction/evaluation concern and is intentionally not
    claimed here.
    """
    if not isinstance(content, str) or not content.strip():
        return [_violation("content_required", "Experience content must be a non-empty string.")]

    body = content.strip()
    errors: list[dict[str, Any]] = []
    if len(content) > EXPERIENCE_MAX_CONTENT_CHARS:
        errors.append(
            _violation(
                "content_too_long",
                f"Experience content exceeds {EXPERIENCE_MAX_CONTENT_CHARS} characters; split it.",
                actual=len(content),
                limit=EXPERIENCE_MAX_CONTENT_CHARS,
            )
        )

    matches = list(_HEADING_RE.finditer(body))
    headings = tuple(match.group(1) for match in matches)
    if headings != _EXPECTED_HEADINGS:
        errors.append(
            _violation(
                "section_structure",
                "Use exactly ## Situation, ## Approach, and ## Reflect in that order.",
                actual=list(headings),
                expected=list(_EXPECTED_HEADINGS),
            )
        )
        return errors

    preamble = body[: matches[0].start()].strip()
    if preamble:
        errors.append(
            _violation(
                "non_section_content",
                "Do not place content before the ## Situation heading.",
                examples=preamble.splitlines()[:3],
            )
        )

    seen_rules: dict[str, str] = {}
    for index, heading in enumerate(_EXPECTED_HEADINGS):
        start = matches[index].end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        lines = [line for line in body[start:end].splitlines() if line.strip()]
        non_bullets = [line for line in lines if not line.startswith("- ")]
        if non_bullets:
            errors.append(
                _violation(
                    "non_bullet_content",
                    f"Every non-empty line in {heading} must be a '- ' bullet.",
                    section=heading,
                    examples=non_bullets[:3],
                )
            )

        bullet_values = [line[2:] for line in lines if line.startswith("- ")]
        empty_bullets = [
            bullet_index
            for bullet_index, bullet in enumerate(bullet_values, start=1)
            if not bullet.strip()
        ]
        if empty_bullets:
            errors.append(
                _violation(
                    "empty_bullet",
                    f"Every {heading} bullet must contain text.",
                    section=heading,
                    bullets=empty_bullets,
                )
            )
        bullets = [bullet.strip() for bullet in bullet_values if bullet.strip()]
        if not bullets:
            errors.append(
                _violation(
                    "section_empty",
                    f"{heading} must contain at least one bullet.",
                    section=heading,
                )
            )
            continue

        limit = EXPERIENCE_SECTION_LIMITS[heading]
        if len(bullets) > limit:
            errors.append(
                _violation(
                    "too_many_bullets",
                    f"{heading} exceeds {limit} bullets; split the experience.",
                    section=heading,
                    actual=len(bullets),
                    limit=limit,
                )
            )

        for bullet_index, bullet in enumerate(bullets, start=1):
            if len(bullet) > EXPERIENCE_MAX_BULLET_CHARS:
                errors.append(
                    _violation(
                        "bullet_too_long",
                        f"{heading} bullet {bullet_index} exceeds {EXPERIENCE_MAX_BULLET_CHARS} characters.",
                        section=heading,
                        bullet=bullet_index,
                        actual=len(bullet),
                        limit=EXPERIENCE_MAX_BULLET_CHARS,
                    )
                )
            normalized = re.sub(r"\s+", " ", bullet).strip().casefold()
            if normalized and normalized in seen_rules:
                errors.append(
                    _violation(
                        "duplicate_rule",
                        "Do not repeat the same rule across experience sections.",
                        section=heading,
                        duplicate_of=seen_rules[normalized],
                        bullet=bullet_index,
                    )
                )
            elif normalized:
                seen_rules[normalized] = heading

    return errors


def validate_experience_operations(
    operations: Iterable[Any], *, validate_existing: bool = True
) -> list[dict[str, Any]]:
    """Validate all experience upserts and attach operation identity to violations."""
    errors: list[dict[str, Any]] = []
    seen_names: dict[str, dict[str, Any]] = {}
    seen_uris: dict[str, dict[str, Any]] = {}
    for operation in operations:
        memory_type = getattr(operation, "memory_type", None)
        fields = getattr(operation, "memory_fields", {}) or {}
        operation_id = {
            "page_id": getattr(operation, "page_id", None),
            "uris": list(getattr(operation, "uris", []) or []),
            "experience_name": fields.get("experience_name"),
        }
        if memory_type != EXPERIENCE_MEMORY_TYPE:
            for uri in operation_id["uris"]:
                if uri_targets_experience_store(uri):
                    errors.append(
                        {
                            **operation_id,
                            **_violation(
                                "experience_uri_type_mismatch",
                                "Only an experiences operation may target /memories/experiences/.",
                                memory_type=memory_type,
                                uri=uri,
                            ),
                        }
                    )
            continue

        if "memory_type" not in fields:
            # The extraction schema omits this redundant field. Normalize it here so the
            # persisted MemoryFile remains strongly typed even on the ordinary extraction path.
            fields["memory_type"] = EXPERIENCE_MEMORY_TYPE
        elif fields.get("memory_type") != EXPERIENCE_MEMORY_TYPE:
            errors.append(
                {
                    **operation_id,
                    **_violation(
                        "experience_payload_type_mismatch",
                        "The persisted memory_fields.memory_type must exactly equal experiences.",
                        memory_type=fields.get("memory_type"),
                    ),
                }
            )

        for violation in validate_experience_content(fields.get("content")):
            errors.append({**operation_id, **violation})

        raw_experience_name = fields.get("experience_name")
        experience_name = str(raw_experience_name or "").strip()
        normalized_name = experience_name.casefold()
        if not isinstance(raw_experience_name, str) or not normalized_name:
            errors.append(
                {
                    **operation_id,
                    **_violation(
                        "experience_name_required",
                        "experience_name must be a non-empty string.",
                    ),
                }
            )
        elif not _safe_experience_name(raw_experience_name):
            errors.append(
                {
                    **operation_id,
                    **_violation(
                        "unsafe_experience_name",
                        "experience_name must be a safe filename stem without path separators, "
                        "reserved or hidden names, Unicode case-fold aliases, control characters, "
                        "percent encoding, Unicode controls, or surrounding whitespace.",
                    ),
                }
            )
        else:
            if normalized_name in seen_names:
                errors.append(
                    {
                        **operation_id,
                        **_violation(
                            "duplicate_experience_name",
                            "Each experience in a batch must have a unique name.",
                            duplicate_of=seen_names[normalized_name],
                        ),
                    }
                )
            else:
                seen_names[normalized_name] = operation_id

        if len(operation_id["uris"]) != 1:
            errors.append(
                {
                    **operation_id,
                    **_violation(
                        "experience_uri_count",
                        "Each experience upsert must resolve to exactly one URI.",
                        actual=len(operation_id["uris"]),
                    ),
                }
            )
        for uri in operation_id["uris"]:
            uri_text = str(uri)
            normalized_uri = _identity_key(uri_text.rstrip("/"))
            if not normalized_uri:
                continue
            filename, uri_error = _experience_uri_filename(uri_text)
            if uri_error:
                errors.append(
                    {
                        **operation_id,
                        **_violation(
                            "experience_uri_scope",
                            uri_error,
                            uri=uri,
                        ),
                    }
                )
            elif not filename or not filename.endswith(".md"):
                errors.append(
                    {
                        **operation_id,
                        **_violation(
                            "experience_uri_filename",
                            "Experience URI must end in a non-empty .md filename.",
                            uri=uri,
                        ),
                    }
                )
            elif normalized_name and filename[:-3] != experience_name:
                errors.append(
                    {
                        **operation_id,
                        **_violation(
                            "experience_uri_name_mismatch",
                            "Experience URI filename must exactly match the NFC experience_name; "
                            "encoded and case aliases are not accepted.",
                            uri=uri,
                        ),
                    }
                )
            if normalized_uri in seen_uris:
                errors.append(
                    {
                        **operation_id,
                        **_violation(
                            "duplicate_experience_uri",
                            "Each experience in a batch must resolve to a unique URI.",
                            uri=uri,
                            duplicate_of=seen_uris[normalized_uri],
                        ),
                    }
                )
            else:
                seen_uris[normalized_uri] = operation_id

        supersedes = str(fields.get("supersedes") or "").strip()
        if supersedes:
            errors.append(
                {
                    **operation_id,
                    **_violation(
                        "automatic_supersession_disabled",
                        "Automatic experience supersession is disabled; leave supersedes empty. "
                        "Use a separately verified administrative migration.",
                        supersedes=supersedes,
                    ),
                }
            )

        old_file = getattr(operation, "old_memory_file_content", None)
        if validate_existing and old_file is not None:
            old_uri = str(getattr(old_file, "uri", None) or operation_id["uris"][0])
            legacy_violations = validate_stored_experience(old_file, old_uri)
            if legacy_violations:
                errors.append(
                    {
                        **operation_id,
                        **_violation(
                            "legacy_experience_read_only",
                            "An experience that predates the current admission policy is read-only. "
                            "Create a bounded new experience and migrate the legacy file separately.",
                            legacy_uri=old_uri,
                            legacy_violation_codes=[
                                violation["code"] for violation in legacy_violations
                            ],
                        ),
                    }
                )
    return errors


def validate_experience_operation_context(
    operations: Iterable[Any], ctx: Any
) -> list[dict[str, Any]]:
    """Bind experience URIs to the authenticated user's canonical experience root."""
    operation_list = list(operations)
    errors: list[dict[str, Any]] = []
    if ctx is None or getattr(getattr(ctx, "user", None), "user_id", None) is None:
        for operation in operation_list:
            memory_type = getattr(operation, "memory_type", None)
            uris = list(getattr(operation, "uris", []) or [])
            if memory_type == EXPERIENCE_MEMORY_TYPE or any(
                uri_targets_experience_store(uri) for uri in uris
            ):
                errors.append(
                    _violation(
                        "experience_context_required",
                        "Experience admission requires an authenticated request context.",
                        memory_type=memory_type,
                        uris=uris,
                    )
                )
        return errors

    expected_root = f"{canonical_user_root(ctx)}/memories/experiences"
    for operation in operation_list:
        memory_type = getattr(operation, "memory_type", None)
        fields = getattr(operation, "memory_fields", {}) or {}
        uris = list(getattr(operation, "uris", []) or [])
        operation_id = {
            "page_id": getattr(operation, "page_id", None),
            "uris": uris,
            "experience_name": fields.get("experience_name"),
        }
        for uri in uris:
            raw_uri = str(uri or "")
            try:
                canonical_uri = canonicalize_uri(raw_uri, ctx)
            except Exception as exc:
                if memory_type == EXPERIENCE_MEMORY_TYPE or uri_targets_experience_store(raw_uri):
                    errors.append(
                        {
                            **operation_id,
                            **_violation(
                                "experience_uri_canonicalization",
                                "Experience URI could not be canonicalized for the current user.",
                                uri=raw_uri,
                                reason=str(exc),
                            ),
                        }
                    )
                continue

            targets_current_experience_store = canonical_uri == expected_root or canonical_uri.startswith(
                f"{expected_root}/"
            )
            if memory_type != EXPERIENCE_MEMORY_TYPE:
                if targets_current_experience_store:
                    errors.append(
                        {
                            **operation_id,
                            **_violation(
                                "experience_uri_type_mismatch",
                                "Only an experiences operation may target the current user's "
                                "canonical experience store.",
                                memory_type=memory_type,
                                uri=raw_uri,
                            ),
                        }
                    )
                continue

            experience_name = fields.get("experience_name")
            expected_uri = f"{expected_root}/{experience_name}.md"
            if canonical_uri != expected_uri:
                errors.append(
                    {
                        **operation_id,
                        **_violation(
                            "experience_uri_context_mismatch",
                            "Experience URI must resolve exactly to the current user's canonical "
                            "experience filename.",
                            uri=raw_uri,
                            canonical_uri=canonical_uri,
                            expected_uri=expected_uri,
                        ),
                    }
                )
    return errors


def validate_stored_experience(memory_file: Any, uri: str) -> list[dict[str, Any]]:
    """Validate the complete identity and body of an already materialized experience file."""
    stored_uri = str(getattr(memory_file, "uri", None) or "")
    fields = dict(getattr(memory_file, "extra_fields", {}) or {})
    fields["memory_type"] = getattr(memory_file, "memory_type", None)
    fields["content"] = (
        memory_file.plain_content()
        if callable(getattr(memory_file, "plain_content", None))
        else str(getattr(memory_file, "content", "") or "")
    )
    operation = SimpleNamespace(
        memory_type=EXPERIENCE_MEMORY_TYPE,
        memory_fields=fields,
        uris=[uri],
        page_id=None,
        old_memory_file_content=None,
    )
    errors = validate_experience_operations([operation], validate_existing=False)
    if stored_uri != uri:
        errors.append(
            _violation(
                "stored_experience_uri_mismatch",
                "Stored experience identity must exactly match the file URI being validated.",
                stored_uri=stored_uri,
                uri=uri,
            )
        )
    return errors
