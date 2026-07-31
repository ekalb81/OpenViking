# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""LiteLLM Embedder Implementation

Uses litellm to provide a unified embedding interface across many providers
(OpenRouter, Ollama, vLLM, and any OpenAI-compatible endpoint).
"""

import asyncio
import os
import weakref
from typing import Any, Dict, List, Optional, Sequence

import litellm

from openviking.models.embedder.base import (
    DenseEmbedderBase,
    EmbedResult,
    truncate_and_normalize,
)
from openviking.models.embedder.micro_batcher import EmbedMicroBatcher
from openviking.telemetry import get_current_telemetry
from openviking_cli.utils import get_logger

logger = get_logger(__name__)


class LiteLLMDenseEmbedder(DenseEmbedderBase):
    """LiteLLM Dense Embedder Implementation

    Routes embedding requests through litellm, supporting dozens of providers
    via a unified interface. Model names use litellm's provider/model format
    (e.g., "openai/text-embedding-3-small", "ollama/nomic-embed-text").

    Example:
        >>> # OpenRouter embeddings
        >>> embedder = LiteLLMDenseEmbedder(
        ...     model_name="openai/text-embedding-3-small",
        ...     api_key="sk-or-...",
        ...     api_base="https://openrouter.ai/api/v1",
        ...     dimension=1536,
        ... )
        >>> result = embedder.embed("Hello world")

        >>> # Local Ollama embeddings
        >>> embedder = LiteLLMDenseEmbedder(
        ...     model_name="ollama/nomic-embed-text",
        ...     api_base="http://localhost:11434",
        ...     dimension=768,
        ... )
        >>> result = embedder.embed("Hello world")
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        dimension: Optional[int] = None,
        query_param: Optional[str] = None,
        document_param: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize LiteLLM Dense Embedder

        Args:
            model_name: Model name in litellm format (e.g., "openai/text-embedding-3-small").
            api_key: API key for the provider. Falls back to provider-specific env vars.
            api_base: Custom API base URL (e.g., "https://openrouter.ai/api/v1").
            dimension: Embedding vector dimension (required).
            query_param: Parameter value for query-side embeddings (non-symmetric mode).
            document_param: Parameter value for document-side embeddings (non-symmetric mode).
            extra_headers: Extra HTTP headers for API requests.
            config: Additional configuration dict.
        """
        super().__init__(model_name, config)
        self.provider = "litellm"

        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

        self.api_key = api_key
        self.api_base = api_base
        self.dimension = dimension
        self.query_param = query_param
        self.document_param = document_param
        self.extra_headers = extra_headers

        if dimension is None:
            raise ValueError(
                "LiteLLM embedding provider requires 'dimension' to be set explicitly. "
                "Check your embedding model's documentation for the correct dimension."
            )
        self._dimension = dimension

        # Micro-batching: coalesce concurrent document embeddings into one
        # array request. Request framing dominates the cost of a single-text
        # call (measured ~213ms per request vs ~8ms per text at batch size 64
        # against the live provider), so this is where the latency goes.
        self.micro_batch_enabled = bool(self.config.get("micro_batch_enabled", True))
        self.micro_batch_size = max(1, int(self.config.get("micro_batch_size", 32)))
        self.micro_batch_wait_ms = max(0.0, float(self.config.get("micro_batch_wait_ms", 10.0)))
        # Pending futures are loop-bound, so keep one batcher per event loop
        # (same pattern as the shared async embed semaphores).
        self._micro_batchers: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, EmbedMicroBatcher]" = (
            weakref.WeakKeyDictionary()
        )

    def _truncate_vector(self, vector: List[float]) -> List[float]:
        """Truncate to the configured dimension and restore unit length.

        Truncating a unit vector leaves it shorter than 1 - dropping 3/4 of a
        Qwen3 embedding gives a norm near 0.5. Every other embedder routes
        through ``truncate_and_normalize``, so vectors already in a collection
        are unit length; emitting unnormalized ones from this path would make
        the index inhomogeneous, and any similarity that is not cosine would
        then score new vectors systematically below old ones with no error.

        Args:
            vector: Input vector from API

        Returns:
            The vector truncated to ``dimension`` and L2-normalized, or
            unchanged when no truncation applies.
        """
        return truncate_and_normalize(vector, self.dimension)

    def _build_kwargs(self, is_query: bool = False) -> Dict[str, Any]:
        """Build kwargs dict for litellm.embedding() call."""
        kwargs: Dict[str, Any] = {"model": self.model_name}

        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        # Don't pass dimensions parameter to API - some models don't support it
        # (e.g., Qwen3-Embedding-4B doesn't support matryoshka representation)
        # Instead, we'll truncate the result vector if needed

        # Non-symmetric embedding support
        active_param = None
        if is_query and self.query_param is not None:
            active_param = self.query_param
        elif not is_query and self.document_param is not None:
            active_param = self.document_param

        if active_param:
            if "=" in active_param:
                # Parse key=value format (e.g., "input_type=query,task=search")
                extra_body = {}
                for part in active_param.split(","):
                    part = part.strip()
                    if "=" in part:
                        key, value = part.split("=", 1)
                        extra_body[key.strip()] = value.strip()
                if extra_body:
                    kwargs["extra_body"] = extra_body
            else:
                kwargs["input_type"] = active_param

        return kwargs

    def _update_telemetry_token_usage(self, response) -> None:
        """Update telemetry and token usage from response."""
        usage = getattr(response, "usage", None)
        if not usage:
            return

        def _usage_value(key: str, default: int = 0) -> int:
            if isinstance(usage, dict):
                return int(usage.get(key, default) or default)
            return int(getattr(usage, key, default) or default)

        prompt_tokens = _usage_value("prompt_tokens", 0)
        total_tokens = _usage_value("total_tokens", prompt_tokens)
        output_tokens = max(total_tokens - prompt_tokens, 0)

        # Update telemetry
        get_current_telemetry().add_token_usage_by_source(
            "embedding",
            prompt_tokens,
            output_tokens,
        )

        # Update token usage tracker
        self.update_token_usage(
            model_name=self.model_name,
            provider="litellm",
            prompt_tokens=prompt_tokens,
            completion_tokens=output_tokens,
        )

    def embed(self, text: str, is_query: bool = False) -> EmbedResult:
        """Perform dense embedding on text via litellm.

        Args:
            text: Input text
            is_query: Flag to indicate if this is a query embedding

        Returns:
            EmbedResult: Result containing dense_vector

        Raises:
            RuntimeError: When embedding call fails
        """

        def _call() -> EmbedResult:
            kwargs = self._build_kwargs(is_query=is_query)
            kwargs["input"] = [text]
            response = litellm.embedding(**kwargs)
            self._update_telemetry_token_usage(response)
            vector = response.data[0]["embedding"]
            # Truncate vector if needed
            vector = self._truncate_vector(vector)
            return EmbedResult(dense_vector=vector)

        try:
            return self._run_with_retry(
                _call,
                logger=logger,
                operation_name="LiteLLM embedding",
            )
        except Exception as e:
            raise RuntimeError(f"LiteLLM embedding failed: {e}") from e

    async def embed_async(self, text: str, is_query: bool = False) -> EmbedResult:
        # Document embeddings coalesce through the micro-batcher. Query
        # embeddings stay on the direct path: they are latency-critical,
        # rarely concurrent, and may carry different request parameters
        # (query_param) than document calls, so they must not share batches.
        if self.micro_batch_enabled and not is_query and isinstance(text, str):
            try:
                return await self._get_micro_batcher().submit(text)
            except Exception as e:
                raise RuntimeError(f"LiteLLM embedding failed: {e}") from e

        async def _call() -> EmbedResult:
            kwargs = self._build_kwargs(is_query=is_query)
            kwargs["input"] = [text]
            response = await litellm.aembedding(**kwargs)
            self._update_telemetry_token_usage(response)
            vector = response.data[0]["embedding"]
            # Truncate exactly as the sync path does. Without this the async
            # path returns the model's native width - 4096 for Qwen3-Embedding
            # - while the configured dimension, and therefore the vector
            # collection, is narrower. Async is the path the server actually
            # uses, so the mismatch reaches storage.
            vector = self._truncate_vector(vector)
            return EmbedResult(dense_vector=vector)

        try:
            return await self._run_with_async_retry(
                _call,
                logger=logger,
                operation_name="LiteLLM async embedding",
            )
        except Exception as e:
            raise RuntimeError(f"LiteLLM embedding failed: {e}") from e

    def _get_micro_batcher(self) -> EmbedMicroBatcher:
        loop = asyncio.get_running_loop()
        batcher = self._micro_batchers.get(loop)
        if batcher is None:
            batcher = EmbedMicroBatcher(
                self._embed_batch_request,
                max_batch_size=self.micro_batch_size,
                max_wait_ms=self.micro_batch_wait_ms,
            )
            self._micro_batchers[loop] = batcher
        return batcher

    async def _embed_batch_request(
        self, texts: Sequence[str], is_query: bool = False
    ) -> List[EmbedResult]:
        """Embed several texts in one provider request, preserving input order."""

        async def _call() -> List[EmbedResult]:
            kwargs = self._build_kwargs(is_query=is_query)
            kwargs["input"] = list(texts)
            response = await litellm.aembedding(**kwargs)
            self._update_telemetry_token_usage(response)
            get_current_telemetry().set("embedding.async.batch_size", len(texts))
            vectors = self._vectors_in_input_order(response.data, expected=len(texts))
            return [
                EmbedResult(dense_vector=self._truncate_vector(vector)) for vector in vectors
            ]

        return await self._run_with_async_retry(
            _call,
            logger=logger,
            operation_name="LiteLLM async batch embedding",
        )

    @staticmethod
    def _vectors_in_input_order(rows: Any, *, expected: int) -> List[Any]:
        """Map response rows back to input positions.

        The OpenAI embeddings contract carries an ``index`` per row and does
        not promise response order. Use the indices when they form a complete
        permutation; otherwise fall back to positional order when the count
        matches, and refuse anything else rather than mis-assign vectors --
        a silently scrambled batch would poison the vector store.
        """
        entries = []
        for position, row in enumerate(rows or []):
            if isinstance(row, dict):
                index, vector = row.get("index"), row.get("embedding")
            else:
                index, vector = getattr(row, "index", None), getattr(row, "embedding", None)
            if vector is None:
                raise RuntimeError(f"Batch embedding row {position} carries no embedding")
            entries.append((index, vector))
        if len(entries) != expected:
            raise RuntimeError(
                f"Batch embedding returned {len(entries)} rows for {expected} inputs"
            )
        indices = [index for index, _ in entries]
        if sorted(indices) == list(range(expected)):
            ordered: List[Any] = [None] * expected
            for index, vector in entries:
                ordered[index] = vector
            return ordered
        return [vector for _, vector in entries]

    async def embed_batch_async(
        self, contents: List[Any], is_query: bool = False
    ) -> List[EmbedResult]:
        """Batch embed via true array requests, chunked to the batch size cap."""
        texts = [c for c in contents]
        if not texts:
            return []
        if not all(isinstance(t, str) for t in texts):
            # Multimodal parts cannot share an array request; fall back.
            return await super().embed_batch_async(contents, is_query=is_query)
        try:
            results: List[EmbedResult] = []
            for start in range(0, len(texts), self.micro_batch_size):
                chunk = texts[start : start + self.micro_batch_size]
                results.extend(await self._embed_batch_request(chunk, is_query=is_query))
            return results
        except Exception as e:
            raise RuntimeError(f"LiteLLM embedding failed: {e}") from e

    def get_dimension(self) -> int:
        """Get embedding dimension.

        Returns:
            int: Vector dimension
        """
        return self._dimension
