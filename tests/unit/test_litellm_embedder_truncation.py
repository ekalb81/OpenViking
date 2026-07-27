"""The async embedding path must truncate to the configured dimension.

LiteLLMDenseEmbedder is told a dimension because the vector collection is built
at that width, and the provider returns the model's native width - 4096 for
Qwen3-Embedding against a configured 1024. The sync path truncated; the async
path did not, and async is what the server uses. The result was a 4x-too-wide
vector heading for a collection that cannot hold it.

Nothing about that failure is loud: the vector is well-formed and semantically
correct, just the wrong shape, so it surfaces as a storage-layer dimension
mismatch far from the cause.
"""

import pytest

from openviking.models.embedder.litellm_embedders import LiteLLMDenseEmbedder

NATIVE_DIM = 4096
CONFIGURED_DIM = 1024


class _Response:
    """Mimics litellm's embedding response shape."""

    def __init__(self, vector):
        self.data = [{"embedding": vector}]
        self.usage = None


def _embedder(dimension=CONFIGURED_DIM):
    return LiteLLMDenseEmbedder(
        model_name="openrouter/qwen/qwen3-embedding-8b",
        api_key="test-key",
        api_base="https://openrouter.ai/api/v1",
        dimension=dimension,
    )


@pytest.mark.asyncio
async def test_async_truncates_to_the_configured_dimension(monkeypatch):
    native = [0.01 * i for i in range(NATIVE_DIM)]

    async def fake_aembedding(**kwargs):
        del kwargs
        return _Response(native)

    import openviking.models.embedder.litellm_embedders as mod

    monkeypatch.setattr(mod.litellm, "aembedding", fake_aembedding)

    result = await _embedder().embed_async("some document text")

    assert len(result.dense_vector) == CONFIGURED_DIM


@pytest.mark.asyncio
async def test_async_and_sync_agree(monkeypatch):
    """They diverged, which is how this shipped. Pin them together."""
    native = [0.01 * i for i in range(NATIVE_DIM)]

    async def fake_aembedding(**kwargs):
        del kwargs
        return _Response(native)

    def fake_embedding(**kwargs):
        del kwargs
        return _Response(native)

    import openviking.models.embedder.litellm_embedders as mod

    monkeypatch.setattr(mod.litellm, "aembedding", fake_aembedding)
    monkeypatch.setattr(mod.litellm, "embedding", fake_embedding)

    e = _embedder()
    async_vec = (await e.embed_async("text")).dense_vector
    sync_vec = e.embed("text").dense_vector

    assert len(async_vec) == len(sync_vec) == CONFIGURED_DIM
    assert async_vec == sync_vec


@pytest.mark.asyncio
async def test_a_vector_already_at_the_configured_width_is_untouched(monkeypatch):
    # Truncation must not corrupt a provider that already returns the right
    # width - it is a ceiling, not a reshape.
    exact = [0.01 * i for i in range(CONFIGURED_DIM)]

    async def fake_aembedding(**kwargs):
        del kwargs
        return _Response(exact)

    import openviking.models.embedder.litellm_embedders as mod

    monkeypatch.setattr(mod.litellm, "aembedding", fake_aembedding)

    result = await _embedder().embed_async("text")

    assert result.dense_vector == exact


@pytest.mark.asyncio
async def test_a_narrower_vector_is_not_padded(monkeypatch):
    """A provider returning fewer dims than configured is a real misconfiguration.

    Silently padding would hide it and poison the index with zero-filled tails;
    passing it through lets the storage layer's dimension check catch it.
    """
    narrow = [0.01 * i for i in range(512)]

    async def fake_aembedding(**kwargs):
        del kwargs
        return _Response(narrow)

    import openviking.models.embedder.litellm_embedders as mod

    monkeypatch.setattr(mod.litellm, "aembedding", fake_aembedding)

    result = await _embedder().embed_async("text")

    assert len(result.dense_vector) == 512


@pytest.mark.asyncio
async def test_the_truncated_vector_is_renormalized(monkeypatch):
    """Truncating a unit vector shortens it; the index expects unit length.

    Every other embedder routes through truncate_and_normalize, so vectors
    already stored are unit length. Dropping 3/4 of a Qwen3 embedding without
    renormalizing yields a norm near 0.5, and under any non-cosine similarity
    that scores new vectors systematically below existing ones - silently.
    """
    import math

    # A genuine unit vector, so the truncation effect is realistic.
    raw = [1.0] * NATIVE_DIM
    norm = math.sqrt(sum(x * x for x in raw))
    unit = [x / norm for x in raw]

    async def fake_aembedding(**kwargs):
        del kwargs
        return _Response(unit)

    import openviking.models.embedder.litellm_embedders as mod

    monkeypatch.setattr(mod.litellm, "aembedding", fake_aembedding)

    v = (await _embedder().embed_async("text")).dense_vector

    assert len(v) == CONFIGURED_DIM
    got = math.sqrt(sum(x * x for x in v))
    assert abs(got - 1.0) < 1e-6, f"expected unit length after truncation, got {got:.4f}"
