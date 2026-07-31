"""LiteLLM embedder batching: array requests, ordering, and routing.

The wiring rules that matter:
- concurrent document embed_async calls share one provider request;
- query embeddings never enter a batch (different latency profile and
  potentially different request parameters via query_param);
- response rows map back to input positions by ``index`` when present,
  because the OpenAI contract does not promise response order;
- a scrambled or short response is an error, never a silent mis-assignment.
"""

import asyncio
from types import SimpleNamespace

import pytest

import openviking.models.embedder.litellm_embedders as mod
from openviking.models.embedder.litellm_embedders import LiteLLMDenseEmbedder


class FakeAembedding:
    """Records every litellm.aembedding call and answers deterministically."""

    def __init__(self):
        self.calls: list[dict] = []
        self.shuffle_indices = False

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        texts = kwargs["input"]
        rows = [
            {"index": i, "embedding": [float(i + 1), 0.0]} for i in range(len(texts))
        ]
        if self.shuffle_indices:
            rows = list(reversed(rows))
        return SimpleNamespace(data=rows, usage={"prompt_tokens": len(texts), "total_tokens": len(texts)})


@pytest.fixture
def fake(monkeypatch):
    fake = FakeAembedding()
    monkeypatch.setattr(mod.litellm, "aembedding", fake)
    return fake


def _embedder(**config):
    return LiteLLMDenseEmbedder(
        model_name="openai/forge-turbo",
        api_key="k",
        api_base="https://example.invalid/v1",
        dimension=2,
        config={"max_retries": 0, "micro_batch_wait_ms": 5.0, **config},
    )


@pytest.mark.asyncio
async def test_concurrent_document_embeds_share_one_provider_call(fake):
    embedder = _embedder()
    results = await asyncio.gather(*(embedder.embed_async(f"doc {i}") for i in range(6)))

    assert len(fake.calls) == 1, "six concurrent documents should be one provider call"
    assert fake.calls[0]["input"] == [f"doc {i}" for i in range(6)]
    # Row i carries [i+1, 0]; caller i must receive exactly row i.
    for i, result in enumerate(results):
        assert result.dense_vector[0] == float(i + 1)


@pytest.mark.asyncio
async def test_shuffled_response_rows_map_back_by_index(fake):
    fake.shuffle_indices = True
    embedder = _embedder()
    results = await asyncio.gather(*(embedder.embed_async(f"doc {i}") for i in range(4)))
    for i, result in enumerate(results):
        assert result.dense_vector[0] == float(i + 1)


@pytest.mark.asyncio
async def test_query_embeddings_bypass_the_batcher(fake):
    embedder = _embedder()
    await asyncio.gather(*(embedder.embed_async(f"q {i}", is_query=True) for i in range(3)))

    assert len(fake.calls) == 3, "queries must not coalesce"
    assert all(len(c["input"]) == 1 for c in fake.calls)


@pytest.mark.asyncio
async def test_disabled_batching_restores_per_call_requests(fake):
    embedder = _embedder(micro_batch_enabled=False)
    await asyncio.gather(*(embedder.embed_async(f"doc {i}") for i in range(3)))
    assert len(fake.calls) == 3


@pytest.mark.asyncio
async def test_embed_batch_async_chunks_to_the_size_cap(fake):
    embedder = _embedder(micro_batch_size=4)
    results = await embedder.embed_batch_async([f"doc {i}" for i in range(10)])

    assert len(results) == 10
    assert [len(c["input"]) for c in fake.calls] == [4, 4, 2]
    # Order survives chunking: chunk-local row 0 is [1.0, 0].
    assert results[0].dense_vector[0] == 1.0
    assert results[4].dense_vector[0] == 1.0


@pytest.mark.asyncio
async def test_short_response_recovers_via_singles(fake, monkeypatch):
    """A provider that mis-counts batch rows degrades to per-text requests.

    The batch response (one row for N inputs) is refused rather than
    mis-assigned; each text is then re-issued alone, where one row for one
    input is well-formed, so callers recover.
    """
    calls: list[int] = []

    async def short(**kwargs):
        calls.append(len(kwargs["input"]))
        return SimpleNamespace(data=[{"index": 0, "embedding": [1.0, 0.0]}], usage=None)

    monkeypatch.setattr(mod.litellm, "aembedding", short)
    embedder = _embedder()
    results = await asyncio.gather(*(embedder.embed_async(f"doc {i}") for i in range(3)))

    assert calls == [3, 1, 1, 1], "one refused batch, then one request per text"
    assert all(r.dense_vector is not None for r in results)


@pytest.mark.asyncio
async def test_short_response_on_a_single_text_is_a_hard_error(fake, monkeypatch):
    async def empty(**kwargs):
        return SimpleNamespace(data=[], usage=None)

    monkeypatch.setattr(mod.litellm, "aembedding", empty)
    embedder = _embedder()
    with pytest.raises(RuntimeError):
        await embedder.embed_async("doc")


@pytest.mark.asyncio
async def test_vectors_are_truncated_to_dimension(fake, monkeypatch):
    async def wide(**kwargs):
        texts = kwargs["input"]
        return SimpleNamespace(
            data=[{"index": i, "embedding": [3.0, 4.0, 9.0, 9.0]} for i in range(len(texts))],
            usage=None,
        )

    monkeypatch.setattr(mod.litellm, "aembedding", wide)
    embedder = _embedder()  # dimension=2
    result = await embedder.embed_async("doc")
    assert len(result.dense_vector) == 2
    # Truncation renormalizes: [3,4] -> unit length.
    assert abs(sum(v * v for v in result.dense_vector) - 1.0) < 1e-9
