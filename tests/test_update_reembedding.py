"""
Tests for skipping re-embedding on metadata-only updates (APP-1535).

`update_long_term_memory` used to re-embed on every write, so toggling `pinned`
paid an embedding call for a vector that should not change -- and let provider
noise drift the stored embedding over time.
"""

from datetime import UTC, datetime

import numpy as np
import pytest
from ulid import ULID

from agent_memory_server.long_term_memory import update_long_term_memory
from agent_memory_server.models import MemoryRecord, MemoryTypeEnum
from agent_memory_server.utils.keys import Keys


class _CountingEmbeddings:
    """Embeddings stub that records every document batch it was asked to embed."""

    def __init__(self, dimensions: int):
        self.dimensions = dimensions
        self._dimensions = dimensions
        self.model = "mock-embedding-model"
        self.embedded_batches: list[list[str]] = []
        self.fill = 0.1

    async def aembed_documents(self, texts):
        self.embedded_batches.append(list(texts))
        return [[self.fill] * self.dimensions for _ in texts]

    async def aembed_query(self, text):
        return [self.fill] * self.dimensions


@pytest.fixture
async def embeddings(use_test_redis_connection):
    if use_test_redis_connection is None:
        pytest.skip("Redis not available")

    import agent_memory_server.memory_vector_db_factory as factory

    return _CountingEmbeddings(factory._get_embedding_dimensions())


@pytest.fixture
async def counting_db(use_test_redis_connection, embeddings):
    """A real Redis-backed vector db whose embedding calls are observable."""
    import agent_memory_server.memory_vector_db_factory as factory

    db = factory.create_redis_memory_vector_db(embeddings)
    # Another suite may have left `memory_idx` behind on a foreign schema, and an
    # index whose vector dims don't match ours silently drops our writes.
    await db.index.create(overwrite=True)
    await db._ensure_index()

    previous = factory._memory_vector_db
    factory._memory_vector_db = db
    try:
        yield db
    finally:
        factory._memory_vector_db = previous


def _record(text: str, **overrides) -> MemoryRecord:
    now = datetime.now(UTC)
    fields = {
        "id": str(ULID()),
        "text": text,
        "namespace": f"ns-{ULID()}",
        "user_id": "acct-1",
        "memory_type": MemoryTypeEnum.SEMANTIC,
        "created_at": now,
        "last_accessed": now,
        "updated_at": now,
    }
    fields.update(overrides)
    return MemoryRecord(**fields)


async def _stored_vector(redis, memory_id: str) -> bytes:
    return await redis.hget(Keys.memory_key(memory_id), "vector")


@pytest.mark.asyncio
async def test_pinning_does_not_re_embed(
    counting_db, embeddings, use_test_redis_connection
):
    """The ticket's case: `pinned` is not part of the text, so it costs no embedding."""
    record = _record("a claim worth keeping")
    await counting_db.add_memories([record])
    embeddings.embedded_batches.clear()

    updated = await update_long_term_memory(record.id, {"pinned": True})

    assert updated is not None
    assert updated.pinned is True
    assert embeddings.embedded_batches == [], "a metadata-only patch must not embed"


@pytest.mark.asyncio
async def test_pinning_preserves_the_stored_vector(
    counting_db, embeddings, use_test_redis_connection
):
    """Reusing the vector is the point -- the bytes in Redis must be byte-identical."""
    record = _record("a claim worth keeping")
    await counting_db.add_memories([record])
    before = await _stored_vector(use_test_redis_connection, record.id)

    # Any re-embed would now produce a different vector.
    embeddings.fill = 0.9

    updated = await update_long_term_memory(record.id, {"pinned": True})

    assert updated is not None, "the patch must actually land"
    after = await _stored_vector(use_test_redis_connection, record.id)
    assert after == before
    assert np.frombuffer(before, dtype=np.float32)[0] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_changing_text_still_re_embeds(
    counting_db, embeddings, use_test_redis_connection
):
    """The skip is conditional -- new text has to get a new vector."""
    record = _record("the original claim")
    await counting_db.add_memories([record])
    embeddings.embedded_batches.clear()
    embeddings.fill = 0.9

    updated = await update_long_term_memory(record.id, {"text": "a revised claim"})

    assert updated is not None
    assert updated.text == "a revised claim"
    assert embeddings.embedded_batches == [["a revised claim"]]

    vector = await _stored_vector(use_test_redis_connection, record.id)
    assert np.frombuffer(vector, dtype=np.float32)[0] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_metadata_only_update_still_rewrites_indexed_fields(
    counting_db, embeddings, use_test_redis_connection
):
    """Skipping the embedding must not skip the write -- the record still changes."""
    record = _record("a claim worth keeping", pinned=False)
    await counting_db.add_memories([record])

    await update_long_term_memory(
        record.id, {"pinned": True, "topics": ["curation"], "namespace": "moved"}
    )

    from agent_memory_server.long_term_memory import get_long_term_memory_by_id

    reloaded = await get_long_term_memory_by_id(record.id)
    assert reloaded is not None
    assert reloaded.pinned is True
    assert reloaded.topics == ["curation"]
    assert reloaded.namespace == "moved"


@pytest.mark.asyncio
async def test_skip_embedding_flag_reuses_every_stored_vector(counting_db, embeddings):
    """Extraction marks whole batches as processed, so the flag has to cover a batch."""
    kept = _record("unchanged one")
    also_kept = _record("unchanged two")
    await counting_db.add_memories([kept, also_kept])
    embeddings.embedded_batches.clear()

    count = await counting_db.update_memories(
        [
            kept.model_copy(update={"discrete_memory_extracted": "t"}),
            also_kept.model_copy(update={"discrete_memory_extracted": "t"}),
        ],
        skip_embedding=True,
    )

    assert count == 2
    assert embeddings.embedded_batches == []


@pytest.mark.asyncio
async def test_update_memories_embeds_by_default(counting_db, embeddings):
    """Re-embedding stays the default -- a caller has to opt out deliberately."""
    record = _record("still worth embedding")
    await counting_db.add_memories([record])
    embeddings.embedded_batches.clear()

    count = await counting_db.update_memories([record])

    assert count == 1
    assert embeddings.embedded_batches == [["still worth embedding"]]


@pytest.mark.asyncio
async def test_extraction_marks_processed_without_embedding(
    counting_db, embeddings, use_test_redis_connection
):
    """The extraction sweep only flips a flag, so it must not pay for a re-embed."""
    from agent_memory_server.long_term_memory import get_long_term_memory_by_id

    record = _record("a claim awaiting extraction")
    await counting_db.add_memories([record])
    before = await _stored_vector(use_test_redis_connection, record.id)
    embeddings.embedded_batches.clear()
    embeddings.fill = 0.9

    await counting_db.update_memories(
        [record.model_copy(update={"discrete_memory_extracted": "t"})],
        skip_embedding=True,
    )

    assert embeddings.embedded_batches == []
    assert await _stored_vector(use_test_redis_connection, record.id) == before

    reloaded = await get_long_term_memory_by_id(record.id)
    assert reloaded is not None
    assert reloaded.discrete_memory_extracted == "t"
