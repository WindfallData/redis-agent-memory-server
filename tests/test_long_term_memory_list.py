"""
Tests for filter-only long-term memory listing, against a redis test container
"""

import asyncio
from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient
from ulid import ULID

from agent_memory_server.filters import Namespace, UserId
from agent_memory_server.long_term_memory import (
    list_long_term_memories,
    update_last_accessed,
)
from agent_memory_server.models import MemoryRecord, MemoryTypeEnum
from agent_memory_server.utils.keys import Keys


class _MockEmbeddings:
    """Embeddings stub -- listing never embeds, but indexing does."""

    def __init__(self, dimensions: int):
        self.dimensions = dimensions
        self._dimensions = dimensions
        self.model = "mock-embedding-model"

    async def aembed_documents(self, texts):
        return [[0.1] * self.dimensions for _ in texts]

    async def aembed_query(self, text):
        return [0.1] * self.dimensions


@pytest.fixture
async def listing_db(use_test_redis_connection):
    """A real Redis-backed vector db with mocked embeddings."""
    if use_test_redis_connection is None:
        pytest.skip("Redis not available")

    import agent_memory_server.memory_vector_db_factory as factory

    embeddings = _MockEmbeddings(factory._get_embedding_dimensions())
    db = factory.create_redis_memory_vector_db(embeddings)
    await db._ensure_index()

    previous = factory._memory_vector_db
    factory._memory_vector_db = db
    try:
        yield db
    finally:
        factory._memory_vector_db = previous


def _record(namespace: str, user_id: str, text: str) -> MemoryRecord:
    """Build a record with an explicit ULID id and timestamps."""
    now = datetime.now(UTC)
    return MemoryRecord(
        id=str(ULID()),
        text=text,
        namespace=namespace,
        user_id=user_id,
        memory_type=MemoryTypeEnum.SEMANTIC,
        created_at=now,
        last_accessed=now,
        updated_at=now,
    )


async def _seed(db, namespace: str, user_id: str, count: int) -> list[MemoryRecord]:
    """Index `count` records, one at a time so ULIDs strictly increase."""
    records = []
    for i in range(count):
        record = _record(namespace, user_id, f"memory number {i}")
        records.append(record)
        await db.add_memories([record])
        # ULIDs are millisecond-resolution; keep creation order unambiguous
        await asyncio.sleep(0.002)
    return records


@pytest.mark.asyncio
async def test_listing_is_deterministic_across_identical_calls(listing_db):
    """Same filter, unchanged corpus -> same records in the same order."""
    namespace = f"ns-{ULID()}"
    user_id = f"user-{ULID()}"
    await _seed(listing_db, namespace, user_id, 12)

    first = await list_long_term_memories(
        namespace=Namespace(eq=namespace), limit=12
    )
    second = await list_long_term_memories(
        namespace=Namespace(eq=namespace), limit=12
    )

    assert [m.id for m in first.memories] == [m.id for m in second.memories]
    assert len(first.memories) == 12


@pytest.mark.asyncio
async def test_listing_is_ordered_by_creation(listing_db):
    """Ordering by ULID id must come back as creation order."""
    namespace = f"ns-{ULID()}"
    user_id = f"user-{ULID()}"
    seeded = await _seed(listing_db, namespace, user_id, 10)

    results = await list_long_term_memories(
        namespace=Namespace(eq=namespace), limit=10
    )

    assert [m.id for m in results.memories] == [r.id for r in seeded]


@pytest.mark.asyncio
async def test_paging_visits_every_record_exactly_once(listing_db):
    """Paging a corpus larger than one page terminates and covers it exactly."""
    namespace = f"ns-{ULID()}"
    user_id = f"user-{ULID()}"
    seeded = await _seed(listing_db, namespace, user_id, 25)

    seen: list[str] = []
    offset: int | None = 0
    pages = 0
    while offset is not None:
        page = await list_long_term_memories(
            namespace=Namespace(eq=namespace), limit=10, offset=offset
        )
        assert page.total == 25, "total must count the corpus, not the page"
        seen.extend(m.id for m in page.memories)
        offset = page.next_offset
        pages += 1
        assert pages <= 5, "paging failed to terminate"

    assert pages == 3, "25 records at 10 per page should be 3 pages"
    assert len(seen) == 25, "every record visited"
    assert len(set(seen)) == 25, "no record visited twice"
    assert seen == [r.id for r in seeded], "page sequence follows creation order"


@pytest.mark.asyncio
async def test_next_offset_is_none_when_page_ends_the_corpus(listing_db):
    """An exactly-full final page must still report no further page."""
    namespace = f"ns-{ULID()}"
    user_id = f"user-{ULID()}"
    await _seed(listing_db, namespace, user_id, 10)

    page = await list_long_term_memories(
        namespace=Namespace(eq=namespace), limit=5, offset=5
    )

    assert len(page.memories) == 5
    assert page.total == 10
    assert page.next_offset is None


@pytest.mark.asyncio
async def test_total_is_corpus_count_not_page_window(listing_db):
    """`total` must not saturate at limit+offset the way it used to."""
    namespace = f"ns-{ULID()}"
    user_id = f"user-{ULID()}"
    await _seed(listing_db, namespace, user_id, 15)

    page = await list_long_term_memories(
        namespace=Namespace(eq=namespace), limit=2
    )

    assert page.total == 15
    assert len(page.memories) == 2
    assert page.next_offset == 2


@pytest.mark.asyncio
async def test_listing_does_not_touch_access_tracking(listing_db, use_test_redis_connection):
    """Listing must not record "someone looked at a list containing this."""
    namespace = f"ns-{ULID()}"
    user_id = f"user-{ULID()}"
    seeded = await _seed(listing_db, namespace, user_id, 5)

    redis = use_test_redis_connection

    async def access_state() -> dict[str, tuple[bytes | None, bytes | None]]:
        state = {}
        for record in seeded:
            key = Keys.memory_key(record.id)
            values = await redis.hmget(key, "last_accessed", "access_count")
            state[record.id] = tuple(values)
        return state

    before = await access_state()
    assert all(v[0] is not None for v in before.values()), (
        "fixture sanity: every record should have a last_accessed to begin with"
    )

    # Page through the whole corpus twice
    for _ in range(2):
        offset = 0
        while offset is not None:
            page = await list_long_term_memories(
                namespace=Namespace(eq=namespace), limit=2, offset=offset
            )
            offset = page.next_offset

    after = await access_state()
    assert after == before, "listing must not mutate last_accessed or access_count"

    # Prove the check above can actually observe access tracking: the same
    # records, put through the tracking path search uses, do change.
    updated = await update_last_accessed(
        [r.id for r in seeded], redis_client=redis, min_interval_seconds=0
    )
    assert updated == len(seeded)
    assert await access_state() != before


@pytest.mark.asyncio
async def test_list_endpoint_returns_filtered_page(listing_db):
    """The HTTP route wires filters, paging and the response envelope."""
    from agent_memory_server.main import app

    namespace = f"ns-{ULID()}"
    user_id = f"user-{ULID()}"
    seeded = await _seed(listing_db, namespace, user_id, 7)
    # A second namespace that must not leak into the results
    await _seed(listing_db, f"other-{ULID()}", user_id, 3)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/long-term-memory/list",
            json={"namespace": {"eq": namespace}, "limit": 5},
        )

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["total"] == 7
    assert body["next_offset"] == 5
    assert [m["id"] for m in body["memories"]] == [r.id for r in seeded[:5]]


@pytest.mark.asyncio
async def test_list_endpoint_batch_fetches_by_id(listing_db):
    """`id.any` doubles as a batch fetch of known ids."""
    from agent_memory_server.main import app

    namespace = f"ns-{ULID()}"
    user_id = f"user-{ULID()}"
    seeded = await _seed(listing_db, namespace, user_id, 6)
    wanted = [seeded[1].id, seeded[4].id]

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/long-term-memory/list",
            json={"id": {"any": wanted}, "limit": 10},
        )

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["total"] == 2
    assert sorted(m["id"] for m in body["memories"]) == sorted(wanted)


@pytest.mark.asyncio
async def test_list_endpoint_rejects_out_of_range_limit(listing_db):
    from agent_memory_server.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/long-term-memory/list", json={"limit": 500}
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_blank_text_search_still_lists_without_sorting(listing_db):
    """The blank-text search path must keep its existing unsorted behavior."""
    from agent_memory_server.long_term_memory import search_long_term_memories

    namespace = f"ns-{ULID()}"
    user_id = f"user-{ULID()}"
    seeded = await _seed(listing_db, namespace, user_id, 4)

    results = await search_long_term_memories(
        text="", namespace=Namespace(eq=namespace), limit=10
    )

    # Same records, ordering unconstrained
    assert {m.id for m in results.memories} == {r.id for r in seeded}
    assert results.total == 4


@pytest.mark.asyncio
async def test_count_memories_is_not_capped_by_page_size(listing_db):
    """count_memories reads `total`, so it must reflect the whole corpus."""
    namespace = f"ns-{ULID()}"
    user_id = f"user-{ULID()}"
    await _seed(listing_db, namespace, user_id, 13)

    count = await listing_db.count_memories(namespace=namespace)

    assert count == 13


@pytest.mark.asyncio
async def test_list_memories_rejects_unsupported_sort_field(listing_db):
    """Sort fields are allowlisted, so no caller string reaches the query."""
    with pytest.raises(ValueError, match="Cannot sort by"):
        await listing_db.list_memories(sort_by="text; DROP")


@pytest.mark.asyncio
async def test_user_id_filter_scopes_the_listing(listing_db):
    namespace = f"ns-{ULID()}"
    mine = f"user-{ULID()}"
    theirs = f"user-{ULID()}"
    seeded = await _seed(listing_db, namespace, mine, 4)
    await _seed(listing_db, namespace, theirs, 6)

    results = await list_long_term_memories(
        namespace=Namespace(eq=namespace), user_id=UserId(eq=mine), limit=50
    )

    assert results.total == 4
    assert [m.id for m in results.memories] == [r.id for r in seeded]
