"""
Tests for durable and queryable pins and provenance (APP-1426).

Two halves of one problem: curation (`pinned`) and citation (`extracted_from`,
`metadata`) must survive the write paths that rewrite records, and must be
expressible as filters so a curation UI doesn't have to fetch everything.
"""

import time
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from ulid import ULID

from agent_memory_server.filters import ExtractedFrom, Namespace, Pinned
from agent_memory_server.llm import ChatCompletionResponse
from agent_memory_server.long_term_memory import (
    _parse_ft_search_docs,
    compact_long_term_memories,
    deduplicate_by_hash,
    deduplicate_by_id,
    deduplicate_by_semantic_search,
    index_long_term_memories,
    list_long_term_memories,
    merge_memories_with_llm,
    update_long_term_memory,
)
from agent_memory_server.models import (
    EditMemoryRecordRequest,
    MemoryRecord,
    MemoryTypeEnum,
)
from agent_memory_server.utils.keys import Keys


class _MockEmbeddings:
    """Embeddings stub -- filtering never embeds, but indexing does."""

    def __init__(self, dimensions: int):
        self.dimensions = dimensions
        self._dimensions = dimensions
        self.model = "mock-embedding-model"

    async def aembed_documents(self, texts):
        return [[0.1] * self.dimensions for _ in texts]

    async def aembed_query(self, text):
        return [0.1] * self.dimensions


@pytest.fixture
async def curation_db(use_test_redis_connection):
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


def _record(namespace: str, text: str, **overrides) -> MemoryRecord:
    now = datetime.now(UTC)
    fields = {
        "id": str(ULID()),
        "text": text,
        "namespace": namespace,
        "user_id": "acct-1",
        "memory_type": MemoryTypeEnum.SEMANTIC,
        "created_at": now,
        "last_accessed": now,
        "updated_at": now,
    }
    fields.update(overrides)
    return MemoryRecord(**fields)


def _merge_response(text: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        content=text,
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        model="gpt-4o-mini",
    )


# ---------------------------------------------------------------------------
# Queryability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pinned_filter_selects_only_pinned_records(curation_db):
    """ "This account's pinned memories" has to be expressible server-side."""
    namespace = f"ns-{ULID()}"
    pinned = _record(namespace, "keep me", pinned=True)
    await curation_db.add_memories(
        [pinned, _record(namespace, "ordinary a"), _record(namespace, "ordinary b")]
    )

    results = await list_long_term_memories(
        namespace=Namespace(eq=namespace), pinned=Pinned(eq=True), limit=10
    )

    assert [m.id for m in results.memories] == [pinned.id]
    assert results.total == 1


@pytest.mark.asyncio
async def test_unpinned_filter_returns_records_written_without_the_field(
    curation_db, use_test_redis_connection
):
    """`pinned=False` is a negation, so records predating the field still match.

    A positive `@pinned:{0}` match would silently drop any record indexed before
    `pinned` joined the schema, because those hashes have no value at all.
    """
    namespace = f"ns-{ULID()}"
    legacy = _record(namespace, "written before pinned existed")
    await curation_db.add_memories([legacy, _record(namespace, "written after")])

    # Simulate the pre-schema record by removing the field entirely
    await use_test_redis_connection.hdel(Keys.memory_key(legacy.id), "pinned")

    results = await list_long_term_memories(
        namespace=Namespace(eq=namespace), pinned=Pinned(eq=False), limit=10
    )

    assert legacy.id in {m.id for m in results.memories}
    assert results.total == 2


@pytest.mark.asyncio
async def test_extracted_from_filter_selects_by_source_handle(curation_db):
    """Removing a document means finding what it produced."""
    namespace = f"ns-{ULID()}"
    from_doc = _record(namespace, "cited claim", extracted_from=["doc-7"])
    await curation_db.add_memories(
        [
            from_doc,
            _record(namespace, "other claim", extracted_from=["doc-9"]),
            _record(namespace, "uncited claim"),
        ]
    )

    results = await list_long_term_memories(
        namespace=Namespace(eq=namespace),
        extracted_from=ExtractedFrom(any=["doc-7"]),
        limit=10,
    )

    assert [m.id for m in results.memories] == [from_doc.id]


@pytest.mark.asyncio
async def test_filters_are_reachable_over_the_list_api(curation_db, client):
    """The filters have to arrive through the request model, not just in Python."""
    namespace = f"ns-{ULID()}"
    pinned = _record(namespace, "pinned and cited", pinned=True, extracted_from=["d-1"])
    await curation_db.add_memories([pinned, _record(namespace, "neither")])

    response = await client.post(
        "/v1/long-term-memory/list",
        json={
            "namespace": {"eq": namespace},
            "pinned": {"eq": True},
            "extracted_from": {"any": ["d-1"]},
            "limit": 10,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert [m["id"] for m in body["memories"]] == [pinned.id]


# ---------------------------------------------------------------------------
# Patchability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extracted_from_is_patchable_and_then_queryable(curation_db):
    """The source handle is set after the fact, so PATCH has to accept it."""
    namespace = f"ns-{ULID()}"
    record = _record(namespace, "a claim needing a citation")
    await curation_db.add_memories([record])

    updated = await update_long_term_memory(record.id, {"extracted_from": ["doc-42"]})

    assert updated is not None
    assert updated.extracted_from == ["doc-42"]

    results = await list_long_term_memories(
        namespace=Namespace(eq=namespace),
        extracted_from=ExtractedFrom(eq="doc-42"),
        limit=10,
    )
    assert [m.id for m in results.memories] == [record.id]


@pytest.mark.asyncio
async def test_patching_an_unknown_field_is_still_rejected(curation_db):
    """Widening the allowlist by one field must not open it up generally."""
    record = _record(f"ns-{ULID()}", "a claim")
    await curation_db.add_memories([record])

    with pytest.raises(ValueError, match="Cannot update fields"):
        await update_long_term_memory(record.id, {"memory_hash": "abc"})


@patch("agent_memory_server.api.long_term_memory.search_long_term_memories")
@pytest.mark.asyncio
async def test_soft_filter_fallback_keeps_pins_and_provenance(mock_search, client):
    """Neither filter is a relevance signal, so the zero-hit retry must keep both.

    The fallback drops relevance filters to preserve recall. Relaxing
    `extracted_from` would quietly widen "what came from this document" to
    "anything that reads a bit like it".
    """
    from agent_memory_server.models import MemoryRecordResults

    mock_search.return_value = MemoryRecordResults(
        memories=[], total=0, next_offset=None
    )

    response = await client.post(
        "/v1/long-term-memory/search",
        json={
            "text": "scarlet dolphin",
            "pinned": {"eq": True},
            "extracted_from": {"eq": "doc-1"},
            # A relaxable filter, so the fallback engages
            "topics": {"eq": "marine biology"},
        },
    )

    assert response.status_code == 200, response.text
    assert mock_search.call_count == 2, "the fallback should have engaged"

    fallback_kwargs = mock_search.call_args_list[1].kwargs
    assert fallback_kwargs["pinned"].eq is True
    assert fallback_kwargs["extracted_from"].eq == "doc-1"
    assert fallback_kwargs.get("topics") is None
    assert "doc-1" not in fallback_kwargs["text"], (
        "a source handle is not a semantic hint"
    )


def test_source_handles_cannot_contain_the_storage_delimiter():
    """Commas are the tag separator, so callers must encode handles first."""
    with pytest.raises(ValueError):
        EditMemoryRecordRequest(extracted_from=["https://x.test/a,b"])


# ---------------------------------------------------------------------------
# Durability: merging
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_carries_curation_and_provenance_forward():
    """A merge builds a new record; the fields it forgets are lost silently."""
    older = datetime.fromtimestamp(int(time.time()) - 500, UTC)
    newer = datetime.fromtimestamp(int(time.time()) - 100, UTC)
    event_older = datetime.now(UTC) - timedelta(days=9)
    event_newer = datetime.now(UTC) - timedelta(days=2)

    memories = [
        MemoryRecord(
            id="1",
            text="A",
            user_id="u",
            namespace="n",
            created_at=older,
            last_accessed=older,
            memory_type=MemoryTypeEnum.SEMANTIC,
            pinned=True,
            metadata={"source": "first", "shared": "old"},
            extracted_from=["doc-1"],
            access_count=3,
            event_date=event_older,
            persisted_at=older,
        ),
        MemoryRecord(
            id="2",
            text="B",
            user_id="u",
            namespace="n",
            created_at=newer,
            last_accessed=newer,
            memory_type=MemoryTypeEnum.SEMANTIC,
            metadata={"shared": "new"},
            extracted_from=["doc-2"],
            access_count=4,
            event_date=event_newer,
        ),
    ]

    with patch(
        "agent_memory_server.long_term_memory.LLMClient.create_chat_completion",
        new_callable=AsyncMock,
        return_value=_merge_response("Merged"),
    ):
        merged = await merge_memories_with_llm(memories)

    assert merged.pinned is True, "a pin held by any member survives the merge"
    assert set(merged.extracted_from) == {"doc-1", "doc-2"}, "citations are unioned"
    assert merged.metadata == {"source": "first", "shared": "new"}, "newest key wins"
    assert merged.access_count == 7, "access counts are summed"
    assert merged.event_date == event_older, "earliest non-null event date"
    assert merged.persisted_at == older, "earliest non-null persistence"


@pytest.mark.asyncio
async def test_pinned_record_is_never_merged_away(curation_db):
    """A human pinned it precisely to stop it being rewritten."""
    namespace = f"ns-{ULID()}"
    pinned = _record(
        namespace,
        "the curated wording",
        pinned=True,
        metadata={"citation": "doc-1"},
        extracted_from=["doc-1"],
    )
    await curation_db.add_memories([pinned, _record(namespace, "the curated wording")])

    with patch(
        "agent_memory_server.long_term_memory.merge_memories_with_llm",
        new_callable=AsyncMock,
    ) as merge:
        result, was_merged = await deduplicate_by_semantic_search(
            memory=pinned, namespace=namespace
        )

    assert was_merged is False
    assert result is pinned
    merge.assert_not_awaited(), "a pinned anchor must not reach the merge at all"


@pytest.mark.asyncio
async def test_pinned_neighbour_is_left_out_of_a_merge(curation_db):
    """Merging around a pinned record must not consume it."""
    namespace = f"ns-{ULID()}"
    pinned = _record(namespace, "shared wording", pinned=True)
    neighbour = _record(namespace, "shared wording")
    await curation_db.add_memories([pinned, neighbour])

    incoming = _record(namespace, "shared wording")

    with patch(
        "agent_memory_server.long_term_memory.LLMClient.create_chat_completion",
        new_callable=AsyncMock,
        return_value=_merge_response("Merged wording"),
    ):
        merged, was_merged = await deduplicate_by_semantic_search(
            memory=incoming, namespace=namespace
        )

    assert was_merged is True, "the unpinned neighbour is still eligible"
    assert merged.pinned is False, "the merge did not inherit the pin"

    surviving = await list_long_term_memories(
        namespace=Namespace(eq=namespace), pinned=Pinned(eq=True), limit=10
    )
    assert [m.id for m in surviving.memories] == [pinned.id], (
        "the pinned record must still be there, untouched"
    )


@pytest.mark.asyncio
async def test_unrelated_write_preserves_a_cited_records_provenance(curation_db):
    """The headline case: semantic dedup fires as a side effect of a normal write.

    `compact_semantic_duplicates` defaults on, so this is the ordinary path, not
    the explicit /compact endpoint.
    """
    namespace = f"ns-{ULID()}"
    cited = _record(
        namespace,
        "the company was founded in 1999",
        metadata={"citation": "page 4"},
        extracted_from=["doc-1"],
    )
    await curation_db.add_memories([cited])

    incoming = _record(namespace, "the company was founded in 1999")

    with patch(
        "agent_memory_server.long_term_memory.LLMClient.create_chat_completion",
        new_callable=AsyncMock,
        return_value=_merge_response("The company was founded in 1999."),
    ):
        await index_long_term_memories([incoming], deduplicate=True)

    surviving = await list_long_term_memories(
        namespace=Namespace(eq=namespace),
        extracted_from=ExtractedFrom(eq="doc-1"),
        limit=10,
    )

    assert surviving.total == 1, "the citation survived the rewrite"
    assert surviving.memories[0].metadata == {"citation": "page 4"}


# ---------------------------------------------------------------------------
# Durability: same-id overwrite and hash duplicates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_id_repost_does_not_clear_curation(curation_db):
    """An idempotent retry of a POST must not unpin or un-cite."""
    namespace = f"ns-{ULID()}"
    stored = _record(
        namespace,
        "original",
        pinned=True,
        metadata={"citation": "page 4"},
        extracted_from=["doc-1"],
    )
    await curation_db.add_memories([stored])

    # The same record as a naive client would re-send it: curation fields absent
    resent = _record(namespace, "original, reworded", id=stored.id)

    deduped, was_overwrite = await deduplicate_by_id(memory=resent, namespace=namespace)

    assert was_overwrite is True
    assert deduped.text == "original, reworded", "the new text still wins"
    assert deduped.pinned is True
    assert deduped.metadata == {"citation": "page 4"}
    assert deduped.extracted_from == ["doc-1"]


@pytest.mark.asyncio
async def test_repost_can_still_replace_curation_it_supplies(curation_db):
    """Preserve-if-absent must not become preserve-always."""
    namespace = f"ns-{ULID()}"
    stored = _record(namespace, "original", extracted_from=["doc-1"])
    await curation_db.add_memories([stored])

    resent = _record(namespace, "original", id=stored.id, extracted_from=["doc-2"])

    deduped, _ = await deduplicate_by_id(memory=resent, namespace=namespace)

    assert deduped.extracted_from == ["doc-2"]


def test_ft_search_reply_parses_two_elements_per_document():
    """The reply is [total, key, fields, key, fields, ...], whatever RETURN asked for."""
    reply = [
        3,
        b"memory:a",
        [b"id_", b"a", b"pinned", b"0"],
        b"memory:b",
        [b"id_", b"b", b"pinned", b"1"],
        b"memory:c",
        [b"id_", b"c", b"pinned", b"0"],
    ]

    docs = _parse_ft_search_docs(reply)

    assert [key for key, _ in docs] == ["memory:a", "memory:b", "memory:c"]
    assert [fields["pinned"] for _, fields in docs] == ["0", "1", "0"]
    assert _parse_ft_search_docs([0]) == []
    assert _parse_ft_search_docs(None) == []


@pytest.mark.asyncio
async def test_hash_duplicate_compaction_keeps_the_pinned_copy(
    curation_db, use_test_redis_connection
):
    """The hash pass deletes rather than merges, so it has to check the pin itself.

    The memory hash covers only text/user/session/namespace/type, so a pinned
    record and an identical unpinned one collide — and the pinned one used to be
    deleted outright whenever it was the older of the two.

    NB: the identifiers here avoid hyphens on purpose. This pass builds its
    FT.AGGREGATE query by string interpolation without escaping tag values, so a
    namespace or session id containing `-` matches nothing and the whole pass
    silently no-ops. That is pre-existing and untouched here.
    """
    namespace = f"ns_{str(ULID()).lower()}"
    session_id = f"sess_{str(ULID()).lower()}"
    shared = {"namespace": namespace, "session_id": session_id}

    pinned = _record(namespace, "identical text", session_id=session_id, pinned=True)
    unpinned = _record(namespace, "identical text", session_id=session_id)
    for record in (pinned, unpinned):
        record.memory_hash = curation_db.generate_memory_hash(record)
    assert pinned.memory_hash == unpinned.memory_hash, "fixture sanity: hashes collide"

    # The pinned record is written first, making it the older of the two
    await curation_db.add_memories([pinned])
    await curation_db.add_memories([unpinned])

    await compact_long_term_memories(
        redis_client=use_test_redis_connection,
        compact_hash_duplicates=True,
        compact_semantic_duplicates=False,
        **shared,
    )

    surviving = await list_long_term_memories(
        namespace=Namespace(eq=namespace), limit=10
    )
    surviving_ids = [m.id for m in surviving.memories]

    assert pinned.id in surviving_ids, "the pinned copy must not be deleted"
    assert unpinned.id not in surviving_ids, "the unpinned duplicate is still compacted"


@pytest.mark.asyncio
async def test_hash_duplicate_compaction_keeps_the_newest_when_none_are_pinned(
    curation_db, use_test_redis_connection
):
    """The pin check must not change behaviour for ordinary duplicates."""
    namespace = f"ns_{str(ULID()).lower()}"
    session_id = f"sess_{str(ULID()).lower()}"

    older = _record(namespace, "identical text", session_id=session_id)
    newer = _record(namespace, "identical text", session_id=session_id)
    for record in (older, newer):
        record.memory_hash = curation_db.generate_memory_hash(record)
    older.last_accessed = datetime.now(UTC) - timedelta(days=1)

    await curation_db.add_memories([older])
    await curation_db.add_memories([newer])

    await compact_long_term_memories(
        redis_client=use_test_redis_connection,
        namespace=namespace,
        session_id=session_id,
        compact_hash_duplicates=True,
        compact_semantic_duplicates=False,
    )

    surviving = await list_long_term_memories(
        namespace=Namespace(eq=namespace), limit=10
    )
    surviving_ids = [m.id for m in surviving.memories]

    assert surviving_ids == [newer.id], "keep the most recently accessed copy"


@pytest.mark.asyncio
async def test_hash_duplicate_write_carries_an_incoming_pin(curation_db):
    """The incoming record is dropped, so its pin has to move to the survivor."""
    namespace = f"ns-{ULID()}"
    stored = _record(namespace, "identical text")
    stored.memory_hash = curation_db.generate_memory_hash(stored)
    await curation_db.add_memories([stored])

    incoming = _record(namespace, "identical text", pinned=True)

    result, was_duplicate = await deduplicate_by_hash(
        memory=incoming, namespace=namespace
    )

    assert was_duplicate is True
    assert result is None, "the incoming duplicate is still dropped"

    pinned = await list_long_term_memories(
        namespace=Namespace(eq=namespace), pinned=Pinned(eq=True), limit=10
    )
    assert [m.id for m in pinned.memories] == [stored.id], (
        "the surviving copy inherited the pin"
    )
