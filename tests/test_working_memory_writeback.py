"""
Background tasks must not roll back a working-memory PUT that lands while they run.

run_delayed_extraction and promote_working_memory_to_long_term read the session,
spend seconds on LLM extraction or indexing, then record what they did on the
messages/memories they read. Writing that back as the whole session they read
restored the session as it was before any PUT made in the meantime, silently
discarding that PUT (its new messages and its ``data``).
"""

from unittest.mock import patch

import pytest
import redis as sync_redis
from ulid import ULID

from agent_memory_server import working_memory as working_memory_module
from agent_memory_server.config import settings
from agent_memory_server.long_term_memory import (
    promote_working_memory_to_long_term,
    run_delayed_extraction,
)
from agent_memory_server.models import (
    MemoryMessage,
    MemoryRecord,
    MemoryTypeEnum,
    WorkingMemory,
)
from agent_memory_server.utils.keys import Keys
from agent_memory_server.working_memory import (
    get_working_memory,
    set_working_memory,
    update_working_memory_items,
)


USER_ID = "test-user"
NAMESPACE = "test"


@pytest.fixture()
def session_id():
    return f"writeback-{ULID()}"


def session(session_id, messages, memories=(), transcript="v1"):
    return WorkingMemory(
        session_id=session_id,
        user_id=USER_ID,
        namespace=NAMESPACE,
        messages=list(messages),
        memories=list(memories),
        data={"transcript": transcript},
        ttl_seconds=3600,
    )


async def load(session_id, redis):
    return await get_working_memory(
        session_id=session_id,
        user_id=USER_ID,
        namespace=NAMESPACE,
        redis_client=redis,
    )


@pytest.mark.asyncio
async def test_delayed_extraction_keeps_put_made_during_extraction(
    requires_redis, session_id
):
    turn_1 = MemoryMessage(id="m1", role="user", content="turn 1")
    turn_2 = MemoryMessage(id="m2", role="user", content="turn 2")
    await set_working_memory(session(session_id, [turn_1]), redis_client=requires_redis)

    async def extract_while_client_puts(**kwargs):
        # the client commits its next turn while the LLM call is in flight
        await set_working_memory(
            session(session_id, [turn_1, turn_2], transcript="v2"),
            redis_client=requires_redis,
        )
        return []

    with patch(
        "agent_memory_server.long_term_memory.extract_memories_from_session_thread",
        side_effect=extract_while_client_puts,
    ):
        await run_delayed_extraction(
            session_id=session_id, namespace=NAMESPACE, user_id=USER_ID
        )

    stored = await load(session_id, requires_redis)
    assert stored.data == {"transcript": "v2"}
    # m1 was extracted; m2 arrived after the read and is left for the next run
    assert [(m.id, m.discrete_memory_extracted) for m in stored.messages] == [
        ("m1", "t"),
        ("m2", "f"),
    ]


@pytest.mark.asyncio
async def test_promotion_keeps_put_made_during_indexing(
    requires_redis, mock_memory_vector_db, session_id
):
    memory = MemoryRecord(
        id="mem-1",
        text="User prefers Redis",
        memory_type=MemoryTypeEnum.SEMANTIC,
        session_id=session_id,
        user_id=USER_ID,
        namespace=NAMESPACE,
    )
    turn_1 = MemoryMessage(id="m1", role="user", content="turn 1")
    turn_2 = MemoryMessage(id="m2", role="user", content="turn 2")
    await set_working_memory(
        session(session_id, [turn_1], [memory]), redis_client=requires_redis
    )

    puts = []

    async def index_while_client_puts(memories, **kwargs):
        if not puts:
            # the client commits its next turn while promotion is indexing
            await set_working_memory(
                session(session_id, [turn_1, turn_2], [memory], transcript="v2"),
                redis_client=requires_redis,
            )
            puts.append(True)

    with (
        patch.object(settings, "index_all_messages_in_long_term_memory", True),
        patch.object(settings, "enable_discrete_memory_extraction", False),
        patch(
            "agent_memory_server.long_term_memory.index_long_term_memories",
            side_effect=index_while_client_puts,
        ),
    ):
        promoted = await promote_working_memory_to_long_term(
            session_id=session_id,
            user_id=USER_ID,
            namespace=NAMESPACE,
            redis_client=requires_redis,
        )

    assert promoted == 2
    assert puts

    stored = await load(session_id, requires_redis)
    assert stored.data == {"transcript": "v2"}
    # the items promotion read are stamped; m2 arrived after the read and is left for the next run
    assert [(m.id, m.persisted_at is not None) for m in stored.messages] == [
        ("m1", True),
        ("m2", False),
    ]
    assert [(m.id, m.persisted_at is not None) for m in stored.memories] == [
        ("mem-1", True),
    ]

    key = Keys.working_memory_key(
        session_id=session_id, user_id=USER_ID, namespace=NAMESPACE
    )
    assert 0 < await requires_redis.ttl(key) <= 3600


@pytest.mark.asyncio
async def test_update_items_skips_items_rewritten_or_removed_since_read(
    requires_redis, session_id
):
    kept = MemoryMessage(id="m1", role="user", content="kept")
    rewritten = MemoryMessage(id="m2", role="assistant", content="original")
    removed = MemoryMessage(id="m3", role="user", content="removed")
    await set_working_memory(
        session(session_id, [kept, rewritten, removed]), redis_client=requires_redis
    )
    read = await load(session_id, requires_redis)

    await set_working_memory(
        session(
            session_id,
            [kept, rewritten.model_copy(update={"content": "edited"})],
            transcript="v2",
        ),
        redis_client=requires_redis,
    )

    updated = await update_working_memory_items(
        session_id=session_id,
        user_id=USER_ID,
        namespace=NAMESPACE,
        messages=[
            (m, m.model_copy(update={"discrete_memory_extracted": "t"}))
            for m in read.messages
        ],
        redis_client=requires_redis,
    )

    assert updated == 1
    stored = await load(session_id, requires_redis)
    assert stored.data == {"transcript": "v2"}
    assert [
        (m.id, m.content, m.discrete_memory_extracted) for m in stored.messages
    ] == [("m1", "kept", "t"), ("m2", "edited", "f")]


@pytest.mark.asyncio
async def test_update_items_rechecks_items_written_between_read_and_exec(
    requires_redis, redis_url, session_id
):
    await set_working_memory(
        session(session_id, [MemoryMessage(id="m1", role="user", content="turn 1")]),
        redis_client=requires_redis,
    )
    read = await load(session_id, requires_redis)

    key = Keys.working_memory_key(
        session_id=session_id, user_id=USER_ID, namespace=NAMESPACE
    )
    other_client = sync_redis.Redis.from_url(redis_url)
    real_model = working_memory_module.MemoryMessage
    compared = []

    def compare_while_client_writes(**raw):
        if not compared:
            # a write lands after the update's read but before its EXEC
            other_client.json().set(key, "$.messages[0].content", "edited")
        compared.append(raw["content"])
        return real_model(**raw)

    with patch.object(
        working_memory_module, "MemoryMessage", compare_while_client_writes
    ):
        updated = await update_working_memory_items(
            session_id=session_id,
            user_id=USER_ID,
            namespace=NAMESPACE,
            messages=[
                (m, m.model_copy(update={"discrete_memory_extracted": "t"}))
                for m in read.messages
            ],
            redis_client=requires_redis,
        )
    other_client.close()

    # the first attempt aborted on the WATCH; the retry saw the edit and skipped the item
    assert compared == ["turn 1", "edited"]
    assert updated == 0
    stored = await load(session_id, requires_redis)
    assert [
        (m.id, m.content, m.discrete_memory_extracted) for m in stored.messages
    ] == [("m1", "edited", "f")]
