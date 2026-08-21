package com.redis.agentmemory.models.longtermemory;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.redis.agentmemory.models.common.TagFilter;
import org.jetbrains.annotations.Nullable;

import java.util.List;

/** Request payload for long-term memory listing operations. */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ListRequest {

    @Nullable
    @JsonProperty("session_id")
    private TagFilter sessionId;

    @Nullable
    private TagFilter namespace;

    @Nullable
    private TagFilter topics;

    @Nullable
    private TagFilter entities;

    @Nullable
    @JsonProperty("user_id")
    private TagFilter userId;

    @Nullable
    @JsonProperty("memory_type")
    private TagFilter memoryType;

    @Nullable
    @JsonProperty("extraction_strategy")
    private TagFilter extractionStrategy;

    @Nullable
    @JsonProperty("memory_hash")
    private TagFilter memoryHash;

    @Nullable
    private TagFilter id;

    @Nullable
    @JsonProperty("discrete_memory_extracted")
    private TagFilter discreteMemoryExtracted;

    private int limit = 10;

    private int offset = 0;

    public ListRequest() {
    }

    @Nullable
    public TagFilter getSessionId() {
        return sessionId;
    }

    public void setSessionId(@Nullable TagFilter sessionId) {
        this.sessionId = sessionId;
    }

    @Nullable
    public TagFilter getNamespace() {
        return namespace;
    }

    public void setNamespace(@Nullable TagFilter namespace) {
        this.namespace = namespace;
    }

    @Nullable
    public TagFilter getTopics() {
        return topics;
    }

    public void setTopics(@Nullable TagFilter topics) {
        this.topics = topics;
    }

    @Nullable
    public TagFilter getEntities() {
        return entities;
    }

    public void setEntities(@Nullable TagFilter entities) {
        this.entities = entities;
    }

    @Nullable
    public TagFilter getUserId() {
        return userId;
    }

    public void setUserId(@Nullable TagFilter userId) {
        this.userId = userId;
    }

    @Nullable
    public TagFilter getMemoryType() {
        return memoryType;
    }

    public void setMemoryType(@Nullable TagFilter memoryType) {
        this.memoryType = memoryType;
    }

    @Nullable
    public TagFilter getExtractionStrategy() {
        return extractionStrategy;
    }

    public void setExtractionStrategy(@Nullable TagFilter extractionStrategy) {
        this.extractionStrategy = extractionStrategy;
    }

    @Nullable
    public TagFilter getMemoryHash() {
        return memoryHash;
    }

    public void setMemoryHash(@Nullable TagFilter memoryHash) {
        this.memoryHash = memoryHash;
    }

    @Nullable
    public TagFilter getId() {
        return id;
    }

    public void setId(@Nullable TagFilter id) {
        this.id = id;
    }

    @Nullable
    public TagFilter getDiscreteMemoryExtracted() {
        return discreteMemoryExtracted;
    }

    public void setDiscreteMemoryExtracted(@Nullable TagFilter discreteMemoryExtracted) {
        this.discreteMemoryExtracted = discreteMemoryExtracted;
    }

    public int getLimit() {
        return limit;
    }

    public void setLimit(int limit) {
        this.limit = limit;
    }

    public int getOffset() {
        return offset;
    }

    public void setOffset(int offset) {
        this.offset = offset;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private final ListRequest request = new ListRequest();

        public Builder sessionId(@Nullable String sessionId) {
            request.sessionId = sessionId != null ? TagFilter.eq(sessionId) : null;
            return this;
        }

        public Builder sessionId(@Nullable TagFilter sessionId) {
            request.sessionId = sessionId;
            return this;
        }

        public Builder namespace(@Nullable String namespace) {
            request.namespace = namespace != null ? TagFilter.eq(namespace) : null;
            return this;
        }

        public Builder namespace(@Nullable TagFilter namespace) {
            request.namespace = namespace;
            return this;
        }

        public Builder topics(@Nullable List<String> topics) {
            var present = topics != null && !topics.isEmpty();
            request.topics = present ? TagFilter.any(topics) : null;
            return this;
        }

        public Builder topics(@Nullable TagFilter topics) {
            request.topics = topics;
            return this;
        }

        public Builder entities(@Nullable List<String> entities) {
            var present = entities != null && !entities.isEmpty();
            request.entities = present ? TagFilter.any(entities) : null;
            return this;
        }

        public Builder entities(@Nullable TagFilter entities) {
            request.entities = entities;
            return this;
        }

        public Builder userId(@Nullable String userId) {
            request.userId = userId != null ? TagFilter.eq(userId) : null;
            return this;
        }

        public Builder userId(@Nullable TagFilter userId) {
            request.userId = userId;
            return this;
        }

        public Builder memoryType(@Nullable String memoryType) {
            request.memoryType = memoryType != null ? TagFilter.eq(memoryType) : null;
            return this;
        }

        public Builder memoryType(@Nullable TagFilter memoryType) {
            request.memoryType = memoryType;
            return this;
        }

        public Builder extractionStrategy(@Nullable String extractionStrategy) {
            request.extractionStrategy = extractionStrategy != null ? TagFilter.eq(extractionStrategy) : null;
            return this;
        }

        public Builder extractionStrategy(@Nullable TagFilter extractionStrategy) {
            request.extractionStrategy = extractionStrategy;
            return this;
        }

        public Builder memoryHash(@Nullable String memoryHash) {
            request.memoryHash = memoryHash != null ? TagFilter.eq(memoryHash) : null;
            return this;
        }

        public Builder memoryHash(@Nullable TagFilter memoryHash) {
            request.memoryHash = memoryHash;
            return this;
        }

        /** Filter to a single memory id. */
        public Builder id(@Nullable String id) {
            request.id = id != null ? TagFilter.eq(id) : null;
            return this;
        }

        /** Filter to a set of memory ids, i.e. batch fetch by id. */
        public Builder ids(@Nullable List<String> ids) {
            var present = ids != null && !ids.isEmpty();
            request.id = present ? TagFilter.any(ids) : null;
            return this;
        }

        public Builder id(@Nullable TagFilter id) {
            request.id = id;
            return this;
        }

        public Builder discreteMemoryExtracted(@Nullable String discreteMemoryExtracted) {
            request.discreteMemoryExtracted =
                    discreteMemoryExtracted != null ? TagFilter.eq(discreteMemoryExtracted) : null;
            return this;
        }

        public Builder discreteMemoryExtracted(@Nullable TagFilter discreteMemoryExtracted) {
            request.discreteMemoryExtracted = discreteMemoryExtracted;
            return this;
        }

        public Builder limit(int limit) {
            request.limit = limit;
            return this;
        }

        public Builder offset(int offset) {
            request.offset = offset;
            return this;
        }

        public ListRequest build() {
            return request;
        }
    }

    @Override
    public String toString() {
        return "ListRequest{"
                + "sessionId=" + sessionId
                + ", namespace=" + namespace
                + ", topics=" + topics
                + ", entities=" + entities
                + ", userId=" + userId
                + ", memoryType=" + memoryType
                + ", extractionStrategy=" + extractionStrategy
                + ", memoryHash=" + memoryHash
                + ", id=" + id
                + ", discreteMemoryExtracted=" + discreteMemoryExtracted
                + ", limit=" + limit
                + ", offset=" + offset
                + '}';
    }
}
