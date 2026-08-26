package com.redis.agentmemory.models.common;

import com.fasterxml.jackson.annotation.JsonInclude;
import org.jetbrains.annotations.Nullable;

/**
 * Filter for boolean fields.
 *
 * <p>
 * Example — list the memories a human has pinned:
 * <pre>{@code
 * ListRequest.builder()
 *     .namespace("account-123")
 *     .pinned(BoolFilter.isTrue())
 *     .build()
 * }</pre>
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class BoolFilter {

    @Nullable
    private Boolean eq;

    private BoolFilter() {}

    public static BoolFilter eq(boolean value) {
        BoolFilter f = new BoolFilter();
        f.eq = value;
        return f;
    }

    public static BoolFilter isTrue() {
        return eq(true);
    }

    public static BoolFilter isFalse() {
        return eq(false);
    }

    @Nullable
    public Boolean getEq() { return eq; }

    @Override
    public String toString() {
        return "BoolFilter{eq=" + eq + '}';
    }
}
