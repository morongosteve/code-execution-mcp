"""Tests for the _ModelRegistry and _ModelEntry classes in huggingface_tools.py."""

import time
from unittest.mock import patch

import pytest

from huggingface_tools import _ModelEntry, _ModelRegistry

# ============================================================================
# _ModelEntry unit tests
# ============================================================================


class TestModelEntry:
    """Basic smoke tests for the metadata wrapper."""

    def test_initial_state(self):
        entry = _ModelEntry(model="obj", ttl=60, backend="api", repo_id="r/1")
        assert entry.model == "obj"
        assert entry.ttl == 60
        assert entry.backend == "api"
        assert entry.repo_id == "r/1"
        # loaded_at and last_accessed should be very close to now
        assert entry.age_seconds < 1.0
        assert not entry.is_expired

    def test_touch_updates_last_accessed(self):
        entry = _ModelEntry(model="obj", ttl=3600, backend="api", repo_id="r/1")
        first_access = entry.last_accessed
        # Advance monotonic clock slightly
        time.sleep(0.01)
        entry.touch()
        assert entry.last_accessed > first_access

    def test_is_expired_with_mocked_time(self):
        """Verify TTL expiration using monkeypatched time.monotonic."""
        base_time = 1000.0
        with patch("huggingface_tools.time") as mock_time:
            # Construction at t=1000
            mock_time.monotonic.return_value = base_time
            entry = _ModelEntry(model="obj", ttl=10, backend="api", repo_id="r/1")

            # At t=1005 (5s elapsed, TTL=10) -- not expired
            mock_time.monotonic.return_value = base_time + 5
            assert not entry.is_expired

            # At t=1011 (11s elapsed, TTL=10) -- expired
            mock_time.monotonic.return_value = base_time + 11
            assert entry.is_expired

    def test_age_seconds_with_mocked_time(self):
        base_time = 5000.0
        with patch("huggingface_tools.time") as mock_time:
            mock_time.monotonic.return_value = base_time
            entry = _ModelEntry(model="x", ttl=3600, backend="local", repo_id="r/2")

            mock_time.monotonic.return_value = base_time + 42.5
            assert abs(entry.age_seconds - 42.5) < 0.01

    def test_is_expired_exactly_at_boundary(self):
        """At exactly TTL seconds, should NOT be expired (uses > not >=)."""
        base_time = 2000.0
        with patch("huggingface_tools.time") as mock_time:
            mock_time.monotonic.return_value = base_time
            entry = _ModelEntry(model="obj", ttl=10, backend="api", repo_id="r/1")

            # At exactly TTL boundary
            mock_time.monotonic.return_value = base_time + 10
            assert not entry.is_expired

            # Just past TTL boundary
            mock_time.monotonic.return_value = base_time + 10.001
            assert entry.is_expired

    def test_touch_resets_expiration_clock(self):
        """After touch(), the TTL countdown restarts from the new last_accessed."""
        base_time = 3000.0
        with patch("huggingface_tools.time") as mock_time:
            mock_time.monotonic.return_value = base_time
            entry = _ModelEntry(model="obj", ttl=10, backend="api", repo_id="r/1")

            # At t+8, touch
            mock_time.monotonic.return_value = base_time + 8
            entry.touch()

            # At t+8+9 = t+17, should not be expired (9s < 10 TTL since last touch)
            mock_time.monotonic.return_value = base_time + 17
            assert not entry.is_expired

            # At t+8+11 = t+19, expired (11s > 10 TTL since last touch)
            mock_time.monotonic.return_value = base_time + 19
            assert entry.is_expired

    def test_age_seconds_unaffected_by_touch(self):
        """age_seconds measures from loaded_at, not from last_accessed."""
        base_time = 4000.0
        with patch("huggingface_tools.time") as mock_time:
            mock_time.monotonic.return_value = base_time
            entry = _ModelEntry(model="obj", ttl=3600, backend="api", repo_id="r/1")

            mock_time.monotonic.return_value = base_time + 50
            entry.touch()

            mock_time.monotonic.return_value = base_time + 100
            assert abs(entry.age_seconds - 100.0) < 0.01

    def test_slots_prevent_arbitrary_attributes(self):
        """_ModelEntry uses __slots__, so arbitrary attrs should raise."""
        entry = _ModelEntry(model="obj", ttl=60, backend="api", repo_id="r/1")
        with pytest.raises(AttributeError):
            entry.nonexistent_attr = "boom"

    def test_model_stores_arbitrary_objects(self):
        """The model field can store any Python object."""
        obj = {"key": [1, 2, 3]}
        entry = _ModelEntry(model=obj, ttl=60, backend="api", repo_id="r/1")
        assert entry.model is obj


# ============================================================================
# _ModelRegistry -- basic operations
# ============================================================================


class TestRegistryBasicOps:
    """Tests for put / get / remove / __contains__ / __len__."""

    def test_empty_registry(self, fresh_registry):
        reg = fresh_registry(max_models=3)
        assert len(reg) == 0
        assert "anything" not in reg
        assert reg.get("anything") is None
        assert reg.keys() == []

    def test_put_and_get(self, fresh_registry):
        reg = fresh_registry(max_models=5)
        evicted = reg.put("m1", "obj1", ttl=3600, backend="api", repo_id="repo/1")
        assert evicted == []
        assert "m1" in reg
        assert len(reg) == 1
        assert reg.get("m1") == "obj1"

    def test_put_returns_empty_eviction_list_when_room(self, fresh_registry):
        reg = fresh_registry(max_models=3)
        for i in range(3):
            evicted = reg.put(f"m{i}", f"obj{i}", ttl=3600, backend="api", repo_id=f"r/{i}")
            assert evicted == []
        assert len(reg) == 3

    def test_get_returns_none_for_missing_key(self, populated_registry):
        assert populated_registry.get("nonexistent") is None

    def test_contains_true_for_present_key(self, populated_registry):
        assert "model-0" in populated_registry

    def test_contains_false_for_missing_key(self, populated_registry):
        assert "model-99" not in populated_registry

    def test_len_reflects_entries(self, populated_registry):
        # populated_registry has 3 models in a capacity-5 registry
        assert len(populated_registry) == 3

    def test_remove_existing_key_returns_true(self, populated_registry):
        assert populated_registry.remove("model-0") is True
        assert "model-0" not in populated_registry
        assert len(populated_registry) == 2

    def test_remove_missing_key_returns_false(self, populated_registry):
        assert populated_registry.remove("ghost") is False

    def test_remove_idempotent(self, populated_registry):
        assert populated_registry.remove("model-1") is True
        assert populated_registry.remove("model-1") is False

    def test_keys_returns_current_keys(self, populated_registry):
        keys = populated_registry.keys()
        assert set(keys) == {"model-0", "model-1", "model-2"}

    def test_put_multiple_then_get_each(self, fresh_registry):
        reg = fresh_registry(max_models=5)
        for i in range(5):
            reg.put(f"k{i}", f"v{i}", ttl=3600, backend="api", repo_id=f"r/{i}")
        for i in range(5):
            assert reg.get(f"k{i}") == f"v{i}"

    def test_remove_all_results_in_empty_registry(self, populated_registry):
        for key in ["model-0", "model-1", "model-2"]:
            populated_registry.remove(key)
        assert len(populated_registry) == 0
        assert populated_registry.keys() == []


# ============================================================================
# _ModelRegistry -- TTL expiration
# ============================================================================


class TestRegistryTTLExpiration:
    """Verify that models expire after their TTL elapses."""

    def test_expired_models_evicted_on_get(self, fresh_registry):
        """put() a model with short TTL, advance time, get() should return None."""
        base = 1000.0
        with patch("huggingface_tools.time") as mock_time:
            mock_time.monotonic.return_value = base
            reg = fresh_registry(max_models=5)
            reg.put("short", "obj-short", ttl=10, backend="api", repo_id="r/s")

            # Before expiration
            mock_time.monotonic.return_value = base + 5
            assert reg.get("short") == "obj-short"

            # After expiration (> 10s since last access at t=base+5)
            mock_time.monotonic.return_value = base + 5 + 11
            assert reg.get("short") is None
            assert len(reg) == 0

    def test_expired_models_evicted_on_keys(self, fresh_registry):
        base = 2000.0
        with patch("huggingface_tools.time") as mock_time:
            mock_time.monotonic.return_value = base
            reg = fresh_registry(max_models=5)
            reg.put("a", "obj-a", ttl=5, backend="api", repo_id="r/a")
            reg.put("b", "obj-b", ttl=100, backend="api", repo_id="r/b")

            # Advance past a's TTL but not b's
            mock_time.monotonic.return_value = base + 6
            keys = reg.keys()
            assert "a" not in keys
            assert "b" in keys
            assert len(reg) == 1

    def test_expired_models_evicted_on_put(self, fresh_registry):
        """Expired entries should be cleaned before capacity check on put()."""
        base = 3000.0
        with patch("huggingface_tools.time") as mock_time:
            mock_time.monotonic.return_value = base
            reg = fresh_registry(max_models=2)
            reg.put("a", "obj-a", ttl=5, backend="api", repo_id="r/a")
            reg.put("b", "obj-b", ttl=5, backend="api", repo_id="r/b")
            assert len(reg) == 2

            # Advance past TTL so both expire
            mock_time.monotonic.return_value = base + 6
            # This put should first evict the expired ones, then add the new one
            reg.put("c", "obj-c", ttl=100, backend="api", repo_id="r/c")
            # The expired entries are cleaned silently by _evict_expired,
            # not returned by put() (put only returns LRU evictions)
            assert len(reg) == 1
            assert reg.get("c") == "obj-c"

    def test_touch_resets_ttl_clock(self, fresh_registry):
        """Accessing a model via get() should reset the TTL countdown."""
        base = 4000.0
        with patch("huggingface_tools.time") as mock_time:
            mock_time.monotonic.return_value = base
            reg = fresh_registry(max_models=5)
            reg.put("m", "obj", ttl=10, backend="api", repo_id="r/m")

            # At t+8, touch via get()
            mock_time.monotonic.return_value = base + 8
            assert reg.get("m") == "obj"  # touch happens

            # At t+8+9 = t+17, should still be alive (9s since last touch < 10 TTL)
            mock_time.monotonic.return_value = base + 17
            assert reg.get("m") == "obj"

            # At t+17+11 = t+28, now expired (11s since last touch at t+17)
            mock_time.monotonic.return_value = base + 28
            assert reg.get("m") is None

    def test_mixed_ttl_only_expired_evicted(self, fresh_registry):
        """With different TTLs, only the truly expired entries are evicted."""
        base = 5000.0
        with patch("huggingface_tools.time") as mock_time:
            mock_time.monotonic.return_value = base
            reg = fresh_registry(max_models=5)
            reg.put("short", "obj-s", ttl=5, backend="api", repo_id="r/s")
            reg.put("medium", "obj-m", ttl=50, backend="api", repo_id="r/m")
            reg.put("long", "obj-l", ttl=500, backend="api", repo_id="r/l")

            mock_time.monotonic.return_value = base + 10
            keys = reg.keys()
            assert "short" not in keys
            assert "medium" in keys
            assert "long" in keys

    def test_evict_expired_returns_evicted_keys(self, fresh_registry):
        """_evict_expired() returns the list of evicted keys."""
        base = 6000.0
        with patch("huggingface_tools.time") as mock_time:
            mock_time.monotonic.return_value = base
            reg = fresh_registry(max_models=5)
            reg.put("a", "obj-a", ttl=5, backend="api", repo_id="r/a")
            reg.put("b", "obj-b", ttl=5, backend="api", repo_id="r/b")
            reg.put("c", "obj-c", ttl=100, backend="api", repo_id="r/c")

            mock_time.monotonic.return_value = base + 10
            evicted = reg._evict_expired()
            assert set(evicted) == {"a", "b"}
            assert len(reg) == 1


# ============================================================================
# _ModelRegistry -- LRU eviction
# ============================================================================


class TestRegistryLRUEviction:
    """When the registry is at capacity, the least-recently-accessed model is evicted."""

    def test_lru_evicts_oldest_on_put(self, fresh_registry):
        """Fill registry to capacity, put one more -- the oldest should be evicted."""
        base = 5000.0
        with patch("huggingface_tools.time") as mock_time:
            mock_time.monotonic.return_value = base
            reg = fresh_registry(max_models=3)

            # Insert 3 models at slightly different times
            for i in range(3):
                mock_time.monotonic.return_value = base + i
                reg.put(f"m{i}", f"obj{i}", ttl=3600, backend="api", repo_id=f"r/{i}")

            assert len(reg) == 3
            # m0 was last accessed at t=base (oldest)
            # Adding m3 should evict m0
            mock_time.monotonic.return_value = base + 10
            evicted = reg.put("m3", "obj3", ttl=3600, backend="api", repo_id="r/3")
            assert evicted == ["m0"]
            assert "m0" not in reg
            assert "m3" in reg
            assert len(reg) == 3

    def test_get_touch_changes_eviction_order(self, fresh_registry):
        """Touching a model via get() should prevent it from being LRU-evicted."""
        base = 6000.0
        with patch("huggingface_tools.time") as mock_time:
            mock_time.monotonic.return_value = base
            reg = fresh_registry(max_models=3)

            for i in range(3):
                mock_time.monotonic.return_value = base + i
                reg.put(f"m{i}", f"obj{i}", ttl=3600, backend="api", repo_id=f"r/{i}")

            # Touch m0 (the oldest) so m1 becomes the LRU
            mock_time.monotonic.return_value = base + 10
            reg.get("m0")

            # Now add m3 -- m1 should be evicted (it's the least recently accessed)
            mock_time.monotonic.return_value = base + 11
            evicted = reg.put("m3", "obj3", ttl=3600, backend="api", repo_id="r/3")
            assert evicted == ["m1"]
            assert "m1" not in reg
            assert "m0" in reg

    def test_put_same_key_does_not_evict(self, fresh_registry):
        """Re-putting the same key at capacity should overwrite, not evict another."""
        reg = fresh_registry(max_models=2)
        reg.put("a", "obj-a", ttl=3600, backend="api", repo_id="r/a")
        reg.put("b", "obj-b", ttl=3600, backend="api", repo_id="r/b")
        assert len(reg) == 2

        # Re-put "a" -- no eviction needed since "a" is already counted
        evicted = reg.put("a", "obj-a-v2", ttl=3600, backend="api", repo_id="r/a-v2")
        assert evicted == []
        assert len(reg) == 2
        assert reg.get("a") == "obj-a-v2"

    def test_evict_oldest_on_empty_returns_none(self, fresh_registry):
        """_evict_oldest() on empty registry returns None."""
        reg = fresh_registry(max_models=5)
        result = reg._evict_oldest()
        assert result is None


# ============================================================================
# _ModelRegistry -- max concurrent cap
# ============================================================================


class TestRegistryMaxConcurrentCap:
    """Verify the cap constants for models (5) and embeddings (3)."""

    def test_model_cap_is_five(self):
        from huggingface_tools import MAX_CONCURRENT_MODELS
        assert MAX_CONCURRENT_MODELS == 5

    def test_embedding_cap_is_three(self):
        from huggingface_tools import MAX_CONCURRENT_EMBEDDINGS
        assert MAX_CONCURRENT_EMBEDDINGS == 3

    def test_default_ttl_is_3600(self):
        from huggingface_tools import DEFAULT_TTL_SECONDS
        assert DEFAULT_TTL_SECONDS == 3600

    def test_cannot_exceed_max_without_eviction(self, fresh_registry):
        """Adding models beyond capacity always triggers eviction to stay at max."""
        reg = fresh_registry(max_models=3)
        for i in range(3):
            reg.put(f"m{i}", f"obj{i}", ttl=3600, backend="api", repo_id=f"r/{i}")

        # Adding 4th, 5th -- each evicts the LRU
        evicted_4 = reg.put("m3", "obj3", ttl=3600, backend="api", repo_id="r/3")
        assert len(evicted_4) == 1
        assert len(reg) == 3

        evicted_5 = reg.put("m4", "obj4", ttl=3600, backend="api", repo_id="r/4")
        assert len(evicted_5) == 1
        assert len(reg) == 3

    def test_global_model_registry_has_cap_five(self):
        from huggingface_tools import _model_registry
        assert _model_registry._max_models == 5

    def test_global_embedding_registry_has_cap_three(self):
        from huggingface_tools import _embedding_registry
        assert _embedding_registry._max_models == 3


# ============================================================================
# _ModelRegistry -- get() touches access time
# ============================================================================


class TestRegistryGetTouches:
    """Verify that get() updates the last_accessed timestamp."""

    def test_get_updates_last_accessed(self, fresh_registry):
        base = 7000.0
        with patch("huggingface_tools.time") as mock_time:
            mock_time.monotonic.return_value = base
            reg = fresh_registry(max_models=5)
            reg.put("m", "obj", ttl=3600, backend="api", repo_id="r/m")

            # Record initial access time
            initial_access = reg._entries["m"].last_accessed

            mock_time.monotonic.return_value = base + 50
            reg.get("m")

            assert reg._entries["m"].last_accessed == base + 50
            assert reg._entries["m"].last_accessed > initial_access

    def test_get_nonexistent_key_does_not_create_entry(self, fresh_registry):
        reg = fresh_registry(max_models=5)
        reg.get("nope")
        assert len(reg) == 0


# ============================================================================
# _ModelRegistry -- put() returns evicted keys
# ============================================================================


class TestRegistryPutEviction:
    """put() should return a list of keys that were LRU-evicted."""

    def test_put_returns_empty_list_when_room(self, fresh_registry):
        reg = fresh_registry(max_models=5)
        evicted = reg.put("a", "obj", ttl=3600, backend="api", repo_id="r/a")
        assert evicted == []

    def test_put_returns_evicted_keys_when_full(self, fresh_registry):
        base = 8000.0
        with patch("huggingface_tools.time") as mock_time:
            mock_time.monotonic.return_value = base
            reg = fresh_registry(max_models=2)

            mock_time.monotonic.return_value = base
            reg.put("a", "obj-a", ttl=3600, backend="api", repo_id="r/a")
            mock_time.monotonic.return_value = base + 1
            reg.put("b", "obj-b", ttl=3600, backend="api", repo_id="r/b")

            mock_time.monotonic.return_value = base + 2
            evicted = reg.put("c", "obj-c", ttl=3600, backend="api", repo_id="r/c")
            assert evicted == ["a"]  # a was oldest

    def test_put_can_evict_multiple_if_cap_is_one(self):
        """Edge case: capacity=1, already has one entry -- adding another evicts it."""
        reg = _ModelRegistry(max_models=1)
        reg.put("a", "obj-a", ttl=3600, backend="api", repo_id="r/a")
        evicted = reg.put("b", "obj-b", ttl=3600, backend="api", repo_id="r/b")
        assert evicted == ["a"]
        assert len(reg) == 1
        assert "b" in reg


# ============================================================================
# _ModelRegistry -- items_info()
# ============================================================================


class TestRegistryItemsInfo:
    """items_info() should return properly-structured metadata dicts."""

    def test_items_info_empty_registry(self, fresh_registry):
        reg = fresh_registry(max_models=5)
        assert reg.items_info() == {}

    def test_items_info_contains_expected_keys(self, fresh_registry):
        base = 9000.0
        with patch("huggingface_tools.time") as mock_time:
            mock_time.monotonic.return_value = base
            reg = fresh_registry(max_models=5)
            reg.put("m", "a-string-object", ttl=100, backend="local", repo_id="repo/m")

            mock_time.monotonic.return_value = base + 10
            info = reg.items_info()

        assert "m" in info
        meta = info["m"]
        assert meta["type"] == "str"
        assert meta["repo_id"] == "repo/m"
        assert meta["backend"] == "local"
        assert isinstance(meta["age_seconds"], float)
        assert isinstance(meta["ttl_remaining"], float)

    def test_items_info_ttl_remaining_decreases(self, fresh_registry):
        base = 10000.0
        with patch("huggingface_tools.time") as mock_time:
            mock_time.monotonic.return_value = base
            reg = fresh_registry(max_models=5)
            reg.put("m", "obj", ttl=100, backend="api", repo_id="r/m")

            mock_time.monotonic.return_value = base + 30
            info = reg.items_info()
            # TTL remaining should be approximately 100 - 30 = 70
            assert 69.0 <= info["m"]["ttl_remaining"] <= 71.0

    def test_items_info_ttl_remaining_clamped_to_zero(self, fresh_registry):
        """When model is nearly expired, ttl_remaining should be >= 0 (clamped)."""
        base = 11000.0
        with patch("huggingface_tools.time") as mock_time:
            mock_time.monotonic.return_value = base
            reg = fresh_registry(max_models=5)
            reg.put("m", "obj", ttl=10, backend="api", repo_id="r/m")

            # Advance to exactly the TTL boundary
            mock_time.monotonic.return_value = base + 10
            info = reg.items_info()
            assert info["m"]["ttl_remaining"] == 0.0

    def test_items_info_excludes_expired(self, fresh_registry):
        """items_info() calls _evict_expired internally, so expired entries vanish."""
        base = 12000.0
        with patch("huggingface_tools.time") as mock_time:
            mock_time.monotonic.return_value = base
            reg = fresh_registry(max_models=5)
            reg.put("alive", "obj-a", ttl=100, backend="api", repo_id="r/a")
            reg.put("dead", "obj-d", ttl=5, backend="api", repo_id="r/d")

            mock_time.monotonic.return_value = base + 6
            info = reg.items_info()
            assert "dead" not in info
            assert "alive" in info

    def test_items_info_type_uses_class_name(self, fresh_registry):
        """items_info 'type' field reflects the model object's class name."""
        reg = fresh_registry(max_models=5)
        reg.put("dict-model", {"a": 1}, ttl=100, backend="api", repo_id="r/d")
        reg.put("list-model", [1, 2, 3], ttl=100, backend="api", repo_id="r/l")
        info = reg.items_info()
        assert info["dict-model"]["type"] == "dict"
        assert info["list-model"]["type"] == "list"

    def test_items_info_multiple_entries(self, fresh_registry):
        """items_info returns info for all non-expired entries."""
        base = 13000.0
        with patch("huggingface_tools.time") as mock_time:
            mock_time.monotonic.return_value = base
            reg = fresh_registry(max_models=5)
            for i in range(3):
                reg.put(f"m{i}", f"obj{i}", ttl=3600, backend="api", repo_id=f"r/{i}")

            mock_time.monotonic.return_value = base + 1
            info = reg.items_info()
            assert len(info) == 3
            for i in range(3):
                assert f"m{i}" in info
