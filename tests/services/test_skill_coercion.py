"""
Tests for _coerce_to_str_list utility — edge cases for LLM-returned skill lists.
TDD: written before the implementation.
"""

import pytest
from app.services.langchain_agents import LangChainAgents


@pytest.fixture
def agents():
    """Minimal LangChainAgents instance (no LLMs needed for utility tests)."""
    import unittest.mock as mock
    with mock.patch.object(LangChainAgents, "__init__", lambda self, *a, **kw: None):
        instance = LangChainAgents.__new__(LangChainAgents)
    return instance


class TestCoerceToStrList:
    """Tests for the LangChainAgents._coerce_to_str_list helper."""

    def test_plain_string_list(self, agents):
        """Standard list of strings passes through unchanged."""
        result = agents._coerce_to_str_list(["Python", "Java", "SQL"])
        assert result == ["Python", "Java", "SQL"]

    def test_dict_with_skill_key(self, agents):
        """Dicts with a 'skill' key should yield the value."""
        result = agents._coerce_to_str_list([{"skill": "Python"}, {"skill": "Java"}])
        assert result == ["Python", "Java"]

    def test_dict_with_name_key(self, agents):
        """Dicts with a 'name' key should yield the value."""
        result = agents._coerce_to_str_list([{"name": "Machine Learning"}])
        assert result == ["Machine Learning"]

    def test_mixed_list(self, agents):
        """Mixed list: strings, dicts, None, ints all handled gracefully."""
        result = agents._coerce_to_str_list(["Python", {"skill": "Java"}, None, 42, "Go"])
        assert result == ["Python", "Java", "42", "Go"]

    def test_none_values_skipped(self, agents):
        """None values are silently dropped."""
        result = agents._coerce_to_str_list([None, None, "Python"])
        assert result == ["Python"]

    def test_empty_list(self, agents):
        """Empty list returns empty list."""
        result = agents._coerce_to_str_list([])
        assert result == []

    def test_none_input(self, agents):
        """None as input returns empty list."""
        result = agents._coerce_to_str_list(None)
        assert result == []

    def test_integer_list(self, agents):
        """Integer list items get cast to str."""
        result = agents._coerce_to_str_list([1, 2, 3])
        assert result == ["1", "2", "3"]

    def test_dict_with_unknown_key_uses_first_value(self, agents):
        """Dict with unknown structure uses the first string value."""
        result = agents._coerce_to_str_list([{"technology": "Kubernetes"}])
        assert result == ["Kubernetes"]

    def test_nested_list_flattened(self, agents):
        """Nested lists are flattened one level."""
        result = agents._coerce_to_str_list([["Python", "Java"], "Go"])
        assert "Go" in result
        # Nested list items should be extracted
        assert "Python" in result
        assert "Java" in result

    def test_empty_string_preserved(self, agents):
        """Empty strings are dropped to avoid polluting skill lists."""
        result = agents._coerce_to_str_list(["", "Python", ""])
        assert result == ["Python"]
