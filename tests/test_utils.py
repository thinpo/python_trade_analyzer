"""Tests for utility functions."""

import pytest

from trade_analyzer.utils import parse_sample_size

def test_parse_sample_size_percentage():
    """Test parsing percentage sample sizes."""
    assert parse_sample_size("10%", 1000) == 100
    assert parse_sample_size("50%", 1000) == 500
    assert parse_sample_size("100%", 1000) == 1000

def test_parse_sample_size_absolute():
    """Test parsing absolute sample sizes."""
    assert parse_sample_size("100", 1000) == 100
    assert parse_sample_size("500", 1000) == 500
    assert parse_sample_size("1000", 1000) == 1000

def test_parse_sample_size_invalid():
    """Test parsing invalid sample sizes."""
    with pytest.raises(ValueError):
        parse_sample_size("0%", 1000)
    with pytest.raises(ValueError):
        parse_sample_size("101%", 1000)
    with pytest.raises(ValueError):
        parse_sample_size("-10%", 1000)
    with pytest.raises(ValueError):
        parse_sample_size("0", 1000)
    with pytest.raises(ValueError):
        parse_sample_size("-100", 1000)
    with pytest.raises(ValueError):
        parse_sample_size("invalid", 1000)

def test_parse_sample_size_none():
    """Test parsing None sample size."""
    assert parse_sample_size(None, 1000) == 1000 