"""Tests for data models."""

from trade_analyzer.models import Trade

def test_trade_creation():
    """Test Trade object creation."""
    trade = Trade(
        timestamp="2024-02-01 09:30:00",
        symbol="AAPL",
        price=189.50,
        volume=100,
        side="REGULAR"
    )
    assert trade.timestamp == "2024-02-01 09:30:00"
    assert trade.symbol == "AAPL"
    assert trade.price == 189.50
    assert trade.volume == 100
    assert trade.side == "REGULAR"

def test_trade_string_representation():
    """Test Trade string representation."""
    trade = Trade(
        timestamp="2024-02-01 09:30:00",
        symbol="AAPL",
        price=189.50,
        volume=100,
        side="REGULAR"
    )
    expected = "Trade(timestamp='2024-02-01 09:30:00', symbol='AAPL', price=189.5, volume=100, side='REGULAR')"
    assert str(trade) == expected 