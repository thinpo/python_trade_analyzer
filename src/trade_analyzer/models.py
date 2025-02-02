"""Data models for Trade Analyzer."""

from dataclasses import dataclass
from typing import Optional

@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: str
    symbol: str
    price: float
    volume: int
    side: str

    def __str__(self) -> str:
        """String representation of a trade."""
        return f"Trade(timestamp='{self.timestamp}', symbol='{self.symbol}', price={self.price}, volume={self.volume}, side='{self.side}')" 