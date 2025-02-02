"""Trade data writer module."""

import logging
from typing import List

import duckdb
import pandas as pd

from .models import Trade

logger = logging.getLogger(__name__)

def save_trades_to_csv(trades: List[Trade], output_file: str) -> None:
    """Save trades to a CSV file for later reuse."""
    logger.info(f"Saving {len(trades)} trades to {output_file}")
    
    # Convert trades to pandas DataFrame
    df = pd.DataFrame([{
        'timestamp': t.timestamp,
        'symbol': t.symbol,
        'price': t.price,
        'volume': t.volume,
        'side': t.side
    } for t in trades])
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    logger.info(f"Successfully saved trades to {output_file}")

def save_trades_to_parquet(trades: List[Trade], output_file: str) -> None:
    """Save trades to a Parquet file for later reuse."""
    logger.info(f"Saving {len(trades)} trades to {output_file}")
    
    # Convert trades to pandas DataFrame
    df = pd.DataFrame([{
        'timestamp': t.timestamp,
        'symbol': t.symbol,
        'price': t.price,
        'volume': t.volume,
        'side': t.side
    } for t in trades])
    
    # Use DuckDB to save to Parquet with compression
    con = duckdb.connect(database=':memory:')
    con.register('trades_df', df)
    con.execute(f"""
        COPY (SELECT * FROM trades_df) 
        TO '{output_file}' (FORMAT 'parquet', COMPRESSION 'ZSTD')
    """)
    logger.info(f"Successfully saved trades to {output_file}") 