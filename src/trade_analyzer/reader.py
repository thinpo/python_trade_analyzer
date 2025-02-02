"""Trade data reader module."""

import logging
from typing import List, Optional

import duckdb
import pandas as pd

from .models import Trade
from .utils import parse_sample_size

logger = logging.getLogger(__name__)

def read_trades(file_path: str, sample_size: Optional[str] = None, symbols: Optional[List[str]] = None) -> List[Trade]:
    """Read trades from a file (CSV or Parquet)."""
    logger.info(f"Reading trades from: {file_path}")
    if symbols:
        logger.info(f"Filtering for symbols: {', '.join(symbols)}")
        symbols = [s.strip().upper() for s in symbols]
    
    try:
        # Initialize DuckDB connection
        con = duckdb.connect(database=':memory:')
        
        # Create a table from the input file
        logger.info("Loading data into DuckDB...")
        if file_path.endswith('.parquet'):
            con.execute("""
                CREATE TABLE trades AS 
                SELECT * FROM read_parquet(?)
            """, [file_path])
        else:
            con.execute("""
                CREATE TABLE trades AS 
                SELECT * FROM read_csv_auto(?, delim='|', header=true, sample_size=1000000)
            """, [file_path])
        
        # Get total count
        total_count = con.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        logger.info(f"Total trades in file: {total_count:,}")
        
        # Show schema
        schema = con.execute("DESCRIBE trades").fetchall()
        logger.info("\nTable schema:")
        for col in schema:
            logger.info(f"  {col[0]}: {col[1]}")
        
        # Build the sampling query
        if symbols:
            symbol_list = ", ".join(f"'{s}'" for s in symbols)
            symbol_filter = f"WHERE Symbol IN ({symbol_list})"
            
            # Get count per symbol
            symbol_counts = con.execute(f"""
                SELECT Symbol, COUNT(*) as count 
                FROM trades 
                WHERE Symbol IN ({symbol_list})
                GROUP BY Symbol
            """).fetchall()
            
            logger.info("\nTrades available per symbol:")
            for symbol, count in symbol_counts:
                logger.info(f"  {symbol}: {count:,} trades")
                
            # Calculate sample size per symbol if percentage
            if sample_size:
                abs_sample_size = parse_sample_size(sample_size, symbol_counts[0][1])
                logger.info(f"Sampling {abs_sample_size:,} trades per symbol" + 
                          (f" ({sample_size})" if str(sample_size).endswith('%') else ""))
        else:
            symbol_filter = ""
            # Calculate total sample size if percentage
            if sample_size:
                abs_sample_size = parse_sample_size(sample_size, total_count)
                logger.info(f"Sampling {abs_sample_size:,} trades total" + 
                          (f" ({sample_size})" if str(sample_size).endswith('%') else ""))
        
        # Construct the sampling query
        if sample_size:
            abs_size = parse_sample_size(sample_size, total_count)
            if symbols:
                # Sample per symbol using window functions
                query = f"""
                    WITH ranked AS (
                        SELECT *,
                            ROW_NUMBER() OVER (PARTITION BY Symbol ORDER BY random()) as rn
                        FROM trades
                        {symbol_filter}
                    )
                    SELECT 
                        Time as timestamp,
                        Symbol as symbol,
                        "Trade Price" as price,
                        "Trade Volume" as volume,
                        CASE 
                            -- Regular trades
                            WHEN "Sale Condition" = '' OR "Sale Condition" LIKE '%@%' THEN 'REGULAR'
                            
                            -- Opening/Closing trades
                            WHEN "Sale Condition" LIKE '%O%' THEN 'OPENING'
                            WHEN "Sale Condition" LIKE '%M%' OR "Sale Condition" LIKE '%6%' THEN 'CLOSING'
                            WHEN "Sale Condition" LIKE '%5%' THEN 'REOPENING'
                            
                            -- Special trade types
                            WHEN "Sale Condition" LIKE '%F%' THEN 'ISO'
                            WHEN "Sale Condition" LIKE '%I%' THEN 'ODD_LOT'
                            WHEN "Sale Condition" LIKE '%X%' THEN 'CROSS'
                            WHEN "Sale Condition" LIKE '%B%' OR "Sale Condition" LIKE '%W%' THEN 'AVG_PRICE'
                            WHEN "Sale Condition" LIKE '%C%' THEN 'CASH'
                            WHEN "Sale Condition" LIKE '%V%' THEN 'CONTINGENT'
                            WHEN "Sale Condition" LIKE '%7%' THEN 'QCT'
                            WHEN "Sale Condition" LIKE '%4%' THEN 'DERIVATIVE'
                            
                            -- Extended hours
                            WHEN "Sale Condition" LIKE '%T%' OR "Sale Condition" LIKE '%U%' THEN 'EXTENDED'
                            
                            -- Out of sequence
                            WHEN "Sale Condition" LIKE '%L%' OR "Sale Condition" LIKE '%Z%' THEN 'OUT_OF_SEQ'
                            
                            -- Other special conditions
                            WHEN "Sale Condition" LIKE '%H%' THEN 'PRICE_VARIATION'
                            WHEN "Sale Condition" LIKE '%P%' THEN 'PRIOR_REF'
                            WHEN "Sale Condition" LIKE '%R%' THEN 'SELLER'
                            WHEN "Sale Condition" LIKE '%E%' THEN 'AUTO_EXEC'
                            WHEN "Sale Condition" LIKE '%K%' THEN 'RULE_127_155'
                            WHEN "Sale Condition" LIKE '%9%' THEN 'CORRECTED_CLOSE'
                            
                            -- If none of the above, keep the original code
                            ELSE 'OTHER_' || REPLACE("Sale Condition", ' ', '')
                        END as side
                    FROM ranked
                    WHERE rn <= {abs_size}
                    ORDER BY timestamp
                """
            else:
                # Sample from entire dataset
                query = f"""
                    SELECT timestamp, Symbol as symbol, "Trade Price" as price, 
                           "Trade Volume" as volume, "Sale Condition" as side
                    FROM trades
                    {symbol_filter}
                    ORDER BY RANDOM()
                    LIMIT {abs_size}
                """
        else:
            # No sampling, just filter by symbols if provided
            query = f"""
                SELECT timestamp, Symbol as symbol, "Trade Price" as price, 
                       "Trade Volume" as volume, "Sale Condition" as side
                FROM trades
                {symbol_filter}
                ORDER BY timestamp
            """
        
        # Execute query and convert to DataFrame
        df = con.execute(query).df()
        
        # Convert to Trade objects
        trades = []
        for _, row in df.iterrows():
            trade = Trade(
                timestamp=str(row['timestamp']),
                symbol=str(row['symbol']),
                price=float(row['price']),
                volume=int(row['volume']),
                side=str(row['side'])
            )
            trades.append(trade)
        
        logger.info(f"\nCollected {len(trades):,} trades")
        if len(trades) > 0:
            logger.info("\nFirst few trades:")
            for trade in trades[:5]:
                logger.info(f"  {trade}")
        
        return trades
        
    except Exception as e:
        logger.error(f"Error reading trades: {e}")
        raise
    finally:
        if 'con' in locals():
            con.close()

def load_trades_from_csv(input_file: str) -> List[Trade]:
    """Load trades from a previously saved CSV file."""
    logger.info(f"Loading trades from {input_file}")
    
    # Read CSV file
    df = pd.read_csv(input_file)
    
    # Convert back to Trade objects
    trades = []
    for _, row in df.iterrows():
        trade = Trade(
            timestamp=str(row['timestamp']),
            symbol=str(row['symbol']),
            price=float(row['price']),
            volume=int(row['volume']),
            side=str(row['side'])
        )
        trades.append(trade)
    
    logger.info(f"Successfully loaded {len(trades)} trades from {input_file}")
    return trades

def load_trades_from_parquet(input_file: str) -> List[Trade]:
    """Load trades from a previously saved Parquet file."""
    logger.info(f"Loading trades from {input_file}")
    
    # Use DuckDB to read Parquet file
    con = duckdb.connect(database=':memory:')
    df = con.execute(f"""
        SELECT * FROM read_parquet('{input_file}')
    """).df()
    
    # Convert back to Trade objects
    trades = []
    for _, row in df.iterrows():
        trade = Trade(
            timestamp=str(row['timestamp']),
            symbol=str(row['symbol']),
            price=float(row['price']),
            volume=int(row['volume']),
            side=str(row['side'])
        )
        trades.append(trade)
    
    logger.info(f"Successfully loaded {len(trades)} trades from {input_file}")
    return trades 