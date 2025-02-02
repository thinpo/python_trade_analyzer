#!/usr/bin/env python3

import os
import sys
import json
import logging
import argparse
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
import requests
import pandas as pd
import duckdb
from dotenv import load_dotenv
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enable debug logging for trade type mapping if LOG_LEVEL is DEBUG
if os.getenv('LOG_LEVEL', '').upper() == 'DEBUG':
    logger.setLevel(logging.DEBUG)

@dataclass
class Trade:
    timestamp: str
    symbol: str
    price: float
    volume: int
    side: str

    def __str__(self):
        return f"Trade(timestamp='{self.timestamp}', symbol='{self.symbol}', price={self.price}, volume={self.volume}, side='{self.side}')"

class ApiType(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    NVIDIA = "nvidia"

    @staticmethod
    def from_url(url: str) -> 'ApiType':
        if "ollama" in url:
            return ApiType.OLLAMA
        elif "nvidia" in url:
            return ApiType.NVIDIA
        else:
            return ApiType.OPENAI

class LLMClient:
    def __init__(self, api_url: str, api_key: str, model: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.client = None
        
        # Initialize OpenAI client for NVIDIA and OpenAI
        if 'nvidia' in api_url.lower() or 'openai' in api_url.lower():
            self.client = OpenAI(
                base_url=api_url,
                api_key=api_key
            )

    def analyze_trades(self, trades: List[Trade]) -> str:
        # Prepare the prompt
        prompt = self._create_analysis_prompt(trades)
        
        try:
            if self.client:  # For NVIDIA and OpenAI
                messages = [{"role": "user", "content": prompt}]
                
                # Create completion with streaming
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.6,
                    top_p=0.7,
                    max_tokens=4096,
                    stream=True
                )
                
                # Handle streaming response
                logger.info("Receiving streaming response...")
                full_response = []
                for chunk in completion:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        full_response.append(content)
                print()  # New line after streaming
                return "".join(full_response)
                
            else:  # For Ollama
                response = requests.post(
                    self.api_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                response.raise_for_status()
                return response.json()["response"]
                
        except Exception as e:
            logger.error(f"Error during API call: {str(e)}")
            raise

    def _create_analysis_prompt(self, trades: List[Trade]) -> str:
        # Calculate basic statistics
        total_volume = sum(t.volume for t in trades)
        avg_volume = total_volume / len(trades) if trades else 0
        median_volume = sorted([t.volume for t in trades])[len(trades)//2] if trades else 0
        
        # Group trades by symbol
        symbols = {}
        for trade in trades:
            if trade.symbol not in symbols:
                symbols[trade.symbol] = {
                    'trades': [],
                    'volume': 0,
                    'min_price': float('inf'),
                    'max_price': float('-inf'),
                    'sides': {},
                    'buy_volume': 0,
                    'sell_volume': 0,
                    'sweep_volume': 0,
                    'odd_lot_volume': 0,
                    'regular_volume': 0,
                    'price_points': set(),
                    'price_levels': [],
                    'last_price': None,
                    'price_trend': []
                }
            
            data = symbols[trade.symbol]
            data['trades'].append(trade)
            data['volume'] += trade.volume
            data['min_price'] = min(data['min_price'], trade.price)
            data['max_price'] = max(data['max_price'], trade.price)
            data['price_points'].add(trade.price)
            data['price_levels'].append(trade.price)
            
            # Track volume by side
            if trade.side == 'BUY':
                data['buy_volume'] += trade.volume
            elif trade.side == 'SELL':
                data['sell_volume'] += trade.volume
            elif trade.side == 'SWEEP':
                data['sweep_volume'] += trade.volume
            elif trade.side == 'ODD_LOT':
                data['odd_lot_volume'] += trade.volume
            elif trade.side == 'REGULAR':
                data['regular_volume'] += trade.volume
            
            # Track price trend
            if data['last_price'] is not None:
                if trade.price > data['last_price']:
                    data['price_trend'].append(1)  # Up
                elif trade.price < data['last_price']:
                    data['price_trend'].append(-1)  # Down
                else:
                    data['price_trend'].append(0)  # Flat
            data['last_price'] = trade.price
            
            if trade.side not in data['sides']:
                data['sides'][trade.side] = 0
            data['sides'][trade.side] += trade.volume

        # Create symbol summaries with imbalance analysis
        symbol_summaries = []
        for symbol, data in symbols.items():
            trades = data['trades']
            avg_price = sum(t.price for t in trades) / len(trades)
            
            # Calculate imbalance metrics
            buy_sell_ratio = data['buy_volume'] / data['sell_volume'] if data['sell_volume'] > 0 else float('inf')
            net_flow = data['buy_volume'] - data['sell_volume']
            imbalance_percent = (net_flow / data['volume'] * 100) if data['volume'] > 0 else 0
            
            # Calculate price volatility and trend
            price_volatility = (data['max_price'] - data['min_price']) / avg_price * 100 if avg_price > 0 else 0
            price_trend = sum(data['price_trend']) / len(data['price_trend']) if data['price_trend'] else 0
            
            # Calculate BBO-related statistics
            price_points = sorted(data['price_points'])
            price_levels = data['price_levels']
            top_levels = price_points[-3:] if len(price_points) >= 3 else price_points
            bottom_levels = price_points[:3] if len(price_points) >= 3 else price_points
            
            summary = f"""
### {symbol}:
Volume Statistics:
- Total Volume: {data['volume']:,} shares
- Buy Volume: {data['buy_volume']:,} shares
- Sell Volume: {data['sell_volume']:,} shares
- Sweep Volume: {data['sweep_volume']:,} shares
- Odd Lot Volume: {data['odd_lot_volume']:,} shares
- Regular Volume: {data['regular_volume']:,} shares

Price Statistics:
- Range: ${data['min_price']:.3f} to ${data['max_price']:.3f}
- Average: ${avg_price:.3f}
- Volatility: {price_volatility:.2f}%
- Price Trend: {"Upward" if price_trend > 0.2 else "Downward" if price_trend < -0.2 else "Sideways"}

Trade Imbalance:
- Buy/Sell Ratio: {buy_sell_ratio:.2f}
- Net Flow: {net_flow:+,d} shares
- Imbalance: {imbalance_percent:.1f}%

Price Levels:
- Top Levels: {', '.join(f'${p:.3f}' for p in reversed(top_levels))}
- Bottom Levels: {', '.join(f'${p:.3f}' for p in bottom_levels)}
- Total Price Points: {len(price_points)}

Trade Types: {', '.join(f"{side}({vol:,} shares)" for side, vol in data['sides'].items())}
"""
            symbol_summaries.append(summary)

        # Create the analysis prompt
        prompt = f"""Please analyze these trades and provide insights in the following format. For each section, include the calculation formulas used:

### 1. Volume Analysis
- Total Volume: {total_volume:,} shares
  Formula: sum(trade_volume) for all trades
- Average Trade Size: {avg_volume:.1f} shares
  Formula: total_volume / number_of_trades
- Median Trade Size: {median_volume} shares
  Formula: middle value of sorted trade volumes
- Distribution by trade type (SWEEP, ODD_LOT, REGULAR)
  Formula: percentage = (type_volume / total_volume) * 100
- Buy/Sell Imbalance Analysis:
  * Volume Ratio = buy_volume / sell_volume
  * Net Flow = buy_volume - sell_volume
  * Imbalance % = (net_flow / total_volume) * 100

### 2. Price Analysis
{chr(10).join(symbol_summaries)}

### 3. Statistical Measures
For each symbol, please calculate and explain:
- VWAP (Volume-Weighted Average Price)
  Formula: sum(price * volume) / sum(volume)
- Price Volatility
  Formula: (high - low) / average_price * 100
- Trade Size Distribution
  * Mean, Median, Mode
  * Standard Deviation
  * Skewness (to identify size bias)
- Time-weighted metrics
  * Price momentum (average price change per minute)
  * Volume momentum (average volume change per minute)

### 4. Pattern Analysis
Please identify and quantify:
- Price Impact Ratio
  Formula: abs(price_change) / volume for large trades
- Trade Size Clustering
  Formula: histogram of trade sizes with standard deviation bands
- Temporal Patterns
  * Trade frequency over time
  * Volume distribution over time
- Price Level Support/Resistance
  Formula: frequency of trades at each price level

### 5. Market Microstructure
Analyze and provide formulas for:
- Bid-Ask Bounce
  Formula: frequency of price reversals
- Trade Sign Analysis
  * Buy/Sell ratio over time
  * Sequential trade correlation
- Market Impact
  Formula: price_change / square_root(volume)
- Liquidity Analysis
  * Volume at each price level
  * Price elasticity (price change per unit volume)

Please provide a thorough analysis with:
1. Exact calculations and formulas used
2. Statistical significance where applicable
3. Comparative analysis between symbols
4. Market microstructure implications
5. Trading strategy considerations

Trade Details (first 10 trades for reference):
{chr(10).join(f"- {trade.timestamp}: {trade.symbol} {trade.side} {trade.volume:,} shares @ ${trade.price:.3f}" for trade in trades[:10])}
{f'... and {len(trades)-10:,} more trades' if len(trades) > 10 else ''}"""

        return prompt

def parse_sample_size(sample_arg: str, total_size: int) -> int:
    """Convert sample size argument to absolute number."""
    if not sample_arg:
        return total_size
        
    if str(sample_arg).endswith('%'):
        # Handle percentage
        try:
            percentage = float(sample_arg.rstrip('%'))
            if percentage <= 0 or percentage > 100:
                raise ValueError("Percentage must be between 0 and 100")
            return int(total_size * percentage / 100)
        except ValueError as e:
            logger.error(f"Invalid percentage format: {sample_arg}")
            raise
    else:
        # Handle absolute number
        try:
            size = int(sample_arg)
            if size <= 0:
                raise ValueError("Sample size must be positive")
            return size
        except ValueError:
            logger.error(f"Invalid sample size format: {sample_arg}")
            raise

def read_trades(file_path: str, sample_size: str = None, symbols: List[str] = None) -> List[Trade]:
    logger.info(f"Reading trades from: {file_path}")
    if symbols:
        logger.info(f"Filtering for symbols: {', '.join(symbols)}")
        symbols = [s.strip().upper() for s in symbols]  # Normalize symbols
    
    try:
        # Initialize DuckDB connection
        con = duckdb.connect(database=':memory:')
        
        # Create a table from the CSV file
        logger.info("Loading data into DuckDB...")
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
                            WHEN "Sale Condition" LIKE '%F%' THEN 'ISO'  -- Intermarket Sweep Order
                            WHEN "Sale Condition" LIKE '%I%' THEN 'ODD_LOT'
                            WHEN "Sale Condition" LIKE '%X%' THEN 'CROSS'
                            WHEN "Sale Condition" LIKE '%B%' OR "Sale Condition" LIKE '%W%' THEN 'AVG_PRICE'
                            WHEN "Sale Condition" LIKE '%C%' THEN 'CASH'
                            WHEN "Sale Condition" LIKE '%V%' THEN 'CONTINGENT'
                            WHEN "Sale Condition" LIKE '%7%' THEN 'QCT'  -- Qualified Contingent Trade
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
                    WHERE rn <= ?
                    ORDER BY timestamp
                """
                params = [abs_size]
            else:
                # Simple random sampling
                query = f"""
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
                            WHEN "Sale Condition" LIKE '%F%' THEN 'ISO'  -- Intermarket Sweep Order
                            WHEN "Sale Condition" LIKE '%I%' THEN 'ODD_LOT'
                            WHEN "Sale Condition" LIKE '%X%' THEN 'CROSS'
                            WHEN "Sale Condition" LIKE '%B%' OR "Sale Condition" LIKE '%W%' THEN 'AVG_PRICE'
                            WHEN "Sale Condition" LIKE '%C%' THEN 'CASH'
                            WHEN "Sale Condition" LIKE '%V%' THEN 'CONTINGENT'
                            WHEN "Sale Condition" LIKE '%7%' THEN 'QCT'  -- Qualified Contingent Trade
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
                    FROM trades
                    {symbol_filter}
                    ORDER BY random()
                    LIMIT ?
                """
                params = [abs_size]
        else:
            # No sampling, just filter by symbols if provided
            query = f"""
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
                        WHEN "Sale Condition" LIKE '%F%' THEN 'ISO'  -- Intermarket Sweep Order
                        WHEN "Sale Condition" LIKE '%I%' THEN 'ODD_LOT'
                        WHEN "Sale Condition" LIKE '%X%' THEN 'CROSS'
                        WHEN "Sale Condition" LIKE '%B%' OR "Sale Condition" LIKE '%W%' THEN 'AVG_PRICE'
                        WHEN "Sale Condition" LIKE '%C%' THEN 'CASH'
                        WHEN "Sale Condition" LIKE '%V%' THEN 'CONTINGENT'
                        WHEN "Sale Condition" LIKE '%7%' THEN 'QCT'  -- Qualified Contingent Trade
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
                FROM trades
                {symbol_filter}
                ORDER BY Time
            """
            params = []

        # Execute the query and convert to trades
        logger.info("Executing sampling query...")
        df = con.execute(query, params).df()
        
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

        # Log results
        symbol_counts = df.groupby('symbol').size()
        logger.info("\nTrades collected per symbol:")
        for symbol, count in symbol_counts.items():
            logger.info(f"  {symbol}: {count:,} trades")

        logger.info(f"\nSuccessfully read {len(trades):,} total trades")
        if trades:
            logger.info("\nFirst few trades per symbol:")
            shown_symbols = set()
            for trade in trades:
                if trade.symbol not in shown_symbols:
                    logger.info(f"{trade.symbol}: {trade}")
                    shown_symbols.add(trade.symbol)
                if len(shown_symbols) >= 3:  # Show at most 3 symbols
                    break

        return trades

    except Exception as e:
        logger.error(f"Error reading trades: {e}")
        logger.error("This might be due to:")
        logger.error("1. Invalid CSV format")
        logger.error("2. Missing or incorrect columns")
        logger.error("3. File access issues")
        logger.error("4. Memory constraints")
        raise
    finally:
        if 'con' in locals():
            con.close()

def get_provider_defaults(provider: str):
    """Get default URL and model for a provider."""
    defaults = {
        'openai': {
            'url': os.getenv('OPENAI_URL', 'https://api.openai.com/v1/chat/completions'),
            'model': os.getenv('DEFAULT_OPENAI_MODEL', 'gpt-4'),
            'key_env': 'OPENAI_API_KEY'
        },
        'nvidia': {
            'url': os.getenv('NVIDIA_URL', 'https://api.nvidia.com/v1/chat/completions'),
            'model': os.getenv('DEFAULT_NVIDIA_MODEL', 'deepseek-ai/deepseek-r1'),
            'key_env': 'NVIDIA_API_KEY'
        },
        'ollama': {
            'url': os.getenv('OLLAMA_URL', 'http://localhost:11434/api/generate'),
            'model': os.getenv('DEFAULT_OLLAMA_MODEL', 'llama2'),
            'key_env': None  # Ollama doesn't need a key
        }
    }
    return defaults.get(provider.lower())

def save_trades_to_csv(trades: List[Trade], output_file: str):
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

def main():
    parser = argparse.ArgumentParser(description='Analyze trade data using LLM')
    parser.add_argument('csv_file', help='Path to the CSV file containing trade data')
    parser.add_argument('--provider', choices=['openai', 'nvidia', 'ollama'], 
                      default='openai', help='LLM provider to use')
    parser.add_argument('--api-url', help='Override default API endpoint URL')
    parser.add_argument('--api-key', help='Override default API key')
    parser.add_argument('--model', help='Override default model name')
    parser.add_argument('--sample', help='Number of trades to sample per symbol (e.g., 1000 or 10%%)')
    parser.add_argument('--symbols', help='Comma-separated list of symbols to analyze (e.g., AAPL,MSFT,GOOGL)')
    parser.add_argument('--save-sample', help='Save the sampled trades to this CSV file')
    parser.add_argument('--load-sample', help='Load previously sampled trades from this CSV file')
    
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Get provider defaults
    provider_config = get_provider_defaults(args.provider)
    if not provider_config:
        logger.error(f"Invalid provider: {args.provider}")
        logger.error("Supported providers: openai, nvidia, ollama")
        sys.exit(1)

    # Set up configuration with overrides
    api_url = args.api_url or provider_config['url']
    model = args.model or provider_config['model']
    
    logger.info(f"Using provider: {args.provider}")
    logger.info(f"Using API URL: {api_url}")
    logger.info(f"Using model: {model}")

    # Get API key with priority: command line > LLM_TOKEN > provider-specific key
    api_key = (args.api_key or 
               os.getenv('LLM_TOKEN') or 
               (os.getenv(provider_config['key_env']) if provider_config['key_env'] else None))
    
    if not api_key and provider_config['key_env']:  # Only check if provider needs a key
        logger.error(f"No API key found for {args.provider}. Please either:")
        logger.error("1. Set LLM_TOKEN environment variable, or")
        logger.error(f"2. Set {provider_config['key_env']} environment variable, or")
        logger.error("3. Use --api-key argument")
        sys.exit(1)

    try:
        # Either load from saved sample or read from source
        if args.load_sample:
            trades = load_trades_from_csv(args.load_sample)
        else:
            # Parse symbols if provided
            symbols = None
            if args.symbols:
                symbols = [s.strip() for s in args.symbols.split(',')]
                
            # Read trades with sampling and symbol filtering
            trades = read_trades(args.csv_file, args.sample, symbols)
            
            # Save sample if requested
            if args.save_sample:
                save_trades_to_csv(trades, args.save_sample)
        
        # Initialize LLM client
        client = LLMClient(api_url, api_key, model)
        
        # Analyze trades
        logger.info("\nSending trades to LLM for analysis...")
        analysis = client.analyze_trades(trades)
        
        logger.info("\nLLM Analysis:")
        print(analysis)
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        logger.error("This might be due to:")
        logger.error("1. API endpoint not responding correctly")
        logger.error("2. Response format mismatch")
        logger.error("3. Authentication issues")
        logger.error("4. Network connectivity problems")
        sys.exit(1)

if __name__ == "__main__":
    main() 