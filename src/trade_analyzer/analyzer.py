"""Trade analysis module using LLMs."""

import logging
from typing import List

import requests
from openai import OpenAI

from .models import Trade

logger = logging.getLogger(__name__)

def create_analysis_prompt(trades: List[Trade]) -> str:
    """Create a prompt for LLM analysis."""
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

def analyze_trades(trades: List[Trade], api_url: str, api_key: str, model: str) -> str:
    """Analyze trades using LLM."""
    # Prepare the prompt
    prompt = create_analysis_prompt(trades)
    
    try:
        if 'nvidia' in api_url.lower() or 'openai' in api_url.lower():
            # Initialize OpenAI client for NVIDIA and OpenAI
            client = OpenAI(
                base_url=api_url,
                api_key=api_key
            )
            
            messages = [{"role": "user", "content": prompt}]
            
            # Create completion with streaming
            completion = client.chat.completions.create(
                model=model,
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
                api_url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["response"]
            
    except Exception as e:
        logger.error(f"Error during API call: {str(e)}")
        raise 