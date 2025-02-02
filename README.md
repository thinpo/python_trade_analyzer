# NYSE TAQ Trade Analyzer

A powerful tool for analyzing trade data using Large Language Models (LLMs). This tool supports reading trade data from both CSV and Parquet files, with features for sampling, filtering, and detailed analysis.

## Features

- **Multiple File Format Support**:
  - CSV files (pipe-delimited)
  - Parquet files (with ZSTD compression)

- **Flexible Data Sampling**:
  - Sample by absolute number (e.g., `--sample 1000`)
  - Sample by percentage (e.g., `--sample 10%`)
  - Per-symbol or global sampling

- **Symbol Filtering**:
  - Filter trades for specific symbols
  - Support for multiple symbols

- **Multiple LLM Providers**:
  - OpenAI
  - NVIDIA AI
  - Ollama (local)

- **Trade Analysis**:
  - Volume analysis
  - Price analysis
  - Statistical measures
  - Pattern recognition
  - Market microstructure insights

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd python_trade_analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and preferences
```

## Usage

### Basic Usage

```bash
python src/trade_analyzer.py trades.csv --sample 1000 --symbols AAPL,MSFT
```

### Using Parquet Files

```bash
# Read from Parquet file
python src/trade_analyzer.py trades.parquet --sample 1000 --symbols AAPL,MSFT

# Save sample to Parquet
python src/trade_analyzer.py trades.csv --sample 1000 --symbols AAPL,MSFT --save-sample sample.parquet

# Load from saved Parquet sample
python src/trade_analyzer.py trades.csv --load-sample sample.parquet
```

### Sampling Options

```bash
# Sample 1000 trades per symbol
python src/trade_analyzer.py trades.csv --sample 1000 --symbols AAPL,MSFT

# Sample 10% of trades per symbol
python src/trade_analyzer.py trades.csv --sample 10% --symbols AAPL,MSFT

# Sample from all symbols
python src/trade_analyzer.py trades.csv --sample 1000
```

### LLM Provider Options

```bash
# Use OpenAI (default)
python src/trade_analyzer.py trades.csv --provider openai

# Use NVIDIA AI
python src/trade_analyzer.py trades.csv --provider nvidia

# Use local Ollama
python src/trade_analyzer.py trades.csv --provider ollama
```

## Environment Variables

- `LLM_TOKEN`: Default API key for LLM providers
- `OPENAI_API_KEY`: OpenAI API key
- `NVIDIA_API_KEY`: NVIDIA AI API key
- `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)
- `DEFAULT_OPENAI_MODEL`: Default model for OpenAI
- `DEFAULT_NVIDIA_MODEL`: Default model for NVIDIA
- `DEFAULT_OLLAMA_MODEL`: Default model for Ollama

## Input File Format

### CSV Format
- Pipe-delimited (|)
- Required columns:
  - Time
  - Symbol
  - Trade Price
  - Trade Volume
  - Sale Condition

### Parquet Format
- Same column structure as CSV
- ZSTD compression for optimal storage

## Analysis Output

The tool provides comprehensive trade analysis including:

1. Volume Analysis
   - Total volume
   - Average trade size
   - Distribution by trade type
   - Buy/Sell imbalance

2. Price Analysis
   - Price range
   - VWAP
   - Volatility
   - Price trends

3. Statistical Measures
   - Mean, median, mode
   - Standard deviation
   - Time-weighted metrics

4. Pattern Analysis
   - Price impact ratio
   - Trade size clustering
   - Support/resistance levels

5. Market Microstructure
   - Bid-ask bounce
   - Trade sign analysis
   - Market impact
   - Liquidity analysis

## Sale Condition Codes

The tool recognizes various sale condition codes:

1. Regular Trades:
   - Empty condition or '@' = REGULAR

2. Opening/Closing:
   - 'O' = OPENING
   - 'M' or '6' = CLOSING
   - '5' = REOPENING

3. Special Trade Types:
   - 'F' = ISO (Intermarket Sweep Order)
   - 'I' = ODD_LOT
   - 'X' = CROSS
   - 'B' or 'W' = AVG_PRICE
   - 'C' = CASH
   - 'V' = CONTINGENT
   - '7' = QCT (Qualified Contingent Trade)
   - '4' = DERIVATIVE

4. Extended Hours:
   - 'T' or 'U' = EXTENDED

5. Out of Sequence:
   - 'L' or 'Z' = OUT_OF_SEQ

6. Other Special Conditions:
   - 'H' = PRICE_VARIATION
   - 'P' = PRIOR_REF
   - 'R' = SELLER
   - 'E' = AUTO_EXEC
   - 'K' = RULE_127_155
   - '9' = CORRECTED_CLOSE

## Performance Considerations

- Uses DuckDB for efficient data processing
- Parquet format provides better compression and faster read times
- Sampling is done at the database level for memory efficiency
- Supports processing large files through efficient streaming

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - See LICENSE file for details 