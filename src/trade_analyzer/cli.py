"""Command-line interface for Trade Analyzer."""

import argparse
import logging
import os
import sys
from typing import Optional

from dotenv import load_dotenv

from .analyzer import analyze_trades
from .config import get_provider_defaults
from .reader import read_trades, load_trades_from_csv, load_trades_from_parquet
from .writer import save_trades_to_csv, save_trades_to_parquet

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the trade analyzer CLI."""
    parser = argparse.ArgumentParser(description='Analyze trade data using LLM')
    parser.add_argument('input_file', help='Path to the input file containing trade data (CSV or Parquet)')
    parser.add_argument('--provider', choices=['openai', 'nvidia', 'ollama'], 
                      default='openai', help='LLM provider to use')
    parser.add_argument('--api-url', help='Override default API endpoint URL')
    parser.add_argument('--api-key', help='Override default API key')
    parser.add_argument('--model', help='Override default model name')
    parser.add_argument('--sample', help='Number of trades to sample per symbol (e.g., 1000 or 10%%)')
    parser.add_argument('--symbols', help='Comma-separated list of symbols to analyze (e.g., AAPL,MSFT,GOOGL)')
    parser.add_argument('--save-sample', help='Save the sampled trades to this file (use .parquet or .csv extension)')
    parser.add_argument('--load-sample', help='Load previously sampled trades from this file (Parquet or CSV)')
    
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
            if args.load_sample.endswith('.parquet'):
                trades = load_trades_from_parquet(args.load_sample)
            else:
                trades = load_trades_from_csv(args.load_sample)
        else:
            # Parse symbols if provided
            symbols = None
            if args.symbols:
                symbols = [s.strip() for s in args.symbols.split(',')]
                
            # Read trades with sampling and symbol filtering
            trades = read_trades(args.input_file, args.sample, symbols)
            
            # Save sample if requested
            if args.save_sample:
                if args.save_sample.endswith('.parquet'):
                    save_trades_to_parquet(trades, args.save_sample)
                else:
                    save_trades_to_csv(trades, args.save_sample)
        
        # Analyze trades
        logger.info("\nSending trades to LLM for analysis...")
        analysis = analyze_trades(trades, api_url, api_key, model)
        
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