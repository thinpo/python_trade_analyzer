"""Utility functions for Trade Analyzer."""

import logging

logger = logging.getLogger(__name__)

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