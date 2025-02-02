"""Configuration management for Trade Analyzer."""

import os
from typing import Dict, Optional

def get_provider_defaults(provider: str) -> Optional[Dict[str, str]]:
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