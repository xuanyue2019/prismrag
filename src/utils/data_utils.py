"""
Data utilities for PrismRAG
"""

import json
import os
from typing import Any, Dict, List, Optional


def load_json(filepath: str) -> Any:
    """Load data from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """Save data to JSON file"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def split_text(
    text: str,
    max_length: int,
    overlap: int = 0,
    split_on: str = " "
) -> List[str]:
    """
    Split text into chunks with optional overlap.
    
    Args:
        text: Text to split
        max_length: Maximum length per chunk
        overlap: Number of characters to overlap between chunks
        split_on: Character/string to split on
        
    Returns:
        List of text chunks
    """
    
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_length
        
        if end >= len(text):
            # Last chunk
            chunks.append(text[start:])
            break
        
        # Find a good split point
        split_point = text.rfind(split_on, start, end)
        if split_point == -1 or split_point <= start:
            # No good split point found, use max_length
            split_point = end
        
        chunks.append(text[start:split_point])
        
        # Move start position with overlap
        start = max(split_point - overlap, start + 1)
    
    return chunks


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """Merge two dictionaries recursively"""
    
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Flatten a nested dictionary"""
    
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)


def count_tokens_approximate(text: str) -> int:
    """Approximate token count (rough estimate)"""
    # Very rough approximation: 1 token ≈ 4 characters for English
    # For Chinese text, it's closer to 1 token ≈ 1.5 characters
    
    # Simple heuristic: if text contains mostly Chinese characters
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    total_chars = len(text)
    
    if chinese_chars > total_chars * 0.5:
        # Mostly Chinese text
        return int(total_chars / 1.5)
    else:
        # Mostly English/other text
        return int(total_chars / 4)


def truncate_text(text: str, max_tokens: int) -> str:
    """Truncate text to approximate token limit"""
    
    current_tokens = count_tokens_approximate(text)
    
    if current_tokens <= max_tokens:
        return text
    
    # Calculate approximate character limit
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    total_chars = len(text)
    
    if chinese_chars > total_chars * 0.5:
        # Mostly Chinese text
        char_limit = int(max_tokens * 1.5)
    else:
        # Mostly English/other text
        char_limit = int(max_tokens * 4)
    
    return text[:char_limit]