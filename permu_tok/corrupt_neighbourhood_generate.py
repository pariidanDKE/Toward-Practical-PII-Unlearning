
############################################### OPTIMAL TOKENIZER ###########################################
# utils.py - Vocabulary indices and caching utilities

import os
import pickle
import hashlib
import json
from pathlib import Path
from logging_utils import get_logger, should_log_stats, add_subject_lengths
from utils import get_config
import Levenshtein

def setup_optimized_tokenizer(tokenizer, memory_mode="balanced", precompute_tokens=1000, cache_path=None):
    """
    One-time setup for optimized token corruption
    Call this once after loading your tokenizer
    
    Args:
        tokenizer: The tokenizer object
        memory_mode: Memory usage mode ("minimal", "balanced", "performance", "unlimited")
        precompute_tokens: Number of tokens to precompute (0 to skip)
        cache_path: Path to save/load precomputed neighborhoods (None to skip persistent caching)
    """
    logger = get_logger()
    logger.info("Setting up optimized tokenizer...")
    
    # Configure memory usage
    configure_memory_usage(memory_mode)
    
    # Build vocabulary indices
    build_vocab_indices(tokenizer)
    
    # Handle persistent caching
    if cache_path is not None:
        cache_path = Path(cache_path)
        if load_precomputed_neighbors(tokenizer, cache_path):
            print("Loaded precomputed neighbors from cache")
        else:
            # Precompute neighbors for common tokens
            if precompute_tokens > 0:
                if memory_mode == "minimal":
                    print("Skipping precomputation in minimal memory mode")
                else:
                    memory_efficient_precompute(tokenizer, target_memory_mb=50)
                    actual_cache_path = save_precomputed_neighbors(tokenizer, cache_path)
                    print(f"Saved precomputed neighbors to {actual_cache_path}")
    else:
        # Precompute neighbors for common tokens (no persistent caching)
        if precompute_tokens > 0:
            if memory_mode == "minimal":
                print("Skipping precomputation in minimal memory mode")
            else:
                memory_efficient_precompute(tokenizer, target_memory_mb=50)
    
    print(f"Setup complete. Memory usage: {get_memory_usage_mb():.1f} MB")

import re
from collections import defaultdict, OrderedDict

# Global caches and indices
_vocab_by_length = None
_vocab_by_first_char = None
_latin_token_cache = {}
_neighbor_cache = {}

class LimitedCache(OrderedDict):
    """LRU cache with size limit to control memory usage"""
    def __init__(self, max_size=10000):
        super().__init__()
        self.max_size = max_size
    
    def __setitem__(self, key, value):
        if key in self:
            # Move to end
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            # Remove oldest item
            self.popitem(last=False)

def get_tokenizer_hash(tokenizer):
    """
    Generate a hash to identify the tokenizer vocabulary
    This helps ensure we're loading the correct cached data
    """
    # Create a hash based on vocabulary size and some sample tokens
    vocab_items = list(tokenizer.vocab.items())
    sample_size = min(1000, len(vocab_items))

    sorted_vocab_items = sorted(vocab_items)
    sample_items = sorted_vocab_items[:sample_size]
    
    # Include vocabulary size and sample tokens in hash
    hash_data = {
        'vocab_size': len(tokenizer.vocab),
        'sample_tokens': sample_items
    }
    
    hash_string = json.dumps(hash_data, sort_keys=True)
    return hashlib.md5(hash_string.encode()).hexdigest()


def save_precomputed_neighbors(tokenizer, cache_path):
    """
    Save precomputed neighbors to disk
    
    Args:
        tokenizer: The tokenizer object
        cache_path: Path where to save the cache
    """
    cache_path = Path(cache_path)
    
    # If cache_path is a directory or has no extension, create a default filename
    if cache_path.is_dir() or (not cache_path.suffix):
        cache_path = cache_path / f"neighbors_{get_tokenizer_hash(tokenizer)}.pkl"
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create cache data structure
    cache_data = {
        'tokenizer_hash': get_tokenizer_hash(tokenizer),
        'vocab_by_length': dict(_vocab_by_length) if _vocab_by_length else {},
        'vocab_by_first_char': dict(_vocab_by_first_char) if _vocab_by_first_char else {},
        'neighbor_cache': dict(_neighbor_cache),
        'latin_token_cache': dict(_latin_token_cache),
        'version': '1.0'
    }
    
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Successfully saved {len(_neighbor_cache)} precomputed neighbors to {cache_path}")
        return cache_path  # Return the actual path used
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")


def load_precomputed_neighbors(tokenizer, cache_path):
    """
    Load precomputed neighbors from disk
    
    Args:
        tokenizer: The tokenizer object
        cache_path: Path to load the cache from
        
    Returns:
        bool: True if successfully loaded, False otherwise
    """
    global _vocab_by_length, _vocab_by_first_char, _latin_token_cache, _neighbor_cache
    
    cache_path = Path(cache_path)
    
    # If cache_path is a directory or has no extension, look for the default filename
    if cache_path.is_dir() or (not cache_path.suffix):
        cache_path = cache_path / f"neighbors_{get_tokenizer_hash(tokenizer)}.pkl"
        get_logger().info(f'Cache Path resolved to: {cache_path}')

    if not cache_path.exists():
        get_logger().info(f"Cache file {cache_path} does not exist")
        return False
    
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Verify this cache is for the correct tokenizer
        current_hash = get_tokenizer_hash(tokenizer)
        cached_hash = cache_data.get('tokenizer_hash')
        
        if current_hash != cached_hash:
            print(f"Cache tokenizer hash mismatch. Current: {current_hash}, Cached: {cached_hash}")
            return False
        
        # Load the cached data
        _vocab_by_length = defaultdict(list, cache_data.get('vocab_by_length', {}))
        _vocab_by_first_char = defaultdict(list, cache_data.get('vocab_by_first_char', {}))
        _latin_token_cache.update(cache_data.get('latin_token_cache', {}))
        
        # Load neighbor cache based on current memory mode
        cached_neighbors = cache_data.get('neighbor_cache', {})
        if isinstance(_neighbor_cache, LimitedCache):
            # If we have a limited cache, load up to its limit
            for key, value in list(cached_neighbors.items())[:_neighbor_cache.max_size]:
                _neighbor_cache[key] = value
        else:
            # Unlimited cache
            _neighbor_cache.update(cached_neighbors)
        
        print(f"Successfully loaded {len(cached_neighbors)} precomputed neighbors from {cache_path}")
        return True
        
    except Exception as e:
        print(f"Warning: Failed to load cache from {cache_path}: {e}")
        return False


def debug_cache_path(tokenizer, cache_path):
    """
    Debug function to show what file path will actually be used
    
    Args:
        tokenizer: The tokenizer object
        cache_path: The input cache path
    """
    original_path = Path(cache_path)
    print(f"Original cache_path: {original_path}")
    print(f"Is directory: {original_path.is_dir()}")
    print(f"Has suffix: {bool(original_path.suffix)}")
    print(f"Exists: {original_path.exists()}")
    
    # Apply the same logic as the functions
    if original_path.is_dir() or (not original_path.suffix):
        final_path = original_path / f"neighbors_{get_tokenizer_hash(tokenizer)}.pkl"
    else:
        final_path = original_path
        
    print(f"Final cache_path: {final_path}")
    print(f"Final path exists: {final_path.exists()}")
    
    if final_path.exists():
        print(f"File size: {final_path.stat().st_size} bytes")
    
    return final_path


def clear_cache_file(cache_path):
    """
    Remove the cache file from disk
    
    Args:
        cache_path: Path to the cache file to remove
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        try:
            cache_path.unlink()
            print(f"Removed cache file: {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to remove cache file {cache_path}: {e}")
    else:
        print(f"Cache file {cache_path} does not exist")


def is_latin_alphabet_only(text):
    """
    Check if a token contains only Latin alphabet characters, numbers, 
    common punctuation, and whitespace/special tokenizer symbols.
    Cached version for better performance.
    """
    if text in _latin_token_cache:
        return _latin_token_cache[text]
    
    # Remove common tokenizer prefixes/symbols
    cleaned_text = text.replace('▁', '').replace('Ġ', '').replace('##', '')
    
    # Allow Latin letters, numbers, common punctuation, and whitespace
    latin_pattern = r'^[a-zA-Z0-9\s\.,;:!?\-\'\"()\[\]{}/@#$%^&*+=<>|\\~`_]*$'
    
    result = bool(re.match(latin_pattern, cleaned_text))
    _latin_token_cache[text] = result
    return result

def build_vocab_indices(tokenizer):
    """Build optimized indices for the vocabulary - call this once at startup"""
    global _vocab_by_length, _vocab_by_first_char
    
    if _vocab_by_length is not None:
        return  # Already built
    
    print("Building vocabulary indices...")
    _vocab_by_length = defaultdict(list)
    _vocab_by_first_char = defaultdict(list)
    
    # Pre-filter and index Latin tokens
    for vocab_token, vocab_token_id in tokenizer.vocab.items():
        if is_latin_alphabet_only(vocab_token):
            token_length = len(vocab_token)
            _vocab_by_length[token_length].append((vocab_token, vocab_token_id))
            
            # Index by first alphabetic character
            first_char = ''
            for char in vocab_token:
                if char.isalpha():
                    first_char = char
                    break
            if first_char:
                _vocab_by_first_char[first_char].append((vocab_token, vocab_token_id))
    
    print(f"Vocabulary indices built: {len(_vocab_by_length)} length buckets, "
          f"{len(_vocab_by_first_char)} first-char buckets")

def get_vocab_by_length():
    """Get the vocabulary grouped by length"""
    return _vocab_by_length

def get_vocab_by_first_char():
    """Get the vocabulary grouped by first character"""
    return _vocab_by_first_char

def get_neighbor_cache():
    """Get the neighbor cache"""
    return _neighbor_cache

def set_neighbor_cache(cache):
    """Set the neighbor cache (useful for switching cache types)"""
    global _neighbor_cache
    _neighbor_cache = cache

def clear_caches():
    """Call this if you switch tokenizers or want to free memory"""
    global _vocab_by_length, _vocab_by_first_char, _latin_token_cache, _neighbor_cache
    print("Clearing all caches...")
    _vocab_by_length = None
    _vocab_by_first_char = None
    _latin_token_cache.clear()
    _neighbor_cache.clear()

def get_memory_usage_mb():
    """Get current memory usage of caches"""
    total_size = 0
    
    # Rough estimation
    if _vocab_by_length:
        total_size += sum(len(tokens) for tokens in _vocab_by_length.values()) * 40
    if _vocab_by_first_char:
        total_size += sum(len(tokens) for tokens in _vocab_by_first_char.values()) * 40
    
    total_size += len(_latin_token_cache) * 30
    total_size += len(_neighbor_cache) * 100  # Rough estimate per cache entry
    
    return total_size / (1024 * 1024)

def configure_memory_usage(mode="balanced"):
    """Configure memory vs speed trade-offs"""
    global _neighbor_cache
    
    if mode == "minimal":
        # Minimize memory usage - no neighbor caching
        _neighbor_cache = {}
        max_precompute = 0
        print("Minimal memory mode: No neighbor caching")
        
    elif mode == "balanced":
        # Balance memory and speed
        _neighbor_cache = LimitedCache(max_size=5000)
        max_precompute = 1000
        print("Balanced mode: Limited caching (5000 entries)")
        
    elif mode == "performance":
        # Maximize speed, higher memory usage
        _neighbor_cache = LimitedCache(max_size=20000)
        max_precompute = 5000
        print("Performance mode: Extended caching (20000 entries)")
        
    elif mode == "unlimited":
        # No limits - maximum speed
        _neighbor_cache = {}
        max_precompute = float('inf')
        print("Unlimited mode: No cache size limits")
    
    return max_precompute



def find_neighbourhood_k_optimized(tokenizer, token_id, k=1):
    """
    Optimized version that uses pre-built indices and caching
    """
    # Build indices if not already done
    build_vocab_indices(tokenizer)
    
    # Get references to the cached data
    vocab_by_length = get_vocab_by_length()
    vocab_by_first_char = get_vocab_by_first_char()
    neighbor_cache = get_neighbor_cache()
    
    # Check cache first
    cache_key = (token_id, k)
    if cache_key in neighbor_cache:
        return neighbor_cache[cache_key]
    
    original_token = tokenizer.convert_ids_to_tokens([token_id])[0]

    if should_log_stats('subject_token_len'):
        add_subject_lengths(original_token)

    # Handle digit tokens specially (unchanged logic)
    if original_token.strip().isdigit():
        neighbors = []
        for digit in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            possible_formats = [digit, f' {digit}', f'▁{digit}']
            for formatted_digit in possible_formats:
                if formatted_digit in tokenizer.vocab:
                    digit_id = tokenizer.vocab[formatted_digit]
                    if digit_id != token_id:
                        neighbors.append(digit_id)
        
        neighbor_cache[cache_key] = neighbors
        return neighbors

    match_first_char = get_config()['match_first_char']
    original_first_char = ''
    if match_first_char:
        for char in original_token:
            if char.isalpha():
                original_first_char = char
                break

    neighbors = []
    
    # Choose search strategy based on constraints
    if match_first_char and original_first_char:
        # Search only tokens with matching first character
        candidate_tokens = vocab_by_first_char.get(original_first_char, [])
    else:
        # For k=1, we can limit search to tokens of similar length (length ± k)
        if k == 1:
            candidate_tokens = []
            original_length = len(original_token)
            for length in range(max(1, original_length - k), original_length + k + 1):
                candidate_tokens.extend(vocab_by_length.get(length, []))
        else:
            # For larger k, search all Latin tokens (fallback to original approach)
            candidate_tokens = []
            for tokens_list in vocab_by_length.values():
                candidate_tokens.extend(tokens_list)
    
    # Check distance for candidate tokens
    for vocab_token, vocab_token_id in candidate_tokens:
        if vocab_token_id == token_id:
            continue
            
        distance = Levenshtein.distance(original_token, vocab_token)
        if distance <= k:
            neighbors.append(vocab_token_id)
    
    # Cache the result
    neighbor_cache[cache_key] = neighbors
    return neighbors

def find_neighbourhood_k_adaptive_strict_optimized(tokenizer, token_id, k=1):
    """
    Optimized version of the adaptive strict function
    """
    # Build indices if not already done
    build_vocab_indices(tokenizer)
    
    # Get references to the cached data
    vocab_by_length = get_vocab_by_length()
    neighbor_cache = get_neighbor_cache()
    
    # Check cache first
    cache_key = (token_id, k, 'adaptive_strict')
    if cache_key in neighbor_cache:
        return neighbor_cache[cache_key]
    
    original_token = tokenizer.convert_ids_to_tokens([token_id])[0]
    if should_log_stats('subject_token_len'):
        add_subject_lengths(original_token)
    
    original_length = len(original_token)
    neighbors = []
    
    # Handle digit tokens
    if original_token.strip().isdigit():
        for digit in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            for fmt in [digit, f' {digit}', f'▁{digit}']:
                if fmt in tokenizer.vocab:
                    digit_id = tokenizer.vocab[fmt]
                    if digit_id != token_id and Levenshtein.distance(original_token, fmt) == original_length:
                        neighbors.append(digit_id)
        
        neighbor_cache[cache_key] = neighbors
        return neighbors
    
    # Only search tokens of the same length (major optimization for adaptive strict)
    candidate_tokens = vocab_by_length.get(original_length, [])
    
    for vocab_token, vocab_token_id in candidate_tokens:
        if (vocab_token_id != token_id and 
            Levenshtein.distance(original_token, vocab_token) == original_length):
            neighbors.append(vocab_token_id)
    
    # Cache the result
    neighbor_cache[cache_key] = neighbors
    return neighbors

def precompute_all_neighbors(tokenizer, k=1, max_cache_size=10000):
    """
    Precompute neighbors for the most common tokens to warm up the cache
    Call this once after loading your tokenizer for maximum performance
    """
    
    build_vocab_indices(tokenizer)
    
    # Get token IDs sorted by frequency (you might need to adjust this based on your tokenizer)
    # This is a simple heuristic - you might want to use actual token frequencies
    common_token_ids = list(range(min(max_cache_size, len(tokenizer.vocab))))
    
    print(f"Precomputing neighbors for {len(common_token_ids)} tokens...")
    for i, token_id in enumerate(common_token_ids):
        if i % 1000 == 0:
            current_memory = get_memory_usage_mb()
            print(f"Progress: {i}/{len(common_token_ids)}, Memory: {current_memory:.1f}MB")
        
        # Precompute both types
        find_neighbourhood_k_optimized(tokenizer, token_id, k=k)
        find_neighbourhood_k_adaptive_strict_optimized(tokenizer, token_id, k=k)
    
    print("Precomputation complete!")

def memory_efficient_precompute(tokenizer, target_memory_mb=50):
    """Precompute neighbors while staying under memory limit"""
    
    build_vocab_indices(tokenizer)
    
    max_tokens = min(len(tokenizer.vocab), target_memory_mb * 200)  # Rough estimate
    common_token_ids = list(range(max_tokens))
    
    print(f"Precomputing neighbors for {len(common_token_ids)} tokens (target: {target_memory_mb}MB)")
    
    for i, token_id in enumerate(common_token_ids):
        if i % 1000 == 0:
            current_memory = get_memory_usage_mb()
            print(f"Progress: {i}/{len(common_token_ids)}, Memory: {current_memory:.1f}MB")
            
            if current_memory > target_memory_mb:
                print(f"Memory limit reached at {i} tokens")
                break
        
        find_neighbourhood_k_optimized(tokenizer, token_id, k=1)

# Example usage functions
def setup_with_persistent_cache(tokenizer, cache_dir="./tokenizer_cache"):



    cache_path = Path(cache_dir) / f"neighbors_{get_tokenizer_hash(tokenizer)}.pkl"
    
    setup_optimized_tokenizer(
        tokenizer=tokenizer,
        memory_mode="balanced",
        precompute_tokens=1000,
        cache_path=cache_path
    )
    
    return cache_path

def reset_tokenizer_cache(tokenizer, cache_dir="./tokenizer_cache"):
    """
    Example function to reset/clear the tokenizer cache
    """
    cache_path = Path(cache_dir) / f"neighbors_{get_tokenizer_hash(tokenizer)}.pkl"
    clear_cache_file(cache_path)
    clear_caches()
    print("Tokenizer cache reset complete")