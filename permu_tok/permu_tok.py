import permu_tok.corrupt_neighbourhood_generate
import Levenshtein
import torch

from permu_tok.corrupt_neighbourhood_generate import (
    build_vocab_indices,
    get_neighbor_cache,
    get_vocab_by_first_char,
    get_vocab_by_length,
    is_latin_alphabet_only
)
from logging_utils import (
    add_corrupted_subject_info,
    add_subject_lengths,
    get_config,
    should_log_stats
)

def longest_common_subsequence_indices(list1, list2):
    """
    Find the longest common subsequence and return indices in list1 where it occurs
    """
    m, n = len(list1), len(list2)
    
    # Create LCS table
    lcs_table = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill LCS table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if list1[i-1] == list2[j-1]:
                lcs_table[i][j] = lcs_table[i-1][j-1] + 1
            else:
                lcs_table[i][j] = max(lcs_table[i-1][j], lcs_table[i][j-1])
    
    # Backtrack to find LCS indices in list1
    lcs_indices = []
    i, j = m, n
    while i > 0 and j > 0:
        if list1[i-1] == list2[j-1]:
            lcs_indices.append(i-1)
            i -= 1
            j -= 1
        elif lcs_table[i-1][j] > lcs_table[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    lcs_indices.reverse()
    return lcs_indices

import random
import re

def is_latin_alphabet_only(text):
    """
    Check if a token contains only Latin alphabet characters, numbers, 
    common punctuation, and whitespace/special tokenizer symbols.
    """
    # Remove common tokenizer prefixes/symbols
    cleaned_text = text.replace('▁', '').replace('Ġ', '').replace('##', '')
    
    # Allow Latin letters, numbers, common punctuation, and whitespace
    # This regex matches: letters (a-zA-Z), numbers (0-9), common punctuation, whitespace
    latin_pattern = r'^[a-zA-Z0-9\s\.,;:!?\-\'\"()\[\]{}/@#$%^&*+=<>|\\~`_]*$'
    
    return bool(re.match(latin_pattern, cleaned_text))

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


#######################################################

def find_neighbourhood_k(tokenizer, token_id, k=1):
    """
    Finds tokens in the tokenizer's vocabulary that are within a Levenshtein 
    distance of k from the given token_id (neighborhood approach).
    Only returns tokens with Latin alphabet characters.
    """
    original_token = tokenizer.convert_ids_to_tokens([token_id])[0]

    if should_log_stats('subject_token_len'):
        add_subject_lengths(original_token)

    def get_first_alpha_char(token):
            for char in token:
                if char.isalpha():
                    return char
            return ''

    # Check if the token is a digit
    if original_token.strip().isdigit():
        # For digit tokens, only return other single digit tokens
        neighbors = []
        decoded_neighbours = []
        
        # Try to find all single digits in the vocabulary
        for digit in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            # Check different possible formats
            possible_formats = [
                digit,  # Just the digit
                f' {digit}',  # Space + digit
                f'▁{digit}',  # Common in sentencepiece tokenizers
            ]
            
            for formatted_digit in possible_formats:
                if formatted_digit in tokenizer.vocab:
                    digit_id = tokenizer.vocab[formatted_digit]
                    if digit_id != token_id:  # Don't include the original token
                        neighbors.append(digit_id)
                        decoded_neighbours.append(formatted_digit)
        

        return neighbors

    match_first_char = get_config()['match_first_char']
    original_first_char = get_first_alpha_char(original_token) if original_token and match_first_char else ''

    neighbors = []
    decoded_neighbours = []
    
    for vocab_token, vocab_token_id in tokenizer.vocab.items():
        if vocab_token_id == token_id:
            continue
            
        # Filter out non-Latin alphabet tokens
        if not is_latin_alphabet_only(vocab_token):
            continue

        if match_first_char and (not vocab_token or get_first_alpha_char(vocab_token) != original_first_char):
            continue

        distance = Levenshtein.distance(original_token, vocab_token)
        if distance <= k:
            neighbors.append(vocab_token_id)
            decoded_neighbours.append(vocab_token)
    return neighbors

def find_neighbourhood_k_adaptive_strict(tokenizer, token_id, k=1):
    original_token = tokenizer.convert_ids_to_tokens([token_id])[0]
    if should_log_stats('subject_token_len'):
        add_subject_lengths(original_token)
    
    original_length = len(original_token)
    neighbors = []
    
    if original_token.strip().isdigit():
        for digit in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            for fmt in [digit, f' {digit}', f'▁{digit}']:
                if fmt in tokenizer.vocab:
                    digit_id = tokenizer.vocab[fmt]
                    if digit_id != token_id and Levenshtein.distance(original_token, fmt) == original_length:
                        neighbors.append(digit_id)
        return neighbors
    
    for vocab_token, vocab_token_id in tokenizer.vocab.items():
        if (vocab_token_id != token_id and 
            is_latin_alphabet_only(vocab_token) and 
            len(vocab_token) == original_length and 
            Levenshtein.distance(original_token, vocab_token) == original_length):
            neighbors.append(vocab_token_id)
    
    return neighbors

def corrupt_single_token_id(tokenizer, token_id, replace_prob=0.6, k=1):
    """
    Corrupts a single token ID with a similar token from the vocabulary
    based on a given probability.
    """

    if get_config()['optimal_neighbours_generation']:
        simple_neighbourhood_func = find_neighbourhood_k_optimized
        adaptive_neighbourhood_func = find_neighbourhood_k_adaptive_strict_optimized
    else:
        simple_neighbourhood_func = find_neighbourhood_k
        adaptive_neighbourhood_func = find_neighbourhood_k_adaptive_strict

    if random.random() < replace_prob:
        if get_config()['use_adaptive_k']:
            #neighbor_ids = find_neighbourhood_k_adaptive_strict(tokenizer, token_id, k=k)
            neighbor_ids = adaptive_neighbourhood_func(tokenizer, token_id, k=k)

        else:
            #neighbor_ids = find_neighbourhood_k(tokenizer, token_id, k=k)
            neighbor_ids = simple_neighbourhood_func(tokenizer, token_id, k=k)
        if neighbor_ids:
            sampled_id = random.choice(neighbor_ids)
            return sampled_id
        else:
            return token_id
    else:
        return token_id

def create_perturbed_subject(tokenizer, inputs_idx, tokens_to_mix, token_replace_prob=0.6, k=1):
    """
    Create perturbed subject with different probabilities for different token ranges.
    tokens_to_mix contains tuples of (start, end, reduce_prob) where reduce_prob is boolean
    """


    original_subjects_decoded = []
    corrupted_subjects_decoded = []


    for item in tokens_to_mix:
        if len(item) == 3:
            b, e, reduce_prob = item
            actual_prob = token_replace_prob / 2 if reduce_prob else token_replace_prob
        else:
            b, e = item
            actual_prob = token_replace_prob
        subject_ids = inputs_idx[b:e]
        original_subjects_decoded.append(tokenizer.decode(subject_ids, skip_special_tokens=True))

        for i in range(len(subject_ids)):
            original_token = subject_ids[i]
            #subject_ids.append(original_token)  
            subject_ids[i] = corrupt_single_token_id(tokenizer, subject_ids[i],  
                                                   replace_prob=actual_prob, k=k)

        inputs_idx[b:e] = subject_ids
        corrupted_subjects_decoded.append(tokenizer.decode(subject_ids, skip_special_tokens=True))

    add_corrupted_subject_info(original_subjects_decoded, corrupted_subjects_decoded)


    return inputs_idx

def add_tokens_to_mix_lcs(missing_tokens, full_text_input_id, subject_id, subject, tokens_to_mix):
    """
    Handle tokenization mismatch using LCS and add ranges to tokens_to_mix (original version)
    """
    lcs_indices = longest_common_subsequence_indices(full_text_input_id, subject_id)
    lcs_coverage = len(lcs_indices) / len(subject_id)  # NEW: Track this

    print(f'Proportion of LCS Indices: {(len(lcs_indices) / len(subject_id)):.2f}')
    if len(lcs_indices) < 0.2 * len(subject_id):
        raise ValueError(
            f"\n❌ Subject tokenization mismatch - too many tokens lost!\n"
            f"Subject: {subject}\n"
            f"Subject token IDs: {subject_id}\n"
            f"Full text token IDs: {full_text_input_id}\n"
            f"Tokens missing from full text: {missing_tokens}\n"
            f"LCS coverage: {len(lcs_indices)}/{len(subject_id)} ({len(lcs_indices)/len(subject_id)*100:.1f}%)\n"
        )
    ranges = []
    if lcs_indices:
        start_idx = lcs_indices[0]
        end_idx = lcs_indices[0]
        for i in range(1, len(lcs_indices)):
            if lcs_indices[i] == lcs_indices[i-1] + 1:
                end_idx = lcs_indices[i]
            else:
                ranges.append((start_idx, end_idx + 1))
                start_idx = lcs_indices[i]
                end_idx = lcs_indices[i]
        ranges.append((start_idx, end_idx + 1))
        for range_start, range_end in ranges:
            tokens_to_mix.append((range_start, range_end))
    return ranges, lcs_coverage

def has_preceding_context(subject_text, full_text):
    preceding_chars = []
    start = 0
    while True:
        subject_pos = full_text.find(subject_text, start)
        if subject_pos == -1:
            break
        if subject_pos > 0:
            preceding_char = full_text[subject_pos - 1]
            if preceding_char not in [' ', '\n', '\t']:
                preceding_chars.append(preceding_char)
        start = subject_pos + 1
    return len(preceding_chars) > 0, preceding_chars

def tokenize_subject_contextual(tokenizer, subject_text, preceding_chars):
    all_tokens = []
    for preceding_char in preceding_chars:
        tokens = tokenizer(preceding_char + subject_text, return_tensors="pt")['input_ids'][0]
        tokens = [tok.item() for tok in tokens if tok not in tokenizer.all_special_ids]
        all_tokens.append(torch.tensor(tokens[1:]))
    return all_tokens
