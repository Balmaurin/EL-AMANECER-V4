"""
Text quality assessment module.

This module provides tools for assessing the quality of text content using
various metrics including:
- Information entropy
- Stopword distribution
- Character and digit ratios
- N-gram shingling
"""

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Set, Union

from unidecode import unidecode


def shingle_words(text: str, k: int = 5) -> Set[str]:
    """Generate word-level k-shingles from text.

    Args:
        text: Input text to generate shingles from
        k: Size of each shingle in words

    Returns:
        Set of k-word shingles

    Note:
        Only considers words with 2 or more characters to reduce noise.
    """
    if not isinstance(text, str):
        return set()

    # Extract valid words
    words = [w for w in re.findall(r"\w{2,}", text.lower())]

    # Generate k-shingles
    return set(" ".join(words[i : i + k]) for i in range(max(0, len(words) - k + 1)))


def entropy(text: str) -> float:
    """Calculate Shannon entropy of text.

    Args:
        text: Input text to calculate entropy for

    Returns:
        Shannon entropy value in bits

    Note:
        Higher entropy indicates more information content and
        less repetition or redundancy.
    """
    if not isinstance(text, str) or not text:
        return 0.0

    # Count character frequencies
    char_counts = Counter(text)
    total_chars = sum(char_counts.values())

    # Calculate entropy
    return -sum(
        (count / total_chars) * math.log2(count / total_chars)
        for count in char_counts.values()
        if count > 0
    )


# Common Spanish stopwords
SPANISH_STOPWORDS = {
    "de",
    "la",
    "que",
    "el",
    "en",
    "y",
    "a",
    "los",
    "del",
    "se",
    "las",
    "por",
    "un",
    "para",
    "con",
    "no",
    "una",
    "su",
    "al",
    "lo",
}


def stopword_ratio(text: str, lang: str = "es") -> float:
    """Calculate ratio of stopwords in text.

    Args:
        text: Input text to analyze
        lang: Language code for stopword set (currently only 'es' supported)

    Returns:
        Ratio of stopwords to total words (0.0-1.0)

    Note:
        A ratio around 0.25 is typical for natural text. Significantly
        higher or lower values may indicate quality issues.
    """
    if not isinstance(text, str) or not text:
        return 0.0

    # Get stopword set for language
    stopwords = SPANISH_STOPWORDS if lang == "es" else set()

    # Extract and count words
    words = [w for w in re.findall(r"\w{2,}", text.lower())]
    if not words:
        return 0.0

    # Calculate ratio
    return sum(1 for w in words if w in stopwords) / len(words)


def digit_ratio(text: str) -> float:
    """Calculate ratio of digit characters in text.

    Args:
        text: Input text to analyze

    Returns:
        Ratio of digits to total characters (0.0-1.0)

    Note:
        High digit ratios may indicate data dumps, logs, or
        other non-narrative content.
    """
    if not isinstance(text, str) or not text:
        return 0.0

    digit_count = sum(char.isdigit() for char in text)
    return digit_count / max(1, len(text))


def quality_score(text: str) -> float:
    """Calculate overall quality score for text.

    Args:
        text: Input text to evaluate

    Returns:
        Quality score between 0.0 and 1.0

    Note:
        The score combines multiple metrics:
        - Information entropy (40%)
        - Stopword distribution (40%)
        - Digit ratio (20%)

        Higher scores indicate more natural, informative text.
    """
    # Basic validation
    if not isinstance(text, str) or len(text) < 200:
        return 0.0

    # Normalize text
    normalized = unidecode(text)

    # Calculate component scores
    entropy_score = min(1.0, entropy(normalized) / 5.0)
    stopword_score = 1.0 - abs(stopword_ratio(normalized) - 0.25)
    digit_score = 1.0 - min(1.0, digit_ratio(normalized) * 2.0)

    # Weighted combination
    score = 0.4 * entropy_score + 0.4 * stopword_score + 0.2 * digit_score

    # Ensure score is in [0,1]
    return max(0.0, min(1.0, score))
