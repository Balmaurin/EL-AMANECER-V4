#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sheily_text_cleaner.py
======================
Simple text cleaning utilities
"""

import re


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""

    # Basic cleaning
    text = text.strip()

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove control characters
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)

    return text
