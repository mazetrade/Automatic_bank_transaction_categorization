"""Cleaning of raw bank transaction labels."""

import re
import unicodedata

DATE_PATTERN = re.compile(r"\b\d{2}/\d{2,4}(/\d{2,4})?\b")  # 12/03, 10/2026, 12/03/2026
SEPA_REF_PATTERN = re.compile(r"\bFR\d{2}ZZZ\d+\b")          # FR84ZZZ123456
CARD_PATTERN = re.compile(r"\bX\d{4}\b")                      # X4821
PUNCTUATION_PATTERN = re.compile(r"[^A-Z0-9 ]")               # anything but letters, digits, spaces
LONG_NUMBER_PATTERN = re.compile(r"\b\d{4,}\b")               # store codes, references
SPACES_PATTERN = re.compile(r"\s+")                           # runs of spaces


def strip_accents(text):
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def clean_label(label):
    """Normalize a raw bank label: remove dates, card numbers, references and punctuation."""
    text = strip_accents(str(label).upper())
    text = DATE_PATTERN.sub(" ", text)
    text = SEPA_REF_PATTERN.sub(" ", text)
    text = CARD_PATTERN.sub(" ", text)
    text = PUNCTUATION_PATTERN.sub(" ", text)
    text = LONG_NUMBER_PATTERN.sub(" ", text)
    return SPACES_PATTERN.sub(" ", text).strip()