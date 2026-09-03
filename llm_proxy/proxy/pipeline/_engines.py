"""Cached Presidio engine singletons.

Loading the AnalyzerEngine triggers spaCy model load (~hundreds of MB
of RAM) and is slow (~1s). We do it once at first use and reuse.
"""

from __future__ import annotations

from functools import lru_cache

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine, DeanonymizeEngine


@lru_cache(maxsize=1)
def get_analyzer() -> AnalyzerEngine:
    return AnalyzerEngine()


@lru_cache(maxsize=1)
def get_anonymizer() -> AnonymizerEngine:
    return AnonymizerEngine()


@lru_cache(maxsize=1)
def get_deanonymizer() -> DeanonymizeEngine:
    return DeanonymizeEngine()
