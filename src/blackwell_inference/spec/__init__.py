"""Speculative decoding components."""

from .ngram import NGramProposer, SpecResult, greedy_generate, speculative_generate

__all__ = ["NGramProposer", "SpecResult", "greedy_generate", "speculative_generate"]
