import pytest


def test_import_asr():
    from blackwell_inference.asr import benchmark_qwen3_asr

    assert callable(benchmark_qwen3_asr)
