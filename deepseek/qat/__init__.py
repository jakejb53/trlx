"""Quantization-aware training support for DeepSeek-V4-Flash.

This package runs inside the NeMo AutoModel environment and must not import
``trlx`` or ``dataset``: their transformers pin differs from NeMo's.
"""
