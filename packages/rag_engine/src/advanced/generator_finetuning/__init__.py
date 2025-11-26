"""
Generator Fine-tuning for RAG
Based on EMNLP 2024 Paper Section A.6

Implements LoRA fine-tuning for RAG generators with:
- Dg method (Document-grounded training)
- QA datasets (ASQA, HotpotQA, NQ, TriviaQA)
- 3 epoch training with optimal hyperparameters
"""

from .generator_finetuner import FinetuningMethod, FinetuningResult, GeneratorFinetuner

__all__ = ["GeneratorFinetuner", "FinetuningResult", "FinetuningMethod"]
