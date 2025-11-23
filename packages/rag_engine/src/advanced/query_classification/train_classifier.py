#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Script for Query Classifier
Based on EMNLP 2024 Paper Section A.1

Trains BERT-base-multilingual-cased classifier to 95% accuracy
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from packages.rag_engine.src.advanced.query_classification import QueryClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Query Classifier for RAG")
    parser.add_argument(
        "--model_name",
        default="bert-base-multilingual-cased",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size (paper: 16)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Learning rate (paper: 1e-5)"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--output_dir",
        default="models/query_classifier",
        help="Output directory for model",
    )
    parser.add_argument(
        "--test_run", action="store_true", help="Run a quick test with small dataset"
    )

    args = parser.parse_args()

    # Initialize classifier
    classifier = QueryClassifier(
        model_name=args.model_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        model_path=args.output_dir,
    )

    # Load base model
    logger.info("Loading base model...")
    if not classifier.load_model():
        logger.error("Failed to load base model")
        return 1

    # Train the model
    logger.info("Starting training...")
    try:
        if args.test_run:
            # Create small test dataset
            test_texts = [
                "What is the capital of France?",
                "Translate hello to Spanish",
                "Please continue writing this paragraph",
                "Search for information about machine learning",
                "Summarize this article",
                "Find the population of Tokyo",
            ]
            test_labels = [1, 0, 0, 1, 0, 1]  # 1=retrieval required, 0=no retrieval

            metrics = classifier.train(
                train_texts=test_texts,
                train_labels=test_labels,
                num_epochs=1,
                save_steps=10,
                eval_steps=10,
            )
        else:
            # Full training with paper dataset
            metrics = classifier.train(
                num_epochs=args.epochs, save_steps=500, eval_steps=500
            )

        logger.info("Training completed!")
        logger.info(f"Final metrics: {metrics}")

        # Test the trained model
        logger.info("Testing trained model...")
        test_queries = [
            "What is artificial intelligence?",
            "Translate 'hello world' to French",
            "Please continue this story",
            "Find information about climate change",
            "Summarize the benefits of exercise",
            "Who won the Nobel Prize in Physics 2023?",
        ]

        for query in test_queries:
            result = classifier.classify(query)
            logger.info(f"Query: '{query[:50]}...'")
            logger.info(f"  Needs retrieval: {result.needs_retrieval}")
            logger.info(f"  Confidence: {result.confidence:.3f}")
            logger.info(f"  Class: {result.predicted_class}")
            logger.info("")

        # Save final model
        classifier.save_model()
        logger.info(f"Model saved to {args.output_dir}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
