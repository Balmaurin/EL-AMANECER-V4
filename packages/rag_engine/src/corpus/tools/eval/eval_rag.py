"""
RAG System Evaluation Module.

This module provides tools for evaluating RAG system performance using:
- Precision and recall metrics
- Mean Reciprocal Rank (MRR)
- Context relevance scoring
- Gold standard comparison

Input Dataset Format (JSONL):
{
    "question": str,           # Query/question text
    "gold": List[str],        # Expected keywords/snippets in retrieved context
    "metadata": Dict          # Optional additional metadata
}

Example:
    python tools/eval/eval_rag.py --data data/eval/dev.jsonl --k 6 --mode hybrid
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from tools.common.paths import CORPUS_DIR
from tools.retrieval.search_unified import unified_search

# Configure logging
log = logging.getLogger("rag.eval")


@dataclass
class EvalQuery:
    """Evaluation query data structure."""

    question: str
    gold: List[str]
    metadata: Dict[str, Any] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalQuery":
        """Create EvalQuery from dictionary."""
        return cls(
            question=data["question"],
            gold=data.get("gold", []),
            metadata=data.get("metadata", {}),
        )


def load_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Load evaluation dataset from JSONL file.

    Args:
        path: Path to JSONL file

    Yields:
        Dictionary containing query data

    Raises:
        FileNotFoundError: If file not found
        json.JSONDecodeError: If file contains invalid JSON
    """
    try:
        content = path.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(f"Invalid JSON in line: {e}")
                continue
    except FileNotFoundError:
        log.error(f"Evaluation file not found: {path}")
        raise


def calculate_mrr(ranks: Sequence[int]) -> float:
    """Calculate Mean Reciprocal Rank.

    Args:
        ranks: List of ranks where relevant items were found

    Returns:
        MRR score between 0.0 and 1.0
    """
    if not ranks:
        return 0.0

    reciprocal_ranks = [1.0 / (rank + 1) for rank in ranks]
    return np.mean(reciprocal_ranks)


@dataclass
class EvalResults:
    """Evaluation results container."""

    precision: float
    recall: float
    mrr: float
    total_queries: int
    relevant_hits: int

    def __str__(self) -> str:
        """Format results for display."""
        return (
            f"Evaluation Results:\n"
            f"  Total Queries:     {self.total_queries}\n"
            f"  Relevant Hits:     {self.relevant_hits}\n"
            f"  Precision:         {self.precision:.3f}\n"
            f"  Recall:           {self.recall:.3f}\n"
            f"  MRR:              {self.mrr:.3f}"
        )


def evaluate_retrieval(
    base: Path,
    queries: List[EvalQuery],
    top_k: int = 6,
    mode: str = "hybrid",
    show_progress: bool = True,
) -> EvalResults:
    """Evaluate retrieval performance on a set of queries.

    Args:
        base: Base directory containing indices
        queries: List of evaluation queries
        top_k: Number of results to retrieve per query
        mode: Search mode (hybrid, bm25, dense)
        show_progress: Whether to show progress bar

    Returns:
        EvalResults containing evaluation metrics
    """
    hit_precision_sum = 0.0
    hit_recall_sum = 0.0
    mrr_ranks = []
    total_relevant = 0

    # Process each query
    for query in tqdm(queries, disable=not show_progress):
        try:
            # Execute search
            hits = unified_search(
                "universal", base, query.question, top_k=top_k, mode=mode
            )

            # Track relevant hits
            relevant_hits = 0
            first_relevant_rank = None

            # Check each result against gold standard
            for i, hit in enumerate(hits):
                text = hit.get("text", "").lower()

                # Check if result contains any gold standard text
                if any(gold.lower() in text for gold in query.gold):
                    relevant_hits += 1
                    if first_relevant_rank is None:
                        first_relevant_rank = i

            # Calculate precision and recall for this query
            precision = relevant_hits / len(hits) if hits else 0.0
            hit_precision_sum += precision

            # Calculate recall based on gold standard coverage
            context = " ".join(h.get("text", "") for h in hits).lower()
            covered_gold = sum(1 for g in query.gold if g.lower() in context)
            recall = covered_gold / len(query.gold) if query.gold else 0.0
            hit_recall_sum += recall

            # Track MRR
            if first_relevant_rank is not None:
                mrr_ranks.append(first_relevant_rank)

            total_relevant += relevant_hits

        except Exception as e:
            log.error(f"Error evaluating query '{query.question[:50]}...': {e}")
            continue

    # Calculate final metrics
    num_queries = len(queries)
    return EvalResults(
        precision=hit_precision_sum / num_queries,
        recall=hit_recall_sum / num_queries,
        mrr=calculate_mrr(mrr_ranks),
        total_queries=num_queries,
        relevant_hits=total_relevant,
    )


def main() -> None:
    """Main evaluation script entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG system retrieval performance"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to evaluation dataset (JSONL)"
    )
    parser.add_argument(
        "--k", type=int, default=6, help="Number of results to retrieve per query"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "bm25", "dense"],
        help="Search mode to evaluate",
    )
    args = parser.parse_args()

    # Configure base directory using absolute path
    base = CORPUS_DIR / "universal"
    latest_ptr = base / "latest.ptr"

    if latest_ptr.exists():
        snapshot = latest_ptr.read_text(encoding="utf-8").strip()
        base = base / snapshot if (base / snapshot).exists() else Path(snapshot)

    # Load and process evaluation dataset
    try:
        queries = [EvalQuery.from_dict(row) for row in load_jsonl(Path(args.data))]
    except Exception as e:
        log.error(f"Error loading evaluation dataset: {e}")
        return

    # Run evaluation
    try:
        results = evaluate_retrieval(
            base=base, queries=queries, top_k=args.k, mode=args.mode
        )
        print(results)

    except Exception as e:
        log.error(f"Evaluation failed: {e}")


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
