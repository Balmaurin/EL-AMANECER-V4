"""
RAPTOR (Recursive Abstractive Processing and Topical Organization for Retrieval) Tree Builder.

This module implements the RAPTOR algorithm for hierarchical document organization.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def build_raptor(base: Path, levels: int = 3, k_per_cluster: int = 8) -> None:
    """Build a RAPTOR tree from document chunks.

    Args:
        base: Base directory containing chunks and where index will be stored
        levels: Number of hierarchical levels in the tree (default: 3)
        k_per_cluster: Target number of documents per cluster (default: 8)
    """
    # Load all chunks from the chunks directory
    chunks_dir = base / "chunks"
    if not chunks_dir.exists():
        print("Error: Chunks directory not found")
        return

    chunks = []
    for f in chunks_dir.glob("*.jsonl"):
        try:
            chunks.append(json.loads(f.read_text(encoding="utf-8")))
        except json.JSONDecodeError as e:
            print(f"Error reading chunk file {f}: {e}")
            continue

    if not chunks:
        print("No chunks found for RAPTOR processing")
        return
    # Create DataFrame and compute TF-IDF features
    df = pd.DataFrame(chunks)
    if "text" not in df.columns:
        print("Error: Chunks must contain 'text' field")
        return

    vectorizer = TfidfVectorizer(
        max_features=4096, ngram_range=(1, 2), strip_accents="unicode"
    )
    X = vectorizer.fit_transform(df["text"])

    # Initialize root node
    current = [{"level": 0, "df": df, "X": X, "parent": None, "name": "L0"}]

    # Create output directory
    outdir = base / "index" / "raptor"
    outdir.mkdir(parents=True, exist_ok=True)

    def generate_cluster_summary(texts: List[str]) -> str:
        """Generate a summary for a cluster using top TF-IDF terms."""
        text = " ".join(texts)
        tfidf = TfidfVectorizer(stop_words="english").fit([text])
        vocab = sorted(
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()],
            key=lambda x: -x[1],
        )[:20]
        return " ".join(w for w, _ in vocab)

    # Build tree level by level
    for lvl in range(1, levels + 1):
        next_level = []

        for node in current:
            # Skip small nodes that don't need further splitting
            if len(node["df"]) <= k_per_cluster:
                continue

            # Determine number of clusters
            k = max(2, len(node["df"]) // k_per_cluster)

            # Perform clustering
            km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(node["X"])
            subdf = node["df"].copy()
            subdf["cluster"] = km.labels_

            # Process each cluster
            for cid, group in subdf.groupby("cluster"):
                # Generate cluster summary
                summary = generate_cluster_summary(group["text"].tolist())

                # Save cluster data
                output_file = outdir / f"L{lvl}_{node.get('name', 'root')}_{cid}.jsonl"
                cluster_data = {
                    "level": lvl,
                    "parent": node.get("name"),
                    "summary": summary,
                    "chunk_ids": group["chunk_id"].tolist(),
                }

                try:
                    output_file.write_text(
                        json.dumps(cluster_data, ensure_ascii=False), encoding="utf-8"
                    )
                except IOError as e:
                    print(f"Error writing cluster file {output_file}: {e}")
                    continue

                # Add node to next level
                next_level.append(
                    {
                        "level": lvl,
                        "df": group,
                        "X": X[group.index],
                        "parent": node.get("name"),
                        "name": f"L{lvl}_{cid}",
                    }
                )

        current = next_level

    print("[OK] RAPTOR tree construction completed")
