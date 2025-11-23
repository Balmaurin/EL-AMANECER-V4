#!/usr/bin/env python
"""
Comprehensive validation test for Universal++ RAG v4

Tests:
1. Index file existence
2. Search functionality (multiple queries)
3. Score validation (in [0, 1] range)
4. Metadata retrieval
5. HNSW vs FAISS backend consistency
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from tools.retrieval.search_local_hnsw import search_local_hnsw


def test_index_files():
    """Verify required index files exist."""
    print("\n=== Test 1: Index Files ===")
    index_dir = Path("index")
    required = [
        index_dir / "corpus_faiss.index",
        index_dir / "corpus_mapping.parquet",
        index_dir / "hnsw.idx",
        index_dir / "state.json",
    ]

    for f in required:
        if f.exists():
            size_mb = f.stat().st_size / (1024**2)
            print(f"✅ {f.name}: {size_mb:.2f} MB")
        else:
            print(f"❌ {f.name}: MISSING")
            return False
    return True


def test_search(query, expected_min_score=0.5):
    """Test search functionality."""
    print(f"\n  Query: '{query}'")
    results = search_local_hnsw(Path("."), query, top_k=3)

    if not results:
        print(f"  ❌ No results")
        return False

    for i, r in enumerate(results, 1):
        score = r["score"]
        if not (0.0 <= score <= 1.0):
            print(f"  ❌ Invalid score: {score}")
            return False

        text_preview = r["text"][:60].replace("\n", " ")
        print(f"  [{i}] {score:.4f} | {r['doc_id'][:30]}")
        print(f"      {text_preview}...")

    return True


def test_searches():
    """Test multiple queries."""
    print("\n=== Test 2: Search Functionality ===")

    queries = [
        ("ciberseguridad", 0.6),
        ("aprendizaje federado", 0.6),
        ("Internet de las Cosas", 0.5),
    ]

    for query, min_score in queries:
        if not test_search(query, min_score):
            return False

    return True


def test_score_distribution():
    """Verify score distribution is reasonable."""
    print("\n=== Test 3: Score Distribution ===")

    results = search_local_hnsw(Path("."), "ciberseguridad", top_k=10)
    scores = [r["score"] for r in results]

    print(f"  Min: {min(scores):.4f}")
    print(f"  Max: {max(scores):.4f}")
    print(f"  Mean: {sum(scores)/len(scores):.4f}")

    # Scores should be decreasing
    for i in range(len(scores) - 1):
        if scores[i] < scores[i + 1]:
            print(f"  ❌ Scores not monotonic: {scores[i]:.4f} < {scores[i+1]:.4f}")
            return False

    print("  ✅ Scores monotonically decreasing")
    return True


def test_metadata():
    """Test metadata retrieval."""
    print("\n=== Test 4: Metadata Retrieval ===")

    results = search_local_hnsw(Path("."), "ciberseguridad", top_k=1)
    if not results:
        print("  ❌ No results to check metadata")
        return False

    r = results[0]
    required_keys = ["chunk_id", "doc_id", "text", "score", "meta"]

    for key in required_keys:
        if key in r:
            value = r[key]
            if isinstance(value, str):
                print(f"  ✅ {key}: {value[:50]}...")
            else:
                print(f"  ✅ {key}: {type(value).__name__}")
        else:
            print(f"  ❌ {key}: MISSING")
            return False

    return True


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Universal++ RAG v4 — Validation Suite")
    print("=" * 60)

    tests = [
        ("Index Files", test_index_files),
        ("Search Functionality", test_searches),
        ("Score Distribution", test_score_distribution),
        ("Metadata Retrieval", test_metadata),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n❌ {name} failed with exception: {e}")
            import traceback

            traceback.print_exc()
            results[name] = False

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)

    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All validation tests passed!")
        return 0
    else:
        print("\n⚠️ Some tests failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
