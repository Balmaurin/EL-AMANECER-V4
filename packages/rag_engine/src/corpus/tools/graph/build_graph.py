import json
import re
from pathlib import Path

import duckdb
import pandas as pd


def simple_entities(text: str):
    return [
        w for w in re.findall(r"\b([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑ\-]{2,})\b", text) if len(w) > 2
    ]


def build_graph(
    base: Path, min_edge_weight=2, db_path="corpus/_registry/catalog.duckdb"
):
    nodes, edges = {}, {}
    for jf in (base / "chunks").glob("*.jsonl"):
        rec = json.loads(jf.read_text(encoding="utf-8"))
        ents = list(set(simple_entities(rec["text"])))[:30]
        for e in ents:
            nodes[e] = nodes.get(e, 0) + 1
        for i, a in enumerate(ents):
            for b in ents[i + 1 :]:
                key = (a, b) if a < b else (b, a)
                edges[key] = edges.get(key, 0) + 1
    # Ensure DataFrames have defined columns even when empty
    N = pd.DataFrame(
        [{"id": k, "freq": v} for k, v in nodes.items()], columns=["id", "freq"]
    )
    E = pd.DataFrame(
        [
            {"src": a, "dst": b, "weight": w}
            for (a, b), w in edges.items()
            if w >= min_edge_weight
        ],
        columns=["src", "dst", "weight"],
    )
    out = base / "index" / "graph"
    out.mkdir(parents=True, exist_ok=True)
    N.to_parquet(out / "nodes.parquet", index=False)
    E.to_parquet(out / "edges.parquet", index=False)
    # Ensure DuckDB directory exists
    dbp = Path(db_path)
    dbp.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(dbp))
    # Register DataFrames for SQL access
    con.register("N", N)
    con.register("E", E)
    con.execute(
        "CREATE TABLE IF NOT EXISTS kg_nodes(id VARCHAR, snapshot VARCHAR, freq INTEGER)"
    )
    con.execute(
        "CREATE TABLE IF NOT EXISTS kg_edges(src VARCHAR, dst VARCHAR, snapshot VARCHAR, weight INTEGER)"
    )
    con.execute("DELETE FROM kg_nodes WHERE snapshot=?", [base.name])
    con.execute("DELETE FROM kg_edges WHERE snapshot=?", [base.name])
    if not N.empty:
        con.execute(
            "INSERT INTO kg_nodes SELECT id, ? as snapshot, freq FROM N", [base.name]
        )
    if not E.empty:
        con.execute(
            "INSERT INTO kg_edges SELECT src, dst, ? as snapshot, weight FROM E",
            [base.name],
        )
    con.close()
    print("[OK] Grafo construido")
