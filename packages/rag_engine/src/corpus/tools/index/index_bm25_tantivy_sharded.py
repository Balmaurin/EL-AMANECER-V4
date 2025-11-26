from pathlib import Path

import yaml


def build_tantivy_sharded(base: Path):
    try:
        import tantivy
    except Exception:
        print("Tantivy no disponible, omitiendo sharded")
        return
    cfg = yaml.safe_load(open("config/universal.yaml", "r", encoding="utf-8"))
    shards = int(
        cfg.get("retrieval", {}).get("lexical", {}).get("shards", {}).get("count", 4)
    )
    shard_dir = base / "index" / "tantivy_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    # Simple hash-partition por doc_id
    import hashlib
    import json

    buckets = {i: [] for i in range(shards)}
    for jf in (base / "chunks").glob("*.jsonl"):
        rec = json.loads(jf.read_text(encoding="utf-8"))
        key = (
            int(hashlib.sha256(rec["doc_id"].encode("utf-8")).hexdigest(), 16) % shards
        )
        buckets[key].append(rec)
    schema = tantivy.SchemaBuilder()
    schema.add_text_field("chunk_id", stored=True)
    schema.add_text_field("doc_id", stored=True)
    schema.add_text_field("title", stored=True)
    schema.add_text_field("text", stored=True)
    schema = schema.build()
    for s in range(shards):
        sdir = shard_dir / f"shard_{s}"
        if sdir.exists():
            pass
        else:
            sdir.mkdir(parents=True, exist_ok=True)
        index = tantivy.Index.create(schema, sdir)
        writer = index.writer()
        for rec in buckets[s]:
            writer.add_document(
                {
                    "chunk_id": rec["chunk_id"],
                    "doc_id": rec["doc_id"],
                    "title": rec.get("meta", {}).get("title", ""),
                    "text": rec["text"],
                }
            )
        writer.commit()
    print(f"[OK] Tantivy sharded: {shards} shards")
