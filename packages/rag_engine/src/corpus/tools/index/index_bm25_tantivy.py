import json
from pathlib import Path

try:
    import tantivy
except Exception as e:
    tantivy = None


def _schema():
    sb = tantivy.SchemaBuilder()
    sb.add_text_field("chunk_id", stored=True)
    sb.add_text_field("doc_id", stored=True)
    sb.add_text_field("title", stored=True)
    sb.add_text_field("lang", stored=True)
    sb.add_text_field("tags", stored=True)
    sb.add_text_field("text", stored=True)
    return sb.build()


def build_tantivy(base: Path):
    """Build or incrementally update a Tantivy BM25 index under base/index/tantivy."""
    if tantivy is None:
        print("tantivy no instalado; usa Whoosh o instala tantivy.")
        return
    idx_dir = base / "index" / "tantivy"
    idx_dir.mkdir(parents=True, exist_ok=True)
    seen_file = idx_dir / "seen.json"
    try:
        seen = (
            set(json.loads(seen_file.read_text(encoding="utf-8")))
            if seen_file.exists()
            else set()
        )
    except Exception:
        seen = set()

    if (idx_dir / "meta.json").exists():
        index = tantivy.Index.open(idx_dir)
    else:
        schema = _schema()
        index = tantivy.Index(schema, path=idx_dir)

    writer = index.writer()
    added = 0
    for jf in (base / "chunks").glob("*.jsonl"):
        obj = json.loads(jf.read_text(encoding="utf-8"))
        cid = obj.get("chunk_id")
        if cid in seen:
            continue
        meta = obj.get("meta", {})
        writer.add_document(
            {
                "chunk_id": str(cid),
                "doc_id": str(obj.get("doc_id", "")),
                "title": str(meta.get("title", "")),
                "lang": str(meta.get("lang", "")),
                "tags": ",".join(meta.get("tags", [])),
                "text": str(obj.get("text", "")),
            }
        )
        seen.add(cid)
        added += 1
    writer.commit()
    seen_file.write_text(json.dumps(sorted(list(seen))), encoding="utf-8")
    if added:
        print(f"[OK] Tantivy BM25 incremental: +{added} chunks")
    else:
        print("Tantivy BM25 ya actualizado.")
