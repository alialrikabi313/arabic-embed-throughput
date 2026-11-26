# -*- coding: utf-8 -*-
"""المرحلة 3: بناء مجموعة الجودة (500 استعلام dev حقيقي + 20 ألف مقطع) لقياس nDCG@10.

يقرأ: كوربس MIRACL-ar وأحكام الصلة (arabic-retrieval-finetune). ينتج في QUALITY_DIR:
quality_subset.jsonl (المقاطع) و quality_queries.json (الاستعلامات مع أحكامها). بلا GPU.
"""

import gzip
import json
import sys

from config import (
    CORPUS_SHARDS,
    QRELS_TSV,
    QUALITY_DIR,
    QUALITY_N_PASSAGES,
    QUALITY_N_QUERIES,
    QUALITY_QUERIES_JSON,
    QUALITY_SUBSET_JSONL,
    QUERIES_TSV,
    SEED,
)
from quality import choose_subset, read_qrels


def main() -> None:
    if not QRELS_TSV.exists():
        sys.exit(f"لا توجد بيانات MIRACL: شغّل 01_download_data.py في arabic-retrieval-finetune ({QRELS_TSV})")
    qrels = read_qrels(QRELS_TSV)
    topics = {}
    with QUERIES_TSV.open(encoding="utf-8") as f:
        for line in f:
            qid, query = line.rstrip("\n").split("\t", 1)
            topics[qid] = query

    # مرور أول: كل المعرّفات فقط (الاختيار يحتاج القائمة كاملة)
    docids: list[str] = []
    for shard in CORPUS_SHARDS:
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            docids += [json.loads(line)["docid"] for line in f]
    qids, keep = choose_subset(qrels, docids, QUALITY_N_QUERIES, QUALITY_N_PASSAGES, SEED)
    print(f"← {len(qids)} استعلاماً، {len(keep)} مقطعاً من {len(docids):,}")

    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    with QUALITY_SUBSET_JSONL.open("w", encoding="utf-8") as out:
        for shard in CORPUS_SHARDS:
            with gzip.open(shard, "rt", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    if d["docid"] in keep:
                        out.write(json.dumps(d, ensure_ascii=False) + "\n")
                        n += 1
    QUALITY_QUERIES_JSON.write_text(
        json.dumps({q: {"query": topics[q], "rels": qrels[q]} for q in qids}, ensure_ascii=False),
        encoding="utf-8")
    print(f"✓ {n} مقطع ← {QUALITY_SUBSET_JSONL}")


if __name__ == "__main__":
    main()
