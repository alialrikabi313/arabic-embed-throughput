# -*- coding: utf-8 -*-
"""قياس جودة الاسترجاع الفعلي (nDCG@10) على مجموعة فرعية من MIRACL-ar dev. دوال نقية بلا GPU."""

import math
import random


def read_qrels(path) -> dict[str, dict[str, int]]:
    """qid → {docid: صلة}. الصيغة الفعلية: qid<TAB>Q0<TAB>docid<TAB>rel."""
    qrels: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, _, docid, rel = line.split()
            qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels


def choose_subset(qrels: dict[str, dict[str, int]], all_docids: list[str], n_queries: int,
                  n_passages: int, seed: int = 0) -> tuple[list[str], set[str]]:
    """يختار استعلامات لها موجب واحد على الأقل، ويبني مجموعة مقاطع تحوي كل موجباتها وسلبياتها المُحكَّمة،
    ثم يكمّل بمشتّتات عشوائية حتى n_passages. التثبيت بالبذرة فتتطابق المجموعة بين التركيبات."""
    rng = random.Random(seed)
    eligible = sorted(q for q, r in qrels.items() if any(v > 0 for v in r.values()))
    qids = sorted(rng.sample(eligible, min(n_queries, len(eligible))))
    keep = {d for q in qids for d in qrels[q]}  # الموجبة والمُحكَّمة السلبية معاً
    if len(keep) > n_passages:
        raise ValueError(f"المقاطع المحكَّمة ({len(keep)}) تتجاوز الحجم المطلوب ({n_passages})")
    pool = [d for d in all_docids if d not in keep]
    keep |= set(rng.sample(pool, n_passages - len(keep)))
    return qids, keep


def ndcg_at_k(ranked: list[str], rels: dict[str, int], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if rels.get(d, 0) > 0)
    n_rel = sum(1 for r in rels.values() if r > 0)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(n_rel, k)))
    return dcg / ideal if ideal else 0.0
