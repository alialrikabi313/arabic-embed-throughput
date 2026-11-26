# -*- coding: utf-8 -*-
"""اختبارات مجموعة الجودة. المعرّفات بصيغة MIRACL الفعلية (docid من الشكل «رقم#فقرة»)."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quality import choose_subset, ndcg_at_k, read_qrels

_QRELS = {"7": {"1#0": 1, "2#0": 0}, "9": {"3#0": 1}, "11": {"4#0": 0}}  # «11» بلا موجب
_DOCS = [f"{i}#0" for i in range(1, 60)]


def test_subset_keeps_every_judged_passage_of_chosen_queries():
    qids, keep = choose_subset(_QRELS, _DOCS, n_queries=2, n_passages=20)
    assert qids == ["7", "9"]  # الاستعلام 11 بلا موجب يُستبعد (nDCG غير معرَّف له)
    assert {"1#0", "2#0", "3#0"} <= keep and len(keep) == 20


def test_subset_is_deterministic_for_a_seed():
    a = choose_subset(_QRELS, _DOCS, 2, 20, seed=3)
    b = choose_subset(_QRELS, _DOCS, 2, 20, seed=3)
    assert a == b


def test_subset_too_small_for_judged_passages_raises():
    with pytest.raises(ValueError):
        choose_subset(_QRELS, _DOCS, 2, 2)


def test_ndcg_hand_values():
    assert ndcg_at_k(["a"], {"a": 1}) == 1.0
    assert math.isclose(ndcg_at_k(["x", "y", "a"], {"a": 1}), 0.5)
    # سلبي مُحكَّم (rel=0) لا يُحتسب موجباً ولا يدخل المثالي
    assert ndcg_at_k(["n", "a"], {"n": 0, "a": 1}) == 1 / math.log2(3)


def test_read_qrels_real_miracl_format(tmp_path):
    p = tmp_path / "q.tsv"
    p.write_text("7\tQ0\t1#0\t1\n7\tQ0\t2#0\t0\n", encoding="utf-8")
    assert read_qrels(p) == {"7": {"1#0": 1, "2#0": 0}}
