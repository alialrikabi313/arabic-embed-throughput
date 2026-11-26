# -*- coding: utf-8 -*-
"""اختبارات التكافؤ بقيم تُحسب يدوياً."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parity import cosine_stats, topk_overlap


def test_identical_embeddings_have_cosine_one():
    x = np.array([[3.0, 4.0], [1.0, 0.0]])
    assert cosine_stats(x, x)["cos_min"] > 1 - 1e-12


def test_orthogonal_rows_have_cosine_zero_and_min_catches_it():
    ref = np.array([[1.0, 0.0], [1.0, 0.0]])
    other = np.array([[1.0, 0.0], [0.0, 1.0]])
    s = cosine_stats(ref, other)
    # المتوسط 0.5 يُخفي الصف السيئ؛ الأدنى يكشفه
    assert abs(s["cos_mean"] - 0.5) < 1e-12 and abs(s["cos_min"]) < 1e-12


def test_topk_overlap_full_and_partial():
    p = np.eye(4)
    q_ref = np.array([[4.0, 3.0, 2.0, 1.0]])  # ترتيب المرجع: 0,1,2,3
    q_same = q_ref.copy()
    q_swap = np.array([[4.0, 1.0, 2.0, 3.0]])  # أعلى اثنين: 0 ثم 3
    assert topk_overlap(q_ref, p, q_same, p, k=2) == 1.0
    # أعلى اثنين للمرجع {0,1} وللآخر {0,3}: تقاطع 1 من 2
    assert topk_overlap(q_ref, p, q_swap, p, k=2) == 0.5
