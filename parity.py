# -*- coding: utf-8 -*-
"""مقاييس التكافؤ بين تضمينين (مرجع مقابل دقة أقل). نقية بلا GPU فتُختبر بسرعة."""

import numpy as np


def cosine_stats(ref: np.ndarray, other: np.ndarray) -> dict[str, float]:
    """تشابه الكوساين بين كل صف ونظيره؛ الأدنى هو ما يكشف الحالات السيئة لا المتوسط."""
    a = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    b = other / np.linalg.norm(other, axis=1, keepdims=True)
    cos = (a * b).sum(axis=1)
    return {"cos_mean": float(cos.mean()), "cos_min": float(cos.min())}


def topk_overlap(ref_q: np.ndarray, ref_p: np.ndarray, q: np.ndarray, p: np.ndarray, k: int = 10) -> float:
    """متوسط نسبة تقاطع أعلى k مقاطع بين ترتيب المرجع وترتيب التضمين الآخر لنفس الاستعلامات."""
    top_ref = np.argsort(-(ref_q @ ref_p.T), axis=1)[:, :k]
    top_oth = np.argsort(-(q @ p.T), axis=1)[:, :k]
    shared = [len(set(r) & set(o)) / k for r, o in zip(top_ref, top_oth)]
    return float(np.mean(shared))
