# -*- coding: utf-8 -*-
"""المرحلة 1: قياس إنتاجية وزمن استدلال نماذج التضمين على GPU.

يقرأ: مقاطع واستعلامات MIRACL-ar الحقيقية. ينتج: results/throughput.json.
لكل (نموذج، دقة، انتباه): تضمين مرجعي fp32/eager للمقارنة ← إنتاجية لكل حجم دفعة
(وسيط عدة جولات مع إحماء) ← زمن استعلام مفرد ← ذروة الذاكرة ← تكافؤ الجودة مع المرجع.
التركيبة غير المدعومة تُسجَّل بسببها ولا تُخفى.
"""

import gzip
import json
import statistics
import sys
import time

import numpy as np

from config import (
    ATTENTION,
    BATCH_SIZES,
    CORPUS_SHARD,
    LATENCY_QUERIES,
    MAX_SEQ_LEN,
    MODELS,
    N_PASSAGES,
    N_QUERIES,
    PRECISIONS,
    QUERIES_TSV,
    RESULTS_JSON,
    SEED,
    TIMED_RUNS,
    WARMUP_RUNS,
)
from parity import cosine_stats, topk_overlap


def load_inputs() -> tuple[list[str], list[str]]:
    passages = []
    with gzip.open(CORPUS_SHARD, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            passages.append(f"{d['title']} {d['text']}")
            if len(passages) == N_PASSAGES:
                break
    queries = [l.split("\t", 1)[1].strip() for l in QUERIES_TSV.open(encoding="utf-8")]
    rng = np.random.default_rng(SEED)
    return passages, [queries[i] for i in rng.choice(len(queries), N_QUERIES, replace=False)]


def timed(fn, sync) -> float:
    sync()
    t0 = time.perf_counter()
    fn()
    sync()
    return time.perf_counter() - t0


def main() -> None:
    if not CORPUS_SHARD.exists():
        sys.exit(f"لا توجد بيانات MIRACL: شغّل 01_download_data.py في arabic-retrieval-finetune ({CORPUS_SHARD})")
    # استيراد كسول: مكتبات ثقيلة
    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        sys.exit("يتطلب GPU (CUDA غير متاح الآن)")
    sync = torch.cuda.synchronize
    dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    passages, queries = load_inputs()
    gpu = torch.cuda.get_device_name(0)
    results: list[dict] = []

    for name, ppref in MODELS.items():
        ref_p = ref_q = None
        docs = [ppref + p for p in passages]
        for prec in PRECISIONS:
            for attn in ATTENTION:
                cfg = {"model": name, "precision": prec, "attention": attn}
                print(f"◄ {name.split('/')[-1]} {prec} {attn}", flush=True)
                try:
                    model = SentenceTransformer(
                        name, device="cuda",
                        model_kwargs={"torch_dtype": dtypes[prec], "attn_implementation": attn},
                    )
                except (ValueError, NotImplementedError) as e:
                    results.append({**cfg, "error": str(e)[:160]})
                    print(f"  ✗ غير مدعوم: {str(e)[:100]}", flush=True)
                    continue
                model.max_seq_length = MAX_SEQ_LEN
                torch.cuda.reset_peak_memory_stats()
                row = {**cfg, "gpu": gpu, "throughput": {}}

                for bs in BATCH_SIZES:
                    enc = lambda: model.encode(docs, batch_size=bs, show_progress_bar=False)
                    for _ in range(WARMUP_RUNS):
                        enc()
                    secs = [timed(enc, sync) for _ in range(TIMED_RUNS)]
                    row["throughput"][str(bs)] = {
                        "passages_per_s": N_PASSAGES / statistics.median(secs),
                        "min": N_PASSAGES / max(secs), "max": N_PASSAGES / min(secs),
                    }
                row["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9

                lat = []
                for q in queries[:LATENCY_QUERIES]:
                    lat.append(timed(lambda: model.encode([q], show_progress_bar=False), sync) * 1000)
                row["latency_ms_p50"] = float(np.percentile(lat, 50))
                row["latency_ms_p95"] = float(np.percentile(lat, 95))

                p_emb = model.encode(docs, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
                q_emb = model.encode(queries, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
                p_emb, q_emb = p_emb.astype(np.float32), q_emb.astype(np.float32)
                if ref_p is None:  # أول تركيبة fp32/eager هي المرجع
                    ref_p, ref_q = p_emb, q_emb
                row.update(cosine_stats(ref_p, p_emb))
                row["top10_overlap"] = topk_overlap(ref_q, ref_p, q_emb, p_emb)
                results.append(row)
                del model
                torch.cuda.empty_cache()

    RESULTS_JSON.parent.mkdir(exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"✓ {len(results)} تركيبة ← {RESULTS_JSON}")


if __name__ == "__main__":
    main()
