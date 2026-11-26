# -*- coding: utf-8 -*-
"""المرحلة 4: تقنيات الاستدلال (torch.compile و ONNX Runtime و TensorRT وكمّية INT8/FP8) على GPU.

يعمل داخل WSL (Triton غير متاح على ويندوز). نفس منهجية 01_bench.py: 4000 مقطع حقيقي، طول 256،
وسيط 3 جولات بعد إحماء مع cuda.synchronize؛ ويضيف nDCG@10 الفعلي على مجموعة الجودة (03).
كل تركيبة تُسجَّل بنجاحها أو سبب فشلها. قابل للاستئناف (يتخطى (نموذج، تقنية) المسجَّلة).
القفل المشترك للـGPU يُمسَك أثناء كل تركيبة وحدها كي لا تفسد قياسات الآخرين زمنها.
"""

import contextlib
import gzip
import json
import os
import statistics
import sys
import tempfile
import time

import numpy as np

from config import (
    BATCH_SIZES,
    CORPUS_SHARD,
    GPU_LOCK,
    LATENCY_QUERIES,
    MAX_SEQ_LEN,
    MODELS,
    N_PASSAGES,
    N_QUERIES,
    QUALITY_QUERIES_JSON,
    QUALITY_SUBSET_JSONL,
    QUERIES_TSV,
    QUERY_PREFIX,
    SEED,
    SYSTEMS_RESULTS_JSON,
    TECHNIQUES,
    TIMED_RUNS,
    WARMUP_RUNS,
)
from parity import cosine_stats, topk_overlap
from quality import ndcg_at_k

LOCK_POLL_S = 1  # فحص كل ثانية: بفحص 20ث خسرنا سباق التحرير ساعة كاملة أمام من يعيد الإمساك فوراً


@contextlib.contextmanager
def gpu_lock():
    """قفل مجلد ذرّي مشترك: لا يُزال قفل غيرنا أبداً؛ ننتظر ثم نمسك ونُفرج في finally."""
    while True:
        try:
            os.mkdir(GPU_LOCK)
            break
        except FileExistsError:
            pass
            time.sleep(LOCK_POLL_S)
    try:
        yield
    finally:
        os.rmdir(GPU_LOCK)


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class OrtEncoder:
    """ONNX Runtime: يصدّر النموذج fp16 مرة (التجميع داخل الرسم) ثم يرمّز بترتيب الأطوال نفسه في ST."""

    def __init__(self, st_model, name: str, providers: list, tmp_dir: str):
        import onnxruntime as ort
        import torch

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        pooling = st_model[1].pooling_mode  # mean (MiniLM, e5) أو cls (bge-m3)؛ في sentence-transformers 6 خاصية لا دالة
        self.tok = st_model.tokenizer
        base = st_model[0].auto_model

        class Wrap(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = base

            def forward(self, input_ids, attention_mask):
                h = self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                if pooling == "cls":
                    return h[:, 0]
                mask = attention_mask.unsqueeze(-1).to(h.dtype)
                return (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        path = os.path.join(tmp_dir, name.replace("/", "_") + ".onnx")
        w = Wrap().half().cuda().eval()
        ids = torch.ones(2, 8, dtype=torch.long, device="cuda")
        torch.onnx.export(
            w, (ids, ids), path, input_names=["input_ids", "attention_mask"], output_names=["emb"],
            dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "emb": {0: "b"}},
            opset_version=17, dynamo=False,
        )
        del w
        torch.cuda.empty_cache()
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(path, so, providers=providers)
        wanted = providers[0] if isinstance(providers[0], str) else providers[0][0]
        if wanted not in self.sess.get_providers():
            raise RuntimeError(f"المزوّد المطلوب لم يُفعَّل (الفعّال: {self.sess.get_providers()})")

    def encode(self, texts: list[str], bs: int) -> np.ndarray:
        order = np.argsort([-len(t) for t in texts])  # الأطول أولاً كما يفعل ST
        parts = []
        for i in range(0, len(texts), bs):
            chunk = [texts[j] for j in order[i : i + bs]]
            enc = self.tok(chunk, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="np")
            e = self.sess.run(None, {"input_ids": enc["input_ids"].astype(np.int64),
                                     "attention_mask": enc["attention_mask"].astype(np.int64)})[0]
            parts.append(e.astype(np.float32))
        out = _normalize(np.concatenate(parts))
        res = np.empty_like(out)
        res[order] = out
        return res


def make_encoder(name: str, technique: str, tmp_dir: str):
    """يعيد (encode(texts, bs) → np.float32 مطبَّع، وقت التحضير بالثواني)."""
    import torch
    from sentence_transformers import SentenceTransformer

    fw, rest = technique.split("-", 1)
    t0 = time.perf_counter()
    if fw == "ort":
        st = SentenceTransformer(name, device="cuda",
                                 model_kwargs={"torch_dtype": torch.float16, "attn_implementation": "eager"})
        st.max_seq_length = MAX_SEQ_LEN
        if "tensorrt" in rest:
            # أشكال ديناميكية: بلا ملفات أشكال يعيد TensorRT بناء المحرك لكل شكل جديد فيصير القياس بناءً لا استدلالاً
            shapes = lambda b, n: f"input_ids:{b}x{n},attention_mask:{b}x{n}"
            providers = [("TensorrtExecutionProvider", {
                            "trt_fp16_enable": True, "trt_engine_cache_enable": True,
                            "trt_engine_cache_path": os.path.join(tmp_dir, "trt_cache"),
                            "trt_profile_min_shapes": shapes(1, 1),
                            "trt_profile_opt_shapes": shapes(64, 128),
                            "trt_profile_max_shapes": shapes(256, MAX_SEQ_LEN)}),
                         "CUDAExecutionProvider"]
        else:
            providers = ["CUDAExecutionProvider"]
        enc = OrtEncoder(st, name, providers, tmp_dir)
        del st
        torch.cuda.empty_cache()
        return (lambda texts, bs: enc.encode(texts, bs)), time.perf_counter() - t0

    prec = rest.split("-")[0].split("+")[0]
    dtype = {"fp32": torch.float32, "fp16": torch.float16}.get(prec, torch.bfloat16)
    st = SentenceTransformer(name, device="cuda", model_kwargs={"torch_dtype": dtype, "attn_implementation": "sdpa"})
    st.max_seq_length = MAX_SEQ_LEN
    inner = st[0].auto_model
    if "int8dyn" in rest or "fp8dyn" in rest:
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            Int8DynamicActivationInt8WeightConfig,
            quantize_,
        )
        cfg = Int8DynamicActivationInt8WeightConfig() if "int8dyn" in rest else Float8DynamicActivationFloat8WeightConfig()
        quantize_(inner, cfg)
    if "compile" in rest:
        # نغلّف forward نفسه: إسناد st[0].auto_model لا يبدّل ما تستدعيه ST فتبقى الترجمة معطّلة بصمت
        from torch._dynamo.utils import counters

        # نفس البنية والأشكال (MiniLM و e5-small) تعيد استخدام رسم مخزَّن فلا يُسجَّل رسم جديد؛ نصفّر الذاكرة لكل صف
        torch._dynamo.reset()
        counters.clear()
        inner.forward = torch.compile(inner.forward, dynamic=True)

    def encode(texts: list[str], bs: int) -> np.ndarray:
        return st.encode(texts, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=False).astype(np.float32)

    return encode, time.perf_counter() - t0


def timed(fn, sync) -> float:
    sync()
    t0 = time.perf_counter()
    fn()
    sync()
    return time.perf_counter() - t0


def measure(name: str, technique: str, inputs: dict, ref: dict, tmp_dir: str) -> dict:
    import torch

    sync = torch.cuda.synchronize
    encode, prep_s = make_encoder(name, technique, tmp_dir)
    torch.cuda.reset_peak_memory_stats()
    row = {"model": name, "technique": technique, "prep_seconds": round(prep_s, 1), "throughput": {}}
    docs, queries = inputs["docs"], inputs["queries"]
    for bs in BATCH_SIZES:
        t_warm = time.perf_counter()
        for _ in range(WARMUP_RUNS):
            encode(docs, bs)
        if "compile" in technique and bs == BATCH_SIZES[0]:
            row["first_encode_seconds"] = round(time.perf_counter() - t_warm, 1)  # يشمل الترجمة
            from torch._dynamo.utils import counters

            # حارس: صف باسم +compile بلا أي رسم مترجَم كان سيُنشر باسم خاطئ (حدث فعلاً قبل إصلاح التغليف)
            if not counters["stats"].get("unique_graphs"):
                raise RuntimeError("torch.compile لم يُسجّل أي رسم: الترجمة غير فعّالة")
        secs = [timed(lambda: encode(docs, bs), sync) for _ in range(TIMED_RUNS)]
        row["throughput"][str(bs)] = {"passages_per_s": N_PASSAGES / statistics.median(secs),
                                      "min": N_PASSAGES / max(secs), "max": N_PASSAGES / min(secs)}
    row["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9  # لا يشمل ذاكرة ORT/TensorRT الخارجية
    lat = [timed(lambda: encode([q], 1), sync) * 1000 for q in queries[:LATENCY_QUERIES]]
    row["latency_ms_p50"], row["latency_ms_p95"] = float(np.percentile(lat, 50)), float(np.percentile(lat, 95))

    p, q = encode(docs, 64), encode(queries, 64)
    if technique == TECHNIQUES[0]:
        ref.update({"p": p, "q": q})
    elif "p" not in ref:
        # الاستئناف: صف fp32 أُنجز في عملية سابقة فلا مرجع في الذاكرة؛ نعيد حسابه (غير مُقاس زمنياً)
        ref_encode, _ = make_encoder(name, TECHNIQUES[0], tmp_dir)
        ref.update({"p": ref_encode(docs, 64), "q": ref_encode(queries, 64)})
        del ref_encode
        torch.cuda.empty_cache()
    row.update(cosine_stats(ref["p"], p))
    row["top10_overlap"] = topk_overlap(ref["q"], ref["p"], q, p)

    # جودة فعلية: nDCG@10 على مجموعة الجودة (استعلامات dev حقيقية ومقاطعها)
    qp, qq = encode(inputs["q_docs"], 64), encode(inputs["q_queries"], 64)
    top = torch.from_numpy(qq).cuda() @ torch.from_numpy(qp).cuda().T
    idx = top.topk(10, dim=1).indices.cpu().numpy()
    ids = inputs["q_docids"]
    row["ndcg@10"] = float(np.mean([ndcg_at_k([ids[j] for j in r], inputs["q_rels"][i]) for i, r in enumerate(idx)]))
    return row


def load_inputs(name: str) -> dict:
    passages = []
    with gzip.open(CORPUS_SHARD, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            passages.append(f"{d['title']} {d['text']}")
            if len(passages) == N_PASSAGES:
                break
    queries = [l.split("\t", 1)[1].strip() for l in QUERIES_TSV.open(encoding="utf-8")]
    rng = np.random.default_rng(SEED)
    queries = [queries[i] for i in rng.choice(len(queries), N_QUERIES, replace=False)]
    pp, qp = MODELS[name], QUERY_PREFIX[name]
    sub = [json.loads(l) for l in QUALITY_SUBSET_JSONL.open(encoding="utf-8")]
    qs = json.loads(QUALITY_QUERIES_JSON.read_text(encoding="utf-8"))
    qids = sorted(qs)
    return {
        "docs": [pp + p for p in passages], "queries": [qp + q for q in queries],
        "q_docs": [pp + f"{d['title']} {d['text']}" for d in sub], "q_docids": [d["docid"] for d in sub],
        "q_queries": [qp + qs[q]["query"] for q in qids], "q_rels": [qs[q]["rels"] for q in qids],
    }


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        sys.exit("يتطلب GPU (CUDA غير متاح)")
    if not QUALITY_SUBSET_JSONL.exists():
        sys.exit(f"شغّل 03_build_quality_subset.py أولاً ({QUALITY_SUBSET_JSONL})")
    SYSTEMS_RESULTS_JSON.parent.mkdir(exist_ok=True)
    results = json.loads(SYSTEMS_RESULTS_JSON.read_text(encoding="utf-8")) if SYSTEMS_RESULTS_JSON.exists() else []
    done = {(r["model"], r["technique"]) for r in results}
    tmp_dir = tempfile.mkdtemp(prefix="aet_")
    gpu = torch.cuda.get_device_name(0)

    for name in MODELS:
        inputs, ref = load_inputs(name), {}
        for tech in TECHNIQUES:
            if (name, tech) in done:
                continue
            print(f"◄ {name.split('/')[-1]} {tech}", flush=True)
            try:
                with gpu_lock():
                    row = measure(name, tech, inputs, ref, tmp_dir)
                row["gpu"] = gpu
            except Exception as e:  # تقنية غير مدعومة تُسجَّل بسببها (لا تُخفى ولا توقف الباقي)
                row = {"model": name, "technique": tech, "error": f"{type(e).__name__}: {str(e)[:200]}"}
                print(f"  ✗ {row['error']}", flush=True)
                if tech == TECHNIQUES[0]:
                    sys.exit("فشل المرجع fp32؛ لا معنى لمقارنة التكافؤ")
            torch.cuda.empty_cache()
            results.append(row)
            SYSTEMS_RESULTS_JSON.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"✓ {len(results)} تركيبة ← {SYSTEMS_RESULTS_JSON}")


if __name__ == "__main__":
    main()
