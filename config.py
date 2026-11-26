# -*- coding: utf-8 -*-
"""الإعدادات. عدّل هنا فقط عند تغيير النماذج أو شبكة التجارب أو حجم العيّنة."""

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# مقاطع حقيقية من MIRACL-ar (مخرج arabic-retrieval-finetune): الأطوال الواقعية هي ما يهمّ هنا؛
# نصوص اصطناعية بطول ثابت تُخفي أثر الحشو (padding) فتُضخّم الأرقام
MIRACL_DIR = Path(os.environ.get("AET_MIRACL_DIR", _HERE.parent / "arabic-retrieval-finetune-data" / "raw"))
CORPUS_SHARD = MIRACL_DIR / "miracl-corpus-v1.0-ar" / "docs-4.jsonl.gz"
QUERIES_TSV = MIRACL_DIR / "miracl-v1.0-ar" / "topics" / "topics.miracl-v1.0-ar-dev.tsv"
RESULTS_JSON = _HERE / "results" / "throughput.json"
RESULTS_MD = _HERE / "results.md"

N_PASSAGES = 4000  # يكفي لاستقرار القياس ويبقى التشغيل الكامل بدقائق
N_QUERIES = 200
MAX_SEQ_LEN = 256  # نفس طول arabic-retrieval-finetune كي تُقارَن الأرقام
SEED = 0

# نموذج → بادئة المقطع (e5 يتطلب «passage: »)
MODELS: dict[str, str] = {
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "",
    "intfloat/multilingual-e5-small": "passage: ",
    "BAAI/bge-m3": "",
}

# شبكة التجارب. torch.compile غير مُجرَّب: triton غير متاح على ويندوز هنا
PRECISIONS = ("fp32", "fp16", "bf16")
ATTENTION = ("eager", "sdpa")
BATCH_SIZES = (16, 64, 256)

WARMUP_RUNS = 1
TIMED_RUNS = 3  # الوسيط مع min/max؛ الاهتزاز يظهر على لابتوب (حرارة/طاقة)
LATENCY_QUERIES = 100  # زمن استعلام مفرد (دفعة = 1): السيناريو الحيّ للبحث

# ── المرحلة الثانية: تقنيات الاستدلال (WSL) وقياس الجودة الفعلي ──
# مجموعة جودة: استعلامات dev الحقيقية + كل موجباتها + سلبياتها المُحكَّمة + مشتّتات عشوائية.
# 20 ألف مقطع بدل 2.06 مليون: يكفي لكشف فرق الجودة بين الدقات ويبقى التشغيل بدقائق.
# المطلق لا يُقارَن برقم MIRACL الكامل (الكوربس أصغر فالأرقام أعلى)؛ المقارنة بين التركيبات فقط
QUALITY_DIR = Path(os.environ.get("AET_QUALITY_DIR", _HERE.parent / "arabic-embed-throughput-data"))
QUALITY_SUBSET_JSONL = QUALITY_DIR / "quality_subset.jsonl"
QUALITY_QUERIES_JSON = QUALITY_DIR / "quality_queries.json"
QRELS_TSV = MIRACL_DIR / "miracl-v1.0-ar" / "qrels" / "qrels.miracl-v1.0-ar-dev.tsv"
CORPUS_SHARDS = [MIRACL_DIR / "miracl-corpus-v1.0-ar" / f"docs-{i}.jsonl.gz" for i in range(5)]
QUALITY_N_QUERIES = 500
QUALITY_N_PASSAGES = 20_000
SYSTEMS_RESULTS_JSON = _HERE / "results" / "systems.json"
SYSTEMS_RESULTS_MD = _HERE / "results_systems.md"

# بادئة الاستعلام (e5 يتطلب «query: » وإلا تنخفض الجودة بصمت)
QUERY_PREFIX: dict[str, str] = {
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "",
    "intfloat/multilingual-e5-small": "query: ",
    "BAAI/bge-m3": "",
}
# التقنيات بالترتيب؛ الأولى (fp32) مرجع التكافؤ. الاسم: إطار-دقة[+تحويل]
TECHNIQUES = (
    "pt-fp32-sdpa",
    "pt-fp16-sdpa",
    "pt-fp16-sdpa+compile",
    "pt-bf16-sdpa",
    "ort-fp16-cuda",
    "ort-fp16-tensorrt",
    "pt-bf16+int8dyn",
    "pt-bf16+int8dyn+compile",
    "pt-bf16+fp8dyn",
    "pt-bf16+fp8dyn+compile",
)
# قفل GPU مشترك بين العمال (mkdir ذرّي)؛ المسار يختلف بين ويندوز و WSL لنفس المجلد
_LOCK_DEFAULT = "/mnt/d/projects/.gpu_lock" if sys.platform.startswith("linux") else "D:/projects/.gpu_lock"
GPU_LOCK = Path(os.environ.get("AET_GPU_LOCK", _LOCK_DEFAULT))
