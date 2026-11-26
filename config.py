# -*- coding: utf-8 -*-
"""الإعدادات. عدّل هنا فقط عند تغيير النماذج أو شبكة التجارب أو حجم العيّنة."""

import os
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
