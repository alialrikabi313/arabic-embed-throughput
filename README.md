# arabic-embed-throughput

How much faster can an embedding model encode Arabic text on a GPU, and what does the speed-up
cost in ranking quality? A controlled sweep of **precision × attention implementation × batch size**
on real Arabic passages, with a parity check against fp32 for every configuration.

Models: `paraphrase-multilingual-MiniLM-L12-v2` (118M), `multilingual-e5-small` (118M), `bge-m3` (568M).
Hardware: one laptop RTX 5090 (see caveats). Full table: [results.md](results.md).

## Findings

| | MiniLM | e5-small | bge-m3 |
|---|---|---|---|
| fp32 / eager (baseline), passages/s | 1,533 | 1,491 | 206 |
| **fp16 / SDPA**, passages/s | **4,538** (2.96×) | **4,487** (3.01×) | **765** (3.71×) |
| top-10 overlap with fp32 ranking (fp16) | 0.998 | 0.985 | 0.997 |
| bf16 / SDPA, top-10 overlap | 0.978 | **0.916** | 0.977 |

1. **fp16 + SDPA is ~3–3.7× faster than strict fp32 with eager attention**, while keeping the top-10
   ranking within 0.985–0.998 of fp32. It also cuts peak memory (MiniLM 2.79 → 1.14 GB).
2. **bf16 is as fast as fp16 but less faithful.** Mean/min cosine to fp32 looks fine (min ≥ 0.9975),
   yet the retrieved top-10 diverges more — for e5-small only 91.6% overlap. Cosine similarity of
   the embeddings alone would have hidden this; measure the ranking. (More mantissa bits matter when
   the scores of neighbouring passages are close.)
3. **SDPA attention alone gives ~1.2–1.4×** in both fp32 and fp16, plus lower memory.
4. **Single-query latency is flat**: ~4.3–5.3 ms for the small models and ~8–9 ms for bge-m3 at
   batch size 1, regardless of precision or attention. It is launch-bound, so these optimisations
   matter for *indexing*, not for query-time latency.
5. **Bigger batches are not always faster**: for bge-m3 fp16/SDPA, batch 64 (765/s) beat batch 256
   (664/s). I did not isolate the cause; padding to the longest sequence in a batch is the likely one.

## Method

- 4,000 real passages (title + text) from the MIRACL-ar corpus; 200 real dev queries.
- `max_seq_len = 256` for every model and configuration.
- Throughput = median of 3 timed runs after 1 warm-up, with `cuda.synchronize` around the timer;
  min–max across runs is reported (spread was ≤ 2% except one bge-m3/MiniLM outlier run).
- Parity: every configuration is compared with the model's own fp32/eager embeddings — cosine
  (mean and **min**) and the overlap of the top-10 passages for the 200 queries.
- Configurations a model rejects are recorded with the reason, not dropped.

```bash
pip install sentence-transformers
# data: run 01_download_data.py of arabic-retrieval-finetune (or set AET_MIRACL_DIR)
python 01_bench.py       # ~25 min on a laptop RTX 5090
python 02_report.py      # → results.md
pytest tests
```

## Caveats (read before quoting the numbers)

- **One machine, one laptop GPU**, possibly power- or thermally limited; absolute numbers will differ
  elsewhere. Ratios between configurations are the transferable part.
- **fp32 means strict fp32** (PyTorch default, TF32 off). TF32 was not tested, so the headline ratio
  is against the slowest reasonable baseline.
- **Top-10 overlap is not retrieval quality.** It measures agreement with fp32, on 4,000 passages. I
  did not measure nDCG here; the quality cost of each precision on a full benchmark is untested.
- **Not tried:** TensorRT, ONNX Runtime, `torch.compile` (Triton is unavailable on this Windows setup),
  INT8/FP8 quantisation, flash-attention kernels beyond PyTorch's SDPA.
- Single run of the whole sweep; no repeat across sessions.

MIT licensed.
