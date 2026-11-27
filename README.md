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
2. **bf16 is as fast as fp16 but changes the ranking more.** Mean/min cosine to fp32 looks fine (min >= 0.9975), yet the retrieved top-10 agrees less with fp32's — for e5-small only 91.6% overlap. This is agreement, not quality: in the later systems sweep (`results_systems.md`, 500 queries) bf16/SDPA nDCG@10 stays within ±0.0013 of fp32. So "bf16 reorders neighbours" is measured; "bf16 hurts retrieval" is not. Cosine alone would have hidden the reordering; measure the ranking, and then measure quality.
3. **SDPA attention alone gives ~1.1–1.4×** (smallest for bge-m3 in fp32: 1.11×), plus lower memory.
4. **Single-query latency is flat**: ~4.3–5.3 ms for the small models and ~8–9 ms for bge-m3 at
   batch size 1, regardless of precision or attention. It is launch-bound, so these optimisations
   matter for *indexing*, not for query-time latency.
5. **Bigger batches are not always faster**: for bge-m3 fp16/SDPA, batch 64 (765/s) beat batch 256
   (664/s). I did not isolate the cause; padding to the longest sequence in a batch is the likely one.

## Second sweep: TensorRT, torch.compile, ONNX Runtime, FP8/INT8 (WSL2)

The first sweep ran on Windows, where Triton is unavailable. Everything below ran **inside WSL2** (PyTorch 2.15 + CUDA 13,
ONNX Runtime 1.32 with TensorRT 10.16, torchao) on the same laptop GPU, with the same 4,000 passages and `max_seq_len = 256`,
and adds a **measured quality number**: nDCG@10 on a subset of MIRACL-ar dev (500 real queries over 20,000 passages: every
judged passage of those queries plus random distractors). Absolute nDCG is higher than full-corpus MIRACL because the corpus is
smaller; compare rows, not published numbers. Full table: [results_systems.md](results_systems.md).
Speed-ups are vs fp32/SDPA **in this same WSL session** (1,668 / 1,693 / 227 passages/s), so they are not comparable to the Windows sweep.

| Technique | MiniLM | e5-small | bge-m3 | nDCG@10 change vs fp32 (MiniLM / e5 / bge-m3) |
|---|---|---|---|---|
| PyTorch fp16 + SDPA | 2.49× | 2.33× | 3.30× | +0.0006 / +0.0009 / +0.0001 |
| fp16 + `torch.compile` | 2.90× | 2.80× | 3.43× | +0.0004 / −0.0006 / −0.0001 |
| ONNX Runtime, CUDA EP, fp16 | 2.16× | 2.07× | 2.92× | −0.0003 / +0.0007 / +0.0000 |
| **ONNX Runtime, TensorRT EP, fp16** | **4.19×** | 3.68× | **5.12×** | −0.0003 / +0.0001 / +0.0006 |
| FP8 dynamic (torchao), eager | 1.22× | 1.35× | 2.04× | +0.0051 / −0.0056 / −0.0036 |
| **FP8 dynamic + `torch.compile`** | 3.13× | 3.15× | 5.03× | +0.0022 / −0.0002 / +0.0007 |
| INT8 dynamic (torchao), eager | 0.34× | 0.34× | 0.89× | +0.0011 / −0.0019 / −0.0008 |
| INT8 dynamic + `torch.compile` | failed | failed | failed | — |

What the numbers say:

1. **TensorRT (via the ONNX Runtime TensorRT provider, fp16) is the fastest tested path: 3.7–5.1× over fp32** with
   nDCG@10 within ±0.0006 and top-10 overlap ≥ 0.992. It also has the lowest single-query latency (p50 2.4–4.9 ms).
2. **FP8 only pays off when compiled.** Eager FP8 reaches 1.2–2.0× and its single-query latency is worse than fp32
   (p50 48–80 ms vs ~6–10 ms); with `torch.compile` it reaches 3.1–5.0×, close to TensorRT fp16. It needs a GPU with FP8
   support (this Blackwell laptop GPU has it).
3. **Quantised paths change the top-10 order but not nDCG@10**: top-10 overlap with fp32 falls to 0.87–0.93 for FP8, yet nDCG
   moves by at most 0.006. Differences of that size are what I would expect from noise on 500 queries; I did not compute
   confidence intervals, so I do not claim any FP8/bf16 quality *gain*.
4. **`torch.compile` on fp16 adds only +4% to +20% over eager SDPA** (bge-m3 +4%, the 118M models +16–20%), and costs
   a one-time 7–56 s on the first encode (compile time included).
5. **INT8 dynamic quantisation is slower than fp32 here** (0.34× for the small models) and fails with
   `torch.compile` for all three models: `self.size(0) needs to be greater than 16`. The int8 matmul used by torchao requires a
   batch dimension above 16, and the smallest sweep batch is 16. I did not work around it.
6. **ONNX Runtime's plain CUDA provider is slower than PyTorch fp16 SDPA for the small models** (3,596 vs 4,160 passages/s for
   MiniLM) but has lower single-query latency (2.7–3.5 ms vs 7–10 ms).

How this second sweep was kept honest (and what went wrong on the way):

- **My first `+compile` rows were not compiled.** Assigning `st[0].auto_model = torch.compile(...)` did not change what
  sentence-transformers calls, so those rows measured eager PyTorch under a wrong label. I noticed because the compile
  "first encode" took 3 s; the fix wraps `forward` itself, and the harness now fails any `+compile` row that records no Dynamo graph.
- Two harness bugs (a pooling-mode API change in sentence-transformers 6, and a missing fp32 reference when resuming) produced
  false failures; those rows were deleted and re-run.
- A CUDA "unknown error" killed the sweep once, and the GPU later vanished from WSL mid-run. Each configuration now runs in its
  own subprocess, and rows that failed only because the GPU was gone were discarded rather than reported as technique failures.
- TensorRT needs explicit dynamic-shape profiles (min/opt/max); without them the engine is rebuilt for every new shape.

```bash
# inside WSL2 with CUDA; installs torch, sentence-transformers, onnxruntime-gpu, torchao, tensorrt-cu13<11
python 03_build_quality_subset.py   # 500 dev queries + 20k passages (CPU)
python 04_bench_systems.py          # resumable; one subprocess per configuration; shared-GPU lock
python 05_report_systems.py         # writes results_systems.md
```

## Method

- 4,000 real passages (title + text) from the MIRACL-ar corpus; 200 real dev queries.
- `max_seq_len = 256` for every model and configuration.
- Throughput = median of 3 timed runs after 1 warm-up, with `cuda.synchronize` around the timer;
  min–max across runs is reported (spread was within ~2% except one MiniLM fp16/eager batch-256 run, 2,252–3,214/s).
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

- **One machine, one laptop GPU**, possibly power- or thermally limited, shared with other jobs, with timed sections serialised through a lock; absolute numbers will differ
  elsewhere. Ratios between configurations are the transferable part.
- **fp32 means strict fp32** (PyTorch default, TF32 off). TF32 was not tested, so the headline ratio
  is against the slowest reasonable baseline.
- **Top-10 overlap is not retrieval quality.** In the first sweep it is the only quality proxy, on 4,000 passages.
  The second sweep measures nDCG@10 directly, on a 20k-passage subset.
- Single run of each sweep; no repeat across sessions.
- **Not tried:** TensorRT-LLM or native `trtexec` engines, FP8 weight-only or static FP8 calibration, INT8 with static
  calibration or SmoothQuant, flash-attention kernels beyond PyTorch SDPA, batch sizes other than 16/64/256. Peak GPU
  memory is not reported for ONNX Runtime/TensorRT rows (PyTorch does not track it). TensorRT engine build time is not
  included in throughput.

MIT licensed.
