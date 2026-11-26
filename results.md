# Results

GPU: NVIDIA GeForce RTX 5090 Laptop GPU · 4,000 real MIRACL-ar passages · max_seq_len 256 · throughput = median of 3 timed runs after 1 warm-up, best of batch sizes 16/64/256 (range = min–max across runs).

| Model | Precision | Attention | Passages/s (best batch) | × vs fp32/eager | Query latency p50 / p95 (ms) | Peak mem (GB) | Top-10 overlap vs fp32 | Min cosine vs fp32 |
|---|---|---|---|---|---|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 | fp32 | eager | 1,533 (b=64; 1,522–1,539) | 1.00× | 4.8 / 5.7 | 2.79 | 1.000 | 1.0000 |
| paraphrase-multilingual-MiniLM-L12-v2 | fp32 | sdpa | 1,902 (b=64; 1,895–1,916) | 1.24× | 4.3 / 5.0 | 2.20 | 1.000 | 1.0000 |
| paraphrase-multilingual-MiniLM-L12-v2 | fp16 | eager | 3,333 (b=64; 3,329–3,345) | 2.17× | 5.0 / 5.6 | 2.36 | 0.998 | 1.0000 |
| paraphrase-multilingual-MiniLM-L12-v2 | fp16 | sdpa | 4,538 (b=256; 4,527–4,581) | 2.96× | 4.5 / 4.9 | 1.14 | 0.998 | 1.0000 |
| paraphrase-multilingual-MiniLM-L12-v2 | bf16 | eager | 3,392 (b=64; 3,387–3,409) | 2.21× | 5.1 / 5.7 | 1.89 | 0.982 | 0.9996 |
| paraphrase-multilingual-MiniLM-L12-v2 | bf16 | sdpa | 4,551 (b=256; 4,531–4,578) | 2.97× | 4.6 / 5.2 | 1.61 | 0.978 | 0.9993 |
| multilingual-e5-small | fp32 | eager | 1,491 (b=64; 1,474–1,496) | 1.00× | 4.8 / 5.8 | 2.79 | 1.000 | 1.0000 |
| multilingual-e5-small | fp32 | sdpa | 1,879 (b=64; 1,878–1,887) | 1.26× | 4.4 / 5.8 | 1.73 | 1.000 | 1.0000 |
| multilingual-e5-small | fp16 | eager | 3,200 (b=64; 3,199–3,248) | 2.15× | 5.1 / 5.7 | 1.41 | 0.988 | 1.0000 |
| multilingual-e5-small | fp16 | sdpa | 4,487 (b=256; 4,453–4,499) | 3.01× | 4.4 / 4.8 | 0.89 | 0.985 | 1.0000 |
| multilingual-e5-small | bf16 | eager | 3,282 (b=64; 3,274–3,304) | 2.20× | 5.3 / 6.0 | 1.41 | 0.916 | 0.9999 |
| multilingual-e5-small | bf16 | sdpa | 4,561 (b=256; 4,550–4,596) | 3.06× | 4.7 / 5.4 | 0.89 | 0.916 | 1.0000 |
| bge-m3 | fp32 | eager | 214 (b=16; 214–215) | 1.00× | 8.7 / 10.4 | 6.67 | 1.000 | 1.0000 |
| bge-m3 | fp32 | sdpa | 228 (b=256; 227–228) | 1.06× | 7.8 / 9.4 | 5.55 | 1.000 | 1.0000 |
| bge-m3 | fp16 | eager | 626 (b=16; 626–631) | 2.92× | 9.0 / 11.5 | 3.35 | 0.998 | 0.9999 |
| bge-m3 | fp16 | sdpa | 765 (b=64; 765–766) | 3.57× | 8.1 / 9.4 | 2.80 | 0.997 | 0.9999 |
| bge-m3 | bf16 | eager | 631 (b=16; 626–631) | 2.94× | 9.0 / 12.0 | 3.35 | 0.978 | 0.9975 |
| bge-m3 | bf16 | sdpa | 777 (b=64; 774–777) | 3.62× | 8.2 / 9.3 | 2.80 | 0.977 | 0.9982 |
