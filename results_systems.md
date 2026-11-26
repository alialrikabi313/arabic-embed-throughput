# Systems results (WSL2, PyTorch + ONNX Runtime + torchao)

GPU: NVIDIA GeForce RTX 5090 Laptop GPU · 4,000 real MIRACL-ar passages · max_seq_len 256 · median of 3 timed runs after warm-up, best of batch 16/64/256 · quality = nDCG@10 on a 20k-passage subset (500 real dev queries; absolute values are higher than full-corpus MIRACL because the corpus is smaller, so compare rows, not to published numbers).

| Model | Technique | Passages/s (best batch) | × vs fp32 | nDCG@10 | Δ nDCG vs fp32 | Top-10 overlap | Query p50 (ms) | Compile/prep (s) | Notes |
|---|---|---|---|---|---|---|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 | pt-fp32-sdpa | 1,668 (b=256; 1,651–1,792) | 1.00× | 0.3812 | +0.0000 | 1.000 | 6.7 | 30.0 | |
| paraphrase-multilingual-MiniLM-L12-v2 | pt-fp16-sdpa | 4,160 (b=256; 4,101–4,185) | 2.49× | 0.3818 | +0.0006 | 0.998 | 7.1 | 2.0 | |
| paraphrase-multilingual-MiniLM-L12-v2 | pt-bf16-sdpa | 3,999 (b=256; 3,909–4,437) | 2.40× | 0.3825 | +0.0013 | 0.980 | 6.5 | 2.5 | |
| paraphrase-multilingual-MiniLM-L12-v2 | pt-bf16+int8dyn | 562 (b=256; 492–582) | 0.34× | 0.3823 | +0.0011 | 0.934 | 425.2 | 2.2 | |
| paraphrase-multilingual-MiniLM-L12-v2 | pt-bf16+fp8dyn | 2,034 (b=256; 2,032–2,046) | 1.22× | 0.3863 | +0.0051 | 0.934 | 65.7 | 2.1 | |
| multilingual-e5-small | pt-fp32-sdpa | 1,693 (b=256; 1,652–1,791) | 1.00× | 0.8560 | +0.0000 | 1.000 | 6.2 | 27.7 | |
| multilingual-e5-small | pt-fp16-sdpa | 3,953 (b=256; 3,942–3,984) | 2.33× | 0.8569 | +0.0009 | 0.993 | 9.9 | 4.1 | |
| multilingual-e5-small | pt-bf16-sdpa | 3,986 (b=256; 3,967–4,059) | 2.35× | 0.8553 | -0.0006 | 0.931 | 10.4 | 4.1 | |
| multilingual-e5-small | pt-bf16+int8dyn | 572 (b=256; 516–577) | 0.34× | 0.8541 | -0.0019 | 0.882 | 415.0 | 4.9 | |
| bge-m3 | pt-fp32-sdpa | 227 (b=256; 227–227) | 1.00× | 0.8895 | +0.0000 | 1.000 | 9.8 | 3.6 | |
| bge-m3 | pt-fp16-sdpa | 748 (b=64; 747–749) | 3.30× | 0.8896 | +0.0001 | 0.996 | 7.3 | 3.3 | |
| bge-m3 | pt-bf16-sdpa | 768 (b=64; 766–769) | 3.39× | 0.8884 | -0.0011 | 0.979 | 9.1 | 3.4 | |
| bge-m3 | pt-bf16+int8dyn | 201 (b=256; 201–202) | 0.89× | 0.8887 | -0.0008 | 0.812 | 411.3 | 3.2 | |
| bge-m3 | pt-bf16+fp8dyn | 462 (b=64; 462–473) | 2.04× | 0.8859 | -0.0036 | 0.874 | 79.6 | 3.6 | |
| paraphrase-multilingual-MiniLM-L12-v2 | ort-fp16-cuda | 3,596 (b=64; 3,507–3,609) | 2.16× | 0.3809 | -0.0003 | 0.998 | 3.5 | 5.4 | |
| multilingual-e5-small | ort-fp16-cuda | 3,502 (b=64; 3,473–3,526) | 2.07× | 0.8566 | +0.0007 | 0.998 | 2.7 | 5.4 | |
| multilingual-e5-small | pt-bf16+fp8dyn | 2,279 (b=256; 2,049–2,280) | 1.35× | 0.8504 | -0.0056 | 0.880 | 47.6 | 3.6 | |
| bge-m3 | ort-fp16-cuda | 662 (b=16; 661–666) | 2.92× | 0.8895 | +0.0000 | 0.995 | 7.1 | 11.2 | |
| paraphrase-multilingual-MiniLM-L12-v2 | pt-fp16-sdpa+compile | 4,832 (b=256; 4,767–4,909) | 2.90× | 0.3816 | +0.0004 | 0.998 | 2.9 | 3.0 (+7.0 first encode) | |
| paraphrase-multilingual-MiniLM-L12-v2 | pt-bf16+int8dyn+compile | — | — | — | — | — | — | — | **failed:** RuntimeError: self.size(0) needs to be greater than 16, but got 16 |
| paraphrase-multilingual-MiniLM-L12-v2 | pt-bf16+fp8dyn+compile | 5,216 (b=256; 5,153–5,360) | 3.13× | 0.3834 | +0.0022 | 0.932 | 7.0 | 1.5 (+35.1 first encode) | |
| multilingual-e5-small | pt-bf16+int8dyn+compile | — | — | — | — | — | — | — | **failed:** RuntimeError: self.size(0) needs to be greater than 16, but got 16 |
| bge-m3 | pt-fp16-sdpa+compile | 777 (b=64; 770–780) | 3.43× | 0.8894 | -0.0001 | 0.996 | 7.5 | 3.7 (+31.0 first encode) | |
| bge-m3 | pt-bf16+int8dyn+compile | — | — | — | — | — | — | — | **failed:** RuntimeError: self.size(0) needs to be greater than 16, but got 16 |
| paraphrase-multilingual-MiniLM-L12-v2 | ort-fp16-tensorrt | 6,992 (b=256; 6,989–7,088) | 4.19× | 0.3809 | -0.0003 | 0.998 | 2.4 | 46.5 | |
| multilingual-e5-small | pt-fp16-sdpa+compile | 4,740 (b=256; 4,620–4,770) | 2.80× | 0.8554 | -0.0006 | 0.990 | 3.0 | 3.5 (+11.4 first encode) | |
| multilingual-e5-small | ort-fp16-tensorrt | 6,223 (b=256; 6,150–6,309) | 3.68× | 0.8561 | +0.0001 | 0.997 | 3.2 | 26.3 | |
| multilingual-e5-small | pt-bf16+fp8dyn+compile | 5,338 (b=256; 5,258–5,366) | 3.15× | 0.8558 | -0.0002 | 0.880 | 3.9 | 5.3 (+42.3 first encode) | |
| bge-m3 | ort-fp16-tensorrt | 1,161 (b=64; 1,160–1,167) | 5.12× | 0.8901 | +0.0006 | 0.992 | 4.9 | 66.7 | |
| bge-m3 | pt-bf16+fp8dyn+compile | 1,140 (b=64; 1,136–1,140) | 5.03× | 0.8902 | +0.0007 | 0.874 | 11.3 | 5.0 (+55.9 first encode) | |
