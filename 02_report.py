# -*- coding: utf-8 -*-
"""المرحلة 2: يحوّل results/throughput.json إلى results.md (أفضل حجم دفعة لكل تركيبة)."""

import json

from config import RESULTS_JSON, RESULTS_MD


def main() -> None:
    rows = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    out = [
        "# Results",
        "",
        f"GPU: {next(r['gpu'] for r in rows if 'gpu' in r)} · 4,000 real MIRACL-ar passages · max_seq_len 256 · "
        "throughput = median of 3 timed runs after 1 warm-up, best of batch sizes 16/64/256 "
        "(range = min–max across runs).",
        "",
        "| Model | Precision | Attention | Passages/s (best batch) | × vs fp32/eager | Query latency p50 / p95 (ms) | Peak mem (GB) | Top-10 overlap vs fp32 | Min cosine vs fp32 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    base: dict[str, float] = {}
    for r in rows:
        if "error" in r:
            out.append(f"| {r['model'].split('/')[-1]} | {r['precision']} | {r['attention']} | unsupported: {r['error']} | | | | | |")
            continue
        bs, t = max(r["throughput"].items(), key=lambda kv: kv[1]["passages_per_s"])
        speed = t["passages_per_s"]
        base.setdefault(r["model"], speed)  # أول تركيبة لكل نموذج = fp32/eager
        out.append(
            f"| {r['model'].split('/')[-1]} | {r['precision']} | {r['attention']} | {speed:,.0f} (b={bs}; {t['min']:,.0f}–{t['max']:,.0f}) "
            f"| {speed / base[r['model']]:.2f}× | {r['latency_ms_p50']:.1f} / {r['latency_ms_p95']:.1f} | {r['peak_mem_gb']:.2f} "
            f"| {r['top10_overlap']:.3f} | {r['cos_min']:.4f} |"
        )
    RESULTS_MD.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("✓ results.md")


if __name__ == "__main__":
    main()
