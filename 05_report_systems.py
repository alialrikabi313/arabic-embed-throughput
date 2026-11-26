# -*- coding: utf-8 -*-
"""المرحلة 5: يحوّل results/systems.json إلى results_systems.md (أفضل حجم دفعة، جودة، تكافؤ، فشل مسجَّل)."""

import json

from config import SYSTEMS_RESULTS_JSON, SYSTEMS_RESULTS_MD


def main() -> None:
    rows = json.loads(SYSTEMS_RESULTS_JSON.read_text(encoding="utf-8"))
    gpu = next((r["gpu"] for r in rows if "gpu" in r), "?")
    out = [
        "# Systems results (WSL2, PyTorch + ONNX Runtime + torchao)",
        "",
        f"GPU: {gpu} · 4,000 real MIRACL-ar passages · max_seq_len 256 · median of 3 timed runs after warm-up, "
        "best of batch 16/64/256 · quality = nDCG@10 on a 20k-passage subset (500 real dev queries; absolute values "
        "are higher than full-corpus MIRACL because the corpus is smaller, so compare rows, not to published numbers).",
        "",
        "| Model | Technique | Passages/s (best batch) | × vs fp32 | nDCG@10 | Δ nDCG vs fp32 | Top-10 overlap | "
        "Query p50 (ms) | Compile/prep (s) | Notes |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    ref: dict[str, dict] = {}
    for r in rows:
        m = r["model"].split("/")[-1]
        if "error" in r:
            out.append(f"| {m} | {r['technique']} | — | — | — | — | — | — | — | **failed:** {r['error']} |")
            continue
        bs, t = max(r["throughput"].items(), key=lambda kv: kv[1]["passages_per_s"])
        speed = t["passages_per_s"]
        if r["technique"].startswith("pt-fp32"):
            ref[r["model"]] = {"speed": speed, "ndcg": r["ndcg@10"]}
        b = ref.get(r["model"])
        prep = f"{r['prep_seconds']}" + (f" (+{r['first_encode_seconds']} first encode)" if "first_encode_seconds" in r else "")
        out.append(
            f"| {m} | {r['technique']} | {speed:,.0f} (b={bs}; {t['min']:,.0f}–{t['max']:,.0f}) "
            f"| {speed / b['speed']:.2f}× | {r['ndcg@10']:.4f} | {r['ndcg@10'] - b['ndcg']:+.4f} "
            f"| {r['top10_overlap']:.3f} | {r['latency_ms_p50']:.1f} | {prep} | |"
        )
    SYSTEMS_RESULTS_MD.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("✓ results_systems.md")


if __name__ == "__main__":
    main()
