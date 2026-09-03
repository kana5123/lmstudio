"""지연시간·메모리 측정 (지시문 14장).

같은 하드웨어·같은 배치·워밍업 후 각 단계를 **따로** 잰다.
  GPU: torch.cuda.Event + synchronize   /  CPU: time.perf_counter
길이 64/128/256/512 각각에서 재고, 최대 GPU 메모리도 기록한다.
"""
import argparse, json, sys, time
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pg2_decompx.decompx_utils import DecompXConfig
from pg2_decompx.modeling_deberta_v2_decompx import DecompXDebertaV2
from models.global_delta_verifier import GlobalDeltaVerifier
from models.token_decomp_verifier import TokenDecompVerifier

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "artifacts/results"
MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"
LENS = (64, 128, 256, 512)


def timeit(fn, n=20, warmup=5, cuda=True):
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(n)]
        for a, b in ev:
            a.record(); fn(); b.record()
        torch.cuda.synchronize()
        ms = sorted(a.elapsed_time(b) for a, b in ev)
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
    else:
        ms = []
        for _ in range(n):
            t = time.perf_counter(); fn(); ms.append((time.perf_counter() - t) * 1000)
        ms.sort(); peak = float("nan")
    return {"mean_ms": float(np.mean(ms)), "median_ms": float(np.median(ms)),
            "p95_ms": float(ms[int(len(ms) * .95)]), "peak_mib": peak}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--with_mdeberta", action="store_true")
    a = ap.parse_args()
    dev, cuda = a.device, a.device == "cuda"

    m = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval().to(dev)
    wrap = DecompXDebertaV2(m)
    DC = DecompXConfig(output_all_layers=True, output_encoder=None, output_classifier=True)
    vA = GlobalDeltaVerifier(768).eval().to(dev)
    vB = TokenDecompVerifier(768).eval().to(dev)
    mdeb = None
    if a.with_mdeberta:
        mdeb = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/mdeberta-v3-base", num_labels=2).eval().to(dev)

    out = {"device": dev, "batch": a.bs,
           "params": {"pg2": sum(p.numel() for p in m.parameters()),
                      "verifierA": sum(p.numel() for p in vA.parameters()),
                      "verifierB": sum(p.numel() for p in vB.parameters())}}
    if mdeb:
        out["params"]["mdeberta"] = sum(p.numel() for p in mdeb.parameters())
    print("파라미터 수:", {k: f"{v:,}" for k, v in out["params"].items()})

    for N in LENS:
        ids = torch.randint(1000, 50000, (a.bs, N), device=dev)
        msk = torch.ones(a.bs, N, dtype=torch.long, device=dev)
        r = {}
        with torch.no_grad():
            r["pg2_forward"] = timeit(lambda: m(input_ids=ids, attention_mask=msk), cuda=cuda)
            r["pg2_forward_hidden"] = timeit(
                lambda: m(input_ids=ids, attention_mask=msk, output_hidden_states=True), cuda=cuda)
            r["decompx_full"] = timeit(lambda: wrap.forward(ids, msk, DC), n=5, warmup=2, cuda=cuda)
            g = torch.randn(a.bs, 768, device=dev); nm = torch.randn(a.bs, 2, device=dev)
            bA = {"global": g, "numeric": nm}
            r["verifierA"] = timeit(lambda: vA(bA), n=50, cuda=cuda)
            bB = {"global": g, "numeric": nm,
                  "delta_c": torch.randn(a.bs, N, 768, device=dev),
                  "directional": torch.randn(a.bs, N, device=dev),
                  "margin": torch.randn(a.bs, N, device=dev),
                  "mask": torch.ones(a.bs, N, dtype=torch.bool, device=dev)}
            r["verifierB"] = timeit(lambda: vB(bB), n=50, cuda=cuda)
            if mdeb:
                r["mdeberta_verifier"] = timeit(
                    lambda: mdeb(input_ids=ids, attention_mask=msk), cuda=cuda)
        # 파생: 오버헤드와 전체 경로
        r["hidden_overhead"] = {"mean_ms": r["pg2_forward_hidden"]["mean_ms"] - r["pg2_forward"]["mean_ms"]}
        r["decompx_overhead"] = {"mean_ms": r["decompx_full"]["mean_ms"] - r["pg2_forward"]["mean_ms"]}
        r["path_A_total"] = {"mean_ms": r["pg2_forward_hidden"]["mean_ms"] + r["verifierA"]["mean_ms"]}
        r["path_B_total"] = {"mean_ms": r["decompx_full"]["mean_ms"] + r["verifierB"]["mean_ms"]}
        if mdeb:
            r["path_mdeberta_total"] = {"mean_ms": r["pg2_forward"]["mean_ms"]
                                        + r["mdeberta_verifier"]["mean_ms"]}
        out[f"N{N}"] = r
        print(f"N={N:4}  PG2 {r['pg2_forward']['mean_ms']:7.2f}ms  "
              f"+은닉 {r['hidden_overhead']['mean_ms']:6.2f}ms  "
              f"DecompX {r['decompx_full']['mean_ms']:9.2f}ms ({r['decompx_full']['mean_ms']/r['pg2_forward']['mean_ms']:6.1f}x)  "
              f"검증기A {r['verifierA']['mean_ms']:5.2f}ms  검증기B {r['verifierB']['mean_ms']:5.2f}ms  "
              f"| 경로A {r['path_A_total']['mean_ms']:7.2f}ms  경로B {r['path_B_total']['mean_ms']:9.2f}ms"
              + (f"  mDeBERTa {r['mdeberta_verifier']['mean_ms']:7.2f}ms" if mdeb else ""), flush=True)

    RES.mkdir(parents=True, exist_ok=True)
    tag = f"latency_{dev}_bs{a.bs}.json"
    (RES / tag).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"저장 -> {RES/tag}")


if __name__ == "__main__":
    main()
