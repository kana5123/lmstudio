"""토큰 수준 방향성 기여 이동 분석 (지시문 14~17절).

입력: artifacts/directional_alignment/dir_{split}_*of8.pt
  a      (n, T, 512)  a_k^(l) = dot(v_U^(l), D_k^(l))   Directional Token Contribution Shift
  recon  (n, 12)      ||sum_k C_k^(l) - h^(l)|| / ||h^(l)||
  cons_* (n, T)       sum_k D_k^(l) vs g^(l)  (절대/상대/코사인)
  p, a_sum, pj_*      사영 보존 검증

산출:
  reconstruction_by_layer.csv     Q1
  conservation_check.csv          Q2
  projection_conservation.csv     Q6
  token_directional_summary.csv   Q7
  punctuation_comparison.csv      17절
  대표 표본 Token x Layer 히트맵 + CSV   16절
"""
import argparse, csv, glob, sys
from collections import Counter
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/directional_alignment"
RES = ROOT / "results/directional_alignment"
PLOT = ROOT / "plots/directional_alignment"
MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"
PUNCT = set(".,?!:;'\"()[]{}-—…/\\|`~@#$%^&*_+=<>")


def load(split):
    fs = sorted(glob.glob(str(ART / f"dir_{split}_*of*.pt")))
    assert fs, f"{split} 샤드 없음"
    ds = [torch.load(f, weights_only=False) for f in fs]
    out = {k: torch.cat([d[k] for d in ds]) for k in
           ("a", "mask", "input_ids", "gt", "recon", "cons_abs", "cons_rel",
            "cons_cos", "p", "a_sum", "pj_abs", "pj_rel", "g_norm")}
    out["sample_id"] = [s for d in ds for s in d["sample_id"]]
    out["T"] = ds[0]["n_transitions"]
    return out


def wcsv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def kind(tokstr):
    t = tokstr.strip()
    if not t:
        return "empty"
    if t.startswith("[") and t.endswith("]"):
        return "special"
    core = t.lstrip("▁")
    if core and all(c in PUNCT for c in core):
        return "punct"
    return "content"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval_test")
    ap.add_argument("--topk", type=int, default=10)
    a_ = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True); PLOT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    d = load(a_.split)
    T, y = d["T"], d["gt"].numpy()
    trans = [f"L{l+1}->L{l+2}" for l in range(T)]
    print(f"{a_.split}: n={len(y)} TP={int(y.sum())} FP={int((1-y).sum())}  전이 {T}개")

    # ---------- Q1 층별 복원 ----------
    rows = []
    for l in range(d["recon"].shape[1]):
        r = d["recon"][:, l].numpy()
        rows.append({"layer": l + 1, "mean": float(r.mean()), "median": float(np.median(r)),
                     "p95": float(np.percentile(r, 95)), "max": float(r.max())})
    wcsv(RES / "reconstruction_by_layer.csv", rows)
    print("\n=== Q1 층별 복원 ||sum_k C_k^(l) - h^(l)|| / ||h^(l)|| ===")
    for r in rows:
        print(f"  층 {r['layer']:2}  평균 {r['mean']:.3e}  중앙 {r['median']:.3e}  "
              f"p95 {r['p95']:.3e}  최대 {r['max']:.3e}")

    # ---------- Q2 보존 ----------
    rows = []
    for l in range(T):
        ab, rl, cs = (d["cons_abs"][:, l].numpy(), d["cons_rel"][:, l].numpy(),
                      d["cons_cos"][:, l].numpy())
        rows.append({"transition": trans[l],
                     "abs_mean": float(ab.mean()), "abs_max": float(ab.max()),
                     "rel_mean": float(rl.mean()), "rel_median": float(np.median(rl)),
                     "rel_p95": float(np.percentile(rl, 95)), "rel_max": float(rl.max()),
                     "cos_min": float(cs.min()), "cos_mean": float(cs.mean()),
                     "g_norm_mean": float(d["g_norm"][:, l].numpy().mean())})
    wcsv(RES / "conservation_check.csv", rows)
    print("\n=== Q2 보존  sum_k D_k^(l)  vs  g^(l) ===")
    for r in rows:
        print(f"  {r['transition']:9} 상대오차 평균 {r['rel_mean']:.3e} 최대 {r['rel_max']:.3e}"
              f"  코사인 최소 {r['cos_min']:.8f}")

    # ---------- Q6 사영 보존 ----------
    rows = []
    for l in range(T):
        ab, rl = d["pj_abs"][:, l].numpy(), d["pj_rel"][:, l].numpy()
        rows.append({"transition": trans[l], "abs_mean": float(ab.mean()),
                     "abs_max": float(ab.max()), "rel_mean": float(rl.mean()),
                     "rel_max": float(rl.max()),
                     "corr_sum_a_vs_p": float(np.corrcoef(d["a_sum"][:, l].numpy(),
                                                          d["p"][:, l].numpy())[0, 1])})
    wcsv(RES / "projection_conservation.csv", rows)
    print("\n=== Q6 사영 보존  sum_k a_k^(l)  vs  dot(v,g^(l)) ===")
    for r in rows:
        print(f"  {r['transition']:9} 절대 평균 {r['abs_mean']:.3e} 최대 {r['abs_max']:.3e}"
              f"  상관 {r['corr_sum_a_vs_p']:.8f}")

    # ---------- Q7 + 17절 토큰 종류 분포 ----------
    A, M, ID = d["a"].numpy(), d["mask"].numpy(), d["input_ids"].numpy()
    kinds = {}
    for t in np.unique(ID):
        kinds[int(t)] = kind(tok.convert_ids_to_tokens(int(t)))
    trow, prow = [], []
    for l in range(T):
        for g, lbl in ((1, "TP"), (0, "FP")):
            sel = np.flatnonzero(y == g)
            cpos, cneg, cabs = Counter(), Counter(), Counter()
            mags = []
            for i in sel:
                m = M[i]
                if m.sum() == 0:
                    continue
                v = A[i, l][m]; ids = ID[i][m]
                k = min(a_.topk, len(v))
                for j in np.argsort(-v)[:k]:
                    cpos[kinds[int(ids[j])]] += 1
                for j in np.argsort(v)[:k]:
                    cneg[kinds[int(ids[j])]] += 1
                for j in np.argsort(-np.abs(v))[:k]:
                    cabs[kinds[int(ids[j])]] += 1
                mags.append(np.abs(v).mean())
            for nm, c in (("top_positive_a", cpos), ("top_negative_a", cneg), ("top_abs_a", cabs)):
                tot = sum(c.values()) or 1
                prow.append({"transition": trans[l], "label": lbl, "ranking": nm,
                             "punct_share": c["punct"]/tot, "content_share": c["content"]/tot,
                             "special_share": c["special"]/tot, "n_counted": tot})
            trow.append({"transition": trans[l], "label": lbl, "n": len(sel),
                         "mean_abs_a": float(np.mean(mags)) if mags else float("nan")})
    wcsv(RES / "token_directional_summary.csv", trow)
    wcsv(RES / "punctuation_comparison.csv", prow)
    print("\n=== 17절 방향성 기여 이동 a 의 상위 토큰 종류 (이전 UNSAFE-margin 분석과 별개) ===")
    print(f"{'전이':10} {'라벨':4} {'|a|상위 구두점':>14} {'내용어':>8} {'특수':>7}")
    for r in prow:
        if r["ranking"] == "top_abs_a":
            print(f"{r['transition']:10} {r['label']:4} {r['punct_share']*100:13.1f}% "
                  f"{r['content_share']*100:7.1f}% {r['special_share']*100:6.1f}%")

    # ---------- 16절 대표 표본 Token x Layer ----------
    for g, lbl in ((1, "TP"), (0, "FP")):
        sel = np.flatnonzero(y == g)
        # 토큰 수가 40~90 인 것 중 |a| 총합이 중앙값에 가까운 표본
        cand = [i for i in sel if 40 <= M[i].sum() <= 90]
        if not cand:
            cand = list(sel[:1])
        tot = np.array([np.abs(A[i]).sum() for i in cand])
        i = cand[int(np.argsort(tot)[len(tot)//2])]
        m = M[i]; ids = ID[i][m]; mat = A[i][:, m]          # (T, n_tok)
        toks = [tok.convert_ids_to_tokens(int(t)).replace("▁", "_") for t in ids]
        with open(RES / f"representative_{lbl.lower()}_token_layer.csv", "w",
                  newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(["token"] + trans)
            for j, t in enumerate(toks):
                w.writerow([t] + [f"{mat[l, j]:.6f}" for l in range(T)])
        # 전이마다 |a| 크기가 자릿수로 다르다(후반 층의 g 노름이 훨씬 크다).
        # 공통 색눈금을 쓰면 앞쪽 층 구조가 전부 0 처럼 보이므로 **행마다 정규화**한다.
        rowmax = np.abs(mat).max(axis=1, keepdims=True)
        matn = mat / np.where(rowmax > 0, rowmax, 1.0)
        fig, ax = plt.subplots(figsize=(max(8, len(toks) * .16), 5))
        im = ax.imshow(matn, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_yticks(range(T)); ax.set_yticklabels(trans, fontsize=7)
        ax.set_xticks(range(len(toks))); ax.set_xticklabels(toks, rotation=90, fontsize=5)
        ax.set_title(f"{lbl}  sample_id={d['sample_id'][i]}   "
                     f"cell = a_k^(l) = dot(v_U^(l), D_k^(l)), row-normalised", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=.8)
        plt.tight_layout()
        plt.savefig(PLOT / f"representative_{lbl.lower()}_heatmap.png", dpi=145); plt.close()
        print(f"  대표 {lbl} 표본 {d['sample_id'][i]} (토큰 {len(toks)}개) 저장")
    print(f"\n저장 -> {RES}, {PLOT}")


if __name__ == "__main__":
    main()
