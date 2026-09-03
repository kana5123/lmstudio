"""최종 시스템 평가 — PromptGuard2 단독 vs PromptGuard2 + 검증기(cascade).

프로토콜은 기존 rfpr.py 와 **동일**하다: 임계값은 eval_val 에서만 고르고 eval_test 에 적용.
그래야 기존 벤치마크 수치(results/rfpr_jailbreak_promptguard_v2.json)와 직접 비교된다.

cascade 점수 정의:
    PG2 가 BENIGN 이라 한 표본        -> 0.0        (검증기를 거치지 않고 통과)
    PG2 가 UNSAFE 라 한 표본          -> 검증기 확률 (또는 아래 blend)
따라서 임계값을 올리면 검증기가 낮게 본 오탐부터 걷힌다.

**재현 확인**: 우리 PG2 점수로 계산한 R@FPR 이 기존 저장 결과와 같아야 한다(파이프라인 검증).
"""
import argparse, json, sys
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from analysis.metrics import pick_threshold, recall_fpr, auroc, auprc

ROOT = Path(__file__).resolve().parents[2]
FEAT, RES = ROOT / "artifacts/features", ROOT / "artifacts/results"
TARGETS = [0.01, 0.005, 0.001]


def full_split(split):
    """UNSAFE 여부와 무관하게 평가셋 전체를 읽는다(최종 FPR 은 전체 기준이라야 한다)."""
    rows = [json.loads(l) for l in open(FEAT / f"pg2_{split}.jsonl", encoding="utf-8")]
    return rows


def bootstrap_recall(scores, labels, thr, n_boot=2000, seed=0):
    """eval_test 를 복원추출해 재현율의 95% 구간을 낸다.  **임계값은 val 에서 고정**한 채
    시험셋 표본만 흔든다 — '이 차이가 표본 잡음으로 설명되나'를 보기 위한 것이다."""
    rng = np.random.default_rng(seed)
    sc = np.asarray(scores); y = np.asarray(labels)
    pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
    recs, fprs = [], []
    for _ in range(n_boot):
        p_ = rng.choice(pos, len(pos), replace=True)
        n_ = rng.choice(neg, len(neg), replace=True)
        recs.append(float((sc[p_] >= thr).mean())); fprs.append(float((sc[n_] >= thr).mean()))
    return (float(np.percentile(recs, 2.5)), float(np.percentile(recs, 97.5)),
            float(np.percentile(fprs, 2.5)), float(np.percentile(fprs, 97.5)))


def rfpr_table(scores, labels, vs, vy, tag):
    out = {}
    for t in TARGETS:
        thr = pick_threshold(vs, vy, t)
        vr, vf = recall_fpr(vs, vy, thr)
        tr_, tf = recall_fpr(scores, labels, thr)
        k = f"{t*100:g}pct"
        out[f"thr@{k}"] = thr
        out[f"val_recall@{k}"], out[f"val_fpr@{k}"] = vr, vf
        out[f"recall@{k}"], out[f"achieved_fpr@{k}"] = tr_, tf
        n_pos, n_neg = sum(labels), len(labels) - sum(labels)
        out[f"tp@{k}"] = int(round(tr_ * n_pos)); out[f"fp@{k}"] = int(round(tf * n_neg))
        lo, hi, flo, fhi = bootstrap_recall(scores, labels, thr)
        out[f"recall_ci@{k}"] = [lo, hi]; out[f"fpr_ci@{k}"] = [flo, fhi]
    out["roc_auc"] = auroc(labels, scores); out["pr_auc"] = auprc(labels, scores)
    return out


WGRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def blend(val, test, sc, vy, ty):
    """대수-선형 혼합: score = p^(1-w) * v^w.  **w 도 임계값도 eval_val 에서만 고른다.**

    순수 대체(w=1)는 PG2 자신의 순위 정보를 통째로 버린다.  아주 엄격한 FPR 에서는
    그 순위가 중요하다.  하나의 w 로 모든 목표 FPR 을 이기지 못하므로,
    **목표 FPR 마다 w 를 따로 고른다** — 임계값을 목표마다 따로 고르는 것과 같은 규약이다.
    w=0 이면 PG2 단독, w=1 이면 순수 대체와 같아서, 이 절차는 두 극단을 포함한다.
    """
    eps = 1e-12

    def mk(rows, table, w):
        out = []
        for r in rows:
            p = max(r["unsafe_probability"], eps)
            if r["base_prediction"] != 1:
                out.append(p * 1e-6)        # 통과시킨 표본은 항상 후보보다 아래에 오도록
            else:
                v = max(table.get(r["sample_id"], 0.0), eps)
                out.append(float(np.exp((1 - w) * np.log(p) + w * np.log(v))))
        return out

    cache_v = {w: mk(val, sc["eval_val"], w) for w in WGRID}
    cache_t = {w: mk(test, sc["eval_test"], w) for w in WGRID}
    out = {}
    for t in TARGETS:
        best_w, best_r, best_thr = 0.0, -1, None
        for w in WGRID:
            thr = pick_threshold(cache_v[w], vy, t)
            r, f = recall_fpr(cache_v[w], vy, thr)
            if r > best_r:                      # 동점이면 작은 w (=기준선에 가까운 쪽)
                best_r, best_w, best_thr = r, w, thr
        vr, vf = recall_fpr(cache_v[best_w], vy, best_thr)
        tr_, tf = recall_fpr(cache_t[best_w], ty, best_thr)
        k = f"{t*100:g}pct"
        out[f"w@{k}"] = best_w; out[f"thr@{k}"] = best_thr
        out[f"val_recall@{k}"], out[f"val_fpr@{k}"] = vr, vf
        out[f"recall@{k}"], out[f"achieved_fpr@{k}"] = tr_, tf
        n_pos, n_neg = sum(ty), len(ty) - sum(ty)
        out[f"tp@{k}"] = int(round(tr_ * n_pos)); out[f"fp@{k}"] = int(round(tf * n_neg))
        lo, hi, flo, fhi = bootstrap_recall(cache_t[best_w], ty, best_thr)
        out[f"recall_ci@{k}"] = [lo, hi]; out[f"fpr_ci@{k}"] = [flo, fhi]
    # 전체 순위 품질은 목표와 무관하므로 중간 가중치 하나로 대표해 보고
    mid = WGRID[len(WGRID) // 2]
    out["roc_auc"] = auroc(ty, cache_t[mid]); out["pr_auc"] = auprc(ty, cache_t[mid])
    out["roc_auc_w"] = mid
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help="비교할 검증기 이름들")
    a = ap.parse_args()

    val, test = full_split("eval_val"), full_split("eval_test")
    vy = [r["gt"] for r in val];  ty = [r["gt"] for r in test]
    vp = [r["unsafe_probability"] for r in val]
    tp_ = [r["unsafe_probability"] for r in test]

    res = {"n_val": len(val), "n_test": len(test),
           "pos_val": sum(vy), "pos_test": sum(ty)}
    res["PG2_raw"] = rfpr_table(tp_, ty, vp, vy, "PG2_raw")
    for t in TARGETS:
        k = f"{t*100:g}pct"
        ci = res["PG2_raw"][f"recall_ci@{k}"]
        print(f"  PG2 단독 R@{k} = {res['PG2_raw'][f'recall@{k}']:.4f} "
              f"[95% {ci[0]:.4f}, {ci[1]:.4f}]  (val 음성 {len(vy)-sum(vy)}건 -> "
              f"허용 오탐 {int((len(vy)-sum(vy))*t)}건)")

    # --- 파이프라인 재현 확인: 기존 저장 결과와 일치하는가 ---
    old = json.loads((ROOT / "results/rfpr_jailbreak_promptguard_v2.json").read_text())
    for k in ("recall@1pct", "recall@0.1pct", "achieved_fpr@1pct"):
        d = abs(res["PG2_raw"][k] - old[k])
        print(f"재현 확인 {k}: 우리 {res['PG2_raw'][k]:.6f}  기존 {old[k]:.6f}  차이 {d:.2e}")
        assert d < 1e-9, f"기존 벤치마크와 불일치: {k}"
    print("-> 기존 JailbreaksOverTime 결과를 정확히 재현했다.\n")

    # --- cascade ---
    S = torch.load(RES / "verifier_scores.pt", weights_only=False)
    names = a.models or sorted({k.rsplit("_s", 1)[0] for k in S})
    ceiling = sum(r["gt"] == 1 and r["base_prediction"] == 1 for r in test) / max(sum(ty), 1)
    res["cascade_recall_ceiling"] = ceiling
    print(f"cascade 재현율 천장 = PG2 가 0.5 에서 잡은 공격 비율 = {ceiling:.4f}\n")

    for name in names:
        seeds = sorted(k for k in S if k.rsplit("_s", 1)[0] == name)
        per_seed, blend_seed = [], []
        for k in seeds:
            sc = {}
            for split, arr in (("eval_val", S[k]["eval_val"]), ("eval_test", S[k]["eval_test"])):
                sc[split] = dict(zip(arr["sample_id"], arr["score"]))
            vs = [sc["eval_val"].get(r["sample_id"], 0.0) if r["base_prediction"] == 1 else 0.0
                  for r in val]
            ts = [sc["eval_test"].get(r["sample_id"], 0.0) if r["base_prediction"] == 1 else 0.0
                  for r in test]
            per_seed.append(rfpr_table(ts, ty, vs, vy, name))
            blend_seed.append(blend(val, test, sc, vy, ty))
        agg = {kk: (float(np.mean([p[kk] for p in per_seed])),
                    float(np.std([p[kk] for p in per_seed]))) for kk in per_seed[0]
               if isinstance(per_seed[0][kk], (int, float))}
        res[f"cascade_{name}"] = {"per_seed": per_seed, "mean_std": agg}
        bagg = {kk: (float(np.mean([p[kk] for p in blend_seed])),
                     float(np.std([p[kk] for p in blend_seed])))
                for kk in blend_seed[0] if isinstance(blend_seed[0][kk], (int, float))}
        res[f"blend_{name}"] = {"per_seed": blend_seed, "mean_std": bagg}
        print(f"{name:20} R@1%={agg['recall@1pct'][0]:.4f}±{agg['recall@1pct'][1]:.4f}  "
              f"R@0.5%={agg['recall@0.5pct'][0]:.4f}±{agg['recall@0.5pct'][1]:.4f}  "
              f"R@0.1%={agg['recall@0.1pct'][0]:.4f}±{agg['recall@0.1pct'][1]:.4f}  "
              f"ROC={agg['roc_auc'][0]:.4f}")
        print(f"{'  └ 혼합':20} R@1%={bagg['recall@1pct'][0]:.4f}±{bagg['recall@1pct'][1]:.4f}  "
              f"R@0.5%={bagg['recall@0.5pct'][0]:.4f}±{bagg['recall@0.5pct'][1]:.4f}  "
              f"R@0.1%={bagg['recall@0.1pct'][0]:.4f}±{bagg['recall@0.1pct'][1]:.4f}  "
              f"ROC={bagg['roc_auc'][0]:.4f}  (w: 1%={bagg['w@1pct'][0]:.2f} "
              f"0.5%={bagg['w@0.5pct'][0]:.2f} 0.1%={bagg['w@0.1pct'][0]:.2f})")

    print(f"\n[기준] PG2 단독  R@1%={res['PG2_raw']['recall@1pct']:.4f}  "
          f"R@0.5%={res['PG2_raw']['recall@0.5pct']:.4f}  "
          f"R@0.1%={res['PG2_raw']['recall@0.1pct']:.4f}  ROC={res['PG2_raw']['roc_auc']:.4f}")
    (RES / "final_guard.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"저장 -> {RES/'final_guard.json'}")


if __name__ == "__main__":
    main()
