"""PHASE E0 실행: population / split / gate / route rate / 추출 비용 산출물 저장."""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.high_recall_attribution_cascade.config import (DATA, EXCLUDED_UNREVIEWED, GATE_RECALLS,
                                                        RES, TARGET_FPRS)
from src.high_recall_attribution_cascade.gate import build as build_gate
from src.high_recall_attribution_cascade.gate import gate_report, tau_for_recall
from src.high_recall_attribution_cascade.population import build as build_pop

COST = {(0, 32): 0.004, (32, 64): 0.009, (64, 128): 0.044, (128, 256): 0.180, (256, 512): 2.898}


def main():
    RES.mkdir(parents=True, exist_ok=True)
    pop = build_pop(save=True)

    # 1. population -----------------------------------------------------------
    t = pop.pivot_table(index="source_group", columns="gt_attack", values="sample_id",
                        aggfunc="count", fill_value=0)
    t.columns = ["n_benign", "n_attack"]
    t["n"] = t.sum(1); t["attack_rate"] = t.n_attack / t.n
    t["source_group_raw"] = pop.groupby("source_group").source_group_raw.agg(
        lambda s: ";".join(sorted(s.unique())))
    t.to_csv(RES / "population_by_source.csv")

    # 2. split audit ----------------------------------------------------------
    s = pop.pivot_table(index=["source_group", "split"], columns="gt_attack",
                        values="sample_id", aggfunc="count", fill_value=0)
    s.columns = ["n_benign", "n_attack"]; s["n"] = s.sum(1)
    s["groups"] = pop.groupby(["source_group", "split"]).duplicate_group_id.nunique()
    s.to_csv(RES / "split_audit.csv")
    leak = int((pop.groupby("duplicate_group_id").split.nunique() > 1).sum())

    # 3. score distribution ---------------------------------------------------
    rows = []
    for (src, y), g in pop.groupby(["source_group", "gt_attack"]):
        for col in ("p_attack", "margin"):
            x = g[col]
            rows.append(dict(source_group=src, gt_class="attack" if y else "benign", score=col,
                             n=len(x), mean=x.mean(), p05=x.quantile(.05), p25=x.quantile(.25),
                             median=x.median(), p75=x.quantile(.75), p95=x.quantile(.95)))
    pd.DataFrame(rows).to_csv(RES / "base_score_distribution.csv", index=False)

    # 4-7. gate ---------------------------------------------------------------
    taus, G = build_gate(pop)
    G.to_csv(RES / "gate_thresholds.csv", index=False)
    json.dump(taus, open(RES / "tau_gate.json", "w"), indent=1)

    # source 별 단독 tau (어느 source 가 제약을 만드는가)
    calib = pop[pop.split == "gate_calib"]
    dr = []
    for rho in GATE_RECALLS:
        for src, g in calib.groupby("source_group"):
            a = np.sort(g.loc[g.gt_attack == 1, "p_attack"].to_numpy())[::-1]
            k = int(np.ceil(len(a) * rho)); tau = float(a[min(k, len(a)) - 1])
            dr.append(dict(rho=rho, source_group=src, tau_needed=tau,
                           route_rate_if_used=float((pop.p_attack >= tau).mean())))
    pd.DataFrame(dr).to_csv(RES / "gate_binding_source.csv", index=False)

    # 게이트 전제 검사 ---------------------------------------------------------
    pr = []
    for src, g in list(pop.groupby("source_group")) + [("POOLED", pop)]:
        b, a = g.loc[g.gt_attack == 0, "p_attack"], g.loc[g.gt_attack == 1, "p_attack"]
        pr.append(dict(source_group=src, benign_median=b.median(), attack_median=a.median(),
                       attack_below_benign_median=float((a < b.median()).mean()),
                       attack_below_0p01=float((a < .01).mean()),
                       attack_below_0p001=float((a < .001).mean()),
                       benign_above_0p01=float((b >= .01).mean())))
    pd.DataFrame(pr).to_csv(RES / "gate_premise_check.csv", index=False)

    # base 단독 FPR 제약 최대 recall (S1 기준선 예고) --------------------------
    br = []
    for src, g in list(pop.groupby("source_group")) + [("POOLED", pop)]:
        f, tt, thr = roc_curve(g.gt_attack, g.p_attack)
        nb = int((g.gt_attack == 0).sum())
        for al in TARGET_FPRS + (0.05, 0.10):
            ok = f <= al
            br.append(dict(source_group=src, target_fpr=al,
                           max_recall=float(tt[ok].max()) if ok.any() else np.nan,
                           benign_denominator=nb, min_fpr_step=1 / nb,
                           insufficient_resolution=bool(1 / nb > al)))
    pd.DataFrame(br).to_csv(RES / "base_strict_threshold_reference.csv", index=False)

    # 8. 추출 비용 -------------------------------------------------------------
    tau_env = taus[f"rho={max(GATE_RECALLS)}"]
    cand = pop[pop.p_attack >= tau_env]
    tot = sum(int(((cand.token_length > lo) & (cand.token_length <= hi)).sum()) * c
              for (lo, hi), c in COST.items()) / 3600
    tk = int(cand.token_length.sum())
    ex = dict(tau_envelope=tau_env, n_candidates=len(cand), n_total=len(pop),
              route_rate=len(cand) / len(pop), total_tokens=tk,
              gpu_hours_single=tot, gpu_hours_8gpu=tot / 8,
              storage_Y_MB=tk * 2 * 4 / 1e6, storage_ids_MB=tk * 4 / 1e6,
              storage_if_C_stored_GB=tk * 12 * 768 * 4 / 1e9,
              excluded_unreviewed_sources=EXCLUDED_UNREVIEWED)
    json.dump(ex, open(RES / "extraction_estimate.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(dict(n_pop=len(pop), leak=leak, taus=taus, extraction=ex),
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
