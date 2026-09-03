"""분석 P·Q·R + 데이터셋 ID probe — 통제 실험 모음.

P: 분류기 헤드 방향 통제.  hidden error 신호가 그냥 SAFE/UNSAFE 결정축의 재표현인가?
Q: confidence/margin 은 **제안 방법이 아니라 통제**로만.  M0/M1/M2 중첩 비교.
R: 원문 어휘 지름길 통제 (TF-IDF).
+  pooled 표현에서 dataset ID 를 얼마나 잘 맞히는가.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd, torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForSequenceClassification

from failure_structure.common import (
    load, representations, group_split, cell_weights, y_error, auroc, auprc,
    boot_ci, boot_delta_ci, wcsv, CELLS, ALL_ANALYSED, SEEDS, RES, EPS, USABLE)
from failure_structure.probes import fit_logreg, MAIN_REPR

MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"


def head_direction():
    """PG2 분류기 헤드가 SAFE/UNSAFE 결정에 쓰는 방향.
    logit_unsafe - logit_benign = (W[1]-W[0])^T pooled + b.  풀러는 선형(dense)+gelu 이므로
    은닉공간 기준 결정축의 선형 성분은 W_pool^T (W[1]-W[0]) 로 본다."""
    m = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
    Wc = m.classifier.weight.detach().numpy()           # (2, 768)
    Wp = m.pooler.dense.weight.detach().numpy()         # (768, 768)
    v_head_pooled = Wc[1] - Wc[0]                       # 풀러 출력 공간
    v_head_hidden = Wp.T @ v_head_pooled                # 은닉(CLS) 공간으로 되돌림
    return (v_head_hidden / (np.linalg.norm(v_head_hidden) + EPS),
            v_head_pooled / (np.linalg.norm(v_head_pooled) + EPS))


def texts_for(sample_ids):
    """canonical_samples / pg2_*.jsonl 에서 원문을 되찾는다 (TF-IDF 통제용)."""
    ROOT = Path(__file__).resolve().parents[2]
    t = {}
    p = ROOT / "data/multisource_guard/canonical_samples.parquet"
    if p.exists():
        d = pd.read_parquet(p, columns=["sample_id", "text"])
        t.update(dict(zip(d.sample_id, d.text)))
    for sp in ("ver_train", "ver_dev", "eval_val", "eval_test"):
        f = ROOT / f"artifacts/features/pg2_{sp}.jsonl"
        if f.exists():
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                t.setdefault(r["sample_id"], r["text"])
    return [t.get(s) for s in sample_ids]


def main():
    D = load(); R = representations(D["h"])
    v_hid, v_pool = head_direction()
    lm = D["logit_unsafe"] - D["logit_benign"]
    pu = 1.0 / (1.0 + np.exp(-lm))
    ent = -(pu * np.log(pu + EPS) + (1 - pu) * np.log(1 - pu + EPS))
    txt = texts_for(D["sample_id"])
    rows_head, rows_conf, rows_inc, rows_tfidf, rows_dsid = [], [], [], [], []
    t0 = time.time()

    for ds in ALL_ANALYSED:
        dm = D["dataset"] == ds
        cell_d, dup_d = D["cell"][dm], D["dup"][dm]
        lm_d, pu_d, ent_d = lm[dm], pu[dm], ent[dm]
        txt_d = [txt[i] for i in np.flatnonzero(dm)]
        for seed in SEEDS:
            tr, te = group_split(dup_d, cell_d, seed)
            ctr, cte = cell_d[tr], cell_d[te]
            ytr, yte = y_error(ctr), y_error(cte)
            wtr = cell_weights(ctr)

            # ---------- Q. confidence 단독 (통제) ----------
            for fn, v in (("unsafe_probability", pu_d), ("logit_margin", lm_d),
                          ("entropy", ent_d)):
                lo, hi = boot_ci(yte, v[te], seed=seed)
                rows_conf.append(dict(dataset=ds, seed=seed, feature=fn,
                                      auroc_incorrect=auroc(yte, v[te]),
                                      auprc_incorrect=auprc(yte, v[te]),
                                      ci_lo=lo, ci_hi=hi, n_test=int(te.sum())))

            # ---------- R. 원문 어휘 지름길 ----------
            ttr = [txt_d[i] for i in np.flatnonzero(tr)]
            tte = [txt_d[i] for i in np.flatnonzero(te)]
            if all(x is not None for x in ttr) and all(x is not None for x in tte):
                vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                      min_df=3, max_features=100000)
                A = vec.fit_transform(ttr)
                mt = LogisticRegression(C=1.0, max_iter=2000)
                mt.fit(A, ytr, sample_weight=wtr)
                st = mt.decision_function(vec.transform(tte))
                lo, hi = boot_ci(yte, st, seed=seed)
                rows_tfidf.append(dict(dataset=ds, seed=seed, auroc_incorrect=auroc(yte, st),
                                       auprc_incorrect=auprc(yte, st), ci_lo=lo, ci_hi=hi,
                                       n_features=int(A.shape[1])))
            else:
                miss = sum(x is None for x in ttr) + sum(x is None for x in tte)
                rows_tfidf.append(dict(dataset=ds, seed=seed, auroc_incorrect=np.nan,
                                       auprc_incorrect=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                                       n_features=0, note=f"원문 못 찾음 {miss}건"))

            # ---------- P/Q. 은닉 probe + 헤드방향 + 중첩비교 ----------
            for rname in MAIN_REPR:
                Xall, lnames = R[rname]
                Xd = Xall[dm]
                for li, ln in enumerate(lnames):
                    Xtr = Xd[tr][:, li].astype(np.float64)
                    Xte = Xd[te][:, li].astype(np.float64)
                    m_, sc_, C_, _ = fit_logreg(Xtr, ytr, wtr, seed)
                    s_te = m_.decision_function(sc_.transform(Xte))
                    w = m_.coef_[0] / (np.linalg.norm(m_.coef_[0]) + EPS)
                    # 스케일러를 되돌린 원공간 계수
                    w_orig = m_.coef_[0] / (sc_.scale_ + EPS)
                    w_orig = w_orig / (np.linalg.norm(w_orig) + EPS)
                    vh = v_hid if Xtr.shape[1] == len(v_hid) else None
                    cos_head = float(w_orig @ vh) if vh is not None else np.nan

                    # 헤드 방향 성분 제거 후 재적합
                    a_resid = np.nan
                    if vh is not None:
                        Xtr_r = Xtr - np.outer(Xtr @ vh, vh)
                        Xte_r = Xte - np.outer(Xte @ vh, vh)
                        mr, scr, _, _ = fit_logreg(Xtr_r, ytr, wtr, seed)
                        a_resid = auroc(yte, mr.decision_function(scr.transform(Xte_r)))
                    rows_head.append(dict(dataset=ds, repr=rname, seed=seed, layer=ln,
                                          cos_probe_vs_head=cos_head,
                                          auroc_raw=auroc(yte, s_te),
                                          auroc_head_removed=a_resid,
                                          delta=a_resid - auroc(yte, s_te)))

                    # M0 / M1 / M2 중첩 비교
                    F0_tr = np.stack([lm_d[tr], pu_d[tr], ent_d[tr]], 1)
                    F0_te = np.stack([lm_d[te], pu_d[te], ent_d[te]], 1)
                    s_tr = m_.decision_function(sc_.transform(Xtr))
                    F2_tr = np.hstack([F0_tr, s_tr[:, None]])
                    F2_te = np.hstack([F0_te, s_te[:, None]])
                    out = {}
                    for nm, Ftr_, Fte_ in (("M0", F0_tr, F0_te), ("M2", F2_tr, F2_te)):
                        s0 = StandardScaler().fit(Ftr_)
                        mm = LogisticRegression(C=1.0, max_iter=1000)
                        mm.fit(s0.transform(Ftr_), ytr, sample_weight=wtr)
                        p_ = mm.predict_proba(s0.transform(Fte_))[:, 1]
                        out[nm] = (auroc(yte, p_), auprc(yte, p_),
                                   log_loss(yte, np.clip(p_, 1e-6, 1 - 1e-6)),
                                   brier_score_loss(yte, p_), p_)
                    a_m1 = auroc(yte, s_te)
                    dlo, dhi = boot_delta_ci(yte, out["M0"][4], out["M2"][4], seed=seed)
                    rows_inc.append(dict(dataset=ds, repr=rname, seed=seed, layer=ln,
                                         M0_auroc=out["M0"][0], M1_hidden_auroc=a_m1,
                                         M2_auroc=out["M2"][0],
                                         delta_M2_M0=out["M2"][0] - out["M0"][0],
                                         delta_ci_lo=dlo, delta_ci_hi=dhi,
                                         M0_logloss=out["M0"][2], M2_logloss=out["M2"][2],
                                         delta_logloss=out["M2"][2] - out["M0"][2],
                                         M0_brier=out["M0"][3], M2_brier=out["M2"][3],
                                         delta_brier=out["M2"][3] - out["M0"][3]))
            print(f"  {ds:20} seed{seed} ({time.time()-t0:.0f}s)", flush=True)

    # ---------- dataset ID probe (pooled) ----------
    pool = np.isin(D["dataset"], USABLE)
    for rname in MAIN_REPR:
        Xall, lnames = R[rname]
        for li, ln in enumerate(lnames):
            X = Xall[pool][:, li].astype(np.float64)
            lab = D["dataset"][pool]
            dup_p = D["dup"][pool]; cell_p = D["cell"][pool]
            tr, te = group_split(dup_p, cell_p, 0)
            s0 = StandardScaler().fit(X[tr])
            mm = LogisticRegression(C=0.1, max_iter=300, multi_class="multinomial")
            mm.fit(s0.transform(X[tr]), lab[tr])
            acc = float((mm.predict(s0.transform(X[te])) == lab[te]).mean())
            maj = float(pd.Series(lab[te]).value_counts(normalize=True).max())
            rows_dsid.append(dict(repr=rname, layer=ln, accuracy=acc,
                                  majority_baseline=maj, n_classes=len(set(lab)),
                                  n_test=int(te.sum())))
    RES.mkdir(parents=True, exist_ok=True)
    wcsv(RES / "classifier_head_control.csv", rows_head)
    wcsv(RES / "confidence_baselines.csv", rows_conf)
    wcsv(RES / "incremental_information.csv", rows_inc)
    wcsv(RES / "tfidf_control.csv", rows_tfidf)
    wcsv(RES / "dataset_identity_probe.csv", rows_dsid)


if __name__ == "__main__":
    main()
