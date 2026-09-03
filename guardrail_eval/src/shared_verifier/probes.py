"""PHASE1-6/7: 모델별 상한(upper bound) 과 공유 검증기(shared verifier).

과제: base model 의 예측이 맞았는지(correct=TP/TN) 틀렸는지(FP/FN) 를 맞힌다.

두 설정을 같은 코드로 돌려 직접 비교한다.
  per_model  (6단계) -- 모델마다 파라미터를 따로 적합.  MAIN 설정에서는 금지지만
                        "얼마나 잘할 수 있는가" 의 상한 기준선으로 쓴다.
  shared     (7단계) -- 파라미터 한 벌을 전 모델 데이터에 적합.  이게 MAIN 설정이다.

지표는 모델별 AUROC 이다.  6개를 한 통에 넣고 재면 검증기가 "이 모델은 원래 자주
틀린다" 는 모델 정체성만 배워도 점수가 높게 나온다.  모델 안에서 재면 그 지름길이
상수가 되어 사라진다.  AUROC 가 0.5 미만이어도 절대 뒤집지 않는다.
"""
import argparse, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared_verifier.features import check_no_leak, load_all

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/shared_verifier"


def make_clf(kind, seed):
    if kind == "linear":
        return LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    return MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=seed,
                         early_stopping=True, n_iter_no_change=10)


def auroc(y, s):
    """정답이 한 종류뿐이면 정의되지 않는다.  뒤집지 않는다."""
    return np.nan if len(np.unique(y)) < 2 else roc_auc_score(y, s)


def evaluate(tab, score, tag, setting, feat, clf, seed, rows):
    """모델별 / (모델,데이터셋)별 AUROC 를 기록."""
    te = tab.split == "test"
    for m, g in tab[te].groupby("model"):
        blocks = [("model", "ALL", g)]
        # base 예측을 고정한 안에서 재면 "base 출력 재현" 지름길이 사라진다.
        #   pred1 = base 가 공격이라 한 것들 중 TP vs FP  (원래 연구 질문)
        #   pred0 = base 가 정상이라 한 것들 중 TN vs FN
        blocks += [("pred1", "ALL", g[g.pred == 1]), ("pred0", "ALL", g[g.pred == 0])]
        blocks += [("model_dataset", d, gd) for d, gd in g.groupby("dataset")]
        for scope, key, sub in blocks:
            if len(sub) < 20:
                continue
            sc = score[sub.index]
            rows.append(dict(
                setting=setting, features=feat, clf=clf, seed=seed, scope=scope,
                model=m, dataset=key, n=len(sub), n_wrong=int((1 - sub.correct).sum()),
                auroc=auroc(sub.correct.to_numpy(), sc),
                # 라벨 누수 진단: 점수가 사실은 정답 클래스를 맞히고 있는 것 아닌가?
                auroc_vs_label=auroc(sub.y.to_numpy(), sc),
                # base 예측 자체를 재현하는 것 아닌가?
                auroc_vs_pred=auroc(sub.pred.to_numpy(), sc)))


def run_tfidf(tab, seed, rows):
    """글자 3-5그램 TF-IDF -> 로지스틱.  vectorizer 는 train 원문에서만 적합한다.

    같은 원문이 모델 수만큼 중복 등장하므로 고유 원문에 대해서만 변환하고
    행마다 그 고유 원문의 위치를 가리키게 한다(희소행렬 메모리 6배 절약)."""
    uid, inv = np.unique(tab.sample_id.to_numpy(), return_inverse=True)
    utext = tab.text.to_numpy()[np.unique(inv, return_index=True)[1]]
    tr_mask = (tab.split == "train").to_numpy()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=10,
                          max_features=100000, sublinear_tf=True)
    vec.fit(tab.text.to_numpy()[tr_mask & ~pd.Series(inv).duplicated().to_numpy()])
    U = vec.transform(utext)                 # (고유원문 수, 어휘)
    X = U[inv]                               # 행 -> 고유원문
    y = tab.correct.to_numpy()
    for setting in ("per_model", "shared"):
        sc = np.full(len(tab), np.nan)
        if setting == "shared":
            if len(np.unique(y[tr_mask])) > 1:
                c = LogisticRegression(max_iter=1000, random_state=seed).fit(X[tr_mask], y[tr_mask])
                sc[~tr_mask] = c.predict_proba(X[~tr_mask])[:, 1]
        else:
            for m in tab.model.unique():
                mm = (tab.model == m).to_numpy()
                a, b = mm & tr_mask, mm & ~tr_mask
                if len(np.unique(y[a])) < 2:
                    continue
                c = LogisticRegression(max_iter=1000, random_state=seed).fit(X[a], y[a])
                sc[b] = c.predict_proba(X[b])[:, 1]
        evaluate(tab, sc, setting, setting, "tfidf", "linear", seed, rows)
    print(f"  seed{seed} tfidf     linear  완료", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--features", default="geom,raw_last,conf,tfidf")
    ap.add_argument("--clfs", default="linear,mlp")
    args = ap.parse_args()

    tab, feats, gnames = load_all(seed=0)
    ngroup = check_no_leak(tab)
    print(f"행 {len(tab):,}  모델 {tab.model.nunique()}  쌍 {tab.groupby(['model','dataset']).ngroups}  "
          f"중복그룹 {ngroup:,}  분할누수 없음")
    print(f"train {int((tab.split=='train').sum()):,} / test {int((tab.split=='test').sum()):,}  "
          f"오답비율 {1-tab.correct.mean():.3f}\n")

    rows = []
    for seed in range(args.seeds):
        for fname in args.features.split(","):
            if fname == "tfidf":
                # 어휘 대조군: 표현을 전혀 안 보고 원문 글자 n-gram 만 쓴다.
                # 표현이 이걸 못 넘으면 "표현에 정보가 있다" 고 말할 수 없다.
                run_tfidf(tab, seed, rows)
                continue
            X = feats[fname]
            for cname in args.clfs.split(","):
                # --- 6단계: 모델별 상한 -------------------------------------
                sc = np.full(len(tab), np.nan)
                for m, g in tab.groupby("model"):
                    tr, te = g.index[g.split == "train"], g.index[g.split == "test"]
                    if len(np.unique(tab.correct[tr])) < 2:
                        continue
                    s = StandardScaler().fit(X[tr])
                    c = make_clf(cname, seed).fit(s.transform(X[tr]), tab.correct[tr])
                    sc[te] = c.predict_proba(s.transform(X[te]))[:, 1]
                evaluate(tab, sc, "per_model", "per_model", fname, cname, seed, rows)

                # --- 7단계: 공유 파라미터 한 벌 ------------------------------
                tr = tab.index[tab.split == "train"]
                te = tab.index[tab.split == "test"]
                s = StandardScaler().fit(X[tr])          # 표준화도 공유(모델별 금지)
                c = make_clf(cname, seed).fit(s.transform(X[tr]), tab.correct[tr])
                sc = np.full(len(tab), np.nan)
                sc[te] = c.predict_proba(s.transform(X[te]))[:, 1]
                evaluate(tab, sc, "shared", "shared", fname, cname, seed, rows)
                print(f"  seed{seed} {fname:<9} {cname:<7} 완료", flush=True)

    df = pd.DataFrame(rows)
    RES.mkdir(parents=True, exist_ok=True)
    df.to_csv(RES / "probe_results.csv", index=False)

    print("\n=== 모델별 AUROC (시드 평균) — 헤드라인 ===")
    h = (df[df.scope == "model"].groupby(["setting", "features", "clf", "model"])
         .auroc.mean().unstack("model").round(3))
    print(h.to_string())
    print("\n=== 설정별 요약: 모델 6개 AUROC 의 평균 ===")
    print(h.mean(axis=1).round(3).to_string())
    print(f"\n저장 -> {RES/'probe_results.csv'}")


if __name__ == "__main__":
    main()
