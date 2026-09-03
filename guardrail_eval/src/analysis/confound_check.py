"""TP/FP 과제가 실제로는 '어느 말뭉치에서 왔나' 문제인지 확인하는 대조 실험.

발견: JailbreaksOverTime 에서 PG2 의 오탐(FP)은 **전부 wildchat** 이고
정탐(TP)은 전부 탈옥 프롬프트 모음집이다.  즉 TP/FP 라벨이 말뭉치 출처와 100% 일치한다.
그렇다면 층별 표현 탐침이 AUROC 0.99 를 내는 것은 '깊이별 표현에 TP/FP 정보가 있다'는
증거가 아니라 **두 말뭉치가 어휘적으로 다르다**는 사실의 반영일 수 있다.

대조군: 원문 글자 n-gram TF-IDF + 로지스틱 회귀.  표현을 전혀 안 보고 어휘만 본다.
이게 표현 탐침과 비슷한 AUROC 를 내면, 표현 탐침의 높은 점수는 어휘 신호의 재탕이다.
"""
import json, sys
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collections, numpy as np, torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from analysis.metrics import auroc, auprc
from data.splits import _dedup_rows, sid

ROOT = Path(__file__).resolve().parents[2]
FEAT, RES = ROOT / "artifacts/features", ROOT / "artifacts/results"


def unsafe_rows(split):
    return [r for r in (json.loads(l) for l in
                        open(FEAT / f"pg2_{split}.jsonl", encoding="utf-8"))
            if r["base_prediction"] == 1]


def main():
    src = {sid(r["prompt"]): r["source"] for r in _dedup_rows()}
    out = {}

    # 1) 라벨-출처 일치도
    for split in ("ver_train", "ver_dev", "eval_val", "eval_test"):
        u = unsafe_rows(split)
        tp = collections.Counter(src[r["sample_id"]].split("/")[0] for r in u if r["gt"] == 1)
        fp = collections.Counter(src[r["sample_id"]].split("/")[0] for r in u if r["gt"] == 0)
        wc_fp = fp.get("wildchat", 0) / max(sum(fp.values()), 1)
        wc_tp = tp.get("wildchat", 0) / max(sum(tp.values()), 1)
        out[f"source_{split}"] = {"tp": dict(tp), "fp": dict(fp),
                                  "fp_from_wildchat": wc_fp, "tp_from_wildchat": wc_tp}
        print(f"{split:10} FP 중 wildchat 비율 {wc_fp*100:6.2f}%   TP 중 wildchat 비율 {wc_tp*100:5.2f}%")
        print(f"           TP 출처 {dict(tp)}")

    # 2) 어휘만 쓰는 대조군
    tr = unsafe_rows("ver_train")
    Xtr = [r["text"] for r in tr]; ytr = np.array([r["gt"] for r in tr])
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=200000)
    A = vec.fit_transform(Xtr)                    # 어휘 사전도 train 에서만 적합
    lr = LogisticRegression(max_iter=3000, random_state=0).fit(A, ytr)
    print(f"\n어휘 대조군(char 3-5gram TF-IDF, 차원 {A.shape[1]})")
    for split in ("ver_dev", "eval_val", "eval_test"):
        u = unsafe_rows(split)
        y = np.array([r["gt"] for r in u])
        s = lr.predict_proba(vec.transform([r["text"] for r in u]))[:, 1]
        p = np.array([r["unsafe_probability"] for r in u])
        out[f"lexical_{split}"] = {"auroc": auroc(y, s), "auprc": auprc(y, s),
                                   "pg2_score_auroc": auroc(y, p)}
        print(f"  {split:10} 어휘 AUROC={auroc(y,s):.4f} AUPRC={auprc(y,s):.4f}   "
              f"| B0 PG2점수 AUROC={auroc(y,p):.4f}")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "confound_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"저장 -> {RES/'confound_check.json'}")


if __name__ == "__main__":
    main()
