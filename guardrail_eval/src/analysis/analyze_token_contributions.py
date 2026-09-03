"""토큰 수준 분석 (지시문 9장).

TP / FP 각각에서:
    방향성 점수 상위(+) 토큰   = TP 쪽 CLS 이동에 기여
    방향성 점수 하위(-) 토큰   = FP 쪽 CLS 이동에 기여
    UNSAFE 여유 기여 상위 토큰 = 분류기가 UNSAFE 로 밀린 이유

특수토큰([CLS]/[SEP]/PAD)은 **복원에는 포함**하되 사람이 보는 표에서는 기본 제외한다
(--keep-special 로 포함 가능).  하위단어(subword)는 원문 조각으로 되돌려 보여준다.

또 오탐(FP)을 유형별로 나눠 본다: 탈옥 문구 인용 / 보안 설명 / 번역 / 코드 / 로그 / 교육.
"""
import argparse, glob, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
FEAT, RES = ROOT / "artifacts/features", ROOT / "artifacts/results"
MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"

FP_KINDS = {
    "탈옥문구 인용": r"(ignore (all )?previous instructions|jailbreak|DAN\b|prompt injection)",
    "보안 설명":    r"(vulnerab|exploit|security|attack|malware|penetration test)",
    "번역":        r"(translate|translation|번역|traduce|übersetz)",
    "코드":        r"(```|def |class |import |function\s*\(|<\?php|SELECT .* FROM)",
    "로그 분석":    r"(\[ERROR\]|\[INFO\]|stack ?trace|Traceback|log file|exception)",
    "교육/설명":    r"(explain|what is|how does|why (is|does)|teach me|tutorial)",
}


def load_dx(split):
    parts = sorted(glob.glob(str(FEAT / f"decompx_{split}_*of*.pt")))
    ds = [torch.load(p, weights_only=False) for p in parts]
    out = {k: torch.cat([d[k] for d in ds]) for k in
           ("directional", "margin", "mask", "input_ids", "gt", "seq_len", "recon_rel_err")}
    out["sample_id"] = [s for d in ds for s in d["sample_id"]]
    return out


def texts(split):
    return {r["sample_id"]: r["text"] for r in
            (json.loads(l) for l in open(FEAT / f"pg2_{split}.jsonl", encoding="utf-8"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval_test")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--keep-special", action="store_true")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    special = set(tok.all_special_ids)
    d = load_dx(a.split); T = texts(a.split)
    print(f"{a.split}: {len(d['gt'])}건  복원 상대오차 max={d['recon_rel_err'].max():.3e} "
          f"mean={d['recon_rel_err'].mean():.3e}")

    agg = {g: {"dir_pos": Counter(), "dir_neg": Counter(), "margin": Counter()}
           for g in (1, 0)}
    examples = {1: [], 0: []}
    for i in range(len(d["gt"])):
        # seq_len 은 창 폭(패딩 포함)이므로 실제 토큰 수는 mask 로 센다
        n = int(d["mask"][i].sum()); g = int(d["gt"][i])
        ids = d["input_ids"][i, :n]
        keep = torch.ones(n, dtype=torch.bool) if a.keep_special else \
            torch.tensor([int(t) not in special for t in ids])
        if keep.sum() == 0:
            continue
        dr = d["directional"][i, :n].clone(); mg = d["margin"][i, :n].clone()
        k = min(a.topk, int(keep.sum()))
        for nm, vals, sign in (("dir_pos", dr, 1), ("dir_neg", dr, -1), ("margin", mg, 1)):
            # 제외 토큰은 0 이 아니라 -inf 로 눌러야 한다.  0 으로 두면 "상위 k개를 채울
            # 실토큰이 부족한" 짧은 표본에서 [CLS]/[SEP] 가 상위에 끼어든다(실측 확인).
            v = (vals * sign).masked_fill(~keep, float("-inf"))
            idx = torch.topk(v, k).indices
            for j in idx:
                agg[g][nm][tok.decode([int(ids[j])]).strip().lower()] += 1
        if len(examples[g]) < 12:
            kk = min(6, int(keep.sum()))
            top = torch.topk(mg.masked_fill(~keep, float("-inf")), kk).indices.tolist()
            bot = torch.topk((-dr).masked_fill(~keep, float("-inf")), kk).indices.tolist()
            examples[g].append({
                "sample_id": d["sample_id"][i],
                "text_head": T[d["sample_id"][i]][:220],
                "top_unsafe_margin": [(tok.decode([int(ids[j])]).strip(), round(float(mg[j]), 3))
                                      for j in top],
                "top_fp_like_direction": [(tok.decode([int(ids[j])]).strip(), round(float(dr[j]), 3))
                                          for j in bot]})

    # DecompX 기여 상위는 구두점이 지배한다 -- 그 비율을 수치로 남겨 해석 한계를 명시
    PUNCT = set(".,?!:;'\"()[]{}-—…")
    out = {"split": a.split, "keep_special": a.keep_special}
    for g, lbl in ((1, "TP"), (0, "FP")):
        for nm in ("dir_pos", "dir_neg", "margin"):
            tot = sum(agg[g][nm].values())
            pun = sum(c for w, c in agg[g][nm].items() if w and all(ch in PUNCT for ch in w))
            out[f"{lbl}_{nm}_punct_share"] = pun / tot if tot else 0.0
    for g, lbl in ((1, "TP"), (0, "FP")):
        print(f"\n=== {lbl} ({int((d['gt']==g).sum())}건) ===")
        for nm, ko in (("dir_pos", "방향성 상위(+, TP쪽)"), ("dir_neg", "방향성 하위(-, FP쪽)"),
                       ("margin", "UNSAFE 여유 기여 상위")):
            top = agg[g][nm].most_common(20)
            print(f"  {ko:26} " + ", ".join(f"{w}({c})" for w, c in top[:14]))
            out[f"{lbl}_{nm}"] = top
        out[f"{lbl}_examples"] = examples[g]

    # --- 오탐 유형 ---
    fp_ids = [d["sample_id"][i] for i in range(len(d["gt"])) if int(d["gt"][i]) == 0]
    kinds = defaultdict(list)
    for s in fp_ids:
        t = T[s]
        hit = [k for k, pat in FP_KINDS.items() if re.search(pat, t, re.I)]
        for k in (hit or ["기타"]):
            kinds[k].append(s)
    print(f"\n=== 오탐(FP) 유형 (중복 허용, 총 {len(fp_ids)}건) ===")
    for k, v in sorted(kinds.items(), key=lambda x: -len(x[1])):
        print(f"  {k:14} {len(v):4}건 ({len(v)/len(fp_ids)*100:5.1f}%)")
    out["fp_kinds"] = {k: {"n": len(v), "ids": v[:20]} for k, v in kinds.items()}

    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"token_contributions_{a.split}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n저장 -> {RES/f'token_contributions_{a.split}.json'}")


if __name__ == "__main__":
    main()
