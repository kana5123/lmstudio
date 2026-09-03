"""여러 출처를 하나의 표준 형식으로 모은다 (지시문 5·6·7절).

핵심 원칙: **출처(provenance)를 잃지 않는다.**  거대한 merged train set 을 만드는 것이
목적이 아니라, source_group 안에서 benign/attack 이 모두 존재하는지 보려는 것이다.

정식 라벨 분류(canonical taxonomy):
  BENIGN                        평범한 정상 요청
  HARMFUL_DIRECT                유해하지만 조작/탈옥 기법이 아님 (직접 요청)
  JAILBREAK_ADVERSARIAL         모델 제약을 우회하려는 탈옥 프롬프트
  PROMPT_INJECTION              사용자 입력에 새 지시를 주입
  INDIRECT_PROMPT_INJECTION     제3자 문서/도구 출력에 지시를 심음
  HARD_NEGATIVE_JAILBREAK_LIKE  정상인데 탈옥처럼 보임 (역할극·안전 관련 질문 등)
  HARD_NEGATIVE_INJECTION_LIKE  정상인데 주입처럼 보임 (NotInject 등)
  UNKNOWN                       확인 못 함

MAIN 이진 목표(지시문 6절):
  양성 = JAILBREAK_ADVERSARIAL / PROMPT_INJECTION / INDIRECT_PROMPT_INJECTION
  음성 = BENIGN / HARD_NEGATIVE_JAILBREAK_LIKE / HARD_NEGATIVE_INJECTION_LIKE
  HARMFUL_DIRECT 와 UNKNOWN 은 MAIN 에서 제외(보존은 함).
"""
import glob, hashlib, json, os, sys, unicodedata
from pathlib import Path

try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError:
    pass

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/multisource_guard"
PG = Path("/home/kana5123/ETRI/datasets/piguard/datasets")
HUB = os.path.expanduser("~/.cache/huggingface/hub")

B, HD, JA, PI, IPI, HNJ, HNI, UNK = (
    "BENIGN", "HARMFUL_DIRECT", "JAILBREAK_ADVERSARIAL", "PROMPT_INJECTION",
    "INDIRECT_PROMPT_INJECTION", "HARD_NEGATIVE_JAILBREAK_LIKE",
    "HARD_NEGATIVE_INJECTION_LIKE", "UNKNOWN")
MAIN_POS = {JA, PI, IPI}
MAIN_NEG = {B, HNJ, HNI}

# PIGuard train.json 의 source 문자열 -> (label0 정식라벨, label1 정식라벨, 공격계열, 언어)
# 근거는 docs/multisource_guard/label_mapping.md 에 기록한다.
PIGUARD_MAP = {
    "BIPIA":                            (B,   IPI, "indirect_injection", "en"),
    "TaskTracker":                      (B,   IPI, "indirect_injection", "en"),
    "Question Set":                     (HNJ, JA,  "jailbreak_roleplay", "en"),
    "jailbreak-classification":         (B,   JA,  "jailbreak", "en"),
    "prompt-injections":                (B,   PI,  "direct_injection", "multi"),
    "safe-guard-prompt-injection":      (B,   JA,  "jailbreak_mixed", "en"),
    "hackaprompt-dataset":              (B,   PI,  "direct_injection", "en"),
    "InjecAgent":                       (B,   IPI, "indirect_injection", "en"),
    "StruQ":                            (B,   PI,  "direct_injection", "en"),
    "ChatGPT-Jailbreak-Prompts":        (B,   JA,  "jailbreak", "en"),
    "vigil-jailbreak-ada-002":          (B,   JA,  "jailbreak", "en"),
    "Prompt-Injection-Mixed-Techniques":(B,   PI,  "direct_injection", "en"),
    "LLM Augmented set":                (B,   UNK, "UNKNOWN", "en"),
    "Alpaca":                           (B,   JA,  "none", "en"),
    "chatbot_instruction_prompts":      (B,   JA,  "none", "en"),
    "open-instruct":                    (B,   JA,  "none", "en"),
    "ultrachat_200k":                   (B,   JA,  "none", "en"),
    "no_robots":                        (B,   JA,  "none", "en"),
    "grok-conversation-harmless":       (B,   JA,  "none", "en"),
    "awesome-chatgpt-prompts":          (HNJ, JA,  "none", "en"),
    "over-defense":                     (HNJ, JA,  "none", "en"),
    "xtest-v2-copy":                    (HNJ, JA,  "none", "en"),
}
WJ_MAP = {
    "vanilla_benign":      (B,   "none", "vanilla"),
    "vanilla_harmful":     (HD,  "none", "vanilla"),
    "adversarial_benign":  (HNJ, "none", "adversarial"),
    "adversarial_harmful": (JA,  "jailbreak_adversarial", "adversarial"),
}


def norm(t):
    return " ".join(unicodedata.normalize("NFKC", str(t)).lower().split())


def sid(dataset, oid, text):
    return hashlib.sha1(f"{dataset}||{oid}||{norm(text)}".encode()).hexdigest()[:20]


def rec(text, dataset, group, orig_source, split, orig_label, canon,
        family, lang, oid, paired=None, meta=None):
    eligible = canon in MAIN_POS or canon in MAIN_NEG
    return {
        "sample_id": sid(dataset, oid, text), "text": text,
        "canonical_dataset": dataset, "source_group": group,
        "original_source": orig_source, "original_split": split,
        "original_label": str(orig_label), "canonical_label": canon,
        "binary_main_eligible": eligible,
        "binary_main_label": ("UNSAFE" if canon in MAIN_POS else
                              "SAFE" if canon in MAIN_NEG else None),
        "language": lang, "attack_family": family,
        "paired_group_id": paired, "original_sample_id": str(oid),
        "metadata": json.dumps(meta or {}, ensure_ascii=False),
    }


def from_piguard():
    rows = []
    for fn, split in (("train.json", "train"), ("valid.json", "valid")):
        d = json.load(open(PG / fn))
        for i, r in enumerate(d):
            s = r.get("source", "UNKNOWN")
            lab = int(r["label"])
            if fn == "train.json":
                m = PIGUARD_MAP.get(s)
                if m is None:
                    c0, c1, fam, lang = B, UNK, "UNKNOWN", "en"
                else:
                    c0, c1, fam, lang = m
                canon = c1 if lab == 1 else c0
                fam = fam if lab == 1 else "none"
            else:
                # valid.json 은 평가용 소규모 세트 (PINT/NotInject/BIPIA/WildGuard)
                canon = (IPI if s.startswith("BIPIA") else
                         HNI if s.startswith("NotInject") else
                         PI if "injection" in s else
                         JA if "jailbreak" in s else
                         B if lab == 0 else UNK)
                fam, lang = ("indirect_injection" if s.startswith("BIPIA") else "UNKNOWN"), "en"
            rows.append(rec(r["prompt"], "piguard_train_mix", f"piguard:{s}", s,
                            split, lab, canon, fam, lang, f"{fn}:{i}"))
    return rows


def from_notinject():
    rows = []
    for sub in ("one", "two", "three"):
        f = glob.glob(f"{HUB}/datasets--leolee99--NotInject/snapshots/*/data/NotInject_{sub}-*.parquet")
        if not f:
            continue
        df = pd.read_parquet(f[0])
        col = "prompt" if "prompt" in df.columns else df.columns[0]
        for i, t in enumerate(df[col]):
            rows.append(rec(t, "notinject", f"notinject:{sub}", f"NotInject_{sub}",
                            "test", "benign", HNI, "none", "en", f"{sub}:{i}"))
    return rows


WJ_PER_CATEGORY = 10000    # 아래 주석 참조


def from_wildjailbreak():
    """공식 카드가 지정한 방식(delimiter='\t', keep_default_na=False)으로 읽는다.
    전체 261,559행 중 **범주당 10,000건을 seed 0 으로 층화 표본**한다.

    이유: 이번 단계의 목적은 "어느 출처가 같은 출처 안에 TP 와 FP 를 모두 주는가"를
    판정하는 것이고, 범주당 1만 건이면 비율 추정 오차가 ±0.5%p 미만이라 판정에 충분하다.
    전수(26만)를 넣으면 PG2 추론만 몇 시간이 걸리는데 판정은 달라지지 않는다.
    **이 표본추출은 결과 표에 명시한다.**
    """
    f = glob.glob(f"{HUB}/datasets--allenai--wildjailbreak/snapshots/*/train/train.tsv")[0]
    df = pd.read_csv(f, delimiter="\t", dtype=str, keep_default_na=False)
    df = df[df["data_type"].isin(WJ_MAP)]
    df = (df.groupby("data_type", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), WJ_PER_CATEGORY), random_state=0)))
    rows = []
    for i, r in df.iterrows():
        dt = r["data_type"]
        canon, fam, kind = WJ_MAP[dt]
        # adversarial 계열은 adversarial 열이 실제 프롬프트, vanilla 계열은 vanilla 열
        text = r["adversarial"] if kind == "adversarial" and r["adversarial"] else r["vanilla"]
        if not text or not str(text).strip():
            continue
        # source_group 은 **생성 방식(construction)** 기준으로 묶는다.
        # adversarial_benign 과 adversarial_harmful 은 같은 WildTeaming 절차로 만들어졌고,
        # 서로가 서로의 hard negative 다.  둘을 한 그룹으로 둬야 같은 출처 안에서
        # TP/FP 비교가 성립한다(범주별로 쪼개면 각 그룹에 한쪽 라벨만 남는다).
        rows.append(rec(text, "wildjailbreak", f"wildjailbreak:{kind}", dt,
                        "train", dt, canon, fam, "en", f"wj:{i}",
                        meta={"data_type": dt, "construction": kind}))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = from_piguard() + from_notinject() + from_wildjailbreak()
    df = pd.DataFrame(rows)
    df = df[df["text"].astype(str).str.strip().str.len() > 0]
    print(f"총 {len(df)}건")
    print(df.groupby("canonical_label").size().to_string())
    print("\n--- source_group 별 MAIN 이진 라벨 ---")
    e = df[df["binary_main_eligible"]]
    t = e.groupby(["source_group", "binary_main_label"]).size().unstack(fill_value=0)
    for c in ("SAFE", "UNSAFE"):
        if c not in t.columns:
            t[c] = 0
    t["both"] = (t["SAFE"] > 0) & (t["UNSAFE"] > 0)
    print(t.sort_values("both", ascending=False).to_string())
    df.to_parquet(OUT / "canonical_samples.parquet", index=False)
    print(f"\n저장 -> {OUT/'canonical_samples.parquet'}  ({len(df)}행)")


if __name__ == "__main__":
    main()
