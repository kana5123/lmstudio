"""번역본에 최종 필터를 적용해 mDeBERTa 학습용 데이터를 만든다.

필터 6종
  ① 번역 실패  : 한글 없음            (safeguard_ko_clean.json 단계에서 이미 제외)
  ② 내용 잘림  : 한/영 길이비 < 0.3   (동상)
  ③ 원문 복사  : 번역문이 원문과 거의 동일 -- 번역이 일어나지 않음.
                 ★역번역 유사도로는 못 잡음(원문 그대로면 유사도가 오히려 1.0에 가까움)
  ④ 영어 혼입  : 번역문의 영어 문자 비중 > 50%
  ⑤ 협박 뒤집힘: 원문 "or I will/else" 인데 한국어가 순접 조건으로 바뀐 것
                 예) "정보를 주지 않으면 폭로하겠다" -> "정보를 주시면 폭로해 드리겠습니다"
  ⑥ 의미 손실  : 역번역 유사도 < 임계

분할: 원본 train 을 학습/검증으로 층화 분할, 원본 test 는 그대로 시험셋.
"""
import json, random, re, sys
from collections import Counter
from pathlib import Path

try:
    from setproctitle import setproctitle
    setproctitle("kana5123")
except ImportError:
    pass

HERE = Path(__file__).resolve().parent
OUT = HERE / "data_ko"
SEED = 0
VAL_FRAC = 0.1
SIM_MIN = float(sys.argv[1]) if len(sys.argv) > 1 else 0.70

HAN = re.compile(r"[가-힣]")
LAT = re.compile(r"[A-Za-z]")
THREAT = re.compile(r"\bor (?:I(?: will|'ll)|else)\b", re.I)
NEG = re.compile(r"않으면|아니면|그렇지\s*않|안\s*하면|하지\s*않으")
OR_KO = re.compile(r"하거나|주시거나|거나\s")
FLIP = re.compile(r"주시면|하시면|알려주면|제공하면|공유하면|하면")


def threat_flipped(r):
    """협박문이 순접 조건으로 뒤집혔는지. 부정조건이나 '~거나'가 있으면 의미 유지."""
    if not THREAT.search(r["text_en"]):
        return False
    ko = r["text_ko"]
    if NEG.search(ko) or OR_KO.search(ko):
        return False
    return True                      # 순접으로 바뀌었거나 문법이 깨진 것


def copied(en, ko):
    """번역문이 원문과 거의 같은가(번역 미수행). 앞 120자 문자 일치율로 판정."""
    a = re.sub(r"\s+", "", en.lower())
    b = re.sub(r"\s+", "", ko.lower())
    if not a or not b:
        return False
    k = min(120, len(a), len(b))
    return sum(1 for x, y in zip(a[:k], b[:k]) if x == y) / k > 0.8


def eng_ratio(ko):
    h = len(HAN.findall(ko))
    l = len(LAT.findall(ko))
    return l / max(1, h + l)


def main():
    rows = json.loads((HERE / "safeguard_ko_verified.json").read_text())
    reasons = Counter()
    keep = []
    for r in rows:
        if copied(r["text_en"], r["text_ko"]):
            reasons["원문 그대로 복사"] += 1
        elif eng_ratio(r["text_ko"]) > 0.5:
            reasons["영어 혼입"] += 1
        elif threat_flipped(r):
            reasons["협박 뒤집힘"] += 1
        elif r["bt_sim"] < SIM_MIN:
            reasons["의미 손실"] += 1
        else:
            keep.append(r)
    print(f"입력 {len(rows):,}건 → 유지 {len(keep):,}건")
    for k, v in reasons.most_common():
        print(f"   제외 {k}: {v}건")

    # 원본 데이터에 같은 문장이 여러 번 있고 번역 후에도 동일해져 분할 간 누수가 생긴다.
    # 시험셋 우선으로 중복을 제거한다(시험셋은 온전히 남기고, 학습/검증에서 뺀다).
    rng = random.Random(SEED)
    te = []
    seen = set()
    for r in keep:
        if r["split"] == "test" and r["text_ko"] not in seen:
            seen.add(r["text_ko"]); te.append(r)
    tr = []
    for r in keep:
        if r["split"] == "train" and r["text_ko"] not in seen:
            seen.add(r["text_ko"]); tr.append(r)
    dropped = len(keep) - len(te) - len(tr)
    print(f"   분할 간/내 중복 제거: {dropped}건")
    by_label = {0: [], 1: []}
    for r in tr:
        by_label[r["label"]].append(r)
    train, val = [], []
    for lab, g in by_label.items():
        rng.shuffle(g)
        k = int(len(g) * VAL_FRAC)
        val += g[:k]; train += g[k:]
    rng.shuffle(train); rng.shuffle(val)

    OUT.mkdir(exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", te)):
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps({"text": r["text_ko"], "label": r["label"]},
                                   ensure_ascii=False) + "\n")
        c = Counter(r["label"] for r in part)
        print(f"   {name:5} {len(part):5}건  정상 {c[0]:5} 공격 {c[1]:5}")

    # 누수 검사
    s = {n: {r["text_ko"] for r in p} for n, p in
         (("train", train), ("val", val), ("test", te))}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        n = len(s[a] & s[b])
        print(f"   {a}∩{b} 겹침 {n}건" + ("  ← 문제" if n else ""))


if __name__ == "__main__":
    main()
