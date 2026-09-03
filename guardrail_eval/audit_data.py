"""최종 학습 데이터 결함 감사 — 유형별로 정량화한다.

표본 눈검사로는 규모를 모른다. 각 결함을 규칙으로 정의해 전량에서 세고,
유형마다 실제 예시를 뽑아 규칙이 맞는지 확인할 수 있게 한다.
"""
import json, re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
HAN = re.compile(r"[가-힣]")
LAT = re.compile(r"[A-Za-z]")

rows = json.loads((HERE / "safeguard_ko_verified.json").read_text())
# 최종 학습셋에 실제 들어간 것만
used = set()
for n in ("train", "val", "test"):
    for l in open(HERE / "data_ko" / f"{n}.jsonl", encoding="utf-8"):
        used.add(json.loads(l)["text"])
data = [r for r in rows if r["text_ko"] in used]

CHECKS = {}


def check(name):
    def deco(f):
        CHECKS[name] = f
        return f
    return deco


@check("문장 미완결(종결어미 없음)")
def c1(r):
    ko = r["text_ko"].rstrip()
    if len(ko) < 15:
        return False
    # 한국어 종결/문장부호로 안 끝나면 잘렸을 가능성
    return not re.search(r"[.!?…\"'\)\]}:;]$|다$|요$|까$|죠$|음$|함$|임$", ko)


@check("따옴표 짝 불일치")
def c2(r):
    ko = r["text_ko"]
    return (ko.count('"') % 2 == 1) or (ko.count("'") % 2 == 1)


@check("괄호 짝 불일치")
def c3(r):
    ko = r["text_ko"]
    return ko.count("(") != ko.count(")") or ko.count("[") != ko.count("]")


@check("영어 조각 혼입(10~50%)")
def c4(r):
    ko = r["text_ko"]
    h, l = len(HAN.findall(ko)), len(LAT.findall(ko))
    return h > 0 and 0.10 < l / max(1, h + l) <= 0.50


@check("한글 자모 깨짐(ㄱㅏ 등)")
def c5(r):
    return bool(re.search(r"[ㄱ-ㅎㅏ-ㅣ]", r["text_ko"]))


@check("같은 문장 2회 이상 반복")
def c6(r):
    parts = [p.strip() for p in re.split(r"[.!?\n]+", r["text_ko"]) if len(p.strip()) > 12]
    return bool(parts) and Counter(parts).most_common(1)[0][1] >= 2


@check("번역문이 원문보다 김(1.5배 초과)")
def c7(r):
    return len(r["text_ko"]) > len(r["text_en"]) * 1.5


@check("숫자 개수 불일치")
def c8(r):
    a = re.findall(r"\d+", r["text_en"])
    b = re.findall(r"\d+", r["text_ko"])
    return len(a) != len(b) and len(a) > 0


counts = Counter()
examples = {}
for r in data:
    for name, f in CHECKS.items():
        try:
            if f(r):
                counts[name] += 1
                examples.setdefault(name, []).append(r)
        except Exception:
            pass

print(f"감사 대상 {len(data):,}건\n")
print(f"{'결함 유형':32}{'건수':>7}{'비율':>8}")
print("-" * 48)
for name in CHECKS:
    n = counts[name]
    print(f"{name:32}{n:>7}{n/len(data):>8.1%}")

(HERE / "audit_examples.json").write_text(json.dumps(
    {k: [{"en": r["text_en"][:300], "ko": r["text_ko"][:300], "sim": r["bt_sim"]}
         for r in v[:5]] for k, v in examples.items()}, ensure_ascii=False, indent=1))
print("\n예시 → audit_examples.json")
