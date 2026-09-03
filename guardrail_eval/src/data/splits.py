"""JailbreaksOverTime 분할 재사용 + verifier 전용 분할 생성.

기존 rfpr.py 의 jailbreak_split() 이 만든 평가 표본(val 2000 / test 4000)을
**그대로** 재현해 보존하고, 거기에 쓰이지 않은 나머지 행만으로 verifier 학습
데이터를 만든다.  겹침이 0 임을 assert 로 확인한다.

용어:
  평가 검증셋(eval_val)  = 기존 벤치마크의 val 2000건. 임계값 선택용.
  평가 시험셋(eval_test) = 기존 벤치마크의 test 4000건. 최종 보고용, 절대 학습 금지.
  검증기 학습셋(ver_train) / 검증기 개발셋(ver_dev) = 위 6000건에 쓰이지 않은 행에서.
"""
import hashlib, json, random
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
JOT = ("/home/kana5123/.cache/huggingface/hub/datasets--djapp18--JailbreaksOverTime/"
       "snapshots/a5a467cbab4b17d7f1c83e6cd119c61722053868/"
       "jailbreaksovertime_hugging_face.json")
SEED = 0
VER_DEV_FRAC = 0.20


def sid(text: str) -> str:
    """샘플 식별자 = 원문 SHA1 앞 16자. 중복 제거가 원문 기준이라 안전하다."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _dedup_rows():
    """원본 그대로: 원문 기준 중복 제거 -> 22,180행 (rfpr.py:31-37 과 동일)."""
    rows = json.load(open(JOT))
    seen, dd = set(), []
    for r in rows:
        if r["prompt"] not in seen:
            seen.add(r["prompt"]); dd.append(r)
    assert len(dd) == 22180, len(dd)
    return dd


def eval_splits(n_val=2000, n_test=4000):
    """rfpr.py:jailbreak_split() 을 그대로 복사. 난수 소비 순서까지 동일해야 한다."""
    dd = _dedup_rows()
    rng = random.Random(SEED)
    pos = [r for r in dd if r["label"] == 1]
    neg = [r for r in dd if r["label"] == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    n_pos_test = round(len(pos) * 15526 / 22180)
    n_neg_test = 15526 - n_pos_test
    pool = {"test": (pos[:n_pos_test], neg[:n_neg_test]),
            "val": (pos[n_pos_test:], neg[n_neg_test:])}
    rate = len(pos) / len(dd)
    out = {}
    for split, n in (("val", n_val), ("test", n_test)):
        p, q = pool[split]
        k = round(n * rate)
        out[split] = ([(x["prompt"], 1) for x in p[:k]]
                      + [(x["prompt"], 0) for x in q[: n - k]])
        rng.shuffle(out[split])
    return out, pool


def build():
    """평가셋 2종 + 검증기용 학습/개발셋을 반환. 모든 원소는 dict(sample_id/text/gt)."""
    ev, _ = eval_splits()
    used = {t for split in ev.values() for t, _ in split}

    dd = _dedup_rows()
    left = [r for r in dd if r["prompt"] not in used]
    assert len(left) == len(dd) - len(used), "평가셋 제거 개수 불일치"

    # 검증기 학습/개발 분할: 라벨 층화 후 seed 고정 셔플. 평가셋과는 이미 배타적.
    rng = random.Random(SEED + 1)
    pos = [r for r in left if r["label"] == 1]
    neg = [r for r in left if r["label"] == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    kp, kn = int(len(pos) * VER_DEV_FRAC), int(len(neg) * VER_DEV_FRAC)
    parts = {
        "ver_dev":   pos[:kp] + neg[:kn],
        "ver_train": pos[kp:] + neg[kn:],
    }
    out = {}
    for name, rs in parts.items():
        rng.shuffle(rs)
        out[name] = [{"sample_id": sid(r["prompt"]), "text": r["prompt"],
                      "gt": int(r["label"]), "split": name} for r in rs]
    for name, pairs in (("eval_val", ev["val"]), ("eval_test", ev["test"])):
        out[name] = [{"sample_id": sid(t), "text": t, "gt": int(y), "split": name}
                     for t, y in pairs]
    return out


def assert_no_overlap(splits: dict):
    """모든 쌍에 대해 sample_id 교집합 0 을 강제."""
    names = list(splits)
    ids = {n: {d["sample_id"] for d in splits[n]} for n in names}
    for n in names:
        assert len(ids[n]) == len(splits[n]), f"{n} 내부 중복 {len(splits[n])-len(ids[n])}건"
    bad = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            k = len(ids[a] & ids[b])
            print(f"  누수 {a:10} ∩ {b:10} = {k}건")
            if k:
                bad.append((a, b, k))
    assert not bad, f"분할 누수: {bad}"


if __name__ == "__main__":
    s = build()
    total = sum(len(v) for v in s.values())
    for n, v in s.items():
        p = sum(d["gt"] for d in v)
        print(f"{n:10} n={len(v):6}  공격(GT UNSAFE)={p:5}  정상(GT SAFE)={len(v)-p:5}")
    print(f"합계 {total} (원본 중복제거 22180)")
    assert total == 22180, total
    assert_no_overlap(s)
    print("분할 누수 0 확인")
