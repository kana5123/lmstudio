"""분할 누수 검사.

1) 네 분할(ver_train / ver_dev / eval_val / eval_test) 사이 sample_id 교집합 0
2) 우리 평가셋이 **기존 벤치마크(results/rfpr_jailbreak_promptguard_v2_scores.jsonl)**
   와 완전히 동일한 표본인가 — 기존 결과와 직접 비교하려면 이게 성립해야 한다
3) 근사 중복(정규화·앞 80자) 정도를 측정해 보고한다.  정확 일치가 아니어도
   학습셋과 평가셋에 사실상 같은 문장이 있으면 성능이 부풀 수 있다.
"""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.splits import build, sid, assert_no_overlap

ROOT = Path(__file__).resolve().parents[1]
norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
head = lambda s: norm(s)[:80]


def test_exact_no_overlap():
    assert_no_overlap(build())


def test_eval_splits_match_stored_benchmark():
    s = build()
    stored = {"val": set(), "test": set()}
    p = ROOT / "results" / "rfpr_jailbreak_promptguard_v2_scores.jsonl"
    for line in open(p, encoding="utf-8"):
        d = json.loads(line)
        stored[d["split"]].add(sid(d["text"]))
    for mine_name, their in (("eval_val", "val"), ("eval_test", "test")):
        mine = {d["sample_id"] for d in s[mine_name]}
        assert mine == stored[their], (
            f"{mine_name} 이 기존 벤치마크와 다르다: 우리만 {len(mine-stored[their])}, "
            f"기존만 {len(stored[their]-mine)}")
        print(f"  {mine_name}: 기존 벤치마크와 {len(mine)}건 완전 일치")


def test_report_near_duplicates():
    """근사 중복은 assert 로 막지 않고 **측정해 보고**한다(원본 데이터의 성질이므로)."""
    s = build()
    tr = [d["text"] for d in s["ver_train"]] + [d["text"] for d in s["ver_dev"]]
    for ev in ("eval_val", "eval_test"):
        te = [d["text"] for d in s[ev]]
        n_norm = len(set(map(norm, tr)) & set(map(norm, te)))
        n_head = len(set(map(head, tr)) & set(map(head, te)))
        print(f"  검증기학습 ∩ {ev}:  정확 0  정규화 {n_norm}  앞80자 {n_head}"
              f"  (평가셋 {len(te)}건 대비 앞80자 {n_head/len(te)*100:.2f}%)")


if __name__ == "__main__":
    test_exact_no_overlap(); test_eval_splits_match_stored_benchmark()
    test_report_near_duplicates(); print("PASS")
