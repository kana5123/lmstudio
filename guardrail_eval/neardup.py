"""학습셋(2023-02~11)과 시험셋(2023-12) 사이 근접 중복 측정.

정확 일치는 0으로 제거했지만, 제일브레이크는 템플릿 변종이 돌아다닌다.
하드라벨 모델이 그걸 외웠다면 0.9545 는 암기 점수다.
"""
import json, re
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass
HERE=Path(__file__).resolve().parent
J=lambda p:[json.loads(l) for l in open(HERE/p,encoding="utf-8")]
norm=lambda s: re.sub(r"\s+"," ",s.strip().lower())
tr=[norm(r["text"]) for r in J("data_jot/train.jsonl")+J("data_jot/val.jsonl")]
te=J("data_par/en_test.jsonl")
print(f"학습 {len(tr)} · 시험 {len(te)} (공격 {sum(r['label'] for r in te)})")

# ① 앞 80자 일치
h80={s[:80] for s in tr}
m80=[r for r in te if norm(r["text"])[:80] in h80]
print(f"\n① 앞 80자 일치 : {len(m80)}건 ({len(m80)/len(te)*100:.1f}%)"
      f"  — 그중 공격 {sum(r['label'] for r in m80)}")

# ② 문자 5-gram 자카드 — 가장 비슷한 학습 문장과의 유사도
def g5(s, n=5, cap=1500):
    s=s[:cap]; return {s[i:i+n] for i in range(max(len(s)-n+1,0))}
tr_g=[g5(s) for s in tr]
# 후보 축소: 앞 30자가 같은 것만 정밀 비교 (전수 비교는 O(n²))
from collections import defaultdict
buck=defaultdict(list)
for i,s in enumerate(tr): buck[s[:30]].append(i)
sims=[]
for r in te:
    s=norm(r["text"]); G=g5(s)
    cand=buck.get(s[:30],[])
    best=0.0
    for i in cand:
        u=len(G|tr_g[i]);  best=max(best, len(G&tr_g[i])/u if u else 0)
    sims.append((best, r["label"], r["uid"]))
for th in (0.9,0.8,0.7,0.5):
    hit=[x for x in sims if x[0]>=th]
    print(f"② 자카드 ≥{th}: {len(hit):>4}건 ({len(hit)/len(te)*100:>5.1f}%)"
          f"  공격 {sum(x[1] for x in hit):>4}/{sum(r['label'] for r in te)}")
json.dump({u:round(s,4) for s,_,u in sims}, open(HERE/"neardup_sim.json","w"))
print(f"\n저장 → neardup_sim.json")
