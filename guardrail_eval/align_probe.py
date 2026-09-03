"""영어로 학습한 가드 안에서 영어/한국어 표현이 얼마나 맞춰져 있는지 측정.

Libovicky et al.(2020) 의 언어 중심점(centroid) 분해와
Cao et al.(ICLR 2020) 의 병렬 검색(retrieval) 지표를 우리 모델에 적용한다.
"""
import json
from pathlib import Path

try:
    from setproctitle import setproctitle
    setproctitle("kana5123")
except ImportError:
    pass

import torch, torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
import os, sys
CKPT = sys.argv[1] if len(sys.argv)>1 else str(HERE/"models"/"mdeberta_en_guard"/"best")
print("모델:", CKPT)
tok = AutoTokenizer.from_pretrained(CKPT)
from transformers import AutoModel
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        CKPT, output_hidden_states=True).cuda().eval().half()
    HAS_HEAD = True
except Exception:
    model = AutoModel.from_pretrained(CKPT).cuda().eval().half(); HAS_HEAD = False


def embed(texts, bs=64):
    """마지막 층을 attention mask 로 평균 풀링 = 문장 표현."""
    out = []
    for i in range(0, len(texts), bs):
        e = tok(texts[i:i+bs], truncation=True, max_length=512,
                padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = model(**e)
            h = (o.hidden_states[-1] if HAS_HEAD else o.last_hidden_state).float()
        m = e["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).cpu())
    return torch.cat(out)


def risk(texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        e = tok(texts[i:i+bs], truncation=True, max_length=512,
                padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out.append(torch.softmax(model(**e).logits.float(), -1)[:, 1].cpu())
    return torch.cat(out)


def load(p, key="text"):
    return [json.loads(l)[key] for l in open(HERE / p, encoding="utf-8")]


en = load("data_en/test.jsonl")
ko = load("data_ko/test.jsonl")
assert len(en) == len(ko) == 1797

import csv
from rfpr import KOPI
kopi = [r["text_ko"] for r in
        csv.DictReader(open(KOPI + "test_id.csv", encoding="utf-8"))][:1797]

E, K, P = embed(en), embed(ko), embed(kopi)
print(f"영어 {E.shape} · 한국어(번역) {K.shape} · KoPI {P.shape}\n")

def cos(a, b): return F.cosine_similarity(a, b, dim=-1)

# ── 1. 같은 뜻의 영/한 쌍이 얼마나 가까운가 ──
same = cos(E, K)
shuf = cos(E, K[torch.randperm(len(K), generator=torch.Generator().manual_seed(0))])
print("[1] 같은 문장의 영어 표현 ↔ 한국어 표현")
print(f"    같은 쌍   cos {same.mean():.4f}")
print(f"    무작위 쌍 cos {shuf.mean():.4f}   (차이 {same.mean()-shuf.mean():+.4f})")

# ── 2. 병렬 검색: 한국어 문장으로 영어 원문을 찾을 수 있나 ──
En, Kn = F.normalize(E, dim=-1), F.normalize(K, dim=-1)
top1 = (Kn @ En.T).argmax(1)
acc = (top1 == torch.arange(len(K))).float().mean()
print(f"\n[2] 병렬 검색 정확도 (한국어 → 영어 원문 1797개 중 정답)")
print(f"    {acc*100:.2f}%   (무작위면 {100/len(K):.2f}%)")

# ── 3. 언어 중심점 제거 후 (Libovicky et al. 2020) ──
Ec, Kc = E - E.mean(0), K - K.mean(0)
same_c = cos(Ec, Kc)
Ecn, Kcn = F.normalize(Ec, dim=-1), F.normalize(Kc, dim=-1)
acc_c = ((Kcn @ Ecn.T).argmax(1) == torch.arange(len(K))).float().mean()
gap = (E.mean(0) - K.mean(0)).norm() / E.norm(dim=-1).mean()
print(f"\n[3] 언어 중심점(centroid) 분해")
print(f"    영어중심 ↔ 한국어중심 거리 / 평균 벡터 크기 = {gap:.4f}")
print(f"    중심 제거 후  같은 쌍 cos {same_c.mean():.4f}  (제거 전 {same.mean():.4f})")
print(f"    중심 제거 후  검색 정확도 {acc_c*100:.2f}%  (제거 전 {acc*100:.2f}%)")

# ── 4. KoPI 는 어디에 놓이는가 ──
print(f"\n[4] 각 집합이 '영어 학습 분포'에서 얼마나 떨어져 있나")
ec = E.mean(0)
for nm, X in (("영어 원문(학습 분포)", E), ("한국어 번역(우리 시험셋)", K), ("KoPI 한국어(낯선 출처)", P)):
    print(f"    {nm:26} 영어중심과 cos {cos(X, ec.expand_as(X)).mean():.4f}")

# ── 5. 위험 점수가 언어 간에 얼마나 흔들리나 ──
re_, rk = (risk(en), risk(ko)) if HAS_HEAD else (torch.zeros(1), torch.zeros(1))
print(f"\n[5] 같은 문장인데 언어만 바꿨을 때 위험 점수 변화")
print(f"    평균 절대차 {(re_-rk).abs().mean():.4f}   결정 뒤집힘 "
      f"{((re_>=.5)!=(rk>=.5)).float().mean()*100:.2f}%")
