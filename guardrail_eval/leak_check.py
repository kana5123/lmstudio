"""교사·학생·평가 사이 누수 검사.

정확 일치뿐 아니라 정규화 일치(소문자·공백 압축)와 앞 80자 일치까지 본다.
"""
import csv, glob, json, re
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass
from datasets import load_dataset

HERE = Path(__file__).resolve().parent
norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
head = lambda s: norm(s)[:80]

wgt = load_dataset("allenai/wildguardmix", "wildguardtrain")["train"]
A = [r["prompt"] for r in wgt if r["prompt"]]                      # 교사의 학습 데이터
wjb = [json.loads(l)["text"] for l in open(HERE/"data_wjb/train.jsonl", encoding="utf-8")]
B = wjb                                                             # 채점시킬 데이터
mj = glob.glob("/home/kana5123/.cache/huggingface/hub/datasets--DAMO-NLP-SG--MultiJail/"
               "snapshots/*/MultiJail.csv")[0]
rows = list(csv.DictReader(open(mj, encoding="utf-8")))
C = [r["en"].strip() for r in rows if r["en"].strip()]              # 평가셋(영어 원문)
wc = [json.loads(l)["text"] for l in open(HERE/"wildchat_ko.jsonl", encoding="utf-8")]

def cmp(X, Y, nx, ny):
    ex = len(set(X) & set(Y))
    nm = len(set(map(norm, X)) & set(map(norm, Y)))
    hd = len(set(map(head, X)) & set(map(head, Y)))
    print(f"  {nx:34} ∩ {ny:26}  정확 {ex:>5}  정규화 {nm:>5}  앞80자 {hd:>5}")

print(f"교사 학습 WildGuardTrain {len(A)} · 채점대상 WildJailbreak {len(B)} · "
      f"평가 MultiJail(영어) {len(C)} · 정상 WildChat-ko {len(wc)}\n")
cmp(A, B, "① 교사 학습(WildGuardTrain)", "채점대상(WildJailbreak)")
cmp(A, C, "② 교사 학습(WildGuardTrain)", "평가 MultiJail(영어)")
cmp(B, C, "③ 학생 학습(WildJailbreak)",  "평가 MultiJail(영어)")
