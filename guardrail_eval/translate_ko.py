"""safe-guard-prompt-injection 을 NLLB-200 으로 한국어 번역.

번역기로 순수 MT(circulus/canvers-en2ko-v1, BART 기반)를 쓴다. 지시를 따르는 LLM 은 인젝션 데이터를 번역시키면
데이터 안의 공격 지시를 실행해 버린다(Qwen2.5 실측: 번역 대신 "I have been PWNED" 출력).
seq2seq 번역 전용 모델은 이 하이재킹이 원리적으로 불가능하다.
NLLB/MADLAD 도 후보였으나 한국어 품질이 떨어져(반복 생성, 띄어쓰기 붕괴,
MADLAD 는 번역 실패 사례) 이 모델을 골랐다.

최대 입력이 1026 토큰이므로 그보다 긴 레코드(전체의 1%)는 문장 단위로 쪼개
번역한 뒤 다시 이어붙인다.
"""
import json, re, sys, time
from pathlib import Path

try:
    from setproctitle import setproctitle
    setproctitle("kana5123")
except ImportError:
    pass

import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL = "circulus/canvers-en2ko-v1"   # BART 기반 EN->KO 전용 seq2seq
MAX_IN = 900          # 모델 최대 위치 1026 보다 여유
BATCH_TOK = 1500      # 배치당 총 토큰 상한
MAX_BATCH = 24        # 배치당 문장 개수 상한 (beam=4 라 실제로는 x4 가 올라감)
NUM_BEAMS = 4
OUT = Path(__file__).resolve().parent / "safeguard_ko.jsonl"
SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def chunks(text, tok):
    """1024 토큰을 넘는 글은 문장 단위로 쪼갠다. 문장 하나가 넘으면 강제 절단."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= MAX_IN:
        return [text]
    out, cur = [], ""
    for s in SENT.split(text):
        if not s:
            continue
        cand = (cur + " " + s).strip() if cur else s
        if len(tok(cand, add_special_tokens=False)["input_ids"]) <= MAX_IN:
            cur = cand
        else:
            if cur:
                out.append(cur)
            if len(tok(s, add_special_tokens=False)["input_ids"]) > MAX_IN:
                sid = tok(s, add_special_tokens=False)["input_ids"]
                for i in range(0, len(sid), MAX_IN):
                    out.append(tok.decode(sid[i:i + MAX_IN]))
                cur = ""
            else:
                cur = s
    if cur:
        out.append(cur)
    return out


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL, torch_dtype=torch.float16).to("cuda").eval()

    ds = load_dataset("xTRam1/safe-guard-prompt-injection")
    recs = []
    for split in ds:
        for r in ds[split]:
            recs.append({"split": split, "text_en": r["text"], "label": r["label"]})
    if limit:
        recs = recs[:limit]

    # 조각 단위로 펼쳐 길이순 정렬(패딩 낭비 최소화)
    pieces = []
    for i, r in enumerate(recs):
        for j, c in enumerate(chunks(r["text_en"], tok)):
            pieces.append((i, j, c, len(tok(c, add_special_tokens=False)["input_ids"])))
    pieces.sort(key=lambda x: x[3])
    print(f"레코드 {len(recs)} → 번역 조각 {len(pieces)}", flush=True)

    done, t0 = {}, time.time()
    i = 0
    while i < len(pieces):
        n = max(1, min(MAX_BATCH, BATCH_TOK // max(1, pieces[i][3])))
        batch = pieces[i:i + n]
        enc = tok([b[2] for b in batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_IN).to("cuda")
        enc.pop("token_type_ids", None)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=min(1024, int(max(b[3] for b in batch) * 1.6) + 48),
                num_beams=NUM_BEAMS,
                no_repeat_ngram_size=4,      # NLLB 에서 관측된 반복 생성 억제
                repetition_penalty=1.1)
        for b, o in zip(batch, out):
            done[(b[0], b[1])] = tok.decode(o, skip_special_tokens=True)
        i += n
        if (i // 200) != ((i - n) // 200):
            print(f"  {i}/{len(pieces)}  ({time.time()-t0:.0f}s)", flush=True)

    with open(OUT, "w", encoding="utf-8") as fh:
        for i, r in enumerate(recs):
            parts = [done[k] for k in sorted(done) if k[0] == i]
            fh.write(json.dumps({**r, "text_ko": " ".join(parts),
                                 "n_chunks": len(parts)}, ensure_ascii=False) + "\n")
    print(f"\n저장 {len(recs)}건 → {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
