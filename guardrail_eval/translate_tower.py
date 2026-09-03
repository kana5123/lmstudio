"""TowerInstruct-7B-v0.2 로 영어 -> 한국어 번역.

PolyGuard(COLM 2025) 가 PolyGuardMix 번역에 쓴 모델·절차를 그대로 따른다.
긴 텍스트는 문장 단위로 쪼개 조각별로 번역한 뒤 이어붙인다(논문의 blingfire 청킹에 대응).
"""
import json, os, re, sys, time
from pathlib import Path
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
MODEL = "Unbabel/TowerInstruct-7B-v0.2"
CHUNK = int(os.getenv("CHUNK", "900"))       # 조각 최대 문자수
BATCH = int(os.getenv("BATCH", "8"))
INST = "Translate the following text from English into Korean.\nEnglish: {src}\nKorean:"

# 문장 경계: 마침표/물음표/느낌표 뒤 공백, 또는 줄바꿈
SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def chunks(text, limit=CHUNK):
    """문장 경계로 자르되 limit 문자를 넘지 않게 모은다. 한 문장이 limit 보다 길면 그대로 둔다."""
    parts, cur, n = [], [], 0
    for s in SENT.split(text):
        s = s.strip()
        if not s: continue
        if cur and n + len(s) + 1 > limit:
            parts.append(" ".join(cur)); cur, n = [], 0
        cur.append(s); n += len(s) + 1
    if cur: parts.append(" ".join(cur))
    return parts or [text[:limit]]


def main():
    src_file = sys.argv[1]
    out_file = sys.argv[2]
    only_label = int(sys.argv[3]) if len(sys.argv) > 3 else None

    rows = json.load(open(HERE / src_file, encoding="utf-8"))
    if only_label is not None:
        rows = [r for r in rows if r["label"] == only_label]
    sh = os.getenv("SHARD")          # "0/4" 형식
    if sh:
        i, n = (int(x) for x in sh.split("/"))
        rows = rows[i::n]
        print(f"샤드 {i}/{n}", flush=True)
    print(f"번역 대상 {len(rows)}건", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    # 모든 조각을 한 줄로 펼쳐 배치 처리 -> 원래 문서로 되접는다
    flat, owner = [], []
    for i, r in enumerate(rows):
        for c in chunks(r["prompt"]):
            flat.append(c); owner.append(i)
    print(f"조각 {len(flat)}개 (문서당 평균 {len(flat)/len(rows):.1f})", flush=True)

    outs = [None] * len(flat)
    t0 = time.time()
    for b in range(0, len(flat), BATCH):
        batch = flat[b:b + BATCH]
        prompts = [tok.apply_chat_template([{"role": "user", "content": INST.format(src=c)}],
                                           tokenize=False, add_generation_prompt=True)
                   for c in batch]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=1536).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=1024, do_sample=False,
                                 repetition_penalty=1.08, no_repeat_ngram_size=12,
                                 pad_token_id=tok.pad_token_id)
        for j, g in enumerate(gen):
            txt = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            outs[b + j] = txt
        done = min(b + BATCH, len(flat))
        if (b // BATCH) % 10 == 0 or done == len(flat):
            el = time.time() - t0
            print(f"  {done}/{len(flat)}  {el:.0f}초  (남은 예상 {el/done*(len(flat)-done):.0f}초)",
                  flush=True)

    merged = [[] for _ in rows]
    for k, o in enumerate(outs):
        merged[owner[k]].append(o)
    res = []
    for r, parts in zip(rows, merged):
        res.append({**r, "text_ko": " ".join(parts), "n_chunk": len(parts)})
    (HERE / out_file).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장 {len(res)}건 → {out_file}  ({time.time()-t0:.0f}초)")


if __name__ == "__main__":
    main()
