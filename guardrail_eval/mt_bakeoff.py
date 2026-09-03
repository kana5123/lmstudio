"""번역 모델 비교 — 같은 표본에 여러 모델을 돌려 객관 지표로 승자를 가린다.

지표
  fail_rate   한글이 하나도 없는 비율 (번역 실패)
  rep_rate    같은 4-gram 이 3회 이상 반복 (반복 생성)
  short_rate  한국어/영어 길이비 < 0.3 (내용 잘림)
  bt_sim      역번역 유사도 -- 한국어를 영어로 되돌려 원문과 bge-m3 코사인.
              역번역기는 NLLB-3.3B 로 고정해 모든 후보에 같은 심판을 쓴다.
"""
import json, re, sys, time
from collections import Counter
from pathlib import Path

try:
    from setproctitle import setproctitle
    setproctitle("kana5123")
except ImportError:
    pass

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "mt_sample.json"
HANGUL = re.compile(r"[가-힣]")

# (이름, 모델id, 종류)  종류: nllb | seq2seq | llm
CANDIDATES = [
    ("circulus",   "circulus/canvers-en2ko-v1",        "seq2seq"),
    ("nhndq",      "NHNDQ/nllb-finetuned-en2ko",       "nllb"),
    ("nllb3.3b",   "facebook/nllb-200-3.3B",           "nllb"),
    ("madlad3b",   "google/madlad400-3b-mt",           "seq2seq_pfx"),
    ("llama3.1-8b", "meta-llama/Llama-3.1-8B-Instruct", "llm"),
    ("qwen2.5-7b",  "Qwen/Qwen2.5-7B-Instruct",         "llm"),
    ("qwen3-8b",    "Qwen/Qwen3-8B",                    "llm"),
    ("gemma2-9b",   "google/gemma-2-9b-it",             "llm"),
    ("gemma3-12b",  "google/gemma-3-12b-it",            "llm"),
    ("exaone3.5-7.8b", "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct", "llm"),
    ("kanana1.5-8b", "kakaocorp/kanana-1.5-8b-instruct-2505", "llm"),
    ("mistral-7b",  "mistralai/Mistral-7B-Instruct-v0.3", "llm"),
]


def repetitive(t, n=4, k=3):
    w = t.split()
    if len(w) < n * 2:
        return False
    c = Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    return max(c.values(), default=0) >= k


def load(mid, kind):
    if kind == "llm":
        tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        m = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16,
                                                 device_map="auto",
                                                 trust_remote_code=True).eval()
    else:
        tok = AutoTokenizer.from_pretrained(mid, src_lang="eng_Latn") if kind == "nllb" \
            else AutoTokenizer.from_pretrained(mid)
        m = AutoModelForSeq2SeqLM.from_pretrained(mid, torch_dtype=torch.float16).to("cuda").eval()
    return tok, m


def translate(tok, m, kind, texts, bs=8):
    out = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        if kind == "llm":
            prompts = [tok.apply_chat_template(
                [{"role": "user", "content": f"Translate the following text into Korean. "
                                             f"Output only the translation.\n\n{t}"}],
                tokenize=False, add_generation_prompt=True) for t in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=1024, add_special_tokens=False).to(m.device)
            with torch.no_grad():
                o = m.generate(**enc, max_new_tokens=512, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            n = enc.input_ids.shape[1]
            out += [tok.decode(x[n:], skip_special_tokens=True).strip() for x in o]
        else:
            src = [f"<2ko> {t}" for t in chunk] if kind == "seq2seq_pfx" else chunk
            enc = tok(src, return_tensors="pt", padding=True, truncation=True,
                      max_length=900).to("cuda")
            enc.pop("token_type_ids", None)
            kw = {"forced_bos_token_id": tok.convert_tokens_to_ids("kor_Hang")} \
                if kind == "nllb" else {}
            with torch.no_grad():
                o = m.generate(**enc, max_new_tokens=512, num_beams=4,
                               no_repeat_ngram_size=4, repetition_penalty=1.1, **kw)
            out += [tok.decode(x, skip_special_tokens=True) for x in o]
    return out


def make_sample(n=500):
    """라벨 × 길이구간 층화 표본."""
    import random
    import warnings; warnings.filterwarnings("ignore")
    from datasets import load_dataset
    ds = load_dataset("xTRam1/safe-guard-prompt-injection")
    recs = [{"split": s, "text_en": r["text"], "label": r["label"]}
            for s in ds for r in ds[s]]
    def bucket(t):
        L = len(t)
        return 0 if L < 100 else 1 if L < 400 else 2
    groups = {}
    for r in recs:
        groups.setdefault((r["label"], bucket(r["text_en"])), []).append(r)
    rng = random.Random(0)
    out, per = [], max(1, n // len(groups))
    for k in sorted(groups):
        g = groups[k]; rng.shuffle(g)
        out += g[:per]
    rng.shuffle(out)
    SAMPLE.write_text(json.dumps(out, ensure_ascii=False))
    print(f"표본 {len(out)}건 저장 (구간 {len(groups)}개 × {per})")
    return out


def main():
    """인자로 받은 모델 하나만 번역한다(모델별 독립 프로세스용)."""
    want = sys.argv[1]
    sample = json.loads(SAMPLE.read_text())
    texts = [r["text_en"] for r in sample]
    hit = [c for c in CANDIDATES if c[0] == want]
    if not hit:
        print(f"알 수 없는 모델: {want}"); sys.exit(2)
    name, mid, kind = hit[0]
    out_p = HERE / f"mt_out_{name}.json"
    if out_p.exists():
        print(f"{name}: 이미 있음, 건너뜀"); return
    t0 = time.time()
    tok, m = load(mid, kind)
    ko = translate(tok, m, kind, texts)
    out_p.write_text(json.dumps(
        [{**r, "text_ko": k} for r, k in zip(sample, ko)], ensure_ascii=False))
    fail = sum(1 for k in ko if not HANGUL.search(k))
    print(f"{name:14} 실패 {fail/len(ko):6.2%}  {time.time()-t0:5.0f}s  → {out_p.name}")


if __name__ == "__main__":
    main()
