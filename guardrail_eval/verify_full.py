"""전량 번역본 검증 — 역번역 유사도 계산 후 저장.

safeguard_ko_clean.json (한글 유무·길이비 1차 통과분) 을 받아
각 레코드의 역번역문과 원문 유사도를 붙여 저장한다.
"""
import json
from pathlib import Path
try:
    from setproctitle import setproctitle
    setproctitle("kana5123")
except ImportError:
    pass
import torch
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
BT, EMB = "facebook/nllb-200-3.3B", "BAAI/bge-m3"


def run_bt(texts, bs=16):
    tok = AutoTokenizer.from_pretrained(BT, src_lang="kor_Hang")
    m = AutoModelForSeq2SeqLM.from_pretrained(BT, torch_dtype=torch.float16).to("cuda").eval()
    bos = tok.convert_tokens_to_ids("eng_Latn")
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            o = m.generate(**enc, forced_bos_token_id=bos, max_new_tokens=384,
                           num_beams=1, no_repeat_ngram_size=4)
        out += [tok.decode(x, skip_special_tokens=True) for x in o]
        if i % 1600 == 0:
            print(f"  역번역 {i}/{len(texts)}", flush=True)
    del m; torch.cuda.empty_cache()
    return out


def run_emb(texts, bs=32):
    tok = AutoTokenizer.from_pretrained(EMB)
    m = AutoModel.from_pretrained(EMB, torch_dtype=torch.float16).to("cuda").eval()
    vs = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            h = m(**enc).last_hidden_state[:, 0]
        vs.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu())
    del m; torch.cuda.empty_cache()
    return torch.cat(vs)


rows = json.loads((HERE / "safeguard_ko_clean.json").read_text())
print(f"대상 {len(rows):,}건")
bt = run_bt([r["text_ko"] for r in rows])
sim = torch.nn.functional.cosine_similarity(
    run_emb([r["text_en"] for r in rows]), run_emb(bt)).tolist()
for r, b, s in zip(rows, bt, sim):
    r["back_en"], r["bt_sim"] = b, s
(HERE / "safeguard_ko_verified.json").write_text(json.dumps(rows, ensure_ascii=False))
print(f"저장 완료 → safeguard_ko_verified.json  (평균 유사도 {sum(sim)/len(sim):.4f})")
