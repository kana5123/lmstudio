"""상위 번역 모델들의 레코드별 역번역 유사도를 저장하고 실패 유형을 뽑는다."""
import json, re, sys
from pathlib import Path
try:
    from setproctitle import setproctitle
    setproctitle("kana5123")
except ImportError:
    pass
import torch
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
BT = "facebook/nllb-200-3.3B"
EMB = "BAAI/bge-m3"
MODELS = sys.argv[1:] or ["circulus", "nllb3.3b", "nhndq"]


def back_translate(texts, bs=8):
    tok = AutoTokenizer.from_pretrained(BT, src_lang="kor_Hang")
    m = AutoModelForSeq2SeqLM.from_pretrained(BT, torch_dtype=torch.float16).to("cuda").eval()
    bos = tok.convert_tokens_to_ids("eng_Latn")
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            o = m.generate(**enc, forced_bos_token_id=bos, max_new_tokens=512,
                           num_beams=1, no_repeat_ngram_size=4)
        out += [tok.decode(x, skip_special_tokens=True) for x in o]
    del m; torch.cuda.empty_cache()
    return out


def embed(texts, bs=16):
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


for name in MODELS:
    f = HERE / f"mt_out_{name}.json"
    if not f.exists():
        print(f"{name}: 파일 없음"); continue
    data = json.loads(f.read_text())
    bt = back_translate([r["text_ko"] for r in data])
    en_v = embed([r["text_en"] for r in data])
    bt_v = embed(bt)
    sim = torch.nn.functional.cosine_similarity(en_v, bt_v).tolist()
    for r, b, s in zip(data, bt, sim):
        r["back_en"] = b; r["bt_sim"] = s
    (HERE / f"mt_verify_{name}.json").write_text(json.dumps(data, ensure_ascii=False))
    print(f"{name}: 저장 완료 (평균 {sum(sim)/len(sim):.4f})", flush=True)
