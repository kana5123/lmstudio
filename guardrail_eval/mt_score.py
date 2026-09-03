"""번역 결과 채점 — 표면 지표 + 역번역 의미 보존도.

표면 지표만으로는 "번역은 됐는데 뜻이 달라진" 경우를 못 잡는다.
한국어를 다시 영어로 되돌려(NLLB-3.3B 고정) 원문과 bge-m3 코사인을 잰다.
역번역기와 임베딩을 모든 후보에 동일하게 써야 비교가 공정하다.
"""
import glob, json, re, sys
from collections import Counter
from pathlib import Path

try:
    from setproctitle import setproctitle
    setproctitle("kana5123")
except ImportError:
    pass

import torch
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
HANGUL = re.compile(r"[가-힣]")
BT_MODEL = "facebook/nllb-200-3.3B"
EMB = "BAAI/bge-m3"


def repetitive(t, n=4, k=3):
    w = t.split()
    if len(w) < n * 2:
        return False
    c = Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    return max(c.values(), default=0) >= k


def back_translate(texts, bs=8):
    tok = AutoTokenizer.from_pretrained(BT_MODEL, src_lang="kor_Hang")
    m = AutoModelForSeq2SeqLM.from_pretrained(BT_MODEL, torch_dtype=torch.float16).to("cuda").eval()
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
            h = m(**enc).last_hidden_state[:, 0]          # bge-m3 는 CLS 사용
        vs.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu())
    del m; torch.cuda.empty_cache()
    return torch.cat(vs)


def main():
    files = sorted(glob.glob(str(HERE / "mt_out_*.json")))
    if not files:
        print("결과 파일 없음"); return
    base = json.loads(Path(files[0]).read_text())
    en = [r["text_en"] for r in base]
    en_vec = embed(en)

    rows = []
    for f in files:
        name = Path(f).stem[len("mt_out_"):]
        data = json.loads(Path(f).read_text())
        ko = [r["text_ko"] for r in data]
        n = len(ko)
        fail = sum(1 for k in ko if not HANGUL.search(k)) / n
        rep = sum(1 for k in ko if repetitive(k)) / n
        short = sum(1 for k, r in zip(ko, data)
                    if len(k) / max(1, len(r["text_en"])) < 0.3) / n
        bt = back_translate(ko)
        sim = torch.nn.functional.cosine_similarity(en_vec, embed(bt)).numpy()
        rows.append(dict(name=name, fail=fail, rep=rep, short=short,
                         bt_sim=float(sim.mean()), bt_sim_p10=float(sorted(sim)[int(n * .1)])))
        print(f"  {name} 채점 완료", flush=True)

    rows.sort(key=lambda r: -r["bt_sim"])
    print(f"\n{'모델':16}{'실패율':>9}{'반복률':>8}{'잘림률':>9}{'역번역유사도':>13}{'하위10%':>10}")
    print("-" * 66)
    for r in rows:
        print(f"{r['name']:16}{r['fail']:>8.2%}{r['rep']:>8.2%}{r['short']:>9.2%}"
              f"{r['bt_sim']:>13.4f}{r['bt_sim_p10']:>10.4f}")
    (HERE / "mt_score.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
