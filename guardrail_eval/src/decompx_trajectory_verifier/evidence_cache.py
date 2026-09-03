"""Evidence 캐시 (§22).  base 와 DecompX 는 frozen 이므로 매 epoch 재계산하지 않는다.

ragged 저장: 고정 512 패딩을 표본마다 저장하지 않는다.
  C_flat  [sum_T, L, d]   fp16 (기본).  표본 i 는 C_flat[off_i:off_{i+1}] -> [T,L,d]
  Y_flat  [sum_T, C]      fp32
  a_flat  [sum_T]         fp32
  ids_flat[sum_T]         int32
  offsets [n+1]           int64

fp16 저장은 §22 대로 float32 원본과 subset 비교 후 승인한다.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.decompx_trajectory_verifier.base_adapter import PromptGuard2Adapter
from src.decompx_trajectory_verifier.config import BASE_MODEL, EPS
from src.decompx_trajectory_verifier.decompx_adapter import DecompXAdapter

# DecompX 귀속 텐서는 (B,T,T,d) 라 메모리가 B*T^2 에 비례한다
CELL_BUDGET = int(os.environ.get("DXTV_CELL_BUDGET", 600_000))


def batch_plan(lengths, budget=CELL_BUDGET, max_bs=16):
    plan, i = [], 0
    while i < len(lengths):
        T = int(lengths[i])
        bs = max(1, min(max_bs, int(budget / max(T * T, 1))))
        plan.append((i, min(i + bs, len(lengths))))
        i += bs
    return plan


class EvidenceExtractor:
    def __init__(self, model_name=BASE_MODEL, device="cuda", store_dtype=torch.float32):
        self.base = PromptGuard2Adapter(model_name, device)
        self.dxa = DecompXAdapter(self.base)
        self.L, self.d = self.base.get_num_layers(), self.base.get_hidden_size()
        self.nC = self.base.get_num_labels()
        self.store_dtype = store_dtype

    def _audit(self, ev, am, bi):
        """§4 계약 감사.  절대오차와 정규화오차를 둘 다 낸다.
        정규화 척도는 B1 에서 승인한 사라지지 않는 값을 쓴다:
          encoder -> ||h_CLS^l||,  head/margin -> ||logits||_2
        """
        C, Y, a, lg, hs = ev["C"][bi], ev["Y"][bi], ev["a"][bi], ev["logits"][bi], ev["hidden"]
        m = am[bi].to(C.dtype)
        T = int(m.sum())
        rec = (C * m[None, :, None]).sum(1)                       # [L,d]
        tgt = torch.stack([h[bi, self.base.get_decision_position()] for h in hs[1:]], 0)
        enc_abs = (rec - tgt).norm(dim=-1)                        # [L]
        enc_norm = enc_abs / (tgt.norm(dim=-1) + EPS)
        enc_maxc = (rec - tgt).abs().amax(-1)
        scale = lg.norm()
        Yrec = (Y * m[:, None]).sum(0)                            # [C]
        head_abs = (Yrec - lg).abs()
        head_norm = head_abs / (scale + EPS)
        mg = lg[self.base.get_attack_label_id()] - lg[self.base.get_benign_label_id()]
        mar_abs = ((a * m).sum() - mg).abs()
        mar_norm = mar_abs / (scale + EPS)
        pad = (1 - m)
        padC = float((C * pad[None, :, None]).abs().max()) if T < C.shape[1] else 0.0
        padY = float((Y * pad[:, None]).abs().max()) if T < C.shape[1] else 0.0
        padA = float((a * pad).abs().max()) if T < C.shape[1] else 0.0
        return dict(enc_abs=enc_abs.cpu().numpy(), enc_norm=enc_norm.cpu().numpy(),
                    enc_maxc=enc_maxc.cpu().numpy(),
                    head_abs=head_abs.cpu().numpy(), head_norm=head_norm.cpu().numpy(),
                    mar_abs=float(mar_abs), mar_norm=float(mar_norm),
                    padC=padC, padY=padY, padA=padA, logit_norm=float(scale))

    def _extract_into(self, idx, texts, Cs, Ys, As, Is, logits, audits):
        enc = self.base.encode([texts[i] for i in idx])
        ii, am = enc["input_ids"], enc["attention_mask"]
        ev = self.dxa.extract(ii, am)
        for bi, gi in enumerate(idx):
            T = int(am[bi].sum())
            # C [L,T,d] -> [T,L,d] 로 두어 토큰 축으로 이어붙이기 쉽게 한다
            Cs[gi] = ev["C"][bi, :, :T].permute(1, 0, 2).to(self.store_dtype).cpu()
            Ys[gi] = ev["Y"][bi, :T].float().cpu()
            As[gi] = ev["a"][bi, :T].float().cpu()
            Is[gi] = ii[bi, :T].to(torch.int32).cpu()
            logits[gi] = ev["logits"][bi].float().cpu().numpy()
            audits[gi] = self._audit(ev, am, bi)
        del ev

    def run(self, texts):
        """-> ragged dict.  행 순서는 입력과 동일."""
        n = len(texts)
        tok = self.base.tokenizer
        lens = np.array([len(tok(t, truncation=False)["input_ids"]) for t in texts])
        order = np.argsort(-lens)
        Cs, Ys, As, Is = [None] * n, [None] * n, [None] * n, [None] * n
        audits = [None] * n
        logits = np.zeros((n, self.nC), dtype=np.float32)
        for s, e in batch_plan(lens[order]):
            idx = list(order[s:e])
            try:
                self._extract_into(idx, texts, Cs, Ys, As, Is, logits, audits)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                for i in idx:                      # 메모리가 모자라면 한 개씩
                    self._extract_into([i], texts, Cs, Ys, As, Is, logits, audits)
                    torch.cuda.empty_cache()
            torch.cuda.empty_cache()
        assert all(x is not None for x in Is), "추출되지 않은 표본이 있다"
        off = np.zeros(n + 1, dtype=np.int64)
        off[1:] = np.cumsum([len(x) for x in Is])
        return dict(C_flat=torch.cat(Cs, 0), Y_flat=torch.cat(Ys, 0), a_flat=torch.cat(As, 0),
                    ids_flat=torch.cat(Is, 0),
                    mask_flat=torch.ones(int(off[-1]), dtype=torch.int8),
                    offsets=torch.from_numpy(off),
                    logits=torch.from_numpy(logits), L=self.L, d=self.d, nC=self.nC,
                    store_dtype=str(self.store_dtype), audits=audits)


def save_shard(df, path, ex):
    ev = ex.run(df["text"].tolist())
    ev |= dict(sample_id=df["sample_id"].tolist(),
               source_group=df["source_group"].tolist(),
               source_subgroup=df["source_subgroup"].tolist(),
               original_split=df["original_split"].tolist()
               if "original_split" in df else [None] * len(df),
               dataset=df["dataset"].tolist(),
               duplicate_group_id=df["duplicate_group_id"].tolist(),
               confusion_cell=df["confusion_cell"].tolist(),
               split=df["split"].tolist(),
               y_fp=torch.tensor(df["y_fp"].to_numpy(), dtype=torch.int8),
               base_pred=torch.tensor(df["base_pred"].to_numpy(), dtype=torch.int8),
               gt=torch.tensor(df["gt"].to_numpy(), dtype=torch.int8))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ev, path)
    return ev


def load_sample(blob, i):
    """-> C [L,T,d] float32, Y [T,C], a [T], ids [T]"""
    o0, o1 = int(blob["offsets"][i]), int(blob["offsets"][i + 1])
    C = blob["C_flat"][o0:o1].float().permute(1, 0, 2)      # [L,T,d]
    return C, blob["Y_flat"][o0:o1], blob["a_flat"][o0:o1], blob["ids_flat"][o0:o1]
