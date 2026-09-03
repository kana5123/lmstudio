"""frozen base 로부터 evidence 를 뽑아 오프라인 캐시한다(§27).

학습 중에는 PromptGuard2/DecompX 를 다시 실행하지 않는다.
저장 항목: q, zD_pos, zD_neg, zH_pos, zH_neg [K,d] / mass_pos, mass_neg [K]
          base_prediction, ground_truth, confusion_cell, sample_id, group_id, dataset
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.decompx_verifier_v2.base_adapter import PromptGuard2Adapter
from src.decompx_verifier_v2.config import ART, BASE_MODEL
from src.decompx_verifier_v2.decompx_features import DecompXTransitionExtractor
from src.decompx_verifier_v2.label_margin_retrieval import LabelMarginRetriever
from src.decompx_verifier_v2.q_extractor import MarginSensitivityExtractor

# DecompX 귀속 텐서는 (B,T,T,d) 라 메모리가 B*T^2*d 에 비례한다.
# 따라서 예산은 B*T^2 로 잡아야 한다 (B*T 로 잡으면 긴 문장에서 터진다).
import os
CELL_BUDGET = int(os.environ.get("DV2_CELL_BUDGET", 1_200_000))   # B * T^2 상한


def batch_plan(lengths, budget=CELL_BUDGET, max_bs=16):
    """길이 내림차순 정렬된 길이 배열 -> (시작, 끝) 배치 경계."""
    plan, i = [], 0
    while i < len(lengths):
        T = int(lengths[i])
        bs = max(1, min(max_bs, int(budget / max(T * T, 1))))
        plan.append((i, min(i + bs, len(lengths))))
        i += bs
    return plan


class EvidenceExtractor:
    def __init__(self, model_name=BASE_MODEL, device="cuda"):
        self.ad = PromptGuard2Adapter(model_name, device)
        self.qx = MarginSensitivityExtractor(self.ad)
        self.dx = DecompXTransitionExtractor(self.ad)
        self.ret = LabelMarginRetriever()
        self.K, self.d = self.dx.K, self.ad.get_hidden_size()

    def run(self, texts):
        """-> dict of numpy arrays, 행 순서는 입력과 동일."""
        n = len(texts)
        tok = self.ad.tokenizer
        lens = np.array([len(tok(t, truncation=False)["input_ids"]) for t in texts])
        order = np.argsort(-lens)
        out = {k: np.zeros((n, self.K, self.d), dtype=np.float32)
               for k in ("q", "zD_pos", "zD_neg", "zH_pos", "zH_neg")}
        out |= {k: np.zeros((n, self.K), dtype=np.float32) for k in ("mass_pos", "mass_neg")}
        out["margin"] = np.zeros(n, dtype=np.float32)
        out["logits"] = np.zeros((n, self.ad.cfg.num_labels), dtype=np.float32)
        for s, e in batch_plan(lens[order]):
            idx = order[s:e]
            enc = self.ad.encode([texts[i] for i in idx])
            ii, am = enc["input_ids"], enc["attention_mask"]
            try:
                qo = self.qx.extract(ii, am)
                do = self.dx.extract(ii, am)
                ev = self.ret(qo["q"], do["D"], do["H_pre"], am)
            except torch.cuda.OutOfMemoryError:
                # 메모리가 모자라면 한 개씩 다시 시도한다
                torch.cuda.empty_cache()
                for j in range(len(idx)):
                    sub = slice(j, j + 1)
                    q1 = self.qx.extract(ii[sub], am[sub])
                    d1 = self.dx.extract(ii[sub], am[sub])
                    e1 = self.ret(q1["q"], d1["D"], d1["H_pre"], am[sub])
                    for k in ("zD_pos", "zD_neg", "zH_pos", "zH_neg", "mass_pos", "mass_neg"):
                        out[k][idx[j:j + 1]] = e1[k].float().cpu().numpy()
                    out["q"][idx[j:j + 1]] = q1["q"].float().cpu().numpy()
                    out["margin"][idx[j:j + 1]] = q1["margin"].float().cpu().numpy()
                    out["logits"][idx[j:j + 1]] = q1["logits"].float().cpu().numpy()
                    del q1, d1, e1
                    torch.cuda.empty_cache()
                continue
            for k in ("zD_pos", "zD_neg", "zH_pos", "zH_neg", "mass_pos", "mass_neg"):
                out[k][idx] = ev[k].float().cpu().numpy()
            out["q"][idx] = qo["q"].float().cpu().numpy()
            out["margin"][idx] = qo["margin"].float().cpu().numpy()
            out["logits"][idx] = qo["logits"].float().cpu().numpy()
            del qo, do, ev
            torch.cuda.empty_cache()
        return out


def extract_to_file(df, path, device="cuda", ex=None):
    """df: sample_id, text, confusion_cell, base_pred, gt, group_id, dataset"""
    ex = ex or EvidenceExtractor(device=device)
    ev = ex.run(df["text"].tolist())
    blob = {k: torch.from_numpy(v) for k, v in ev.items()}
    # 주의: df.gt 는 pandas 의 gt() 메서드와 이름이 겹친다 -> 대괄호 접근을 쓴다
    blob |= dict(sample_id=df["sample_id"].tolist(), dataset=df["dataset"].tolist(),
                 group_id=df["group_id"].tolist(),
                 confusion_cell=df["confusion_cell"].tolist(),
                 split=df["split"].tolist() if "split" in df else None,
                 base_pred=torch.tensor(df["base_pred"].to_numpy(), dtype=torch.int8),
                 gt=torch.tensor(df["gt"].to_numpy(), dtype=torch.int8),
                 K=ex.K, d=ex.d)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)
    return blob
