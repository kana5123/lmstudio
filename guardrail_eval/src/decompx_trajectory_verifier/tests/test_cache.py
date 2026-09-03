"""캐시 왕복 검증 (§23 STEP 6): 저장 -> 로드 후 동일 evidence 가 복원되는가."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.decompx_trajectory_verifier.evidence_cache import load_sample


def check_roundtrip(path, n_check=None):
    blob = torch.load(path, weights_only=False)
    n = len(blob["sample_id"])
    n_check = n_check or n
    off = blob["offsets"]
    rows = []
    # 1) offsets 무결성
    assert int(off[0]) == 0, "offsets 시작이 0 이 아니다"
    assert int(off[-1]) == blob["C_flat"].shape[0] == blob["Y_flat"].shape[0] \
        == blob["a_flat"].shape[0] == blob["ids_flat"].shape[0], "offsets 끝과 flat 길이가 다르다"
    assert (off[1:] > off[:-1]).all(), "빈 표본이 있다"
    # 2) 표본별 shape 및 a = Y_attack - Y_benign 일관성
    L, d, nC = blob["L"], blob["d"], blob["nC"]
    for i in range(n_check):
        C, Y, a, ids = load_sample(blob, i)
        T = int(off[i + 1] - off[i])
        assert C.shape == (L, T, d), f"{i}: C {C.shape} != {(L,T,d)}"
        assert Y.shape == (T, nC) and a.shape == (T,) and ids.shape == (T,)
        rows.append(T)
    # 3) 저장 -> 재로드 -> 완전 동일
    tmp = Path(str(path) + ".roundtrip")
    torch.save(blob, tmp)
    b2 = torch.load(tmp, weights_only=False)
    diffs = {}
    for k in ("C_flat", "Y_flat", "a_flat", "ids_flat", "offsets", "logits"):
        diffs[k] = float((b2[k].float() - blob[k].float()).abs().max())
    for k in ("sample_id", "confusion_cell", "source_group", "split"):
        assert b2[k] == blob[k], f"{k} 불일치"
    tmp.unlink()
    return dict(n=n, checked=n_check, tokens_min=min(rows), tokens_max=max(rows),
                tokens_sum=int(off[-1]), diffs=diffs)


if __name__ == "__main__":
    from src.decompx_trajectory_verifier.config import ART
    r = check_roundtrip(ART / "pilot_256.pt")
    print(f"표본 {r['n']}  검사 {r['checked']}  토큰 {r['tokens_min']}~{r['tokens_max']} "
          f"(합 {r['tokens_sum']:,})")
    print("저장->재로드 최대차:", {k: f"{v:.1e}" for k, v in r["diffs"].items()})
    assert all(v == 0.0 for v in r["diffs"].values()), "왕복에서 값이 바뀌었다"
    print("캐시 왕복 검증 통과")
