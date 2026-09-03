"""PHASE B3 검증: 샤드 무결성 / 계약 감사 분포 / 왕복 / 저장 용량."""
import json, sys, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.decompx_trajectory_verifier.config import ART, DATA, RES

TOL = 2e-3
CACHE = ART / "core_cache"
PER_CHUNK_ROUNDTRIP = 3


def main():
    core = pd.read_parquet(DATA / "core_tp_fp.parquet")
    files = sorted(CACHE.glob("core_s*_c*.pt"))
    print(f"청크 파일 {len(files)}개\n")

    meta, rt_rows, bytes_C, bytes_Y, bytes_a, bytes_ids = [], [], 0, 0, 0, 0
    empty, bad_off = 0, 0
    rng = np.random.default_rng(0)
    for f in files:
        b = torch.load(f, weights_only=False)
        sh = int(f.stem.split("_")[1][1:])
        off = b["offsets"]
        n = len(b["sample_id"])
        if int(off[0]) != 0 or int(off[-1]) != b["C_flat"].shape[0]:
            bad_off += 1
        empty += int((off[1:] - off[:-1] <= 0).sum())
        for i in range(n):
            meta.append(dict(shard=sh, chunk=f.stem, sample_id=b["sample_id"][i],
                             source_group=b["source_group"][i],
                             source_subgroup=b["source_subgroup"][i],
                             confusion_cell=b["confusion_cell"][i],
                             n_tokens=int(off[i + 1] - off[i])))
        bytes_C += b["C_flat"].numel() * b["C_flat"].element_size()
        bytes_Y += b["Y_flat"].numel() * b["Y_flat"].element_size()
        bytes_a += b["a_flat"].numel() * b["a_flat"].element_size()
        bytes_ids += b["ids_flat"].numel() * b["ids_flat"].element_size()
        # --- 왕복: 무작위 표본을 저장 -> 재로드 -> 완전 일치 확인 ------------
        for i in rng.choice(n, size=min(PER_CHUNK_ROUNDTRIP, n), replace=False):
            s, e = int(off[i]), int(off[i + 1])
            piece = dict(C=b["C_flat"][s:e].clone(), Y=b["Y_flat"][s:e].clone(),
                         a=b["a_flat"][s:e].clone(), ids=b["ids_flat"][s:e].clone(),
                         mask=b["mask_flat"][s:e].clone(), logits=b["logits"][i].clone(),
                         sid=b["sample_id"][i], sg=b["source_group"][i],
                         ssg=b["source_subgroup"][i], cell=b["confusion_cell"][i])
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as tf:
                torch.save(piece, tf.name)
                r = torch.load(tf.name, weights_only=False)
            d = {k: float((r[k].float() - piece[k].float()).abs().max())
                 for k in ("C", "Y", "a", "ids", "mask", "logits")}
            rt_rows.append(dict(shard=sh, sample_id=piece["sid"], **d,
                                meta_ok=(r["sg"] == piece["sg"] and r["ssg"] == piece["ssg"]
                                         and r["cell"] == piece["cell"])))
        del b
    M = pd.DataFrame(meta)
    RT = pd.DataFrame(rt_rows)
    M.to_csv(RES / "b3_sample_manifest.csv", index=False)
    RT.to_csv(RES / "b3_roundtrip_audit.csv", index=False)

    # --- 1-3 무결성 ---------------------------------------------------------
    exp = set(core.sample_id)
    got = list(M.sample_id)
    print("=== 무결성 ===")
    print(f"  총 표본 {len(got):,} (기대 {len(exp):,})")
    print(f"  고유 sample_id {M.sample_id.nunique():,}")
    print(f"  중복 sample_id {len(got) - M.sample_id.nunique()}")
    print(f"  누락 sample_id {len(exp - set(got))}")
    print(f"  빈 표본 {empty}   잘못된 offsets {bad_off}")
    dup_shard = M.groupby("sample_id").shard.nunique()
    print(f"  두 샤드에 걸친 표본 {int((dup_shard > 1).sum())}\n")

    print("=== 샤드별 ===")
    sm = M.groupby("shard").agg(표본=("sample_id", "size"), 토큰=("n_tokens", "sum"))
    cc = M.pivot_table(index="shard", columns="confusion_cell", values="sample_id",
                       aggfunc="count", fill_value=0)
    print(sm.join(cc).to_string(), "\n")

    print("=== source_group ===")
    g = M.pivot_table(index="source_group", columns="confusion_cell", values="sample_id",
                      aggfunc="count", fill_value=0).reindex(columns=["TP", "FP"], fill_value=0)
    tk = M.groupby("source_group").n_tokens.agg(["sum", "mean", "median",
                                                 lambda x: x.quantile(.95), "max"])
    tk.columns = ["토큰합", "평균", "중앙", "p95", "최대"]
    print(g.join(tk).round(1).to_string(), "\n")
    print("=== source_subgroup ===")
    gs = M.pivot_table(index=["source_group", "source_subgroup"], columns="confusion_cell",
                       values="sample_id", aggfunc="count", fill_value=0)
    print(gs.join(M.groupby(["source_group", "source_subgroup"]).n_tokens.sum()
                  .rename("토큰합")).to_string(), "\n")

    # --- 4-5 계약 감사 분포 --------------------------------------------------
    def dist(name, col="normalized_error"):
        df = pd.concat([pd.read_csv(p) for p in
                        sorted(RES.glob(f"b3_{name}_audit_s*.csv"))], ignore_index=True)
        out = {}
        for c in ("absolute_error", "normalized_error"):
            x = df[c]
            out[c] = dict(count=len(x), mean=x.mean(), median=x.median(),
                          p95=x.quantile(.95), p99=x.quantile(.99), max=x.max())
        nf = int((df[col] > TOL).sum())
        return df, out, nf
    print("=== 계약 감사 분포 (정규화 오차 기준 초과 판정, 허용 2e-3) ===")
    quarantine = []
    for nm, label in (("encoder", "encoder  sum_k C_lk = h_CLS^l"),
                      ("head", "head     sum_k Y_kc = logit_c"),
                      ("margin", "margin   sum_k a_k = z_a - z_b")):
        df, o, nf = dist(nm)
        print(f"\n[{label}]  행 {o['normalized_error']['count']:,}")
        for c in ("absolute_error", "normalized_error"):
            s = o[c]
            print(f"  {c:<18} mean {s['mean']:.3e}  median {s['median']:.3e}  "
                  f"p95 {s['p95']:.3e}  p99 {s['p99']:.3e}  max {s['max']:.3e}")
        print(f"  >2e-3 초과 {nf:,}건  ({nf/len(df)*100:.4f}%)")
        if nf:
            q = df[df.normalized_error > TOL].copy(); q["error_type"] = nm
            quarantine.append(q)
    P = pd.concat([pd.read_csv(p) for p in sorted(RES.glob("b3_padding_audit_s*.csv"))],
                  ignore_index=True)
    print(f"\n[padding]  행 {len(P):,}")
    print(f"  pad_C_max 최대 {P.pad_C_max.max():.3e}   pad_Y_max 최대 {P.pad_Y_max.max():.3e}   "
          f"pad_a_max 최대 {P.pad_a_max.max():.3e}")
    if quarantine:
        Q = pd.concat(quarantine, ignore_index=True)
        Q.to_csv(RES / "b3_quarantine.csv", index=False)
        print(f"\n격리 {len(Q):,}행 (고유 표본 {Q.sample_id.nunique():,}) -> b3_quarantine.csv")
    else:
        pd.DataFrame().to_csv(RES / "b3_quarantine.csv", index=False)
        print("\n격리 0건")

    # --- 6 왕복 -------------------------------------------------------------
    print(f"\n=== 왕복 검증 ({len(RT)}건, 샤드당 {RT.groupby('shard').size().min()}~"
          f"{RT.groupby('shard').size().max()}건) ===")
    mx = {k: RT[k].max() for k in ("C", "Y", "a", "ids", "mask", "logits")}
    print("  최대차:", {k: f"{v:.1e}" for k, v in mx.items()},
          f" 메타 일치 {bool(RT.meta_ok.all())}")
    print(f"  판정: {'통과 (완전 일치)' if all(v == 0.0 for v in mx.values()) and RT.meta_ok.all() else '★실패'}")

    # --- 8 용량 -------------------------------------------------------------
    disk = sum(f.stat().st_size for f in files)
    print("\n=== 저장 용량 ===")
    print(f"  샤드 8개 / 청크 파일 {len(files)}개")
    print(f"  디스크 실측 합계 {disk/1e9:.1f} GB")
    print(f"  C   {bytes_C/1e9:.1f} GB")
    print(f"  Y   {bytes_Y/1e6:.1f} MB")
    print(f"  a   {bytes_a/1e6:.1f} MB")
    print(f"  input_ids {bytes_ids/1e6:.1f} MB")
    print(f"  기타(메타/마스크/로짓) {(disk-bytes_C-bytes_Y-bytes_a-bytes_ids)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
