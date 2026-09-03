import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.decompx_trajectory_verifier.d0.probes import (OUT, SHORT, SRC_ORDER, c_norm_audit,
                                                       fp_tp_direction, layerwise_source_probe,
                                                       length_control, load, source_probe)
pd.set_option("display.width", 220)
OUT.mkdir(parents=True, exist_ok=True)
meta, H, S = load()
print(f"표본 {len(meta):,}  (train {int((meta.c1_split=='train').sum()):,} / "
      f"test {int((meta.c1_split=='test').sum()):,})\n")

print("=== §3 라벨 통제 source probe (마지막 층 h_12, chance macro acc ~0.333) ===")
r = [source_probe(H[:, -1], meta, c) for c in ("TP", "FP")]
print(pd.DataFrame(r).round(4).to_string(index=False))

print("\n=== §4 층별 source probe ===")
lw = layerwise_source_probe(H, meta)
lw.to_csv(OUT / "layer_source_probe.csv", index=False)
p = lw.pivot(index="layer", columns="cell", values=["macro_f1", "balanced_acc"])
print(p.round(4).to_string())

print("\n=== §6 층별 FP-TP 방향 코사인 (h_l) ===")
d = pd.DataFrame([fp_tp_direction(H[:, l], meta, f"h_{l+1}") for l in range(H.shape[1])])
d.to_csv(OUT / "fp_tp_direction_cosines.csv", index=False)
print(d.round(4).to_string(index=False))

print("\n=== §10 C 크기 감사 ===")
cn, smd = c_norm_audit(meta, S)
cn.to_csv(OUT / "c_norm_by_source_layer.csv", index=False)
smd.to_csv(OUT / "c_norm_smd.csv", index=False)
piv = cn.pivot_table(index="layer", columns="source_group", values="cls_norm_median")
print("  ||sum_k C_lk|| 중앙값 (=||h_CLS^l||)"); print(piv.round(3).to_string())
piv2 = cn.pivot_table(index="layer", columns="source_group", values="tok_norm_median")
print("\n  토큰 기여 노름 ||C_lk|| 중앙값"); print(piv2.round(3).to_string())
print("\n  source 쌍별 표준화 평균차 (||h_CLS^l|| 기준)")
print(smd.pivot(index="layer", columns="pair", values="smd_cls_norm").round(3).to_string())

print("\n=== §11 길이 대조군 ===")
lc = length_control(meta); lc.to_csv(OUT / "length_control.csv", index=False)
print(lc.round(4).to_string(index=False))
