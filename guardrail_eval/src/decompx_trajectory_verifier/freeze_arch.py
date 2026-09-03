"""§1-§2 아키텍처 동결.  training 전에 config 와 파라미터 수를 JSON 으로 남긴다."""
import hashlib, json, subprocess, sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.decompx_trajectory_verifier import config as C
from src.decompx_trajectory_verifier.config import ART
from src.decompx_trajectory_verifier.model import DXTV, VARIANTS

SRC = Path(__file__).resolve().parent


def source_tree_hash():
    h = hashlib.blake2b(digest_size=16)
    for p in sorted(SRC.rglob("*.py")):
        h.update(p.relative_to(SRC).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SRC.parents[1],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def module_order(m):
    """모듈 구성 순서를 문자열로.  Sequential 이면 내부 순서까지 편다."""
    if isinstance(m, nn.Sequential):
        return [type(x).__name__ + (f"({x.in_features},{x.out_features})"
                                    if isinstance(x, nn.Linear) else
                                    (f"({x.normalized_shape[0]})" if isinstance(x, nn.LayerNorm)
                                     else "")) for x in m]
    out = []
    for n, x in m.named_children():
        out.append(f"{n}:{type(x).__name__}" + (f"({x.in_features},{x.out_features},"
                                                f"bias={x.bias is not None})"
                                                if isinstance(x, nn.Linear) else ""))
    ps = [n for n, _ in m.named_parameters(recurse=False)]
    if ps:
        out.append("direct_params=" + ",".join(ps))
    return out


def main(d=768, L=12, max_len=512):
    m = DXTV(d, L, max_len, variant="A3", d_v=C.D_V, depth_tf=C.DEPTH_TF, token_tf=C.TOKEN_TF,
             attr_hidden=C.ATTR_HIDDEN, fusion_out=C.FUSION_OUT, head_hidden=C.HEAD_HIDDEN)
    comp = {}
    for name, mod in [("CellProjector", m.proj), ("DepthTrajectoryEncoder", m.depth),
                      ("AttributionAnchor", m.anchor), ("Fusion", m.fusion),
                      ("TokenContextEncoder", m.token), ("Head", m.head)]:
        comp[name] = dict(params=sum(p.numel() for p in mod.parameters()),
                          module_order=module_order(mod),
                          has_layernorm=any(isinstance(x, nn.LayerNorm) for x in mod.modules()),
                          has_activation=any(isinstance(x, (nn.GELU, nn.ReLU, nn.Tanh))
                                             for x in mod.modules()))
    comp["CellProjector"]["note"] = (
        "LayerNorm 없음. Linear(768,128,bias=True)=98,432 에 더해진 1,536 은 "
        "§12 가 요구한 learned depth embedding E_depth[12,128] 이다.")
    cfg = dict(
        variant="A3", hidden_size=d, num_layers=L, max_len=max_len,
        d_v=C.D_V, depth_transformer=C.DEPTH_TF, token_transformer=C.TOKEN_TF,
        attr_hidden=C.ATTR_HIDDEN, fusion_out=C.FUSION_OUT, head_hidden=C.HEAD_HIDDEN,
        target=dict(TP=0, FP=1), output="P(FP | PromptGuard2 predicted ATTACK)",
        total_params=sum(p.numel() for p in m.parameters()), components=comp,
        variants_available=list(VARIANTS), variants_trained_this_phase=["M0", "A0", "A3"],
        source_tree_hash=source_tree_hash(), git_hash=git_hash(),
        cache_store_dtype=C.STORE_DTYPE,
        optimizer=dict(name="AdamW", lr=C.LR, weight_decay=C.WEIGHT_DECAY,
                       max_epochs=C.MAX_EPOCHS, patience=C.PATIENCE, grad_clip=C.GRAD_CLIP),
        seeds=list(C.SEEDS))
    ART.mkdir(parents=True, exist_ok=True)
    p = ART / "frozen_architecture_config.json"
    json.dump(cfg, open(p, "w"), indent=1, ensure_ascii=False)
    return cfg, p


if __name__ == "__main__":
    cfg, p = main()
    print(f"총 파라미터 {cfg['total_params']:,}   source_tree_hash {cfg['source_tree_hash']}")
    print(f"git_hash {cfg['git_hash']}\n")
    for k, v in cfg["components"].items():
        print(f"{k:<24}{v['params']:>9,}  LN={v['has_layernorm']}  act={v['has_activation']}")
        print(f"    {v['module_order']}")
    print(f"\n저장 -> {p}")
