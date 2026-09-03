"""rfpr 결과 → 논문 표 (행=모델, 열=벤치마크별 R@1%FPR / 달성 FPR / 지연)."""
import json
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
ORDER = [
    ("llamafirewall", "LlamaFirewall (PromptGuard-2-86M)", "프레임워크"),
    ("qwen3guard", "Qwen3Guard-Gen-4B", "프레임워크"),
    ("nemoguard", "NemoGuard-8B", "프레임워크"),
    ("promptguard_v1", "PromptGuard-v1 (jailbreak 라벨)", "경량 분류기"),
    ("promptguard_v1_strict", "PromptGuard-v1 (injection+jailbreak)", "경량 분류기"),
    ("protectaiv2", "ProtectAI v2", "경량 분류기"),
    ("piguard", "PIGuard", "경량 분류기"),
    ("deepset", "Deepset", "경량 분류기"),
    ("fmops", "Fmops", "경량 분류기"),
]


def cell(m, key):
    return "–" if m is None else f"{m[key]:.3f}" if isinstance(m[key], float) else str(m[key])


def main():
    for bench, title in (("jailbreak", "JailbreaksOverTime — 사용자 프롬프트 (test 2,000 / 양성 352)"),
                         ("piarena", "PIArena direct — 제3자 문서 문맥 (test 240 / 양성 120)")):
        print(f"\n## {bench}  {title}")
        hdr = f"{'모델':38}{'R@1%FPR':>9}{'달성FPR':>9}{'R@0.1%':>9}{'달성FPR':>9}{'지연 ms':>9}"
        print(hdr); print("-" * len(hdr))
        last = None
        for key, label, grp in ORDER:
            p = RES / f"rfpr_{bench}_{key}.json"
            if not p.exists():
                continue
            m = json.loads(p.read_text())
            if grp != last:
                print(f"[{grp}]"); last = grp
            print(f"{label:38}{m['recall@1pct']:>9.3f}{m['achieved_fpr@1pct']:>9.4f}"
                  f"{m['recall@0.1pct']:>9.3f}{m['achieved_fpr@0.1pct']:>9.4f}"
                  f"{m['latency_mean_ms']:>9.1f}")


if __name__ == "__main__":
    main()
