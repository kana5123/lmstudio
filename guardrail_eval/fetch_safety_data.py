"""영어 안전 학습 데이터 + 한국어 평가 데이터 내려받기."""
import sys
try:
    from setproctitle import setproctitle; setproctitle("kana5123")
except ImportError: pass
from datasets import load_dataset

TARGETS = [
    ("allenai/wildguardmix",  "wildguardtrain", None),
    ("ToxicityPrompts/PolyGuardPrompts", None, None),
    ("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", None, None),
]
for name, cfg, split in TARGETS:
    try:
        d = load_dataset(name, cfg) if cfg else load_dataset(name)
        print(f"✓ {name}" + (f" [{cfg}]" if cfg else ""))
        for k, v in d.items():
            print(f"    {k}: {len(v)}건  열={list(v.features)[:8]}")
    except Exception as e:
        print(f"✗ {name}: {type(e).__name__}: {str(e)[:150]}")
    sys.stdout.flush()
