"""PHASE E 설정."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/high_recall_attribution_cascade"
ART = ROOT / "artifacts/high_recall_attribution_cascade"
RES = ROOT / "results/high_recall_attribution_cascade"
DOCS = ROOT / "docs/high_recall_attribution_cascade"
PLOTS = ROOT / "plots/high_recall_attribution_cascade"

BASE_MODEL = "meta-llama/Llama-Prompt-Guard-2-86M"

# --- source 단위 (§3, §11) ---------------------------------------------------
# promptshield 의 test/train/validation 은 같은 논문(ACM CODASPY 2025)의 세 분할이다.
# 하나를 held-out 으로 두고 다른 하나로 학습하면 진짜 cross-source 가 아니므로
# source 단위는 데이터셋 계열로 잡는다.
SOURCE_MAP = {
    "promptshield:test": "promptshield",
    "promptshield:train": "promptshield",
    "promptshield:validation": "promptshield",
    "wildjailbreak:adversarial": "wildjailbreak_adversarial",
    "piguard:Question Set": "question_set",
}
# 기존 정책(심사 학회 게재 데이터만)을 유지한다.  GT 두 클래스를 가진 8개 source_group 중
# 심사 게재는 위 5개(=3개 계열)뿐이다.  아래는 사용 가능하나 정책상 제외한 것들이다.
EXCLUDED_UNREVIEWED = ["piguard:safe-guard-prompt-injection",
                       "piguard:jailbreak-classification",
                       "piguard:prompt-injections"]

DEV_SOURCES = ["wildjailbreak_adversarial", "promptshield", "question_set"]
SPLIT_FRACS = dict(train=.60, gate_calib=.10, model_val=.10, system_calib=.10, dev_test=.10)
GATE_RECALLS = (0.95, 0.975, 0.99)
TARGET_FPRS = (0.001, 0.005, 0.01)
