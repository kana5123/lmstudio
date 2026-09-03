"""설정.  모델 관련 값은 런타임 config 에서 읽고 하드코딩하지 않는다."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/decompx_verifier_v2"
ART = ROOT / "artifacts/decompx_verifier_v2"
RES = ROOT / "results/decompx_verifier_v2"
DOCS = ROOT / "docs/decompx_verifier_v2"
PLOTS = ROOT / "plots/decompx_verifier_v2"

BASE_MODEL = "meta-llama/Llama-Prompt-Guard-2-86M"
EPS = 1e-12

# --- PHASE A 에서 확정된 MAIN 중 심사 학회 게재 출처만 사용 -------------------
PEER_REVIEWED_SOURCES = {
    "promptshield_test", "promptshield_train", "promptshield_validation",   # ACM CODASPY 2025
    "adversarial_benign", "adversarial_harmful", "vanilla_benign",          # NeurIPS 2024
    "TaskTracker",                                                          # IEEE SaTML 2025
    "hackaprompt-dataset",                                                  # EMNLP 2023
    "ultrachat_200k",                                                       # EMNLP 2023
    "Question Set",                                                         # ACM CCS 2024
    "over-defense", "NotInject_one", "NotInject_two", "NotInject_three",    # ACL 2025
    "WildGuard",                                                            # ACL 2025 재배포
    "BIPIA",                                                                # KDD 2025
    "xtest-v2-copy",                                                        # NAACL 2024
    "StruQ",                                                                # USENIX Security 2025
}

# --- 검증 허용 오차 -----------------------------------------------------------
TOL_LOGIT_MATCH = 1e-4        # inputs_embeds 경로 대 input_ids 경로 로짓 차이
# sum_k a_lk 대 q_l^T g_l (§24).  두 단계로 판정한다.
#  (1) 대수적 정확성은 float64 에서 본다 -- 구현이 옳은가의 문제.
#      실측 절대 4.95e-10 / 상대 4.97e-09.
#  (2) 추출은 float32 로 한다.  11-12층에서 CLS 크기가 12배 뛰어 상쇄 반올림이 커지므로
#      상대오차는 분모가 작을 때 부풀려진다.  절대오차로 판정한다(실측 최대 1.1e-03).
TOL_QD_IDENTITY_REL_F64 = 1e-7
TOL_QD_IDENTITY_ABS_F32 = 1e-2
TOL_QD_IDENTITY_REL = TOL_QD_IDENTITY_ABS_F32

# --- 아키텍처 ----------------------------------------------------------------
PROJ_DIM = 128
FUSION_HIDDEN = 512
D_MODEL = 256
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1
NORM_FIRST = True
DEPTH_POS_ENCODING = "learned"     # learned | sinusoidal  (§21 에서 명시 고정)
