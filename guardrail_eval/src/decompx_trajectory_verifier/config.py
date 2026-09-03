"""DXTV 설정.  모델 관련 값은 adapter 가 런타임 config 에서 읽는다."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/decompx_trajectory_verifier"
ART = ROOT / "artifacts/decompx_trajectory_verifier"
RES = ROOT / "results/decompx_trajectory_verifier"
DOCS = ROOT / "docs/decompx_trajectory_verifier"
PLOTS = ROOT / "plots/decompx_trajectory_verifier"

BASE_MODEL = "meta-llama/Llama-Prompt-Guard-2-86M"
EPS = 1e-12

# --- 재구성 허용 오차 --------------------------------------------------------
# 두 단계로 판정한다.
#  (1) 대수적 정확성은 float64 에서 본다.  DecompX 의 영점기울기(ZO) 활성함수 근사는
#      m = f(x)/x 를 곱하는 방식이라 합을 정확히 보존한다(논문 3.3절).
#      실측: encoder 3.12e-10, head 1.95e-10, margin 3.44e-10.
#  (2) 추출은 float32 로 한다.  분모를 고를 때 주의가 필요하다 --
#      개별 로짓은 0 을 지날 수 있어(gauge) 상대오차가 무한대로 뛴다.
#      실제로 class0 로짓이 ~1e-3 인 표본에서 상대오차 0.96 이 나왔으나 절대오차는 1.2e-3 이었다.
#      따라서 head/margin 은 사라지지 않는 척도 ||logits||_2 로 정규화해 판정한다.
#      encoder 는 ||h_CLS^l|| 이 사라지지 않으므로 그대로 상대오차를 쓴다.
TOL_ENCODER_REL = 2e-3       # |sum_k C_lk - h_CLS^l| / ||h_CLS^l||
TOL_HEAD_SCALED = 2e-3       # |sum_k Y_kc - logit_c| / ||logits||_2
TOL_MARGIN_SCALED = 2e-3     # |sum_k a_k - (z_a - z_b)| / ||logits||_2
TOL_PAD_ABS = 1e-6           # 패딩 토큰 기여
TOL_F64 = 1e-7               # float64 대수 정확성 게이트

# --- 캐시 저장 정밀도 (§22) --------------------------------------------------
# 파일럿 비교 결과 fp16/bf16 압축은 재구성 항등식을 깨뜨린다:
#   Y/a 를 압축하면 sum_k Y_kc 정규화 오차 0.120(fp16) / 0.867(bf16)  -- 허용 2e-3 의 60~430배
#   C 를 압축하면 sum_k C_lk 가 최대 8.8%(fp16) / 60%(bf16) 흔들린다
# 디스크 여유가 충분하므로(CORE 전체 약 89 GB / 684 GB 여유) 전부 float32 로 저장한다.
STORE_DTYPE = "float32"

# 원 상대오차도 기록은 하되 판정에는 쓰지 않는다(분모 소실 때문)
TOL_HEAD_REL = TOL_HEAD_SCALED
TOL_MARGIN_REL = TOL_MARGIN_SCALED

# --- 아키텍처 (§12-§19) -----------------------------------------------------
D_V = 128                    # cell projection 차원
DEPTH_TF = dict(d_model=128, nhead=4, num_layers=2, dim_feedforward=512,
                dropout=0.1, norm_first=True, activation="gelu")
TOKEN_TF = dict(d_model=128, nhead=4, num_layers=2, dim_feedforward=512,
                dropout=0.1, norm_first=True, activation="gelu")
ATTR_HIDDEN = 32
FUSION_OUT = 128
HEAD_HIDDEN = 64

# --- 학습 (§21) --------------------------------------------------------------
LR = 1e-4
WEIGHT_DECAY = 1e-2
MAX_EPOCHS = 30
PATIENCE = 5
GRAD_CLIP = 1.0
SEEDS = (0, 1, 2, 3, 4)
