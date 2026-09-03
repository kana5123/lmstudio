"""검증된 사실만 담은 모델 레지스트리.  값은 전부 config/모델카드/실측으로 확인한 것이다.

점수 규칙 근거:
  - 이진 모델 5종: id2label 이 확정돼 있고 positive=1 을 실측으로 확인
  - PromptGuard v1: 공식 모델카드 코드가 두 함수를 나눠 정의한다
      get_jailbreak_score        = p[2]         (사용자 대화 필터링용)
      get_indirect_injection_score = p[1]+p[2]  (제3자 입력 필터링용)
    따라서 데이터셋이 indirect_injection 계열이면 p1+p2, 아니면 p2 를 쓴다.
"""

MODELS = {
    "pg2":       dict(hf="meta-llama/Llama-Prompt-Guard-2-86M",             layers=12, score="binary"),
    "pgv1":      dict(hf="meta-llama/Prompt-Guard-86M",                     layers=12, score="pgv1"),
    "piguard":   dict(hf="leolee99/PIGuard",                                layers=12, score="binary", custom=True),
    "protectai": dict(hf="protectai/deberta-v3-base-prompt-injection-v2",   layers=12, score="binary"),
    "deepset":   dict(hf="deepset/deberta-v3-base-injection",               layers=12, score="binary"),
    "fmops":     dict(hf="fmops/distilbert-prompt-injection",               layers=6,  score="binary"),
}

# 제3자 콘텐츠 데이터셋 -> PGv1 은 injection+jailbreak 를 쓴다
THIRD_PARTY_FAMILIES = {"indirect_injection"}


def attack_prob(model_key, probs, attack_family):
    """(n, C) 확률 -> (n,) 공격 확률."""
    if MODELS[model_key]["score"] == "binary":
        return probs[:, 1]
    if attack_family in THIRD_PARTY_FAMILIES:
        return probs[:, 1] + probs[:, 2]
    return probs[:, 2]
