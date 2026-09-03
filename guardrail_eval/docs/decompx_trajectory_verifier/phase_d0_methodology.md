# PHASE D0 방법론 — TRAJECTORY OOD FAILURE DIAGNOSIS

질문: "최종 DecompX attribution Y/a 는 일부 source 로 전이되는데,
중간 C trajectory 를 더한 A3 는 왜 cross-source 에서 무너지는가?"

## 쓴 것과 쓰지 않은 것
기존 B3 fp32 cache, C1 checkpoint, C1 prediction 만 사용했다.
새 production verifier, cascade, distillation, gradient q_l, Integrated Gradients,
TF-IDF, lexical 분석, raw-text 분류기, 새 dataset, 추가 DecompX 추출,
A1~A5 full ablation 은 수행하지 않았다.

## 표현
`h_l = sum_k C_lk` 를 B3 memmap 에서 재구성했다.  DecompX 계약상 layer l 의 실제 CLS hidden 과
같다(PHASE B1/B3 에서 검증됨).  A3 내부 표현(tau_k, v_k, VCLS)은 학습된 checkpoint 를
그대로 두고 추론 시점 hook 으로 꺼냈다.

## 라벨 통제 source probe
source 3분류를 **TP-only 와 FP-only 로 나눠** 수행했다.  그래야 source 분류가 TP/FP 라벨
차이를 지름길로 쓸 수 없다.  `class_weight="balanced"` 를 쓰고 macro-F1 / balanced accuracy /
one-vs-rest AUROC 를 보고한다.  우연 macro accuracy 는 약 1/3 이다.
분할은 기존 `seen_source_seed0` manifest 를 그대로 썼다(새 분할을 만들지 않음).

## 추론 시점 개입
학습된 A3 파라미터를 갱신하지 않는다.
- branch zeroing: `tau_k = 0` 또는 `e_attr,k = 0`
- layer occlusion: 해당 층의 `C` 만 0 으로.  depth embedding 은 projection 뒤에 더해지므로
  나머지 층과 depth position 은 그대로 유지된다.

**개입 모델을 재학습한 A0/A2 와 같다고 주장하지 않는다.**  기존 A3 예측이 어느 branch/깊이에
의존하는지 진단할 뿐이다.  occlusion 은 인과 증명이 아니라 후보 탐색이다.

## 방향 코사인
`d_s = mean_FP(source=s) - mean_TP(source=s)` 의 source 쌍별 코사인.
+0.5 이상 / 0 근처 / 음수 라는 표현은 판정 규칙이 아니라 설명용 기준일 뿐이다.

## 전이 행렬
한 source 에서만 학습/검증하고 세 source 의 test 부분에 평가했다.  A0 와 A3 만, seed 0.
분할은 기존 manifest 재사용.

## 연구 데이터 상태
WildJailbreak / PromptShield / Question Set 은 이미 architecture 개발, C1 평가, LOSO 분석,
근본원인 진단에 사용됐다.  따라서 최종 논문에서 이 셋을 **"untouched final external test"
라고 부를 수 없다.**  development / diagnostic benchmark 다.  최종 OOD 배포 주장은 완전히
새로운 external benchmark 와 post-freeze 평가로 따로 검증해야 한다.
