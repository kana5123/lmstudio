# §1 — direction_repro 파이프라인 감사

보고된 수치를 **저장된 artifact 에서 재계산**해 검증했다. 12개 점검 중 주의 3건.
원본 결과는 삭제·수정하지 않았다.

## 결과 → 소스 매핑

| 보고된 결과 | 결과 파일 | 생성 스크립트/함수 | 정확한 수식 | 데이터 분할 |
|---|---|---|---|---|
| 내부 분할 간 cos | `results/direction_repro/internal_partition_cosine.csv` | `src/direction_repro/direction_reproducibility.py::fit_dir` | `v = norm(mean(g\|TP) − mean(g\|FP))`, `cos = <v_a, v_b>` | 각 분할의 `split=="train"` |
| 내부 전이 AUROC | `internal_transfer_auroc.csv` | 같은 파일 `project()` | `q = <v_a, g> − tau_a` | fit: a의 train / eval: b의 val+test |
| cross-dataset cos | `cross_dataset_cosine.csv` | 같은 파일 (B)절 | `cos(v_WJ, v_JOT)` | WJ train / JOT `ver_train` |
| cross-dataset 전이 | `cross_dataset_transfer_auroc.csv` | 같은 파일 | 위와 동일 | WJ val+test |
| 귀무 \|cos\| | `null_cosine_distribution.csv` | 같은 파일, 라벨 셔플 20회 | — | train |
| WJ 은닉표현 | `artifacts/direction_repro/hidden_*.pt` | `src/direction_repro/extract_hidden.py` | `h[:, l]`, best_window = argmax p_unsafe | — |
| JOT 방향 | `artifacts/directional_alignment/v_u.pt` | `src/directional/global_direction.py` | 동일 | JOT `ver_train` |

## 점검 결과

| # | 점검 | 판정 | 근거 |
|---|---|---|---|
| 1 | TP/FP 가 GT + PG2 hard prediction 인가 | **OK** | `confusion_cells.parquet` 에서 재계산해 **8915/8915 일치**. `gt` 필드가 예측이 아니라 정답 라벨임도 확인 |
| 2 | 방향 적합에 held-out 이 섞였는가 | **OK** | train 전용 방향으로 재계산한 cross-dataset cos 가 보고값과 **최대차 8.33e-17**. (전체 데이터로 적합했다면 불일치했을 것) |
| 3 | 분할 정의에 test 분포 정보가 쓰였는가 | **주의** | ★ `make_partitions` 가 held-out 포함 **전체 텍스트**로 TF-IDF/KMeans 를 적합. 길이 분할 기준도 전체 중앙값. 라벨을 안 쓰므로 AUROC 를 직접 부풀리진 않으나 **분할 정의가 test 분포를 본다** |
| 4 | duplicate/paired 가 split 을 넘나드는가 | **OK** | `group_key` 위반 **0개** |
| 5 | WJ split 별 TP/FP | **OK** | train TP 3886/FP 2413, val TP 826/FP 474, test TP 810/FP 506 |
| 6 | JOT `v_u.pt` 의 적합 표본 | **OK** | `fit_split=ver_train`, TP 2207 / FP 1380, `kind=GLOBAL-DIAGNOSTIC`. 은닉표현에서 재계산해 **cos = 1.0000000000** |
| 7 | 부호가 일관되게 TP−FP 인가 | **OK/주의** | 두 방향 모두 `mu_TP − mu_FP`. UNSAFE 가지에서는 이것이 CORRECT−INCORRECT 와 같다. **다만 SAFE 가지(TN/FN)를 이번 단계 전까지 아예 계산하지 않았다** — 그래서 "correctness" 주장을 검증할 수 없었다 |

## 주의 3건이 결과에 미치는 영향

- **#3 (분할 정의 누수)**: 비지도 정보라 AUROC 를 직접 올리지 못한다. 내부 분할 재현성
  결론(cos 0.75~0.99)의 방향은 바뀌지 않을 것으로 보이나, 엄격히는 train 통계로
  다시 정의해야 한다. **이번 단계의 결론(§4~§17)은 이 분할을 쓰지 않으므로 영향 없음.**
- **#7 (SAFE 가지 미계산)**: 이것이 핵심이었다. `delta_U` 만 보면 그것이 정오 신호인지
  정답 클래스 신호인지 **원리적으로 구별할 수 없다.** 이번 단계에서 `delta_S` 를
  계산하자마자 `cos(delta_U, delta_S) ≈ −1` 이 나와 정체가 드러났다.

파일: `results/direction_debug/pipeline_audit.csv`
