# 라벨 매핑 (Label Mapping)

각 원본 라벨을 정식 분류(canonical taxonomy)로 옮긴 근거를 전부 적는다.
**데이터셋마다 같은 단어가 다른 뜻이므로 무조건 이진 매핑하지 않았다.**

## 0. 정식 분류

| 정식 라벨 | 뜻 |
|---|---|
| `BENIGN` | 평범한 정상 요청 |
| `HARMFUL_DIRECT` | 유해하지만 **조작·우회 기법이 아님** (그냥 대놓고 요청) |
| `JAILBREAK_ADVERSARIAL` | 모델 제약을 우회하려는 탈옥 프롬프트 |
| `PROMPT_INJECTION` | 사용자 입력에 새 지시를 주입 |
| `INDIRECT_PROMPT_INJECTION` | 제3자 문서/도구 출력에 지시를 심음 |
| `HARD_NEGATIVE_JAILBREAK_LIKE` | 정상인데 탈옥처럼 보임 |
| `HARD_NEGATIVE_INJECTION_LIKE` | 정상인데 주입처럼 보임 |
| `UNKNOWN` | 확인 못 함 |

## 1. MAIN 이진 목표

```
양성(UNSAFE) = JAILBREAK_ADVERSARIAL | PROMPT_INJECTION | INDIRECT_PROMPT_INJECTION
음성(SAFE)   = BENIGN | HARD_NEGATIVE_JAILBREAK_LIKE | HARD_NEGATIVE_INJECTION_LIKE
제외         = HARMFUL_DIRECT, UNKNOWN   (보존은 하되 MAIN 실험에서 뺀다)
```

`HARMFUL_DIRECT` 를 자동으로 양성 처리하지 않았다. PromptGuard 2 는 유해성 분류기가 아니라
**조작(injection/jailbreak) 탐지기**이므로, "폭탄 만드는 법 알려줘" 같은 직접 유해 요청을
공격 정답으로 두면 모델이 겨냥하지 않는 것을 재는 셈이 된다.

## 2. PIGuard `train.json` (source 필드 보존, 76,735건)

원본 `label` 은 코드상 `0 = 정상`, `1 = 주입/공격` 이다.

| source | label 0 → | label 1 → | 근거 |
|---|---|---|---|
| `BIPIA` | `BENIGN` | `INDIRECT_PROMPT_INJECTION` | 정상은 깨끗한 코드/표 **문서**, 공격은 **같은 문서 + 심어진 지시**. 실측: 공격문 200건 중 192건이 정상 문서를 부분문자열로 포함 → 짝 구조 |
| `TaskTracker` | `BENIGN` | `INDIRECT_PROMPT_INJECTION` | 정상은 NLP 과제 지시문, 공격은 같은 문서에 `<!-- Directive: ... -->` 류가 삽입됨 |
| `Question Set` | `HARD_NEGATIVE_JAILBREAK_LIKE` | `JAILBREAK_ADVERSARIAL` | 정상 쪽이 평범한 문장이 아니라 **역할극 프롬프트**(GameGPT, 콘텐츠 작성자 페르소나)다. 탈옥과 형식이 닮아 hard negative 로 본다 |
| `jailbreak-classification` | `BENIGN` | `JAILBREAK_ADVERSARIAL` | 공격은 `Ignore all previous instructions...` 류. 정상은 페르소나+일반 QA 혼합 |
| `prompt-injections` (deepset) | `BENIGN` | `PROMPT_INJECTION` | 공격이 `Nun folgen neue Anweisungen` 처럼 **직접 지시 주입**. 독일어 다수 → 언어 `multi` |
| `safe-guard-prompt-injection` | `BENIGN` | `JAILBREAK_ADVERSARIAL` | 공격이 DAN 류 탈옥 + 협박형 혼합 |
| `hackaprompt-dataset` | — | `PROMPT_INJECTION` | 공격 전용 |
| `InjecAgent` | — | `INDIRECT_PROMPT_INJECTION` | 도구 출력 경유 공격 |
| `StruQ` | — | `PROMPT_INJECTION` | 공격 전용 |
| `ChatGPT-Jailbreak-Prompts`, `vigil-jailbreak-ada-002` | — | `JAILBREAK_ADVERSARIAL` | 탈옥 모음 |
| `Prompt-Injection-Mixed-Techniques` | — | `PROMPT_INJECTION` | 공격 전용 |
| `LLM Augmented set` | — | **`UNKNOWN`** | 어떤 절차로 증강했는지 확인 못 함 → MAIN 제외 |
| `Alpaca`, `chatbot_instruction_prompts`, `open-instruct`, `ultrachat_200k`, `no_robots`, `grok-conversation-harmless` | `BENIGN` | — | 일반 지시/대화 말뭉치 |
| `awesome-chatgpt-prompts` | `HARD_NEGATIVE_JAILBREAK_LIKE` | — | 페르소나 지정 프롬프트라 탈옥과 형식이 닮음 |
| `over-defense` | `HARD_NEGATIVE_JAILBREAK_LIKE` | — | 과잉거부 유도용 정상문 |
| `xtest-v2-copy` | `HARD_NEGATIVE_JAILBREAK_LIKE` | — | XSTest 안전 프롬프트(위험해 *보이는* 정상문) |

## 3. PIGuard `valid.json` (144건, 평가용 소규모)

`PINT_*`, `NotInject_*`, `BIPIA_*`, `WildGuard` 하위세트. 이름 규칙대로 매핑했고
`BIPIA_*` → `INDIRECT_PROMPT_INJECTION`, `NotInject_*` → `HARD_NEGATIVE_INJECTION_LIKE`,
`PINT_*jailbreak` → `JAILBREAK_ADVERSARIAL`, `PINT_*injection` → `PROMPT_INJECTION` 로 두었다.
표본이 8~16건씩이라 **방향 학습에는 쓸 수 없고** 참고용이다.

## 4. NotInject (leolee99/NotInject, 3개 하위세트 339건)

전부 `HARD_NEGATIVE_INJECTION_LIKE`. 주입처럼 보이지만 정상인 프롬프트로 설계된
과잉거부 측정용 벤치마크다. 공격 라벨이 없다.

## 5. WildJailbreak (allenai/wildjailbreak)

공식 카드 방식(`delimiter='\t'`, `keep_default_na=False`)으로 읽으면 261,559행이고
`data_type` 은 4범주다. **범주당 10,000건을 seed 0 으로 표본**했다(선정 판단에는 충분).

| `data_type` | 정식 라벨 | 근거 |
|---|---|---|
| `vanilla_benign` | `BENIGN` | 평범한 정상 요청 |
| `vanilla_harmful` | **`HARMFUL_DIRECT`** | 유해하지만 적대적 기법이 아님 → **MAIN 제외** |
| `adversarial_benign` | `HARD_NEGATIVE_JAILBREAK_LIKE` | 적대적 문체로 쓴 **정상** 요청 = 핵심 hard negative |
| `adversarial_harmful` | `JAILBREAK_ADVERSARIAL` | 적대적 문체 + 유해 의도 |

**`harmful` 을 자동으로 `jailbreak` 로 매핑하지 않았다.** `vanilla_harmful` 은 제외했고,
적대적 구성(construction)이 붙은 `adversarial_harmful` 만 양성으로 뒀다.

### source_group 을 생성 방식으로 묶은 이유

`wildjailbreak:adversarial` = `adversarial_benign` + `adversarial_harmful`.
둘은 같은 WildTeaming 절차로 만들어졌고 서로가 서로의 hard negative 다.
범주별로 쪼개면 각 그룹에 한쪽 라벨만 남아 같은 출처 내부 TP/FP 비교가 불가능해진다.

**단, 이 그룹은 §14 의 교란 위험이 남는다**: `adversarial_benign` 과 `adversarial_harmful`
은 여전히 서로 다른 생성 파이프라인 분기일 수 있고, 표현 방향이 "의도"가 아니라
"생성 분기"를 잡을 가능성이 있다. `source_subgroup_audit.csv` 에 TVD 로 정량화했다.

## 6. 확인 못 한 것

- `LLM Augmented set` 의 증강 절차 → `UNKNOWN` 처리
- PIGuard `train.json` 의 각 source 가 원본 데이터셋의 어느 split 에서 왔는지 → `UNKNOWN`
- PromptShield 학습 데이터(Alpaca/Dolly/SPP + FourAttacks) 공개 여부 → 별도 조사 중
