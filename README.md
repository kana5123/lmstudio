# LM Studio 설정 (kana5123 서버)

리눅스 서버(node1)에 올린 LM Studio 의 **설정만** 담은 저장소입니다.
가중치·바이너리·로그·API 키는 담지 않습니다 (`.gitignore` 참고).

설치 위치: `~/.lmstudio` (2026-08-12 설치, 실제 크기 5.1GB)

## 담긴 것

| 파일 | 내용 |
|---|---|
| `settings.json` | 앱 설정 전체. 기본 문맥 길이 8192, 모델 저장 폴더, JIT 모델 수명 1시간 등 |
| `hub/models/qwen/qwen3-4b-2507/` | 쓰던 모델의 정의(`model.yaml`)와 의존성 목록(`manifest.json`) |

## 담지 않은 것과 그 이유

| 항목 | 크기 | 제외 이유 |
|---|---|---|
| `credentials/` | 12K | API 키. 공개 저장소에 올리면 안 됨 |
| `models/` | 2.4G | GGUF 가중치. 허깅페이스에서 다시 받으면 됨 |
| `extensions/` | 2.2G | 런타임 백엔드 |
| `llmster/` | 542M | 앱 번들 |
| `bin/lms` | 105M | GitHub 파일 한도 100MB 초과 |
| `server-logs/` | 4.4M | 실험 입력 데이터가 그대로 들어 있음 |

## 쓰던 모델

- **`qwen/qwen3-4b-2507`** — Qwen3 4B Instruct. 실제 파일은 4비트 양자화본
  `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`
  (양자화 = 가중치를 4비트로 줄여 용량·메모리를 아끼는 것)
- **`text-embedding-nomic-embed-text-v1.5`** — 문장을 벡터로 바꾸는 임베딩 모델

권장 생성 설정은 `model.yaml` 에 들어 있습니다: temperature 0.7 / top-k 20 / top-p 0.8.

## 새 서버에 재현하기

```bash
# 1. LM Studio 설치 후 이 저장소 내용을 ~/.lmstudio 에 덮어쓰기
git clone https://github.com/kana5123/lmstudio.git
cp -r lmstudio/settings.json lmstudio/hub ~/.lmstudio/

# 2. 모델 내려받기 (2.4GB)
~/.lmstudio/bin/lms get lmstudio-community/Qwen3-4B-Instruct-2507-GGUF

# 3. 데스크톱 앱 없이 API 서버만 띄우기
~/.lmstudio/bin/lms server start     # 기본 포트 1234
~/.lmstudio/bin/lms server status
```

OpenAI 호환 규격이라 그대로 호출됩니다:

```bash
curl -s http://127.0.0.1:1234/v1/models

curl -s -X POST http://127.0.0.1:1234/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen/qwen3-4b-2507","messages":[{"role":"user","content":"안녕"}]}'
```

## 사용 기록

서버 로그가 2026-08-12 ~ 08-27 까지 남아 있습니다 (저장소에는 미포함).
