"""rfpr 결과 → 논문 스타일 표 페이지."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = HERE / "guardrail_rfpr.html"

GEN, ENC = "생성형 LLM 가드", "경량 인코더 분류기"
JB = [("qwen3guard", "Qwen3Guard-Gen-4B", "4B", GEN),
      ("qwen3guard_punsafe", "Qwen3Guard-Gen-4B", "4B", GEN),
      ("llamafirewall", "Llama Prompt Guard 2", "86M", ENC),
      ("promptguard_v1", "Llama Prompt Guard 1", "86M", ENC),
      ("piguard", "PIGuard", "184M", ENC)]
PI = [("piguard", "PIGuard", "184M", ENC),
      ("llamafirewall", "Llama Prompt Guard 2", "86M", ENC),
      ("protectaiv2", "ProtectAI v2", "184M", ENC),
      ("deepset", "Deepset", "184M", ENC),
      ("promptguard_v1_strict", "Llama Prompt Guard 1", "86M", ENC),
      ("fmops", "Fmops", "67M", ENC)]
CFG = {"promptguard_v1": "jailbreak 라벨", "promptguard_v1_strict": "injection+jailbreak",
       "qwen3guard": "위험도 = 1−P(safe)", "qwen3guard_punsafe": "위험도 = P(unsafe)"}

SCOPE = [  # 모델, 인젝션, 제일브레이크, 근거
    ("Llama Prompt Guard 2", "O", "O", "모델 카드: “detect both prompt injection and jailbreaking attacks”"),
    ("PIGuard", "O", "O", "학습 데이터에 제일브레이크 3종 포함 (논문 Table 5)"),
    ("Llama Prompt Guard 1", "O", "O", "라벨이 분리됨 — injection=제3자 데이터용, jailbreak=사용자 대화용"),
    ("ProtectAI v2", "O", "X", "모델 카드: “It does not detect jailbreak attacks”"),
    ("Deepset · Fmops", "O", "X", "deepset/prompt-injections(영·독 인젝션)만으로 학습"),
    ("Qwen3Guard-Gen", "X", "O", "범주 10종에 Jailbreak 있음, 인젝션 항목 없음"),
    ("NemoGuard 8B", "X", "X", "콘텐츠 안전 23범주, 둘 다 없음"),
]


def load(bench, key, n_test):
    p = RES / f"rfpr_{bench}_{key}.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    return m if m.get("n_test") == n_test else None


CSS = """
:root{
  --paper:#FAFAFB; --surface:#FFFFFF; --ink:#14171C; --muted:#5C6472; --faint:#8B94A3;
  --rule:#E1E4EA; --hair:#C9CFD9; --accent:#0E6B66; --wash:#EDF4F3;
  --good:#2A7A4F; --weak:#B0453D; --mid:#B37A18;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#101318; --surface:#171B21; --ink:#E7EAEE; --muted:#98A1AF; --faint:#6C7686;
  --rule:#272D36; --hair:#39414D; --accent:#54B8B1; --wash:#152B2A;
  --good:#5FBF8C; --weak:#E0796F; --mid:#D9A441;}}
:root[data-theme="dark"]{
  --paper:#101318; --surface:#171B21; --ink:#E7EAEE; --muted:#98A1AF; --faint:#6C7686;
  --rule:#272D36; --hair:#39414D; --accent:#54B8B1; --wash:#152B2A;
  --good:#5FBF8C; --weak:#E0796F; --mid:#D9A441;}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);margin:0;font-size:16px;line-height:1.65;
  padding:clamp(1.5rem,4vw,4rem) 1.25rem 6rem;
  font-family:"IBM Plex Sans KR",system-ui,sans-serif}
.wrap{max-width:60rem;margin:0 auto;display:flex;flex-direction:column;gap:3rem}
.prose{max-width:43rem}
h1,h2,h3{font-family:Archivo,system-ui,sans-serif;margin:0;line-height:1.2;text-wrap:balance}
h1{font-size:clamp(1.8rem,4.2vw,2.7rem);font-weight:700;letter-spacing:-.02em}
h2{font-size:1.3rem;font-weight:600;letter-spacing:-.01em}
h3{font-size:.95rem;font-weight:600}
p{margin:0}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:.7rem;letter-spacing:.16em;
  text-transform:uppercase;color:var(--accent);margin:0 0 .85rem}
.lede{color:var(--muted);font-size:1.04rem;margin-top:.9rem}
header{border-bottom:1px solid var(--rule);padding-bottom:1.8rem}
.meta{display:flex;flex-wrap:wrap;gap:.45rem;margin-top:1.4rem}
.chip{font-family:"IBM Plex Mono",monospace;font-size:.71rem;color:var(--muted);
  border:1px solid var(--rule);border-radius:2px;padding:.26rem .55rem;background:var(--surface)}
section{display:flex;flex-direction:column;gap:1rem}
.scroll{overflow-x:auto}
/* 논문 표: 세로줄 없음, 가로 규칙선만 */
table.paper{border-collapse:collapse;width:100%;min-width:40rem;
  font-size:.87rem;background:none}
table.paper caption{caption-side:top;text-align:left;padding:0 0 .7rem;
  font-size:.85rem;color:var(--muted)}
table.paper thead tr:first-child th{border-top:2px solid var(--ink)}
table.paper thead tr:last-child th{border-bottom:1px solid var(--ink)}
table.paper th,table.paper td{padding:.42rem .8rem;text-align:right;
  font-variant-numeric:tabular-nums;white-space:nowrap}
table.paper th:first-child,table.paper td:first-child{text-align:left;padding-left:0}
table.paper td:last-child,table.paper th:last-child{padding-right:0}
table.paper thead th{font-weight:600;font-size:.78rem;color:var(--ink);
  font-family:"IBM Plex Sans KR",sans-serif}
table.paper thead th.grp{border-bottom:1px solid var(--hair);text-align:center}
table.paper thead th.spacer{border-bottom:none}
table.paper tbody td{font-family:"IBM Plex Mono",monospace;color:var(--ink)}
table.paper tbody td:first-child{font-family:"IBM Plex Sans KR",sans-serif}
table.paper tbody tr:last-child td{border-bottom:2px solid var(--ink)}
table.paper td.best{font-weight:700}
table.paper tr.grouprow td{font-family:"IBM Plex Sans KR",sans-serif;font-size:.72rem;
  letter-spacing:.1em;text-transform:uppercase;color:var(--accent);font-weight:600;
  padding:.85rem 0 .3rem;text-align:left;border-bottom:1px solid var(--rule)}
table.paper tr.grouprow:first-child td{padding-top:.5rem}
table.paper td.mdl{padding-left:.9rem}
table.paper .sep{border-left:1px solid var(--rule)}
table.paper thead th.grp.sep{border-left:1px solid var(--hair)}
.cfg{color:var(--faint);font-size:.76rem;font-family:"IBM Plex Mono",monospace}
.sz{color:var(--faint);font-size:.76rem;font-family:"IBM Plex Mono",monospace}
.over{color:var(--weak)}
.note{font-size:.83rem;color:var(--muted);max-width:43rem}
table.scope{border-collapse:collapse;width:100%;min-width:38rem;font-size:.85rem}
table.scope th,table.scope td{padding:.5rem .7rem;border-bottom:1px solid var(--rule);
  text-align:left;vertical-align:top}
table.scope thead th{border-top:2px solid var(--ink);border-bottom:1px solid var(--ink);
  font-size:.78rem;font-weight:600}
table.scope td.yn{text-align:center;font-family:"IBM Plex Mono",monospace;font-weight:600;width:5rem}
table.scope td.y{color:var(--good)} table.scope td.n{color:var(--weak)}
table.scope td.src{color:var(--muted);font-size:.8rem}
.grid{display:grid;gap:1rem;grid-template-columns:1fr}
@media(min-width:48rem){.grid{grid-template-columns:repeat(2,1fr)}}
.card{border:1px solid var(--rule);border-radius:3px;background:var(--surface);
  padding:1.05rem;display:flex;flex-direction:column;gap:.45rem}
.card p{font-size:.86rem;color:var(--muted)}
dl{margin:0;display:grid;gap:.5rem .85rem;grid-template-columns:auto 1fr;font-size:.85rem}
dt{font-family:"IBM Plex Mono",monospace;font-size:.72rem;color:var(--muted);white-space:nowrap;padding-top:.15rem}
dd{margin:0}
code{font-family:"IBM Plex Mono",monospace;font-size:.84em;background:var(--wash);
  padding:.06em .32em;border-radius:2px}
footer{border-top:1px solid var(--rule);padding-top:1.4rem;color:var(--muted);font-size:.82rem}
"""


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def jb_table():
    rows = [(k, n, sz, g, load("jailbreak", k, 4000)) for k, n, sz, g in JB]
    rows = [r for r in rows if r[4]]
    b1 = max(m["recall@1pct"] for *_, m in rows)
    b01 = max(m["recall@0.1pct"] for *_, m in rows)
    body, last = [], None
    for k, n, sz, grp, m in rows:
        if grp != last:
            body.append(f'<tr class="grouprow"><td colspan="8">{grp}</td></tr>')
            last = grp
        cfg = f' <span class="cfg">{CFG[k]}</span>' if k in CFG else ""
        nf = m["native_fpr"]
        nfc = ' class="over"' if nf > 0.05 else ""
        over = ' class="over"' if m["achieved_fpr@1pct"] > 0.011 else ""
        body.append(
            f'<tr><td class="mdl">{esc(n)}{cfg} <span class="sz">{sz}</span></td>'
            f'<td>{m["native_recall"]:.3f}</td><td{nfc}>{nf*100:.2f}</td>'
            f'<td>{m["native_f1"]:.3f}</td>'
            f'<td class="sep {"best" if m["recall@1pct"]==b1 else ""}">{m["recall@1pct"]:.3f}</td>'
            f'<td{over}>{m["achieved_fpr@1pct"]*100:.2f}</td>'
            f'<td class="sep {"best" if m["recall@0.1pct"]==b01 else ""}">{m["recall@0.1pct"]:.3f}</td>'
            f'<td>{m["latency_mean_ms"]:.0f}</td></tr>')
    return f"""<table class="paper">
<caption>표 1 &middot; 제일브레이크 탐지. JailbreaksOverTime, 검증 2,000 / 시험 4,000 (공격 704 / 정상 3,296).
<b>기본 판정</b>은 제조사 출고 설정 그대로, <b>FPR 고정</b>은 검증셋에서 목표 오탐율에 맞춘 차단선.</caption>
<thead>
<tr><th class="spacer"></th><th class="grp" colspan="3">기본 판정 (제조사 설정)</th>
    <th class="grp sep" colspan="2">FPR 1% 고정</th>
    <th class="grp sep">FPR 0.1%</th><th class="spacer"></th></tr>
<tr><th>모델</th><th>Recall</th><th>FPR %</th><th>F1</th>
    <th class="sep">Recall</th><th>달성 FPR %</th><th class="sep">Recall</th><th>지연 ms</th></tr>
</thead><tbody>{''.join(body)}</tbody></table>"""


def pi_table():
    rows = [(k, n, sz, g, load("piarena", k, 800)) for k, n, sz, g in PI]
    rows = [r for r in rows if r[4]]
    best = max(m["recall@1pct"] for *_, m in rows)
    body, last = [], None
    for k, n, sz, grp, m in rows:
        if grp != last:
            body.append(f'<tr class="grouprow"><td colspan="7">{grp}</td></tr>')
            last = grp
        cfg = f' <span class="cfg">{CFG[k]}</span>' if k in CFG else ""
        nf = m["native_fpr"]
        nfc = ' class="over"' if nf > 0.05 else ""
        over = ' class="over"' if m["achieved_fpr@1pct"] > 0.011 else ""
        body.append(
            f'<tr><td class="mdl">{esc(n)}{cfg} <span class="sz">{sz}</span></td>'
            f'<td>{m["native_recall"]:.3f}</td><td{nfc}>{nf*100:.2f}</td>'
            f'<td>{m["native_f1"]:.3f}</td>'
            f'<td class="sep {"best" if m["recall@1pct"]==best else ""}">{m["recall@1pct"]:.3f}</td>'
            f'<td{over}>{m["achieved_fpr@1pct"]*100:.2f}</td>'
            f'<td>{m["latency_mean_ms"]:.0f}</td></tr>')
    return f"""<table class="paper">
<caption>표 2 &middot; 프롬프트 인젝션 탐지. PIArena direct, 512토큰 이하만, 검증 700 / 시험 800 (공격 400 / 정상 400).
FPR 0.1%는 검증 음성이 350건이라 측정 불가. 생성형 LLM 가드는 인젝션 범주 자체가 없어 해당 없음.</caption>
<thead>
<tr><th class="spacer"></th><th class="grp" colspan="3">기본 판정 (제조사 설정)</th>
    <th class="grp sep" colspan="2">FPR 1% 고정</th><th class="spacer"></th></tr>
<tr><th>모델</th><th>Recall</th><th>FPR %</th><th>F1</th>
    <th class="sep">Recall</th><th>달성 FPR %</th><th>지연 ms</th></tr>
</thead><tbody>{''.join(body)}</tbody></table>"""


def scope_table():
    body = []
    for name, inj, jb, src in SCOPE:
        f = lambda v: f'<td class="yn {"y" if v=="O" else "n"}">{"지원" if v=="O" else "미지원"}</td>'
        body.append(f'<tr><td>{esc(name)}</td>{f(inj)}{f(jb)}<td class="src">{esc(src)}</td></tr>')
    return f"""<table class="scope">
<thead><tr><th>모델</th><th class="yn">인젝션</th><th class="yn">제일브레이크</th>
<th>근거</th></tr></thead><tbody>{''.join(body)}</tbody></table>"""


def build():
    html = f"""<title>ETRI 가드레일 벤치마크</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+KR:wght@400;500;600&display=swap">
<style>{CSS}</style>
<div class="wrap">

<header class="prose">
  <p class="eyebrow">ETRI · 프롬프트 공격 탐지기 평가</p>
  <h1>같은 표에 놓을 수 없는 두 가지 공격</h1>
  <p class="lede">프롬프트 인젝션과 제일브레이킹은 다른 태스크이고, 탐지기마다 잡겠다고 밝힌 범위가 다릅니다.
  두 축을 나눠 재면 <b>어느 한 모델도 양쪽 1등이 아닙니다</b> — 제일브레이크는 Qwen3Guard와 Prompt Guard 2가,
  인젝션은 PIGuard가 각각 앞섭니다.</p>
  <div class="meta">
    <span class="chip">Recall @ 고정 FPR</span>
    <span class="chip">임계값은 검증셋에서만 선택</span>
    <span class="chip">RTX A6000 · 배치 1</span>
    <span class="chip">seed 0 · 점수 전량 보존</span>
  </div>
</header>

<section>
  <h2>지표</h2>
  <p class="prose">탐지기마다 제조사가 정한 차단선이 제각각이라, 그대로 비교하면 모델 실력이 아니라
  차단선 위치를 비교하게 됩니다. 그래서 <b>모든 모델의 차단선을 “정상 100건당 1건만 막는” 지점에
  똑같이 맞춰 놓고</b>, 그 상태의 Recall(공격을 몇 % 잡는가)을 잽니다.</p>
  <p class="note">차단선은 검증셋에서만 고르고 시험셋에는 그대로 적용합니다. 그래서 시험셋에서 실제로 나온
  오탐율(<b>달성 FPR</b>)은 목표 1%에서 벗어날 수 있고, 예산을 넘긴 값은 <span class="over">붉게</span> 표시했습니다 —
  그 행의 Recall 은 그만큼 부풀려진 값입니다. 이 지표는 Meta가 Prompt Guard 2 모델 카드에서 쓰는 대표 지표와 같습니다.</p>
  <p class="note"><b>기본 판정</b> 열은 제조사가 출고한 설정 그대로 쓴 결과입니다 — 분류기는 최대 확률 라벨,
  LlamaFirewall은 프레임워크 기본 차단선 0.9(<code>prompt_guard_scanner.py:22</code>),
  Qwen3Guard는 세 등급 중 최댓값이 <code>Unsafe</code>일 때만 차단. 모델마다 차단선이 다르므로
  <b>이 열끼리의 비교는 모델 실력이 아니라 차단선 위치 비교</b>입니다. 오른쪽 FPR 고정 열이
  그 차이를 제거한 값이고, 두 열의 간극이 곧 “기본 설정이 얼마나 잘 맞춰져 있는가”입니다.</p>
</section>

<section>
  <h2>제일브레이크 탐지</h2>
  <div class="scroll">{jb_table()}</div>
  <p class="note">엄격한 차단선(0.1%)에서 순위가 뒤집힙니다. Prompt Guard 2만 0.493을 유지하고
  나머지는 무너집니다 — PIGuard 0.001. Meta 모델 카드는 그 이유를
  “점수 순위를 안정시키는 맞춤 손실함수” 때문이라고 밝히고 있습니다.
  실제로 공격 704건 중 점수 0.999를 넘는 것이 Prompt Guard 2는 631건인데 Qwen3Guard는 100건뿐입니다.</p>
  <p class="note">Qwen3Guard는 등급이 셋(<code>Safe</code>/<code>Controversial</code>/<code>Unsafe</code>)이라
  위험도를 어떻게 정의하느냐로 결과가 갈립니다. 중간 등급 <code>Controversial</code>을 위험도에 더하면
  정상 문장 점수까지 같이 올라가 상위권이 오염됩니다 — 0.1% 차단선에서 0.195가 0.036으로 떨어집니다.
  1% 지점에서는 차이가 없어(0.942 대 0.940) 정의 선택이 드러나지 않습니다. 두 정의를 모두 실었습니다.</p>
</section>

<section>
  <h2>프롬프트 인젝션 탐지</h2>
  <div class="scroll">{pi_table()}</div>
  <p class="note">PIGuard가 2위를 두 배 이상 앞섭니다. Prompt Guard 1은 인젝션 표에서는
  <code>injection+jailbreak</code> 설정을 씁니다 — 모델 카드가 제3자 데이터에는 두 라벨을 다 켜라고 지정하기 때문입니다.
  같은 모델을 사용자 프롬프트에 그 설정으로 쓰면 정상 문장까지 전부 막힙니다.</p>
</section>

<section>
  <h2>왜 모델마다 표가 다른가</h2>
  <p class="prose">두 공격은 들어오는 경로가 다릅니다. 인젝션은 문서·검색결과 같은
  <b>제3자 데이터에 숨어</b> 들어오고, 제일브레이크는 <b>사용자가 직접</b> 칩니다.
  탐지기들도 그에 맞춰 서로 다른 범위를 잡겠다고 선언했습니다.</p>
  <div class="scroll">{scope_table()}</div>
  <p class="note">각 표에는 제조사가 그 태스크를 지원한다고 밝힌 모델만 넣었습니다.
  ProtectAI v2를 제일브레이크 표에 넣으면 “안 하겠다고 한 것을 못 했다”를 재게 됩니다.</p>
</section>

<section>
  <h2>실험 설정</h2>
  <div class="grid">
    <div class="card"><h3>데이터</h3><dl>
      <dt>제일브레이크</dt><dd>JailbreaksOverTime (AISec 2025, CC-BY-4.0).
        프롬프트 중복 제거 22,180행을 70:30으로 나눈 뒤 층화 추출.
        공격 비율 17.6% 유지</dd>
      <dt>인젝션</dt><dd>PIArena (MIT) 16개 서브셋. 공식 <code>direct</code> 주입 코드를 그대로 호출해
        공격본을 만들고, 같은 문서의 주입 전 문맥을 정상본으로 짝지음</dd>
      <dt>누수 검사</dt><dd>두 벤치마크 모두 검증셋∩시험셋 = 0건 확인</dd>
    </dl></div>
    <div class="card"><h3>왜 512토큰 이하만 쓰나</h3>
      <p>분류기 5종 전부 <code>max_position_embeddings=512</code>가 구조적 한계입니다.
      그보다 긴 문서는 주입 문장이 잘려나가 모델이 <b>볼 기회조차 없습니다</b> —
      PIArena 전체로 재면 공격의 36.7%가 그렇습니다. 그 상태의 재현율은 탐지 능력이 아니라
      입력 길이 제한을 잰 값이라 길이를 맞췄습니다.</p></div>
    <div class="card"><h3>생성형 모델의 점수</h3>
      <p>Qwen3Guard는 <code>Safety: Safe/Unsafe/Controversial</code> 텍스트를 뱉을 뿐 점수가 없습니다.
      차단선을 쓸려면 연속값이 필요해, 판정 단어가 나오는 자리에서 그 토큰들의 확률을 읽어
      위험도로 씁니다(= 1 − P(Safe)). ShieldGemma 모델 카드가 공식으로 쓰는 방식과 같은 원리입니다.</p></div>
    <div class="card"><h3>외부 대조</h3>
      <p>같은 프로토콜을 2026년 8월 초 경량 분류모델 12종 비교에 적용한 기록이 있고,
      Prompt Guard 2가 그때 0.9091 / 이번 0.922, 구버전이 그때 0.0000 / 이번 0.000으로
      재현됐습니다. PIGuard의 0.1% 붕괴(그때 0.0028, 이번 0.001)도 같이 재현됐습니다.</p></div>
  </div>
</section>

<section>
  <h2>이 표가 말하지 않는 것</h2>
  <div class="grid">
    <div class="card"><h3>긴 문서</h3>
      <p>실제 RAG·문서요약은 수천~수만 토큰입니다. 512 제한 때문에 별도 대책(창 분할 등)이 필요하고,
      전체 서브셋으로 재면 최고 성능이 0.792에서 0.336까지 떨어집니다.</p></div>
    <div class="card"><h3>Meta 수치와의 비교 불가</h3>
      <p>Prompt Guard 2 모델 카드의 재현율 97.5%는 <b>비공개 벤치마크</b> 값입니다.
      여기 0.922와 직접 비교할 수 없습니다.</p></div>
    <div class="card"><h3>점수 포화</h3>
      <p>세 모델 모두 공격 점수가 1.0 근처에 몰려 있어 차단선이 소수점 셋째 자리에서 결정됩니다.
      배치 크기를 바꾸면 개별 점수가 최대 0.012 흔들려, 전체를 배치 1로 다시 계산해 확인했습니다 —
      Qwen3Guard 재현율 0.940 대 0.942로 결론은 바뀌지 않았습니다.</p></div>
    <div class="card"><h3>에이전트 환경</h3>
      <p>Meta는 인젝션 평가에 AgentDojo를 쓰고, 지표도 탐지율이 아니라
      실제 에이전트 피해 감소량입니다. 그 축은 아직 재지 않았습니다.</p></div>
  </div>
</section>

<footer class="prose">
  모든 수치는 <code>guardrail_eval/results/rfpr_*.json</code>에서 다시 계산됩니다.
  모델별 원본 점수와 출력 문자열은 같은 폴더의 <code>*_scores.jsonl</code>에 전량 보존돼 있습니다.
</footer>
</div>
"""
    OUT.write_text(html, encoding="utf-8")
    print(f"{OUT}  ({len(html):,} bytes)")


if __name__ == "__main__":
    build()
