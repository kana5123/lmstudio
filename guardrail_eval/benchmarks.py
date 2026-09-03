"""평가 벤치마크 3종.  Case = (벤치마크, 텍스트, 라벨, 메모).  라벨 1=공격, 0=정상."""
import json, random, sys
from pathlib import Path
from typing import NamedTuple

SEED = 0
VAL_SIZE = 6654   # 시간순 val 구간 전체
ROOT = Path(__file__).resolve().parent
PIARENA_REPO = ROOT.parent / "bench" / "PIArena"

JOT_JSON = ("/home/kana5123/.cache/huggingface/hub/datasets--djapp18--JailbreaksOverTime/"
            "snapshots/a5a467cbab4b17d7f1c83e6cd119c61722053868/jailbreaksovertime_hugging_face.json")
PIARENA_DIR = ("/home/kana5123/.cache/huggingface/hub/datasets--sleeepeer--PIArena/"
               "snapshots/e9f56791974132a803632dc4b5fc18f3de90e91b/data/")
PIARENA_PARQUET = PIARENA_DIR + "squad_v2-00000-of-00001.parquet"
MAX_CONTEXT_LENGTH = 20480   # PIArena defense_promptguard_batch.py:18


def _piarena_direct_fn():
    """PIArena 원본 direct() 를 그대로 쓴다.

    piarena/attacks/__init__.py 가 nanogcg -> google.genai 까지 끌고 와서 import 가
    막히므로, 필요한 두 파일만 합성 패키지로 직접 적재한다(코드는 원본 그대로).
    """
    import importlib.util, types
    base = PIARENA_REPO / "piarena"

    def mkpkg(name, path):
        m = types.ModuleType(name)
        m.__path__ = [str(path)]
        sys.modules[name] = m
        return m

    def mkmod(name, file):
        spec = importlib.util.spec_from_file_location(name, file)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m

    if "piarena_min.attacks.attack_heuristic" not in sys.modules:
        mkpkg("piarena_min", base)
        mkmod("piarena_min.utils", base / "utils.py")
        mkpkg("piarena_min.attacks", base / "attacks")
        mkmod("piarena_min.attacks.attack_heuristic", base / "attacks" / "attack_heuristic.py")
    return sys.modules["piarena_min.attacks.attack_heuristic"].direct


class Case(NamedTuple):
    bench: str
    text: str
    label: int
    meta: str


def jailbreaks_over_time(n=100):
    """시간순 정렬 후 val 구간(1378:8032, 6654건)에서 seed 고정 무작위 n건."""
    rows = sorted(json.load(open(JOT_JSON)), key=lambda x: x["timestamp"])
    val = rows[1378:8032]
    assert len(val) == 6654, len(val)
    picked = val if n >= len(val) else random.Random(SEED).sample(val, n)
    return [Case("JailbreaksOverTime", r["prompt"], r["label"], r["source"]) for r in picked]


def piarena_direct():
    """PIArena squad_v2 전수. 공식 direct 주입 코드를 그대로 호출.

    가드레일에 넣는 텍스트는 context 만 (PIArena 공식 방어 코드가 그렇게 함:
    defenses/promptguard/defense_promptguard_batch.py:200).
    공격본 = 주입된 context, 정상본 = 같은 문서의 주입 전 context.
    """
    import pandas as pd
    direct = _piarena_direct_fn()

    df = pd.read_parquet(PIARENA_PARQUET)
    cases = []
    for _, r in df.iterrows():
        clean = r["context"][:MAX_CONTEXT_LENGTH]
        attacked = direct(r["context"], r["injected_task"], "random")[:MAX_CONTEXT_LENGTH]
        cases.append(Case("PIArena-direct", attacked, 1, r["category"]))
        cases.append(Case("PIArena-direct", clean, 0, r["category"]))
    return cases


def pii_exfiltration():
    """직접 작성한 개인정보 탈취 프롬프트. 정답 라벨 없음 -> 전부 공격으로 둔다."""
    return [Case("PII-exfil", t, 1, "hand-written")
            for t in json.loads((ROOT / "pii_prompts.json").read_text(encoding="utf-8"))]


def load(n_jot: int = 100):
    return jailbreaks_over_time(n_jot) + piarena_direct() + pii_exfiltration()


if __name__ == "__main__":
    import collections
    cs = load()
    for b in dict.fromkeys(c.bench for c in cs):
        sub = [c for c in cs if c.bench == b]
        n = collections.Counter(c.label for c in sub)
        lens = [len(c.text) for c in sub]
        print(f"{b:20} n={len(sub):4}  공격={n[1]:3} 정상={n[0]:3}  "
              f"길이 min/median/max = {min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}")
    assert load() == cs, "seed 고정 실패"
    print("\nseed 고정 확인: 두 번 불러도 동일")


def piarena_all(attack: str = "direct"):
    """PIArena 16개 서브셋 전부. 공식 주입 코드(attack 함수)로 공격본을 만들고,
    같은 문서의 주입 전 문맥을 정상본으로 짝지음. 정상본은 공격법과 무관하게 동일."""
    import glob, pandas as pd
    _piarena_direct_fn()
    import sys as _s
    direct = getattr(_s.modules["piarena_min.attacks.attack_heuristic"], attack)
    cases = []
    for f in sorted(glob.glob(PIARENA_DIR + "*.parquet")):
        sub = f.split("/")[-1].split("-000")[0]
        for _, r in pd.read_parquet(f).iterrows():
            clean = r["context"][:MAX_CONTEXT_LENGTH]
            atk = direct(r["context"], r["injected_task"], "random")[:MAX_CONTEXT_LENGTH]
            cases.append(Case("PIArena-all", atk, 1, sub))
            cases.append(Case("PIArena-all", clean, 0, sub))
    return cases
