"""Build a schema-preserving Korean JSONL and repair untranslated MT rows."""

import json
import re
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "safeguard_en.jsonl"
FIRST_PASS = HERE / "safeguard_ko.jsonl"
OUTPUT = HERE / "safeguard_ko_translated.jsonl"
HANGUL = re.compile(r"[가-힣]")
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")
MAX_INPUT_TOKENS = 700

MODEL_CACHE = Path(
    "/home/kana5123/.cache/huggingface/hub/"
    "models--NHNDQ--nllb-finetuned-en2ko"
)
BASE_SNAPSHOT = MODEL_CACHE / "snapshots/27078d9df1c66897e8ba8d8a00176b8a8c1a3fa1"
SAFE_WEIGHTS = (
    MODEL_CACHE
    / "snapshots/f5227be09374ffead4dddbee8a159dc7c812fa7e/model.safetensors"
)


def load_jsonl(path):
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def local_model_dir(temp_dir):
    """Combine the cached config snapshot with its separately cached safe weights."""
    model_dir = Path(temp_dir) / "model"
    model_dir.mkdir()
    for path in BASE_SNAPSHOT.iterdir():
        (model_dir / path.name).symlink_to(path.resolve())
    (model_dir / "model.safetensors").symlink_to(SAFE_WEIGHTS.resolve())
    return model_dir


def chunks(text, tokenizer):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= MAX_INPUT_TOKENS:
        return [text]

    result, current = [], ""
    for sentence in SENTENCE_BREAK.split(text):
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        candidate_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
        if len(candidate_ids) <= MAX_INPUT_TOKENS:
            current = candidate
            continue
        if current:
            result.append(current)
        sentence_ids = tokenizer(sentence, add_special_tokens=False)["input_ids"]
        if len(sentence_ids) > MAX_INPUT_TOKENS:
            for start in range(0, len(sentence_ids), MAX_INPUT_TOKENS):
                result.append(tokenizer.decode(sentence_ids[start : start + MAX_INPUT_TOKENS]))
            current = ""
        else:
            current = sentence
    if current:
        result.append(current)
    return result


def escape_jsonl_separators(serialized):
    # These are valid inside JSON strings, but Python str.splitlines() treats them as
    # record boundaries. Escaping them makes the file robust for line-oriented tools.
    return (
        serialized.replace("\u0085", "\\u0085")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def main():
    source = load_jsonl(SOURCE)
    translated = load_jsonl(FIRST_PASS)
    if len(source) != len(translated):
        raise ValueError("Source and first-pass record counts differ")
    if any(a["text_en"] != b["text_en"] for a, b in zip(source, translated)):
        raise ValueError("Source and first-pass records are not aligned")

    repair_indices = [
        i for i, row in enumerate(translated) if not HANGUL.search(row["text_ko"])
    ]
    print(f"Repairing {len(repair_indices)} untranslated rows", flush=True)

    with tempfile.TemporaryDirectory(prefix="nllb-merge.") as temp_dir:
        model_dir = local_model_dir(temp_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, src_lang="eng_Latn", local_files_only=True
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_dir,
            local_files_only=True,
            dtype=torch.float32,
            use_safetensors=True,
        ).eval()
        korean_bos = tokenizer.convert_tokens_to_ids("kor_Hang")

        pieces = []
        for row_index in repair_indices:
            for part_index, text in enumerate(chunks(source[row_index]["text_en"], tokenizer)):
                token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                pieces.append((row_index, part_index, text, token_count))
        pieces.sort(key=lambda item: item[3])

        repaired_parts = {}
        started = time.time()
        cursor = 0
        while cursor < len(pieces):
            longest = max(1, pieces[cursor][3])
            batch_size = min(8, max(1, 1400 // longest))
            batch = pieces[cursor : cursor + batch_size]
            encoded = tokenizer(
                [item[2] for item in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_INPUT_TOKENS,
            )
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    forced_bos_token_id=korean_bos,
                    max_new_tokens=min(900, int(longest * 1.6) + 64),
                    num_beams=1,
                    num_return_sequences=1,
                    no_repeat_ngram_size=4,
                )
            for item, output_ids in zip(batch, generated):
                repaired_parts[(item[0], item[1])] = tokenizer.decode(
                    output_ids, skip_special_tokens=True
                ).strip()
            cursor += len(batch)
            if cursor % 40 < len(batch) or cursor == len(pieces):
                print(
                    f"  {cursor}/{len(pieces)} pieces ({time.time() - started:.0f}s)",
                    flush=True,
                )

    for row_index in repair_indices:
        keys = sorted(key for key in repaired_parts if key[0] == row_index)
        translated[row_index]["text_ko"] = " ".join(repaired_parts[key] for key in keys)

    with OUTPUT.open("w", encoding="utf-8", newline="\n") as fh:
        for original, korean in zip(source, translated):
            row = {**original, "text_ko": korean["text_ko"]}
            serialized = json.dumps(row, ensure_ascii=False)
            fh.write(escape_jsonl_separators(serialized) + "\n")
    print(f"Wrote {len(source)} records to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
