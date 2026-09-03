"""Stage 8 — deanonymize.

Substitutes the placeholder tokens (`<EMAIL_ADDRESS_1>`, ...) in the LLM's
response back to their original PII values, using the mapping built in
Stage 3.

We do this without going through Presidio's DeanonymizeEngine because it
requires us to know the byte offsets of each token in the LLM's output —
locating them by string search and using simple `str.replace` is both
faster and more robust here.
"""

from __future__ import annotations

from proxy.schemas import (
    Stage3PresidioAnonymizeResult,
    Stage6LLMCallResult,
    Stage7OutputScanResult,
    Stage8DeanonymizeResult,
)


def run_stage_8(
    s3: Stage3PresidioAnonymizeResult,
    s6: Stage6LLMCallResult,
    s7: Stage7OutputScanResult,
) -> Stage8DeanonymizeResult:
    # If output scan masked the text, work on the masked version (we don't
    # want to deanonymize tokens inside leaked-PII context). Otherwise the
    # raw LLM text.
    text = s7.masked_text if s7.masked and s7.masked_text else (s6.response.message.content or "")

    replaced = 0
    # Replace longest tokens first so `<EMAIL_ADDRESS_10>` isn't shadowed
    # by `<EMAIL_ADDRESS_1>`.
    for token in sorted(s3.mapping.keys(), key=len, reverse=True):
        if token in text:
            text = text.replace(token, s3.mapping[token])
            replaced += 1

    return Stage8DeanonymizeResult(deanonymized_text=text, tokens_replaced=replaced)
