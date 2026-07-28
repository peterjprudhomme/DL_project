"""
Prepare a labeled prompt/response dataset for linear-probe training.

Assembles a harmful prompt pool from HarmBench + AdvBench, generates a model
response for each prompt via the configured LLM, classifies every response as
refused / not-refused, and combines the result with a curated benign set. The
three buckets (harmful-refused, harmful-not-refused, benign) are each capped at
TARGET_COUNT on a best-effort basis and written to a per-model, human-readable
JSON file at ``data/processed/{model_slug}/metadata.json``.

Usage:
    python -m data.prepare_dataset        # from the DL_project root
    python data/prepare_dataset.py

Note: HarmBench is gated on the Hub -> run `hf auth login` first.
"""

import json
import re
from pathlib import Path

from tqdm import tqdm

from data.dataset import HarmBenchDataset, AdvBenchDataset, CustomPromptDataset
from src.llm import LLMInterface
from src.config import MODEL_NAME
from src.extractor import ActivationExtractor


# Desired number of records per bucket (harmful-refused, harmful-not-refused, benign).
TARGET_COUNT = 200


def model_slug(model_name: str) -> str:
    """Filesystem-safe identifier derived from MODEL_NAME.

    Replaces path separators and any other non-safe characters with ``_`` and
    strips leading/trailing underscores. Deterministic and idempotent.

    'Qwen/Qwen2.5-1.5B-Instruct' -> 'Qwen_Qwen2.5-1.5B-Instruct'
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")


def classify_refusal(response: str, extractor: ActivationExtractor) -> bool:
    """Classify a model response as a refusal.

    Delegates to the extractor's phrase-based detector, which owns the full
    refusal phrase list. Empty or whitespace-only responses contain none of the
    phrases and therefore classify as not-refused.

    Always returns a native Python ``bool``.
    """
    return bool(extractor.label_refusal(response))


def make_row(prompt, response, harmful, refused):
    """Build the internal row dict for a single prompt/response record.

    The ``harmful`` and ``refused`` flags are coerced to ints (0/1) so the row
    is JSON-integer compatible with the downstream ProbeDataset reader.
    """
    return {
        "prompt": prompt,
        "response": response,
        "harmful": int(harmful),
        "refused": int(refused),
    }


def cap_bucket(name, rows, target=TARGET_COUNT):
    """Cap a bucket to ``target`` records on a best-effort basis.

    Returns the first ``target`` rows when the bucket has at least ``target``
    records. Otherwise prints a shortfall warning naming the bucket with its
    ``available/target`` counts and returns all available rows.
    """
    if len(rows) >= target:
        return rows[:target]
    print(f"WARNING: bucket '{name}' has {len(rows)}/{target} records; "
          f"writing all {len(rows)} available.")
    return rows


def build_harmful_pool():
    """Assemble the combined harmful prompt pool.

    Loads records from ``HarmBenchDataset`` and ``AdvBenchDataset`` (each
    ``get_records()`` returns a ``list[PromptRecord]``) and concatenates them
    into a single list. Every record in both sources already carries
    ``category="harmful"``; the ``harmful=1`` flag is applied later during row
    construction, so this function simply returns the combined list.
    """
    hb = HarmBenchDataset().get_records()
    ab = AdvBenchDataset().get_records()
    return hb + ab


def process_harmful(pool, llm, extractor, target):
    """Generate + classify responses for the harmful prompt pool.

    Iterates ``pool`` (a list of ``PromptRecord``), calling ``llm.generate`` once
    per prompt and routing each response via ``classify_refusal`` into one of two
    harmful buckets: the refused bucket (``harmful=1, refused=1``) or the
    not-refused bucket (``harmful=1, refused=0``). Every harmful record therefore
    lands in exactly one bucket, decided solely by the classifier.

    The loop early-stops once BOTH buckets have reached ``target`` records to
    avoid unnecessary model calls. Progress is reported via ``tqdm``.

    Returns ``(refused, not_refused)`` lists of row dicts built by ``make_row``.
    """
    refused, not_refused = [], []
    for rec in tqdm(pool, desc="Harmful prompts"):
        if len(refused) >= target and len(not_refused) >= target:
            break                       # early stop once both buckets full
        response = llm.generate(rec.prompt)
        if classify_refusal(response, extractor):
            refused.append(make_row(rec.prompt, response, harmful=1, refused=1))
        else:
            not_refused.append(make_row(rec.prompt, response, harmful=1, refused=0))
    return refused, not_refused


def process_benign(llm, extractor, target):
    """Generate + classify responses for the benign prompt set.

    Selects every non-harmful record from ``CustomPromptDataset`` (those whose
    ``category != "harmful"``), then calls ``llm.generate`` once per prompt up to
    ``target`` records. The loop early-stops at the top once ``target`` rows have
    been collected to avoid unnecessary model calls. Each response is classified
    via ``classify_refusal`` and stored with ``harmful=0``. Progress is reported
    via ``tqdm``.

    Returns a list of row dicts built by ``make_row``.
    """
    benign = [r for r in CustomPromptDataset().get_records()
              if r.category != "harmful"]
    rows = []
    for rec in tqdm(benign, desc="Benign prompts"):
        if len(rows) >= target:
            break                       # early stop once target reached
        response = llm.generate(rec.prompt)
        rows.append(make_row(rec.prompt, response, harmful=0,
                             refused=int(classify_refusal(response, extractor))))
    return rows


def write_metadata(slug, harmful_refused, harmful_not_refused, benign):
    """Write the combined dataset to a per-model human-readable JSON file.

    Concatenates the three buckets (harmful-refused, harmful-not-refused,
    benign), assigns each output record a fresh sequential integer ``id`` and
    casts ``harmful``/``refused`` to ints (0/1). Serializes the list of dicts to
    ``data/processed/{slug}/metadata.json`` with ``indent=2`` (human-readable)
    and ``ensure_ascii=False`` (legible prompts/responses); JSON string escaping
    handles commas, quotes, and newlines in prompts/responses automatically.

    Returns the list of record dicts that was written.
    """
    out_dir = Path("data/processed") / slug
    out_dir.mkdir(parents=True, exist_ok=True)     # create dir if missing
    all_rows = harmful_refused + harmful_not_refused + benign
    records = [
        {
            "id": i,                                # fresh sequential unique id
            "prompt": row["prompt"],
            "response": row["response"],
            "harmful": int(row["harmful"]),         # int 0/1
            "refused": int(row["refused"]),         # int 0/1
        }
        for i, row in enumerate(all_rows)
    ]
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return records


def main():
    """Entry point for the data-preparation pipeline.

    Wires the full pipeline: derive the per-model slug, load the LLM and
    activation extractor, assemble the harmful prompt pool, generate + classify
    responses for the harmful and benign sets, cap each of the three buckets to
    TARGET_COUNT, and write the per-model ``metadata.json``.
    """
    slug = model_slug(MODEL_NAME)

    llm = LLMInterface()
    extractor = ActivationExtractor(llm)

    pool = build_harmful_pool()
    harmful_refused, harmful_not_refused = process_harmful(
        pool, llm, extractor, TARGET_COUNT
    )
    benign = process_benign(llm, extractor, TARGET_COUNT)

    harmful_refused = cap_bucket("harmful_refused", harmful_refused, TARGET_COUNT)
    harmful_not_refused = cap_bucket(
        "harmful_not_refused", harmful_not_refused, TARGET_COUNT
    )
    benign = cap_bucket("benign", benign, TARGET_COUNT)

    print(
        f"Pool size: {len(pool)} | "
        f"harmful_refused: {len(harmful_refused)} | "
        f"harmful_not_refused: {len(harmful_not_refused)} | "
        f"benign: {len(benign)} | target: {TARGET_COUNT}"
    )

    write_metadata(slug, harmful_refused, harmful_not_refused, benign)
    print(f"Wrote dataset to data/processed/{slug}/metadata.json")


if __name__ == "__main__":
    main()
