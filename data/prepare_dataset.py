"""
Prepare an unlabeled prompt/response dataset (plus activations) for probing.

Assembles a prompt pool from a harmful source (HarmBench + AdvBench) and a
curated benign source, generates a model response for each prompt via the
configured LLM, and extracts hidden-state activations for each prompt. The
result is written to a per-model directory under ``data/processed/{model_slug}/``
as two id-aligned artifacts:

    metadata.json   -> a JSON list of {id, prompt, response, harmful} objects
    activations.pt  -> a {id: {layer: cpu_tensor}} mapping

The ``harmful`` flag is the prompt's design-time ``expected_harmful`` label,
encoded as ``1`` (harmful) or ``0`` (not harmful / ambiguous), carried straight
through from the source dataset. The ``refused`` flag is a property of the
model's response and is still assigned by a separate downstream labeling script
that reads ``metadata.json``.

Usage:
    python -m data.prepare_dataset        # from the DL_project root
    python data/prepare_dataset.py

Note: HarmBench is gated on the Hub -> run `hf auth login` first.
"""

import json
from pathlib import Path

import torch
from tqdm import tqdm

from data.dataset import (
    HarmBenchDataset,
    AdvBenchDataset,
    CustomPromptDataset,
    JailbreakPromptDataset,
)
from src.llm import LLMInterface
from src.config import MODEL_NAME, LAYERS, model_slug


# Best-effort cap on how many prompts to draw from each source before running
# the model. Tune these to trade dataset size against compute. They only bound
# the pool; downstream labeling decides the final harmful/refused composition.
HARMFUL_COUNT = 400
BENIGN_COUNT = 200

# Jailbreak augmentation (opt-in). When INCLUDE_JAILBREAK is True, the
# jailbreak-wrapped harmful prompts from data/jailbreak_prompts.json are appended
# to the base pool. These supply the harmful+complied (HC) samples the plain pool
# is short on. The file is loaded via JailbreakPromptDataset, exactly like the
# benign set is loaded via CustomPromptDataset -- no code dependency on the
# generator (data/make_jailbreak_prompts.py), which is a one-time authoring tool.
# Set INCLUDE_JAILBREAK = False for the original 3-bucket behavior.
INCLUDE_JAILBREAK = True
JAILBREAK_COUNT = 400


def cap_bucket(name, rows, target):
    """Cap a bucket to ``target`` records on a best-effort basis.

    Returns the first ``target`` rows when the bucket has at least ``target``
    records. Otherwise prints a shortfall warning naming the bucket with its
    ``available/target`` counts and returns all available rows.
    """
    if len(rows) >= target:
        return rows[:target]
    print(f"WARNING: bucket '{name}' has {len(rows)}/{target} records; "
          f"using all {len(rows)} available.")
    return rows


def build_harmful_pool():
    """Assemble the combined harmful prompt pool.

    Loads records from ``HarmBenchDataset`` and ``AdvBenchDataset`` (each
    ``get_records()`` returns a ``list[PromptRecord]``) and concatenates them
    into a single list.
    """
    hb = HarmBenchDataset().get_records()
    ab = AdvBenchDataset().get_records()
    return hb + ab


def build_benign_pool():
    """Return the benign prompt records from ``CustomPromptDataset``.

    Selects every record whose ``category`` is not ``"harmful"``.
    """
    return [r for r in CustomPromptDataset().get_records()
            if r.category != "harmful"]


def build_prompt_pool(harmful_count=HARMFUL_COUNT, benign_count=BENIGN_COUNT):
    """Assemble the capped prompt pool from the harmful and benign sources.

    Each source is capped independently (best-effort) via ``cap_bucket`` and the
    two capped lists are concatenated (harmful first, then benign). The prompt
    text is all that flows downstream; source membership is not recorded, since
    labeling is deferred to a separate script.

    Returns a ``list[PromptRecord]``.
    """
    harmful = cap_bucket("harmful", build_harmful_pool(), harmful_count)
    benign = cap_bucket("benign", build_benign_pool(), benign_count)
    return harmful + benign


# --- Jailbreak augmentation helper (mirrors the CustomPromptDataset pattern;
#     does not alter the base 3-bucket pool above) ---------------------------

def build_jailbreak_pool(jailbreak_count=JAILBREAK_COUNT):
    """Return capped jailbreak-wrapped harmful records (harmful+jailbreak bucket).

    Loads ``jailbreak_prompts.json`` via ``JailbreakPromptDataset`` -- the same
    static-file + dataset-class pattern used for the benign set. Returns an empty
    list with a warning if the file is missing, so a run can still proceed.
    """
    try:
        rows = JailbreakPromptDataset().get_records()
    except FileNotFoundError:
        print("WARNING: jailbreak_prompts.json not found; skipping jailbreak "
              "bucket. Run `python -m data.make_jailbreak_prompts` first.")
        return []
    return cap_bucket("harmful_jailbreak", rows, jailbreak_count)


def assign_ids(pool):
    """Stamp fresh sequential ids onto the prompt pool.

    Iterates ``pool`` (a list of ``PromptRecord``) and returns a list of dicts
    with keys ``id`` (a fresh sequential integer starting at 0), ``prompt``, and
    ``harmful`` (the record's design-time ``expected_harmful`` label encoded as
    ``1`` or ``0``). The ``group`` and ``template`` provenance fields are carried
    through as-is (``None`` for the plain harmful/benign records). Response text
    is attached later, after generation.

    Returns a list of ``{id, prompt, harmful, group, template}`` dicts.
    """
    return [{"id": i, "prompt": rec.prompt,
             "harmful": int(bool(rec.expected_harmful)),
             "group": rec.group, "template": rec.template}
            for i, rec in enumerate(pool)]


def generate_responses_and_activations(records, llm, layers=LAYERS):
    """Generate responses and extract activations in a single model pass.

    Iterates the id-stamped ``records`` list and calls
    ``llm.generate_with_activations(prompt, layers)`` once per prompt, so the
    response text and the prompt's hidden-state activations come from the SAME
    forward pass (instead of running the model twice).

    Returns ``(metadata, activations)`` where ``metadata`` is a list of
    ``{id, prompt, response, harmful, group, template}`` dicts and
    ``activations`` is a ``{id: {layer: cpu_tensor}}`` mapping keyed by id so the
    two artifacts stay id-aligned. Progress is reported via ``tqdm``.
    """
    metadata = []
    activations = {}
    for row in tqdm(records, desc="Generating responses + activations"):
        response, acts = llm.generate_with_activations(row["prompt"], layers)
        metadata.append({
            "id": row["id"], "prompt": row["prompt"],
            "response": response, "harmful": row["harmful"],
            "group": row.get("group"), "template": row.get("template"),
        })
        activations[row["id"]] = acts
    return metadata, activations


def write_metadata(slug, records):
    """Serialize an already-id-stamped records list to a per-model JSON file.

    Serializes ``records`` to ``data/processed/{slug}/metadata.json`` with
    ``indent=2`` (human-readable) and ``ensure_ascii=False`` (legible
    prompts/responses); JSON string escaping handles commas, quotes, and
    newlines automatically. Creates the output directory if missing.

    Returns the list of record dicts that was written.
    """
    out_dir = Path("data/processed") / slug
    out_dir.mkdir(parents=True, exist_ok=True)     # create dir if missing
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return records


def write_activations(slug, activations):
    """Save the id-keyed activations next to the metadata.

    Composes the same per-model output directory as ``write_metadata``
    (``data/processed/{slug}/``), creates it if missing, and serializes the
    ``activations`` mapping to ``activations.pt`` via ``torch.save``. Writing to
    the SAME directory as ``metadata.json`` keeps the two artifacts id-aligned
    and colocated for the downstream ProbeDataset reader.
    """
    out_dir = Path("data/processed") / slug
    out_dir.mkdir(parents=True, exist_ok=True)     # create dir if missing
    torch.save(activations, out_dir / "activations.pt")


def main():
    """Entry point for the data-preparation pipeline.

    Wires the full pipeline: derive the per-model slug, load the LLM, assemble
    the capped prompt pool (optionally augmented with the jailbreak + benign
    wrapper-control buckets), stamp ids, generate responses, extract
    activations, and write the id-aligned ``metadata.json`` and
    ``activations.pt``. No labeling is performed here.
    """
    slug = model_slug(MODEL_NAME)

    llm = LLMInterface()

    pool = build_prompt_pool(HARMFUL_COUNT, BENIGN_COUNT)
    if INCLUDE_JAILBREAK:
        pool = pool + build_jailbreak_pool(JAILBREAK_COUNT)
    records = assign_ids(pool)

    print(f"Pool size: {len(pool)} | "
          f"harmful cap: {HARMFUL_COUNT} | benign cap: {BENIGN_COUNT} | "
          f"jailbreak: {INCLUDE_JAILBREAK}")

    records, activations = generate_responses_and_activations(records, llm, LAYERS)
    write_metadata(slug, records)
    print(f"Wrote dataset to data/processed/{slug}/metadata.json")

    write_activations(slug, activations)
    print(f"Wrote activations to data/processed/{slug}/activations.pt")


if __name__ == "__main__":
    main()
