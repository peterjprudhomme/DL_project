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

from data.dataset import HarmBenchDataset, AdvBenchDataset, CustomPromptDataset
from src.llm import LLMInterface
from src.config import MODEL_NAME, LAYERS, model_slug


# Best-effort cap on how many prompts to draw from each source before running
# the model. Tune these to trade dataset size against compute. They only bound
# the pool; downstream labeling decides the final harmful/refused composition.
HARMFUL_COUNT = 400
BENIGN_COUNT = 200


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


def assign_ids(pool):
    """Stamp fresh sequential ids onto the prompt pool.

    Iterates ``pool`` (a list of ``PromptRecord``) and returns a list of dicts
    with keys ``id`` (a fresh sequential integer starting at 0), ``prompt``, and
    ``harmful`` (the record's design-time ``expected_harmful`` flag encoded as
    ``1`` or ``0``). Response text is attached later, after generation.

    Returns a list of ``{id, prompt, harmful}`` dicts.
    """
    return [{"id": i, "prompt": rec.prompt,
             "harmful": int(bool(rec.expected_harmful))}
            for i, rec in enumerate(pool)]


def generate_responses(records, llm):
    """Generate a model response for each record's prompt.

    Iterates the id-stamped ``records`` list, calling ``llm.generate`` once per
    prompt, and returns a new list of ``{id, prompt, response, harmful}`` dicts
    (the final metadata schema, with ``harmful`` last). The ``harmful`` label is
    carried through unchanged. Progress is reported via ``tqdm``.
    """
    return [
        {"id": row["id"], "prompt": row["prompt"],
         "response": llm.generate(row["prompt"]), "harmful": row["harmful"]}
        for row in tqdm(records, desc="Generating responses")
    ]


def extract_activations(records, llm, layers=LAYERS):
    """Extract hidden-state activations for each record's prompt.

    Iterates the final, id-stamped ``records`` list and calls
    ``llm.get_activations(prompt, layers)`` once per record. Each record's
    activation mapping is keyed by the record's ``id`` so the result aligns with
    the metadata by id.

    ``LLMInterface.get_activations`` returns a ``{layer: tensor}`` mapping with
    each tensor already detached and moved to CPU, so no further device handling
    is needed here.

    Returns ``{id: {layer: cpu_tensor}}``.
    """
    activations = {}
    for row in tqdm(records, desc="Extracting activations"):
        activations[row["id"]] = llm.get_activations(row["prompt"], layers)
    return activations


def write_metadata(slug, records):
    """Serialize an already-id-stamped records list to a per-model JSON file.

    Serializes ``records`` (a list of ``{id, prompt, response, harmful}`` dicts) to
    ``data/processed/{slug}/metadata.json`` with ``indent=2`` (human-readable)
    and ``ensure_ascii=False`` (legible prompts/responses); JSON string escaping
    handles commas, quotes, and newlines automatically. Creates the output
    directory if missing.

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
    the capped prompt pool, stamp ids, generate responses, extract activations,
    and write the id-aligned ``metadata.json`` and ``activations.pt``. No
    labeling is performed here.
    """
    slug = model_slug(MODEL_NAME)

    llm = LLMInterface()

    pool = build_prompt_pool(HARMFUL_COUNT, BENIGN_COUNT)
    records = assign_ids(pool)

    print(f"Pool size: {len(pool)} | "
          f"harmful cap: {HARMFUL_COUNT} | benign cap: {BENIGN_COUNT}")

    records = generate_responses(records, llm)
    write_metadata(slug, records)
    print(f"Wrote dataset to data/processed/{slug}/metadata.json")

    activations = extract_activations(records, llm, LAYERS)
    write_activations(slug, activations)
    print(f"Wrote activations to data/processed/{slug}/activations.pt")


if __name__ == "__main__":
    main()
