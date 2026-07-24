"""
Quick viewer for the prompt datasets.

Usage:
    python -m data.inspect_data            # from the DL_project root
    python data/inspect_data.py

Always prints the custom contrastive set (no network / no login needed).
Prints HarmBench + AdvBench too if `datasets` is installed and you're
logged in (HarmBench is gated: run `hf auth login` first).
"""

import collections

from data.dataset import CustomPromptDataset

try:
    from data.dataset import HarmBenchDataset, AdvBenchDataset
    _HAS_DATASETS = True
except Exception as e:  # noqa: BLE001
    _HAS_DATASETS = False
    _IMPORT_ERR = e


def _preview(records, n=5):
    for r in records[:n]:
        exp = "" if r.expected_refusal is None else f" refuse={r.expected_refusal}"
        print(f"  [{r.category:<11}] {r.prompt[:80]}{exp}")


def show_custom():
    print("=" * 70)
    print("CUSTOM CONTRASTIVE SET  (data/custom_prompts.json)")
    print("=" * 70)
    ds = CustomPromptDataset()
    recs = ds.get_records()
    counts = collections.Counter(r.category for r in recs)
    print(f"total: {len(recs)}  |  by category: {dict(counts)}")
    for cat in ["harmful", "educational", "harmless", "dual_use"]:
        print(f"\n-- {cat} --")
        _preview([r for r in recs if r.category == cat], n=4)


def show_hub():
    print("\n" + "=" * 70)
    print("HUB DATASETS (HarmBench, AdvBench)")
    print("=" * 70)
    if not _HAS_DATASETS:
        print("`datasets` not importable -> skipping.")
        print(f"reason: {_IMPORT_ERR}")
        print("install with:  pip install datasets")
        return

    try:
        hb = HarmBenchDataset().get_records()
        print(f"\nHarmBench: {len(hb)} prompts")
        _preview(hb, n=5)
    except Exception as e:  # noqa: BLE001
        print(f"\nHarmBench failed to load: {e}")
        print("HarmBench is gated -> run `hf auth login` and accept the dataset terms.")

    try:
        ab = AdvBenchDataset().get_records()
        print(f"\nAdvBench: {len(ab)} prompts")
        _preview(ab, n=5)
    except Exception as e:  # noqa: BLE001
        print(f"\nAdvBench failed to load: {e}")


if __name__ == "__main__":
    show_custom()
    show_hub()
