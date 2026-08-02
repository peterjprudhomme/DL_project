import json
import os
from dataclasses import dataclass
from typing import Optional

from datasets import load_dataset


@dataclass
class PromptRecord:
    """
    A single labeled prompt, before it is run through the model.

    'refused' is intentionally NOT here: it is a property of the model's
    response and is measured at activation-extraction time.
    """

    prompt: str
    source: str                      # "harmbench" | "advbench" | "custom" | "jailbreak"
    category: str                    # harmful | harmless | educational | dual_use | <dataset category>
    expected_harmful: Optional[bool] # None when ambiguous (dual_use)
    expected_refusal: Optional[bool] # design-time hypothesis; None when ambiguous
    topic: Optional[str] = None
    id: Optional[str] = None
    # Provenance for the 2x2 (harm x wrapper) design used by the jailbreak
    # bucket and the wrapper-controlled direction analysis. These stay None for
    # the plain harmful/benign records.
    group: Optional[str] = None      # harmful_plain | benign_plain | harmful_jailbreak | benign_jailbreak
    template: Optional[str] = None   # name of the jailbreak template applied (if any)
    base_id: Optional[object] = None # id of the source record the wrap was built from


class HarmBenchDataset:

    def __init__(self, config="standard"):
        # config can be ['contextual', 'copyright', 'standard']
        # standard has 200 prompts, need to test other configs

        # must be logged into HuggingFace with approved account to work
        # I used "hf auth login" in my VS Code terminal to login to my HuggingFace account with a token
        self.dataset = load_dataset(
            "walledai/HarmBench",
            config
        )

    def __len__(self):
        return len(self.dataset["train"]["prompt"])

    def get_example(self, index):
        return self.dataset["train"]["prompt"][index]

    def get_prompts(self):
        # returns list of prompts
        return self.dataset["train"]["prompt"]

    def get_categories(self):
        # returns categories of prompts (in same order as get_prompts list)
        return self.dataset["train"]["category"]

    def get_records(self):
        """HarmBench prompts are all harmful and expected to be refused."""
        prompts = self.get_prompts()
        categories = self.get_categories()
        return [
            PromptRecord(
                prompt=p,
                source="harmbench",
                category="harmful",
                expected_harmful=True,
                expected_refusal=True,
                topic=cat,
                id=f"harmbench_{i}",
            )
            for i, (p, cat) in enumerate(zip(prompts, categories))
        ]


class AdvBenchDataset:
    """
    AdvBench 'harmful_behaviors' split (from the llm-attacks paper).
    Hosted ungated on the Hub as 'walledai/AdvBench'.
    """

    def __init__(self):
        self.dataset = load_dataset("walledai/AdvBench")

    def __len__(self):
        return len(self.dataset["train"]["prompt"])

    def get_prompts(self):
        return self.dataset["train"]["prompt"]

    def get_records(self):
        prompts = self.get_prompts()
        return [
            PromptRecord(
                prompt=p,
                source="advbench",
                category="harmful",
                expected_harmful=True,
                expected_refusal=True,
                topic=None,
                id=f"advbench_{i}",
            )
            for i, p in enumerate(prompts)
        ]


class CustomPromptDataset:
    """
    Loads the hand-curated contrastive set from custom_prompts.json.
    This is the set that decorrelates topic from refusal (via matched
    harmful/educational pairs) so probes can separate the two behaviors.
    """

    def __init__(self, path=None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "custom_prompts.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.schema = data.get("schema", {})
        self._entries = data["prompts"]

    def __len__(self):
        return len(self._entries)

    def get_prompts(self):
        return [e["prompt"] for e in self._entries]

    def get_by_category(self, category):
        return [e for e in self._entries if e["category"] == category]

    def get_records(self):
        return [
            PromptRecord(
                prompt=e["prompt"],
                source="custom",
                category=e["category"],
                expected_harmful=e.get("expected_harmful"),
                expected_refusal=e.get("expected_refusal"),
                topic=e.get("topic"),
                id=e.get("id"),
            )
            for e in self._entries
        ]


class JailbreakPromptDataset:
    """
    Loads the jailbreak-wrapped harmful prompts from jailbreak_prompts.json
    (produced by ``data/make_jailbreak_prompts.py``).

    Every entry is a harmful request wrapped in a generic jailbreak template.
    The wrap raises the chance the model complies, which populates the
    harmful+complied (HC) cell needed for the difference-in-means refusal
    direction. The actual ``refused`` label is still measured downstream from
    the model's response.
    """

    def __init__(self, path=None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "jailbreak_prompts.json")
        self.path = path
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.schema = data.get("schema", {})
        self._entries = data["prompts"]

    def __len__(self):
        return len(self._entries)

    def get_prompts(self):
        return [e["prompt"] for e in self._entries]

    def get_records(self):
        return [
            PromptRecord(
                prompt=e["prompt"],
                source="jailbreak",
                category=e.get("category", "harmful"),
                expected_harmful=e.get("expected_harmful", True),
                expected_refusal=e.get("expected_refusal", False),
                topic=e.get("topic", "jailbreak"),
                id=e.get("id"),
                group="harmful_jailbreak",
                template=e.get("template"),
                base_id=e.get("base_id"),
            )
            for e in self._entries
        ]


def build_combined_records(
    use_harmbench=True,
    use_advbench=False,
    use_custom=True,
    harmbench_limit=None,
    advbench_limit=None,
):
    """
    Assemble a single labeled prompt list from the chosen sources.

    Returns a list[PromptRecord]. Downstream, run each prompt through the
    model, detect whether it refused, and attach activations to build the
    activation dataset.
    """
    records = []

    if use_harmbench:
        hb = HarmBenchDataset().get_records()
        records.extend(hb[:harmbench_limit] if harmbench_limit else hb)

    if use_advbench:
        ab = AdvBenchDataset().get_records()
        records.extend(ab[:advbench_limit] if advbench_limit else ab)

    if use_custom:
        records.extend(CustomPromptDataset().get_records())

    return records
