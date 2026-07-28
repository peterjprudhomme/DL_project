# How to Run DL_project

This project probes a language model's internal activations to study **refusal
behavior** — whether the model refuses a prompt and whether that decision is
linearly decodable from its hidden states.

The pipeline has two stages:

1. **Extraction** (`data/prepare_dataset.py`) — run prompts through the model,
   record each response, label it as refused / harmful, and save the
   hidden-state activations for selected layers. This single command writes
   both `metadata.json` and `activations.pt` under
   `data/processed/{model_slug}/`.
2. **Probing** (`run_probes.py`) — train a logistic-regression probe on those
   saved activations to predict `refused` from each layer, and report accuracy.

---

## 1. Prerequisites

- [conda](https://docs.conda.io/) (Miniconda or Anaconda)
- A HuggingFace account with access to the gated **HarmBench** dataset
  (only needed for the default extraction run)
- Optional: a CUDA GPU. The code runs on CPU too, just slower. GPU uses
  `float16`, CPU uses `float32` (handled automatically in `src/llm.py`).

The model defaults to `Qwen/Qwen2.5-1.5B-Instruct` (see `src/config.py`), which
is downloaded automatically from HuggingFace on first run.

---

## 2. Environment setup

Create and activate the conda environment:

```bash
cd /{your_base}/DL_project
conda env create -f environment.yml
conda activate dl_refusal
```

> **Note:** `data/dataset.py` uses the HuggingFace `datasets` library to load
> HarmBench/AdvBench, but it is **not** currently listed in `environment.yml`.
> If you plan to run the default extraction (which uses HarmBench), install it:
>
> ```bash
> pip install datasets
> ```

### HuggingFace login (required for HarmBench)

HarmBench is a gated dataset. Log in and accept the dataset terms once:

```bash
hf auth login
```

---

## 3. Run the pipeline

All commands are run from the project root (`DL_project/`) so that the
`src` and `data` package imports resolve.

### Step 1 — Prepare the dataset and extract activations

```bash
python -m data.prepare_dataset
```

This single command will:
- assemble the harmful pool (HarmBench + AdvBench) and the benign set,
- generate a response for each prompt with the Qwen model,
- label `harmful` (from the record) and `refused`
  (keyword heuristic in `src/extractor.py`),
- route records into three balanced buckets (harmful-refused,
  harmful-not-refused, benign),
- capture hidden-state activations for layers `[5, 10, 15, 20]`
  (`LAYERS` in `src/config.py`) for every kept record,
- write **both** outputs to `data/processed/{model_slug}/`:
  - `metadata.json` — one record per prompt (id, prompt, response, harmful, refused)
  - `activations.pt` — dict of `{id: {layer: tensor}}`, id-aligned with the metadata

Expected console output ends with:

```
Wrote dataset to data/processed/{model_slug}/metadata.json
Wrote activations to data/processed/{model_slug}/activations.pt
```

### Step 2 — Train the probes

```bash
python run_probes.py
```

This loads `data/processed/{model_slug}/{metadata.json, activations.pt}`, trains
a linear probe per available layer to predict `refused`, and prints the layer number
with its test accuracy, followed by the `refused` / `harmful` label counts.

---

## 4. Inspecting the prompt datasets (optional)

To preview the prompt sets without running the model:

```bash
python -m data.inspect_data
```

The custom contrastive set (`data/custom_prompts.json`) always prints and needs
no network or login. HarmBench/AdvBench only print if `datasets` is installed
and you're logged in.
