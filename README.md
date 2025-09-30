# knowledge-graph-Ontology-health-medical-history (BioBERT + Ollama + BERTopic)

This repository contains a single Jupyter notebook (provided here as MedicalOntology-Code.zip due to size limits, unzip first)

> TL;DR:  
> 1) **Cell 1** (index 3): NER with BioBERT + relation extraction with **Ollama** → writes `relations_<model>.{csv,jsonl}` and caches.  
> 2) **Cell 2** (index 5): Cleans & clusters entities/relations with **BioBERT embeddings + BERTopic** → writes `kg_out_relations/*` (nodes, relation types, aggregated edges, ontology).  
> 3) **Cell 3** (index 9): Builds **document topics**, maps relations to topics, optional **LLM filtering** via Ollama → writes topic CSVs and overview images.

---

## What to upload to the (anonymous) Hugging Face repo

**Minimum (recommended):**
- `MedicalOntology-Code.ipynb` *(the notebook you shared)*
- `README.md` *(this file)*
- `requirements.txt` *(Python deps to run the notebook)*

**Optional (if you want others to see example outputs without running anything):**
- `relations_<model>.csv` and `relations_<model>.jsonl` (from Step 1)  
- `kg_out_relations/` contents (from Step 2):  
  `edges_aggregated.csv`, `edges_aggregated.zip`, `nodes.csv`, `relation_types.csv`, `relation_inventory.csv`, `ontology.json`, `ontology_clean.json`, optionally `graph.graphml`
- Topic artifacts (from Step 3):  
  `doc_texts.csv`, `doc_topics.csv`, `ontology_topics.csv`, `relation_topics.csv`, `relation_topics_all.csv`, `topic_top_relations.csv`, `ontology_overview.{png,pdf}`, `ontology_topics_network.{png,pdf}`, `topic_<id>_ego.png`, `llm_filter_cache.jsonl`
- `LICENSE` (e.g., MIT), if you want to specify terms

> You do **not** need to upload model weights or the dataset; the notebook pulls models from Hugging Face and reads the dataset directly.

---

## Environment & Dependencies

Install Python 3.10+ and run:

```bash
pip install -r requirements.txt
```

### Extra runtime requirements
- **Ollama** running locally if you want LLM-based relation extraction / filtering.  
  Default model in the notebook is `qwen2.5:32b-instruct`. Start it like:
  ```bash
  ollama run qwen2.5:32b-instruct
  ```
- By default the code attempts to use CUDA if available; use `--device cpu` to force CPU mode.


### Useful environment variables (override defaults)
The notebook reads these to tune Ollama calls:

- `OLLAMA_SERVER` (default `http://localhost:11434`)
- `OLLAMA_THREADS` (default `0` = auto)
- `OLLAMA_NUM_CTX` (default `4048`)
- `OLLAMA_NUM_BATCH` (default `64`)
- `OLLAMA_USE_MLOCK` (default `true`)
- `OLLAMA_NUMA` (default `false`)
- `OLLAMA_KEEP_ALIVE` (default `2h`)

---

### Large files note
Some intermediate artifacts and the notebook itself are very large.  
To stay within GitHub size limits:

- **Notebook**:  
  `MedicalOntology-Code.zip` contains the full Jupyter notebook (`MedicalOntology-Code.ipynb`, ~29 MB).  
  Please unzip it locally to open the notebook in Jupyter or VS Code.

- **NER cache**:  
  `biobert_entities.json` is the full NER cache (~360 MB).  
  A small sample (`biobert_entities - a few rows.json`) is included for illustration.  
  Running Cell 1 will regenerate the full cache if missing.


- **Relations**:  
  `relations_qwen2.5_32b-instruct.zip` and `relations_clean.zip` are compressed versions of  
  large relation CSVs. Unzip them before use.

- **Other omitted artifacts**:  
  Files like `raw_texts.json` are not included due to their size but can be regenerated  
  by running the notebook end-to-end.

This keeps the repository lightweight and anonymous, while still providing working examples of the outputs.
---

## Data flow & Outputs (by cell)

### 🧩 Cell 1 — NER + Relation Extraction (streamed per document)
**File:** cell index **3** in the notebook  
**What it does:**
- Downloads (or reuses cached) OCR texts from **`davanstrien/MedicalHistoryofBritishIndia`** into `raw_texts.json` (use `--fresh_bootstrap` to force refresh).
- Runs **BioBERT NER** with `d4data/biomedical-ner-all` and saves `biobert_entities.json`.
- For each document, prompts an **Ollama** LLM (default `qwen2.5:32b-instruct`) to extract relations.
- Streams and **immediately appends** results to CSV/JSONL as it goes.

**Key inputs & knobs:**
- Dataset ID: `davanstrien/MedicalHistoryofBritishIndia` (hardcoded)
- CLI flags (defaults are safe in notebooks):
  - `--max_pages`, `--min_ner_score`, `--topk_ents`
  - `--fresh_bootstrap` (remove `raw_texts.json` and re-download)
  - `--relation_model` (default `qwen2.5:32b-instruct`)
  - `--ner_on_cpu` (force CPU for NER)
  - `--print_relations`, `--print_every`

**Outputs written to the working directory:**
- `raw_texts.json` (cached OCR text)
- `biobert_entities.json` (NER cache)
- `relations_<model>.csv`
- `relations_<model>.jsonl`  
  *where `<model>` is `relation_model` with `/` and `:` replaced by `_` (e.g., `qwen2.5_32b-instruct`).*

> Just **run the cell** with its defaults to produce outputs. It will detect caches automatically.

---

### 🧭 Cell 2 — Semantic KG Builder (full corpus)
**File:** cell index **5** in the notebook  
**What it does:**
- Cleans entities/relations (removes OCR noise, short garbage strings).
- Embeds unique strings with **Sentence-Transformers BioBERT** (`pritamdeka/S-BioBert-snli-multinli-stsb`).
- Clusters entities & relations using **UMAP + HDBSCAN**, then maps relations to a small schema (e.g., "treated_with", "causes", "located_in", "vaccinated_against").
- Writes an aggregated knowledge graph and a human-readable inventory.

**How to run (typical arguments):**
- `--relations_csv <path>`: use the CSV from Cell 1 (e.g., `relations_qwen2.5_32b-instruct.csv`)
- `--out_dir kg_out_relations`
- Optional: `--ner_json biobert_entities.json`
- Optional: `--device cpu` (if no CUDA), `--do_graphml`

**Outputs (under `kg_out_relations/`):**
- `edges_aggregated.csv`
- `edges_aggregated.zip`
- `nodes.csv`
- `relation_types.csv`
- `relation_inventory.csv`
- `ontology.json`
- `ontology_clean.json`
- *(optional)* `graph.graphml`


In notebooks, defaults are prefilled — run the cell and it will produce **kg_out_relations/*.
---

### 🧪 Cell 3 — Topics & Ontology Induction (+ optional LLM filtering)
**File:** cell index **9** in the notebook  
**What it does:**
- Reads \kg_out_relations/edges_aggregated.csv`.`
- Constructs per-document texts and fits **BERTopic** to infer topics.
- Maps relations to topics by document co-occurrence.
- Applies an additional LLM-based quality filter to relation–topic assignments (using the same Ollama setup as in Cell 1).
- Exports CSVs and overview images.

**Outputs (in the working directory):**
- `doc_texts.csv`, `doc_topics.csv`
- `ontology_topics.csv`
- `relation_topics.csv`, `relation_topics_all.csv`
- `topic_top_relations.csv`
- `ontology_overview.{png,pdf}`, `ontology_topics_network.{png,pdf}`
- `topic_<id>_ego.png` (ego plots, one per topic)
- `llm_filter_cache.jsonl` (append-only cache for LLM filter decisions)

Set `KG_DIR` at the top of this cell if you placed outputs elsewhere.

---

## End-to-end quick start (inside the notebook)

1. **Run Cell 1** (index 3) — produces `relations_<model>.csv` and caches.  
   - If you don’t have an Ollama server, you can still run NER; relation extraction will require it.
2. **Run Cell 2** (index 5) — point `--relations_csv` to the CSV from Cell 1; produces `kg_out_relations/*`.
3. **Run Cell 3** (index 9) — reads `kg_out_relations/edges_aggregated.csv` and writes topic CSVs/figures.
---

## Notes & Tips
- **CPU-only runs** are possible (they’re slower): use `--ner_on_cpu` in Cell 1 and `--device cpu` in Cell 2.
- **Model IDs** are Hugging Face names and will be downloaded automatically on first run:
  - NER: `d4data/biomedical-ner-all`
  - Embeddings: `pritamdeka/S-BioBert-snli-multinli-stsb` (used in both Cell 2 and Cell 3)
- **Reproducibility:** The pipeline writes intermediate CSV/JSONL so you can resume downstream steps without re-running everything.
- **Determinism:** Random seeds and embedding caches are fixed to ensure reproducible clustering and canonical labels across runs.


---

## Citation / Credits
- Dataset: `davanstrien/MedicalHistoryofBritishIndia` (Hugging Face Datasets)
- NER model: `d4data/biomedical-ner-all`
- Sentence embeddings: `pritamdeka/S-BioBert-snli-multinli-stsb`
- Topic modeling: **BERTopic**
