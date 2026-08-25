# Information Retrieval from the Bottom Up

This directory contains the local experiment for the article with the same name.

The experiment uses one corpus and three retrieval methods:

1. `exact_keyword`: count the unique query terms that occur in a document.
2. `bm25`: rank term matches with term frequency, inverse document frequency, and document length normalization.
3. `vector_chroma`: encode the texts with `sentence-transformers/all-MiniLM-L6-v2`, store the vectors in an in-memory Chroma collection, and rank by cosine similarity.

The corpus has 12 short technical documents. It has manual relevance labels for one main query and two validation queries. The script uses the same corpus and a top-five cutoff for each method.

## Run

Use Python 3.10 or newer. Install the pinned direct dependencies:

```text
python -m pip install -r requirements.txt
python run_experiment.py
```

The first run downloads the model from Hugging Face. The script does not call an API. It creates a new in-memory Chroma client on every run.

The script writes:

- `results.json`: rankings, scores, labels, metrics, model ID, and package versions.
- `report.md`: a plain-language report for the writer.

## Source and data flow

This is a small experiment. It uses one Python script instead of a package with a `src/` directory.

`run_experiment.py` loads `corpus.json`, runs the three retrieval methods, and writes `results.json` and `report.md`.

The flow is:

```text
corpus.json
    -> exact_keyword and bm25
    -> MiniLM embeddings
    -> in-memory Chroma collection
    -> Recall@5 and MRR
    -> results.json and report.md
```

On the local Windows machine, the final run used the virtual environment and model cache at:

```text
C:\Users\Etimbuk\projects\.projects
C:\Users\Etimbuk\projects\.projects\huggingface
```

The command was:

```powershell
$env:HF_HOME = 'C:\Users\Etimbuk\projects\.projects\huggingface'
$env:HF_HUB_CACHE = 'C:\Users\Etimbuk\projects\.projects\huggingface\hub'
$env:TOKENIZERS_PARALLELISM = 'false'
& 'C:\Users\Etimbuk\projects\.projects\Scripts\python.exe' `
  'C:\Users\Etimbuk\afiavana\experiments\information_retrieval\run_experiment.py'
```

The script uses the title and body as the indexed text. Tags stay metadata. The vector stage uses normalized 384-dimensional vectors and cosine distance. Chroma stores the vectors in memory for the run. The script records the corpus hash, package versions, rankings, labels, metrics, and runtime in `results.json`.

## Reproducibility

The script sets random seeds, uses one CPU thread where the runtime supports it, adds documents in sorted ID order, and sorts every result by descending score and ascending document ID. The model ID, corpus hash, Python version, package versions, and runtime are recorded in `results.json`.

The ranking is deterministic for the same model files and package versions. Runtime can change between runs. Chroma uses no persistent database.

## Metric rules

`Recall@5` is the number of labeled relevant documents in the top five divided by the number of labeled relevant documents.

`MRR` uses the top-five list in this experiment. It is the reciprocal of the rank of the first relevant document. It is `0.0` when the top five contain no relevant document.

The relevance labels are manual study labels. They describe this small corpus and these queries. They do not measure general search quality.
