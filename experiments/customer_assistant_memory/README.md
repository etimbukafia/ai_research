# Customer assistant graph memory

This experiment tests when a customer assistant needs graph memory.

The experiment uses one local Python process. It does not use Docker or an
external service.

It compares four memory conditions over 24 support cases:

- `flat_recent`
- `vector_chroma`
- `graph_ladybug`
- `hybrid_graph_vector`

## Early writer contract

The writer can use these names before the final run:

- Methods: `flat_recent`, `vector_chroma`, `graph_ladybug`,
  `hybrid_graph_vector`.
- Case classes: `direct`, `multi_step_relationship`, `temporal`,
  `conflict_authority_missing_evidence`.
- Metrics: `evidence_recall`, `path_recall`, `answer_accuracy`,
  `temporal_accuracy`, `provenance_precision`, and
  `unsupported_claim_rate`.
- Featured case: `case-multi-01`.
- Entry point: `experiments/customer_assistant_memory/run_experiment.py`.
- Final values: `results.json`, `trace.json`, and `report.md`.

The case schema is:

```text
case_id
customer_id
ticket_id
ticket_text
case_class
intent
required_record_ids
required_path
gold_answer
answerable
allowed_sources
disallowed_records
```

The builder will confirm the exact record IDs, paths, and result values in the
final output files after the run.

## Runtime

Use this Python executable:

```powershell
C:\Users\Etimbuk\projects\.projects\Scripts\python.exe
```

Install the pinned packages with:

```powershell
& 'C:\Users\Etimbuk\projects\.projects\Scripts\python.exe' -m pip install -r experiments\customer_assistant_memory\requirements.txt
```

The vector condition uses the local
`sentence-transformers/all-MiniLM-L6-v2` model and an in-memory Chroma
client. Set `HF_HOME` to the existing local model cache when needed.

Ladybug 0.19.1 has a CPython 3.14 Windows wheel. Its Windows wheel needs the
OpenSSL runtime DLLs. Set `LBUG_DLL_DIRECTORY` to a directory that contains
`libssl-3-x64.dll` and `libcrypto-3-x64.dll` before running. The script adds
that directory only when it imports Ladybug. It uses Ladybug's embedded
`pybind` backend and creates a local database at `customer_memory.lbug` during
the run.

Example command for the current machine:

```powershell
$env:HF_HOME = 'C:\Users\Etimbuk\projects\.projects\huggingface'
$env:HF_HUB_CACHE = 'C:\Users\Etimbuk\projects\.projects\huggingface\hub'
$env:HF_HUB_OFFLINE = '1'
$env:LBUG_DLL_DIRECTORY = 'C:\Program Files\Microsoft OneDrive\26.145.0728.0011'
$env:TOKENIZERS_PARALLELISM = 'false'
& 'C:\Users\Etimbuk\projects\.projects\Scripts\python.exe' `
  'C:\Users\Etimbuk\afiavana\experiments\customer_assistant_memory\run_experiment.py'
```

The generated `customer_memory.lbug` files are local run state. They are not
needed as source data. The script rebuilds the graph from
`customer_records.json` on each run.

## Outputs

- `customer_records.json`: manually authored customers, orders, products,
  warranties, tickets, issues, resolutions, addresses, preferences, and
  policy records.
- `cases.json`: exactly 24 labelled support cases. Each class has six cases.
- `run_experiment.py`: the visible entry point and all fixed skills.
- `results.json`: package versions, case traces, and metrics.
- `trace.json`: the four-method trace for `case-multi-01`.
- `report.md`: the publishable experiment summary and handoff values.

The offline flag requires the model files to exist in the local cache. The
experiment does not call a model service during the run.
