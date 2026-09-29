# Enterprise agent evaluation harness

This experiment supports the article `Why I Built an Evaluation Harness for
Enterprise Agents`.

It uses synthetic Aster Cloud records and a fixed replay fixture. Each of the
12 business intents has a clean case and a distractor case. The three replay
conditions are `raw_schema`, `prose_rag`, and `semantic_catalog`.

Run it from the repository root:

```powershell
& 'C:\Users\Etimbuk\projects\.projects\Scripts\python.exe' enterprise_agent_lab/experiments/agent_harness/run_experiment.py
```

The run makes no Gemini request. The live validation plan is recorded in
`results.json` and remains separate from replay values. It uses the pinned
`gemini-3.5-flash-lite` model, one worker, a five-second request interval, and
a 15 RPM project limit when it is enabled.

The generated files are the article-facing contract:

- `cases.json` and `run_manifest.json` define the fixture;
- `traces.jsonl` and `results.json` contain replay output;
- `comparison.json` records baseline and candidate comparison;
- `promotion_evaluation.json` records hard and soft gates;
- `promotion_decision.json` records the human decision and rollback fixture;
- `report.md` is a short generated report.

The experiment does not change source records. Draft actions remain local
replay objects.

