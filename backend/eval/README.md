# Eval suites

These suites are the product spec for the research agent. They follow the split in [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents): **regression** (near-100%, blocks CI) vs **capability/quality** (may start low, never blocks deploy).

## Suites

| File | Kind | Offline | Production agent |
|---|---|---|---|
| `suites/ask-regression.json` | Retrieval + Ask abstain | Keyword embed + stub LLM | `--llm` uses the real embed/LLM |
| `suites/ask-quality.json` | Semantic retrieval, coverage, faithfulness | Retrieval graders only | `--llm` for citations/faithfulness |
| `suites/chat-regression.json` | Abstain, no fabricated cites, no web-search on library facts | `--check-references` or harness without `--llm` | `--llm` |
| `suites/chat-quality.json` | Grounded answers, `trials: 3` for pass@k / pass^k | References only | `--llm` |
| `suites/acl-regression.json` | Alice/Bob space isolation (set-difference on ids) | Keyword + lexical + SQL filter | No |
| `suites/search-regression.json` | Catalog FTS / doc numbers / space facets; RAG uses the same ACL SQL | Catalog has no embeddings; RAG keyword-axis in CI | No |
| `suites/search-quality.json` | Semantic qrels without unique tokens | Keyword-axis is **not** retrieval SOTA | `--embeddings` for real Ollama vectors |
| `suites/discipline-regression.json` | Un-accessioned not searchable; blank original → Ask abstain; injection isolation | Keyword + SQL accession filter | No |

ZXQ* titles are synthetic so cleanup is safe. A unique token like `ZXQELLA7` makes lexical hit rate look perfect. **Do not report keyword-axis scores as retrieval SOTA.** The harness prints `embedding_mode` (`none` for catalog, `keyword` or `real` for RAG) and `retrieval_backend` (`catalog` vs `rag`).

## Graders

- **Required (gate):** `retrieval_titles`, `retrieval_ids`, `abstain`, `citations`, `transcript`, `must_include_facts`, structured `outcome` (gate status / abstain language / `excludes`).
- **Diagnostic (not required):** `tool_used` — grade what the agent produced, not the exact tool path.
- **Soft:** `faithfulness` (LLM judge). `unchecked` does not fail the task.

Every task has a `reference` payload. `python -m eval.harness --check-references --suite all` proves two experts (here: the author and the grader) agree the task is solvable.

## Metrics

- `pass_at_1` / `pass@k`: at least one of k trials passed. Useful for capability.
- `pass_hat_k` / `pass^k`: every one of k trials passed. Use this for safety-style regression.
- `regression_failed` / `gate`: any regression trial failed. CI and CD should treat `gate=fail` as red.
- Quality `pass_at_1` is a hill to climb, not a merge blocker.

## What blocks deploy

These suites run on every PR and on `main`. A failure fails the CI job, so [Deploy](../../.github/workflows/deploy.yml) never starts (`workflow_run` conclusion ≠ success).

| Suite | Blocks deploy |
|---|---|
| `ask-regression` | Yes |
| `chat-regression` | Yes (without `--llm`) |
| `acl-regression` | Yes |
| `search-regression` | Yes |
| `discipline-regression` | Yes |
| `ask-quality` / `chat-quality` / `search-quality` | No (nightly only) |

`--llm` quality is [eval-nightly.yml](../../.github/workflows/eval-nightly.yml). A bad quality score does **not** roll back Fly.

Transcripts are written to `eval/runs/`. The Eval panel opens the first failing trial. `POST /api/eval/run` is locked on the public demo unless `EVAL_TOKEN` matches.
