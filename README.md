# openarmature examples

Demo projects for [openarmature](https://github.com/LunarCommand/openarmature-python) — a workflow framework for LLM pipelines and tool-calling agents.

## Projects

| Directory | What it demonstrates |
|---|---|
| [`01-linear-pipeline/`](./01-linear-pipeline) | The minimal graph shape: typed `State`, per-field reducers (`append`), static edges, `END`. Runs a two-node `plan → write` pipeline. |
| [`02-routing-and-subgraphs/`](./02-routing-and-subgraphs) | Conditional routing via `add_conditional_edge`, subgraph composition via `add_subgraph_node`, custom `ProjectionStrategy`, and the `merge` reducer. |

## Setup

Each project depends on `openarmature` via an editable path dep to a sibling clone of the library:

```bash
~/code/
├── openarmature-python/
└── openarmature-examples/
    ├── 01-linear-pipeline/
    └── 02-routing-and-subgraphs/
```

Clone both repos as siblings, then:

```bash
cd openarmature-examples/01-linear-pipeline
uv sync
uv run python main.py "the psychology of long walks"
```

Both demos expect an OpenAI-compatible LLM endpoint at `http://localhost:8000/v1` (vLLM, LM Studio, etc.). Edit `VLLM_BASE_URL` and `MODEL` at the top of each `main.py` to point elsewhere.

## License

Apache-2.0 — same as [openarmature-python](https://github.com/LunarCommand/openarmature-python). See [LICENSE](./LICENSE).
