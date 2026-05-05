# openarmature examples

Demo projects for [openarmature](https://github.com/LunarCommand/openarmature-python) — a workflow framework for LLM pipelines and tool-calling agents.

## Projects

### [`01-linear-pipeline/`](./01-linear-pipeline)

**Use case:** Take a topic (e.g. "the psychology of long walks") and produce a short written piece — first plan a few angles, then write the article.

**Demonstrates:** The minimal graph shape — typed `State`, the `append` reducer, static edges, `END`, a two-node linear `plan → write` pipeline.

### [`02-routing-and-subgraphs/`](./02-routing-and-subgraphs)

**Use case:** A question-answering assistant. Classify the question, then either give a one-shot quick answer or run a multi-step research sub-pipeline (plan angles → gather notes → synthesize), then lightly copy-edit the result.

**Demonstrates:** Conditional edges (state-driven routing) via `add_conditional_edge`, subgraph composition via `add_subgraph_node`, a custom `ProjectionStrategy` for the parent ↔ subgraph boundary, and the `merge` reducer for dict accumulation.

### [`03-explicit-subgraph-mapping/`](./03-explicit-subgraph-mapping)

**Use case:** Compare two topics ("rust vs go", "espresso vs drip coffee") by running the same analysis subgraph on each, then synthesizing a verdict.

**Demonstrates:** One compiled subgraph reused at two parent sites with per-site `ExplicitMapping` — the case spec v0.2 / proposal 0002 was written for, and the only way to express "run the same subgraph twice on disjoint parent fields" without per-site projection classes that mirror each other.

### [`04-observer-hooks/`](./04-observer-hooks)

**Use case:** Add observability to a small `draft → review → finalize` pipeline (where `review` is a subgraph) without changing any node code. A graph-attached console tracer prints structured node-boundary lines to stderr; an invocation-scoped metrics collector tallies counts for the current call.

**Demonstrates:** Observer hooks (spec v0.3 / proposal 0003) — `attach_observer`, the `NodeEvent` shape, namespace chaining across the subgraph boundary, the `drain()` call required for short-lived processes, both function-shaped and class-shaped (`__call__`) observers, and how observers see structured pre/post state without nodes having to log anything themselves.

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
