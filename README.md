# Sovara for LangChain

Record LangChain and LangGraph model and tool calls in Sovara, and add relevant lessons to each
model request.

| Package | Install | Docs |
| --- | --- | --- |
| [`python/`](python) | `uv add "sovara-langchain[agent]"` or `pip install "sovara-langchain[agent]"` | [README](python/README.md) |
| [`typescript/`](typescript) | `npm install @sovara/langchain @sovara/sdk` | [README](typescript/README.md) |

Runnable examples are in [`examples/`](examples).

## Development

The packages use the Sovara SDK from a [sovara](https://github.com/SovaraLabs/sovara) checkout
next to this repository (`../sovara`).
CI checks out sovara's `main` branch (set the `SOVARA_REF` repository variable to use
another one) with the `SOVARA_REPO_TOKEN` secret, a token that can read `SovaraLabs/sovara`.

Python (`python/`):

```sh
cd python
uv sync
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
```

TypeScript (`typescript/`): build the SDK first, then run the checks:

```sh
(cd ../sovara/packages/typescript-sdk && npm ci && npm run build)
cd typescript
npm ci
npm run check
```

### End-to-end tests

These run against a real Sovara server, started from the sovara checkout (set
`SOVARA_SOURCE_DIR` to use another one). Run `uv sync` there and start its Postgres with
`./dev.sh` or `docker compose -f deploy/dev/compose.yaml up -d`. No model is called.

```sh
cd python && uv run pytest -q -m e2e
```

Without a sovara checkout they are skipped. CI doesn't run them.

## License

MIT.
