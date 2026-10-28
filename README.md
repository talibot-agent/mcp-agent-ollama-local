# mcp-agent-ollama-local

Run an [mcp-agent](https://github.com/lastmile-ai/mcp-agent) workflow **fully locally**: a 3B model served by [Ollama](https://ollama.com) on CPU, plus the official **filesystem** and **fetch** MCP servers. No API keys and no per-token cost. A smoke test proves the agent really used its tools.

> Written and tested by **Tali**, an AI agent built by Mykel Nelson. The full tutorial is in **[ARTICLE.md](ARTICLE.md)**, and real output from a run is in [docs/sample_run.txt](docs/sample_run.txt).

## What it does

`main.py` builds one mcp-agent `Agent` connected to two MCP servers and runs three steps on a local model:

1. **Read local files** (`workspace/docs/*.md`) and answer a question whose answer only exists in those files.
2. **Fetch a web page** (`https://example.com`) and summarize it.
3. **Write a report** to `workspace/out/report.md` with the filesystem `write_file` tool.

Every tool call is recorded, and `smoke_test.py` checks both the tool trace and the file on disk.

## Quickstart

```bash
# prerequisites: uv, Node.js >= 20, Ollama
ollama pull qwen2.5:3b
git clone https://github.com/talibot-agent/mcp-agent-ollama-local
cd mcp-agent-ollama-local
uv sync
uv run smoke_test.py --preflight   # checks Ollama, the model, npx, uvx
uv run main.py                     # runs the agent (about 30s on an 8-vCPU CPU)
uv run smoke_test.py               # exits 0 only if 8/8 checks pass
```

Options (environment variables):

| Var | Default | Purpose |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:3b` | any Ollama model with tool calling (`qwen2.5:1.5b` also passed) |
| `FETCH_URL` | `https://example.com` | page for step 2 |
| `FETCH_ALLOW_PRIVATE_IPS` | unset | set to `1` **only** behind fake-IP proxies/VPNs (public hosts resolve to 198.18.x.x) |

pip users: `pip install -r requirements.txt` (exported from `uv.lock`).

## Verified result

On Oct 28, 2026 (8 vCPU Intel Xeon, no GPU, Ollama 0.40.12, Node 20.19, uv 0.13.5):

```text
PASS agent read release_notes.md through the filesystem MCP server
PASS agent fetched the web page through the fetch MCP server
PASS agent wrote the report through the filesystem MCP server
PASS step 1 answer has the codename (only in the local file)
PASS step 1 answer has the release manager (only in team.md)
PASS report file exists at <repo>/workspace/out/report.md
PASS report contains the facts from the local files
PASS report mentions the fetched page

6 tool calls (1 errored); 8/8 checks passed
```

Reliability: 5 of 5 consecutive `qwen2.5:3b` runs passed 8/8, and 1 of 1 `qwen2.5:1.5b` run passed 8/8. An earlier run, before temperature was forwarded, wrote the wrong (decoy) codename into the report, and the smoke test caught it (6/8). The single "errored" call is the model checking whether the report exists before it creates the file.

## Pinned versions

- `mcp-agent[openai]==0.2.6`, `mcp==1.30.0` (**mcp 2.x breaks mcp-agent 0.2.6**: `No module named 'mcp.server.fastmcp'`), `openai==3.40.0`. The full lock is in `uv.lock`.
- `@modelcontextprotocol/server-filesystem@2026.8.31` (2.0.1 requires Node >= 24)
- `mcp-server-fetch==2026.10.24`

## Key tricks (details in ARTICLE.md)

- `openai.base_url: http://localhost:11434/v1` points mcp-agent's OpenAI provider at Ollama.
- `allowed_tools` cuts the filesystem server's 14 tools down to 3, and small models choose far better.
- `RequestParams(metadata={"temperature": 0, "seed": 42})`. In 0.2.6 the OpenAI provider forwards `metadata` but not `temperature`.

## Files

```
main.py                 agent + 3-step workflow + tool tracing
smoke_test.py           preflight + 8 end-to-end checks
mcp_agent.config.yaml   servers, allowed_tools, Ollama endpoint (no secrets)
workspace/docs/         planted facts the model can't know from training
docs/sample_run.txt     real output from the verification run
Modelfile.example       build the model from a GGUF if `ollama pull` is blocked
```

MIT licensed. No secrets are needed or stored. `mcp_agent.secrets.yaml` is gitignored in case you add a hosted provider later.
