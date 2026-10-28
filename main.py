"""Run an mcp-agent workflow fully locally: Ollama (qwen2.5) + filesystem & fetch MCP servers.

No API keys. Usage:  uv run main.py   (see README.md for setup)
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

ROOT = Path(__file__).parent.resolve()
WORKSPACE = ROOT / "workspace"
REPORT = WORKSPACE / "out" / "report.md"
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
FETCH_URL = os.environ.get("FETCH_URL", "https://example.com")

# Every MCP tool call the model makes is recorded here so we can prove
# the answers came from tools, not from the model's imagination.
TOOL_LOG: list[dict] = []


class TracedAgent(Agent):
    async def call_tool(self, name, arguments=None, server_name=None):
        t0 = time.time()
        result = await super().call_tool(name, arguments, server_name)
        TOOL_LOG.append(
            {
                "tool": name,
                "args": arguments,
                "is_error": bool(getattr(result, "isError", False)),
                "secs": round(time.time() - t0, 2),
            }
        )
        print(f"  [tool] {name}({json.dumps(arguments)}) error={TOOL_LOG[-1]['is_error']}")
        return result


app = MCPApp(name="local_research_agent")


async def run() -> dict:
    (WORKSPACE / "out").mkdir(parents=True, exist_ok=True)
    if REPORT.exists():
        REPORT.unlink()

    async with app.run() as agent_app:
        ctx = agent_app.context
        # Sandbox the filesystem server to ./workspace only.
        ctx.config.mcp.servers["filesystem"].args.append(str(WORKSPACE))
        # Only needed behind fake-IP proxies/VPNs where public hosts resolve to 198.18.x.x.
        if os.environ.get("FETCH_ALLOW_PRIVATE_IPS") == "1":
            ctx.config.mcp.servers["fetch"].args.append("--allow-private-ips")

        agent = TracedAgent(
            name="researcher",
            instruction=(
                "You are a careful research assistant. You MUST use your tools to "
                "read files and fetch web pages; never guess file contents. "
                f"All files live under {WORKSPACE}. Always use absolute paths. "
                "Answer briefly."
            ),
            server_names=["filesystem", "fetch"],
        )

        async with agent:
            tools = await agent.list_tools()
            print(f"Tools exposed to the model: {[t.name for t in tools.tools]}")

            llm = await agent.attach_llm(OpenAIAugmentedLLM)
            params = RequestParams(
                model=MODEL,
                max_iterations=6,
                maxTokens=1024,
                # The OpenAI provider forwards `metadata` verbatim into the request body,
                # which is how we get deterministic decoding out of Ollama.
                metadata={"temperature": 0, "seed": 42},
            )
            answers = {}

            steps = {
                "local_docs": (
                    f"Read {WORKSPACE}/docs/release_notes.md and {WORKSPACE}/docs/team.md. "
                    "What is the internal codename of v2.4.0 and who was its release manager?"
                ),
                "web_fetch": (
                    f"Fetch {FETCH_URL} and tell me, in one sentence, what that page says it is for."
                ),
                "write_report": (
                    f"Write a short markdown report to {REPORT} using the write_file tool. "
                    "It must contain: the v2.4.0 codename, its release manager, and a one-line "
                    "summary of the web page you fetched. Then reply 'saved'."
                ),
            }
            for key, prompt in steps.items():
                print(f"\n=== step: {key}\n> {prompt}")
                t0 = time.time()
                answers[key] = await llm.generate_str(message=prompt, request_params=params)
                print(f"< {answers[key].strip()}\n  ({time.time() - t0:.1f}s)")

    out = {
        "model": MODEL,
        "answers": answers,
        "tool_calls": TOOL_LOG,
        "report": REPORT.read_text() if REPORT.exists() else None,
    }
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "last_run.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    start = time.time()
    out = asyncio.run(run())
    out["total_secs"] = round(time.time() - start, 1)
    print(f"\nReport file:\n{out['report']}")
    print(f"Total run time: {out['total_secs']}s  (full trace: logs/last_run.json)")
    sys.exit(0 if out["report"] else 1)
