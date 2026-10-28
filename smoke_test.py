"""End-to-end smoke test. Exits 0 only if the local agent really used its MCP tools.

    uv run smoke_test.py            # preflight + full agent run (~1 min on CPU)
    uv run smoke_test.py --preflight # just check Ollama, the model, npx and uvx
"""

import asyncio
import json
import os
import shutil
import sys
import urllib.request

import main

OLLAMA = "http://localhost:11434"


def preflight() -> None:
    for exe in ("npx", "uvx"):
        assert shutil.which(exe), f"{exe} not found on PATH"
    with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=5) as r:
        models = [m["name"] for m in json.load(r)["models"]]
    assert main.MODEL in models, f"model {main.MODEL!r} not pulled; have {models}"
    print(f"preflight ok: ollama up, {main.MODEL} present, npx+uvx on PATH")


def check(cond: bool, msg: str, failures: list) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def full_run() -> int:
    out = asyncio.run(main.run())
    calls = out["tool_calls"]
    ok_calls = [c for c in calls if not c["is_error"]]
    names = [c["tool"] for c in ok_calls]
    report = out["report"] or ""
    f: list[str] = []

    check(any(c["tool"] == "filesystem_read_text_file" and "release_notes.md" in json.dumps(c["args"])
              for c in ok_calls), "agent read release_notes.md through the filesystem MCP server", f)
    check("fetch_fetch" in names, "agent fetched the web page through the fetch MCP server", f)
    check("filesystem_write_file" in names, "agent wrote the report through the filesystem MCP server", f)
    check("BLUE-HERON-7" in out["answers"]["local_docs"], "step 1 answer has the codename (only in the local file)", f)
    check("Priya Raman" in out["answers"]["local_docs"], "step 1 answer has the release manager (only in team.md)", f)
    check(main.REPORT.exists(), f"report file exists at {main.REPORT}", f)
    check("BLUE-HERON-7" in report and "Priya Raman" in report, "report contains the facts from the local files", f)
    check("example" in report.lower() or "documentation" in report.lower(), "report mentions the fetched page", f)

    print(f"\n{len(calls)} tool calls ({len(calls) - len(ok_calls)} errored); "
          f"{8 - len(f)}/8 checks passed")
    return 1 if f else 0


if __name__ == "__main__":
    preflight()
    if "--preflight" in sys.argv:
        sys.exit(0)
    sys.exit(full_run())
