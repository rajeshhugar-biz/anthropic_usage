# Building with the Claude API

Hands-on notebooks for learning to build applications with the Claude API: prompting, evaluation, tool use, the text editor tool, and web search.

## Setup

```bash
uv sync
```

Create a `.env` file in the repo root with your API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Each module's notebooks read/write their data files using relative paths (e.g. `dataset.json`), so run them with the working directory set to the notebook's own folder. In VS Code this is already configured via `.vscode/settings.json` (`jupyter.notebookFileRoot": "${fileDirname}"`).

## Structure

| Folder | Topic | Notebooks |
|---|---|---|
| [01_getting_started/](01_getting_started/) | Claude API basics | `01_claude_api_basics.ipynb` — create a client, send messages, build a simple multi-turn chat loop |
| [02_prompt_evaluations/](02_prompt_evaluations/) | Evaluating prompts | `01_prompt_evals.ipynb` (starter) → `02_prompt_evals_fns.ipynb` (code-graded validation) → `03_prompt_evals_grader.ipynb` (model-graded validation) → `04_prompt_evals_completed.ipynb` (combined solution) |
| [03_prompting_techniques/](03_prompting_techniques/) | Prompt engineering techniques | `01_prompting.ipynb` (starter) → `02_prompting_completed.ipynb` (solution) — a reusable `PromptEvaluator` that generates test datasets, runs a prompt against them, and builds an HTML/JSON report |
| [04_tool_use/](04_tool_use/) | Tool use building blocks | `01_tools.ipynb` (manual walkthrough of tool schemas and the tool-call/tool-result cycle) → `02_tools_conversation_loop.ipynb` (condensed into a `run_conversation` loop) → `03_tool_streaming.ipynb` (starter) → `04_tool_streaming_completed.ipynb` (solution) — adds streaming to the tool-use loop |
| [05_text_editor_tool/](05_text_editor_tool/) | Text editor tool | `01_text_editor_tool.ipynb` — implements Claude's text editor tool (view/create/string-replace/insert) with backups, wired into the agent loop |
| [06_web_search/](06_web_search/) | Web search tool | `01_web_search.ipynb` (starter) → `02_web_search_completed.ipynb` (solution) — uses Claude's built-in web search tool |

Notebooks are ordered as starter → solution within each module; work through a folder top to bottom.

## Other files

- `Claude_api_notes.md` — running notes on the Claude API taken while building these notebooks.
- `main.py` — minimal script entry point.
