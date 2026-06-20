# Claude API & Agent Development — Course Notes

> Notes on the Claude API: models, requests, prompt engineering & evals, tool use, RAG, advanced features, MCP, Claude Code, and agents vs. workflows.

## Table of Contents

- [Overview of Claude Models](#overview-of-claude-models)
- [Accessing the API](#accessing-the-api)
- [Making a Request](#making-a-request)
- [Multi-Turn Conversations](#multi-turn-conversations)
- [System Prompts](#system-prompts)
- [Temperature](#temperature)
- [Response Streaming](#response-streaming)
- [Controlling Model Output](#controlling-model-output)
- [Structured Data](#structured-data)
- [Prompt Evaluation](#prompt-evaluation)
- [A Typical Eval Workflow](#a-typical-eval-workflow)
- [Generating Test Datasets](#generating-test-datasets)
- [Running the Eval](#running-the-eval)
- [Model Based Grading](#model-based-grading)
- [Code Based Grading](#code-based-grading)
- [Prompt Engineering](#prompt-engineering)
- [Being Clear and Direct](#being-clear-and-direct)
- [Being Specific](#being-specific)
- [Structure with XML Tags](#structure-with-xml-tags)
- [Providing Examples](#providing-examples)
- [Introducing Tool Use](#introducing-tool-use)
- [Project Overview](#project-overview)
- [Tool Functions](#tool-functions)
- [Tool Schemas](#tool-schemas)
- [Handling Message Blocks](#handling-message-blocks)
- [Sending Tool Results](#sending-tool-results)
- [Multi-Turn Conversations with Tools](#multi-turn-conversations-with-tools)
- [Implementing Multiple Turns](#implementing-multiple-turns)
- [Using Multiple Tools](#using-multiple-tools)
- [The Batch Tool](#the-batch-tool)
- [Tools for Structured Data](#tools-for-structured-data)
- [Fine Grained Tool Calling](#fine-grained-tool-calling)
- [The Text Edit Tool](#the-text-edit-tool)
- [The Web Search Tool](#the-web-search-tool)
- [Introducing Retrieval Augmented Generation](#introducing-retrieval-augmented-generation)
- [Text Chunking Strategies](#text-chunking-strategies)
- [Text Embeddings](#text-embeddings)
- [The Full RAG Flow](#the-full-rag-flow)
- [Implementing the RAG Flow](#implementing-the-rag-flow)
- [BM25 Lexical Search](#bm25-lexical-search)
- [A Multi-Index RAG Pipeline](#a-multi-index-rag-pipeline)
- [Reranking Results](#reranking-results)
- [Contextual Retrieval](#contextual-retrieval)
- [Extended Thinking](#extended-thinking)
- [Image Support](#image-support)
- [PDF Support](#pdf-support)
- [Citations](#citations)
- [Prompt Caching](#prompt-caching)
- [Rules of Prompt Caching](#rules-of-prompt-caching)
- [Prompt Caching in Action](#prompt-caching-in-action)
- [Code Execution and the Files API](#code-execution-and-the-files-api)
- [Introducing MCP](#introducing-mcp)
- [MCP Clients](#mcp-clients)
- [Project Setup](#project-setup)
- [Defining Tools with MCP](#defining-tools-with-mcp)
- [The Server Inspector](#the-server-inspector)
- [Implementing a Client](#implementing-a-client)
- [Defining Resources](#defining-resources)
- [Accessing Resources](#accessing-resources)
- [Defining Prompts](#defining-prompts)
- [Prompts in the Client](#prompts-in-the-client)
- [Anthropic Apps](#anthropic-apps)
- [Claude Code Setup](#claude-code-setup)
- [Claude Code in Action](#claude-code-in-action)
- [Enhancements with MCP Servers](#enhancements-with-mcp-servers)
- [Parallelizing Claude Code](#parallelizing-claude-code)
- [Automated Debugging](#automated-debugging)
- [Computer Use](#computer-use)
- [How Computer Use Works](#how-computer-use-works)
- [Agents and Workflows](#agents-and-workflows)
- [Parallelization Workflows](#parallelization-workflows)
- [Chaining Workflows](#chaining-workflows)
- [Routing Workflows](#routing-workflows)
- [Agents and Tools](#agents-and-tools)
- [Environment Inspection](#environment-inspection)
- [Workflows vs Agents](#workflows-vs-agents)

---

## Overview of Claude Models

Claude has three model families optimized for different priorities:

- **Opus** — highest intelligence model for complex, multi-step tasks requiring deep reasoning and planning. Trade-off: higher cost and latency.
- **Sonnet** — balanced model with good intelligence, speed, and cost efficiency. Strong coding abilities and precise code editing. Best for most practical use cases.
- **Haiku** — fastest model optimized for speed and cost efficiency. No reasoning capabilities like Opus/Sonnet. Best for real-time user interactions and high-volume processing.

**Selection framework:**
- Intelligence priority → Opus
- Speed priority → Haiku
- Balanced requirements → Sonnet

**Common approach** — use multiple models in the same application based on specific task requirements rather than a single model selection.

All models share core capabilities: text generation, coding, image analysis. The main difference is optimization focus.

---

## Accessing the API

**API Access Flow** — 5-step process from user input to response display.

**Step 1:** Client sends user text to the developer's server (never access the Anthropic API directly from client apps — keep the API key secret).

**Step 2:** Server makes a request to the Anthropic API using an SDK (Python, TypeScript, JavaScript, Go, Ruby) or plain HTTP. Required parameters = API key + model name + messages list + `max_tokens` limit.

**Step 3:** Text generation has 4 stages:
- **Tokenization** — breaking input into tokens (words/word parts/symbols/spaces)
- **Embedding** — converting tokens to number lists representing all possible word meanings
- **Contextualization** — adjusting embeddings based on neighboring tokens to determine precise meaning
- **Generation** — output layer produces probabilities for the next word; model selects using probability + randomness, adds the selected word, repeats

**Step 4:** Model stops when `max_tokens` is reached or a special `end_of_sequence` token is generated.

**Step 5:** API returns the response with generated text + usage counts + `stop_reason` to the server, which sends it to the client for display.

**Key terms:**
- **Token** — text chunk (word/part/symbol)
- **Embedding** — numerical representation of word meanings
- **Contextualization** — meaning refinement using neighboring words
- **max_tokens** — generation length limit
- **stop_reason** — why the model stopped generating

---

## Making a Request

**Making an API Request to Anthropic** — 4 setup steps + understanding message structure.

**Setup steps:**
1. Install packages — `pip install anthropic python-dotenv` in a Jupyter notebook
2. Store API key — create a `.env` file with `ANTHROPIC_API_KEY="your_key"` (ignore in version control)
3. Load environment variable — use `python-dotenv` to securely load the API key
4. Create client — initialize the Anthropic client and define the model variable (`claude-3-sonnet`)

**API request structure:**
- Function — `client.messages.create()`
- Required arguments — `model`, `max_tokens`, `messages`
- `model` — name of the Claude model to use
- `max_tokens` — safety limit for generation length (not a target length)
- `messages` — list containing conversation exchanges

**Message types:**
- User message — `{"role": "user", "content": "your text"}` (human-authored content)
- Assistant message — contains model-generated responses

**Response access:**
- Full response — contains metadata and nested structure
- Text only — `message.content[0].text` extracts just the generated text

**Example request:**

```python
client.messages.create(
    model=model,
    max_tokens=1000,
    messages=[{"role": "user", "content": "What is quantum computing?"}]
)
```

---

## Multi-Turn Conversations

**Multi-Turn Conversations** — conversations with multiple back-and-forth exchanges that maintain context.

**Key limitation:** the Anthropic API stores no messages. Each request is independent with no memory of previous exchanges.

**Solution requires two steps:**
1. Manually maintain a message list in code
2. Send the entire conversation history with every follow-up request

Message structure = list of dictionaries with `role` (`user`/`assistant`) and `content` fields.

**Conversation flow:**
- Send initial user message
- Receive assistant response
- Append assistant response to message history
- Add new user message to history
- Send complete history for context-aware follow-up

**Helper functions needed:**
- `add_user_message(messages, text)` — appends a user message to history
- `add_assistant_message(messages, text)` — appends an assistant response to history
- `chat(messages)` — sends the message history to the API and returns the response

Without message history, responses lack context and continuity. With complete history, Claude maintains conversation context and provides relevant follow-ups.

---

## System Prompts

**System Prompts** — technique to customize Claude's response style and tone by assigning it a specific role or behavior pattern.

- **Implementation** — pass the system prompt as a plain string to the create function using the `system` keyword argument.
- **Purpose** — control *how* Claude responds rather than *what* it responds. Example: a "math tutor" role makes Claude give hints instead of direct answers.
- **Structure** — first line typically assigns a role ("You are a patient math tutor"), followed by specific behavioral instructions.
- **Key principle** — system prompts guide response approach, not content. The same question gets different treatment based on the assigned role.
- **Technical implementation** — create a params dictionary, conditionally add the `system` key if a prompt is provided, pass params to the create function with `**` unpacking. Handle the `None` case by excluding the `system` parameter entirely.
- **Use case example** — a math tutor that gives guidance/hints rather than complete solutions, encouraging student thinking over direct answers.

---

## Temperature

**Temperature** — parameter (0–1) that controls randomness in Claude's text generation by influencing token selection probabilities.

**Text generation process:** input text → tokenization → probability assignment to possible next tokens → token selection based on probabilities → repeat.

**Temperature effects:**
- Temperature 0 — deterministic output, always selects the highest-probability token
- Higher temperature — increases the chance of selecting lower-probability tokens, more creative/unexpected outputs

**Usage guidelines:**
- Low temperature (near 0) — data extraction, factual tasks requiring consistency
- High temperature (near 1) — creative tasks like brainstorming, writing, jokes, marketing

**Implementation:** add a `temperature` parameter to model API calls. Higher values don't guarantee different outputs, just increase the probability of variation.

**Key insight:** temperature directly manipulates the probability distribution of next-token selection, making high-probability tokens more or less dominant in the selection process.

---

## Response Streaming

**Response Streaming** — technique to display AI responses chunk-by-chunk as they're generated instead of waiting for the complete response.

**Problem solved:** AI responses can take 10–30 seconds. Users expect immediate feedback, not just spinners.

**How it works:**
1. Server sends user message to Claude
2. Claude immediately sends an initial response (no text, just acknowledgment)
3. A stream of events follows, each containing text chunks
4. Server forwards chunks to the frontend for real-time display

**Event types:**
- `message_start` — initial acknowledgment
- `content_block_start` — text generation begins
- `content_block_delta` — contains the actual text chunks (most important)
- `content_block_stop` / `message_stop` — generation complete

**Implementation:**
- Basic — `client.messages.create(stream=True)` returns an event iterator
- Simplified — `client.messages.stream()` with `text_stream` property extracts just the text
- Final message — `stream.get_final_message()` assembles all chunks for storage

**Key benefits:** better UX through immediate response visibility, complete message capture for database storage.

---

## Controlling Model Output

Two key techniques beyond prompt modification:

### 1. Pre-filling assistant messages

Manually adding an assistant message at the end of the conversation to steer the response direction.

How it works:
- Assemble a messages list with the user prompt + a manual assistant message
- Claude sees the assistant message as already-authored content
- Claude continues the response from the exact end of the pre-filled text
- The response gets steered toward the pre-filled direction

Key point: Claude continues from the exact endpoint of the pre-fill, not complete sentences. You must stitch together the pre-fill + generated response.

Example: pre-fill "Coffee is better because" → Claude continues with a justification for coffee.

### 2. Stop sequences

Force Claude to halt generation when a specific string appears.

How it works:
- Provide a stop sequence string in the chat function
- When Claude generates that exact string, the response immediately stops
- The generated stop-sequence text is not included in the final output

Example: prompt "count 1 to 10" + stop sequence "five" → output stops at "four, " (five not included).

Refinement: stop sequence ", five" → clean output "one, two, three, four".

Both techniques provide precise control over response direction and length without changing the core prompts.

---

## Structured Data

**Structured Data Generation** — technique using assistant message pre-filling + stop sequences to get raw output without Claude's natural explanatory headers/footers.

**Problem** — Claude automatically adds markdown formatting, headers, and commentary when generating JSON/code/structured content. Users often want just the raw data for copy/paste functionality.

**Solution pattern:**
1. User message — request for structured data
2. Assistant message pre-fill — opening delimiter (e.g., `` ```json ``)
3. Stop sequence — closing delimiter (e.g., `` ``` ``)

**How it works** — Claude sees the pre-filled message, assumes it already started the response, generates only the requested content, and stops when hitting the delimiter.

**Result** — raw structured data output with no extra formatting or commentary.

**Application** — works for any structured data type (JSON, Python code, lists, etc.), not just JSON. Use whenever you need clean, parseable output without explanatory text.

**Key benefit** — output can be directly used/copied without manual selection or parsing of unwanted text.

---

## Prompt Evaluation

**Prompt Engineering** — techniques for writing/editing prompts to help Claude understand requests and desired responses.

**Prompt Evaluation** — automated testing of prompts using objective metrics to measure effectiveness.

**Three paths after writing a prompt:**
1. Test once/twice, deploy to production *(trap)*
2. Test with custom inputs, minor tweaks for corner cases *(trap)*
3. Run through an evaluation pipeline for objective scoring *(recommended)*

**Key takeaway:** engineers commonly under-test prompts. Use evaluation pipelines to get objective performance scores before iterating and deploying prompts.

---

## A Typical Eval Workflow

**Typical Eval Workflow** — 6-step iterative process for prompt improvement.

1. **Write initial prompt draft** — create a baseline prompt to optimize
2. **Create evaluation dataset** — collection of test inputs (can be 3 examples or thousands, hand-written or LLM-generated)
3. **Generate prompt variations** — interpolate each dataset input into the prompt template
4. **Get LLM responses** — feed each prompt variation to Claude, collect outputs
5. **Grade responses** — use a grader system to score each response (e.g., 1–10 scale), average scores for overall prompt performance
6. **Iterate** — modify the prompt based on scores, repeat the entire process, compare versions

**Key points:** no standard methodology exists; many open-source/paid tools are available. Can start simple with a custom implementation. Grading complexity varies. Objective scoring enables systematic prompt improvement through A/B comparison.

---

## Generating Test Datasets

Custom prompt evaluation workflow = build prompt + generate test dataset + evaluate performance.

**Goal** — an AWS code-assistance prompt that outputs only Python, JSON config, or regex without explanations.

**Dataset generation approaches** — manual assembly or automated with Claude (use faster models like Haiku for generation).

**Dataset structure** — array of JSON objects with a `task` property describing user requests.

**Generation process:**
1. Prompt Claude to create test cases
2. Use pre-filling with assistant message `` ```json ``
3. Set stop sequence `` ``` ``
4. Parse response as JSON
5. Save to file

**Key implementation** — a `generate_dataset()` function that sends a prompt to Claude, gets a structured JSON response of test tasks, and saves it to a `dataset.json` file for later evaluation use.

A test dataset enables systematic evaluation by running the prompt against multiple input scenarios to measure performance consistency.

---

## Running the Eval

**Eval execution process** — merging test cases with prompts, running them through the LLM, and grading outputs.

**Test case** — individual record from the dataset (JSON object).

**Three core functions:**
- `run_prompt` — merges test case with prompt, sends to Claude, returns output
- `run_test_case` — calls `run_prompt`, grades the result, returns a summary dictionary
- `run_eval` — loops through the dataset, calls `run_test_case` for each, assembles results

**Basic prompt structure** — `"Please solve the following task: [test_case_task]"` (v1 starting point).

**Current limitations** — no output formatting instructions, hardcoded scoring (`score=10`), verbose Claude responses.

**Runtime** — ~31 seconds with the Haiku model for full dataset execution.

**Output format** — array of objects containing Claude's output, the original test case, and a score.

**Next step** — implement a proper grading system to replace hardcoded scores.

**Eval pipeline core** — dataset + prompt + LLM + grader, with minimal code complexity.

---

## Model Based Grading

**Model Based Grading** — evaluation system that takes model outputs and assigns objective scores (typically a 1–10 scale, 10 = highest quality).

**Three grader types:**
- **Code graders** — programmatic checks (length, word presence, syntax validation, readability scores)
- **Model graders** — additional API call to evaluate the original model output; highly flexible for quality/instruction-following assessment
- **Human graders** — a person evaluates responses; most flexible but time-consuming and tedious

**Key requirements:** must return an objective signal (usually a numerical score). Define evaluation criteria upfront.

**Implementation pattern for model graders:**
- Create a detailed prompt requesting strengths/weaknesses/reasoning/score (not just a score alone, to avoid default middling scores)
- Use JSON response format with a pre-filled assistant message and stop sequences
- Parse the returned JSON for score and reasoning
- Calculate average scores across test cases for the final metric

Model graders offer high flexibility but may be inconsistent. They still provide an objective baseline for prompt optimization.

---

## Code Based Grading

**Code Based Grading** — automated validation system for LLM outputs containing code, JSON, or regex.

**Core implementation:**
- `validate_json()` — attempts JSON parsing, returns 10 if valid, 0 if error
- `validate_python()` — attempts AST parsing, returns 10 if valid, 0 if error
- `validate_regex()` — attempts regex compilation, returns 10 if valid, 0 if error

**Dataset requirements:**
- Must include a `format` key specifying the expected output type (JSON/Python/RegEx)
- Updated via prompt template modification for automated dataset generation

**Prompt engineering:**
- Instruct the model to respond only with raw code/JSON/regex
- No comments, explanations, or commentary
- Use a pre-filled assistant message with `` ```code``` `` blocks
- Add stop sequences to extract clean output

**Scoring system:**
- Final score = `(model_score + syntax_score) / 2`
- Combines semantic evaluation with syntax validation
- Enables measurement of both correctness and technical validity

**Key limitation** — requires a known expected format for proper validator selection.

---

## Prompt Engineering

**Prompt Engineering** — improving prompts to get more reliable, higher-quality outputs from language models.

**Module structure:** start with an initial poor prompt → apply prompt engineering techniques step-by-step → evaluate improvements after each technique → observe performance gains over time.

**Example goal** — generate a one-day meal plan for athletes based on height, weight, physical goal, dietary restrictions.

**Technical setup:**
- Updated eval pipeline with a flexible prompt evaluator class
- Supports concurrency (adjust `max_concurrent_tasks` based on rate limits)
- `generate_dataset()` method creates test cases with specified inputs
- `run_prompt()` function processes each test case individually

**Key components:**
- `prompt_input_spec` — dictionary defining required prompt inputs
- `extra_criteria` — additional validation requirements for model grading
- `output.html` — formatted evaluation report showing test case results and scores

**Process:** write initial prompt → interpolate test case inputs → run evaluation → apply engineering techniques → re-evaluate → repeat until satisfactory performance.

**Initial results:** expect poor scores (example: 2.32) with basic prompts, especially when using less capable models. Scores improve as techniques are applied.

---

## Being Clear and Direct

**Being Clear and Direct** — use simple, direct language with action verbs in the first line of prompts to specify the exact task.

**First line importance** — the most critical part of the prompt; it sets the foundation for the AI response.

**Structure** — action verb + clear task description + output specifications.

**Examples:**
- "Write three paragraphs about how solar panels work"
- "Identify three countries that use geothermal energy and for each include generation stats"
- "Generate a one-day meal plan for an athlete that meets their dietary restrictions"

**Key components** — action verb at start + direct task statement + expected output details.

**Result** — improved prompt performance (example showed a score increase from 2.32 to 3.92).

---

## Being Specific

**Being Specific** — adding guidelines or steps to direct model output in a particular direction.

**Two types of guidelines:**
- **Type A (Attributes)** — list qualities/attributes desired in the output (length, structure, format)
- **Type B (Steps)** — provide specific steps for the model to follow in its reasoning process

Type A controls output characteristics. Type B controls how the model arrives at the answer. Both techniques are often combined in professional prompts.

**When to use:**
- Type A (attributes) — recommended for almost all prompts
- Type B (steps) — use for complex problems where you want the model to consider a broader perspective or additional viewpoints it might not naturally consider

**Example improvement:** the meal planning prompt's score jumped from 3.92 to 7.86 when guidelines were added, demonstrating significant quality improvement through specificity.

---

## Structure with XML Tags

**XML Tags for Prompt Structure** — using XML tags to organize and delineate different content sections within prompts to improve AI comprehension.

**Purpose** — when interpolating large amounts of content into prompts, XML tags help AI models distinguish between different types of information and understand text grouping.

**Implementation** — wrap content sections in descriptive XML tags like `<sales_records></sales_records>` or `<my_code></my_code>` rather than dumping unstructured text.

**Tag naming** — use descriptive, specific tag names (e.g., "sales_records" is better than "data") to provide context about the nature of the content.

**Example use case** — a debugging prompt with mixed code and documentation becomes clearer when separated into `<my_code>` and `<docs>` tags.

**Benefits** — makes prompt structure obvious to the AI, reduces confusion about content boundaries, improves output quality even for smaller content blocks.

**Application** — can wrap any interpolated content, like `<athlete_information>`, even when the content is short, to clarify that it's external input requiring consideration.

---

## Providing Examples

**One-shot/multi-shot prompting** — providing examples in prompts to guide model behavior. One-shot = single example, multi-shot = multiple examples.

**Implementation:** structure examples with XML tags containing sample input and ideal output. Always wrap examples clearly to distinguish them from the actual prompt content.

**Key applications:**
- Corner case handling (sarcasm detection, edge scenarios)
- Complex output formatting (JSON structures, specific formats)
- Clarifying expected response quality/style

**Best practices:**
- Add context for corner cases ("be especially careful with sarcasm")
- Include reasoning explaining why the output is ideal
- Use highest-scoring examples from prompt evaluations as templates
- Place examples after main instructions/guidelines

**Effectiveness boost:** combine examples with explanations of what makes them ideal to reinforce desired output characteristics.

---

## Introducing Tool Use

**Tool use** — method for Claude to access external information beyond its training data.

**Default limitation:** Claude only knows information from training data and lacks current/real-time information.

**Tool use flow:**
1. Send initial request to Claude + instructions for external data access
2. Claude evaluates if external data is needed, requests specific information
3. Server runs code to fetch the requested data from external sources
4. Send follow-up request to Claude with retrieved data
5. Claude generates the final response using the original prompt + external data

**Weather example:** user asks for current weather → Claude requests weather data → server calls weather API → Claude receives weather data → Claude provides an informed weather response.

**Key concept:** tools enable Claude to augment responses with live/current information by orchestrating external data retrieval between Claude's requests.

---

## Project Overview

**Goal** — teach Claude to set time-based reminders through tool implementation in a Jupyter notebook.

**Target interaction:** User: "Set reminder for doctor's appointment, week from Thursday" → Claude: "I will remind you at that point in time."

**Three core problems requiring tools:**
1. Time knowledge gap — Claude knows the current date but not the exact time
2. Time calculation errors — Claude sometimes miscalculates time-based addition (e.g., 379 days from January 13th, 1973)
3. No reminder mechanism — Claude understands the reminder concept but lacks implementation capability

**Three corresponding tools to build:**
1. Current datetime tool — gets the current date + time
2. Duration addition tool — adds a time duration to a datetime (e.g., current date + 20 days)
3. Reminder setting tool — actually sets the reminder

**Implementation approach** — one tool at a time, building toward multi-tool coordination.

---

## Tool Functions

**Tool Functions** — Python functions executed automatically when Claude needs extra information to help users.

**Key characteristics:**
- Plain Python functions called by Claude when it determines additional data is needed
- Must use descriptive function names and argument names
- Should validate inputs and raise errors with meaningful messages
- Error messages are visible to Claude, allowing it to retry with corrected parameters

**Best practices:**
1. Well-named functions and arguments
2. Input validation with immediate error raising for invalid inputs
3. Meaningful error messages that guide correction

**Example implementation pattern:**

```python
def get_current_datetime(date_format="%Y%m%d %H:%M:%S"):
    if not date_format:
        raise ValueError("date format cannot be empty")
    return datetime.now().strftime(date_format)
```

**Tool function workflow:** Claude identifies need for information → calls tool function → receives result or error → may retry with corrections if an error occurred.

**Purpose:** extend Claude's capabilities beyond its training data by providing access to real-time information like current datetime, weather, etc.

---

## Tool Schemas

**Tool Schemas** — JSON schema specifications that describe tool functions and their parameters for language models.

**JSON Schema** — data validation specification (not ML-specific) used to validate JSON data, adopted by the ML community for tool calling.

**Tool schema structure:**
- `name` — tool identifier
- `description` — 3–4 sentences explaining what the tool does, when to use it, and what data it returns
- `input_schema` — the actual JSON schema describing function arguments with types and descriptions

**Schema generation trick:**
1. Take the tool function to Claude.ai
2. Prompt: "write valid JSON schema spec for tool calling for this function, follow best practices in attached documentation"
3. Attach the Anthropic API documentation tool use page
4. Copy the generated schema

**Implementation pattern:**
- Name functions descriptively
- Name schemas as `[function_name]_schema`
- Import `ToolParam` from `anthropic.types`
- Wrap the schema dictionary with `ToolParam()` to prevent type errors

**Purpose** — inform Claude about available tools, required arguments, and usage context through a standardized JSON validation format.

---

## Handling Message Blocks

**Step 3: Making requests to Claude with tools** — include the tool schema in the request alongside the user message using the `tools` keyword argument containing the JSON schema specs.

**Multi-Block Messages**

Content structure change — messages now contain multiple blocks instead of just text blocks.

Tool response format — assistant message with:
- Text block — user-facing explanation
- Tool use block — contains function name + arguments for tool execution

**Message History Management**

Critical requirement — manually maintain conversation history since Claude stores nothing.

Multi-block handling — append the entire `response.content` (all blocks) to the messages list, not just the text.

Helper function updates needed — `add_user_message` and `add_assistant_message` functions must support multiple blocks instead of single text blocks only.

**Conversation flow** — user message → assistant response with tool use block → execute tool → respond back to Claude with full history.

---

## Sending Tool Results

**Tool Results** — results from executed tool functions sent back to Claude in follow-up requests.

**Process:** execute tool function requested by Claude → create a tool result block → send follow-up request with full conversation history.

**Tool result block structure:**
- `tool_use_id` — matches the ID from the original tool use block to pair requests with results
- `content` — tool function output converted to a string (usually JSON)
- `is_error` — boolean flag for function execution errors (default `false`)

**Tool use ID purpose** — links multiple tool requests to the correct results when Claude makes simultaneous tool calls. Each tool use gets a unique ID; tool results must reference matching IDs.

**Follow-up request requirements:**
- Include the complete message history (original user message + assistant tool use message + new user message with tool result)
- Must include the original tool schemas even if not using tools again
- The tool result block goes in a user message, not an assistant message

**Conversation flow:** user request → Claude assistant response (text + tool use blocks) → server executes tool → user message with tool result block → Claude's final response with integrated results.

---

## Multi-Turn Conversations with Tools

**Multi-Turn Tool Conversations** — conversations where Claude uses multiple tools sequentially to answer a single user query.

**Tool chaining process** — user asks a question → Claude requests first tool → tool executed → result returned → Claude requests second tool → tool executed → result returned → Claude provides final answer.

**Example flow** — user asks "what day is 103 days from today" → Claude calls `get_current_datetime` → Claude calls `add_duration_to_datetime` → Claude provides the answer.

**Implementation pattern** — a while loop that continues calling Claude until no more tool requests, checking each response for `tool_use` blocks.

**`run_conversation` function** — takes initial messages, loops through Claude calls, executes requested tools, adds results to the conversation, continues until the final response.

**Required refactors:**
- `add_user_message`/`add_assistant_message` — updated to handle multiple message blocks instead of just plain text
- `chat` function — accepts a `tools` parameter, returns the entire message instead of just the first text block
- `text_from_message` helper — extracts all text blocks from a message with multiple content blocks

**Key insight** — you can't predict how many tools a user query will require, so the system must handle arbitrary chains of tool calls automatically.

---

## Implementing Multiple Turns

**Multiple Turns Implementation** — continuously calling Claude until it stops requesting tools.

**Stop Reason field** — indicates why Claude stopped generating text.
- `stop_reason = "tool_use"` means Claude wants to call a tool
- Other values exist but `tool_use` is most commonly checked

**`run_conversation` function** — main loop that:
1. Calls Claude with messages + available tools
2. Adds the assistant response to conversation history
3. Checks `stop_reason` — if not `"tool_use"`, breaks the loop
4. If `tool_use`, calls the `run_tools` function
5. Adds tool results as a user message
6. Repeats until no more tool requests

**`run_tools` function** — processes multiple tool use blocks:
1. Filters `message.content` for blocks with `type="tool_use"`
2. Iterates through each tool request
3. Runs the appropriate tool function via the `run_tool` helper
4. Creates `tool_result` blocks with: `type="tool_result"`, `tool_use_id=original_id`, `content=JSON_encoded_output`, `is_error=boolean`
5. Returns a list of all tool result blocks

**`run_tool` function** — dispatcher that:
- Takes `tool_name` and `tool_input`
- Uses if-statements to match tool names to functions
- Executes the appropriate tool function
- Scalable for adding multiple tools

**Error handling** — try/except blocks around tool execution:
- Success — `is_error=false`, `content=tool_output`
- Failure — `is_error=true`, `content=error_message`

**Key architecture points:**
- Assistant messages can contain multiple blocks (text + multiple tool_use)
- Each tool_use block gets a separate tool_result response
- Tool results are sent back as a user message containing all results
- The process repeats until Claude provides a final text-only response

---

## Using Multiple Tools

**Multiple Tools Implementation** — adding additional tools to an existing tool system after the initial framework setup.

**Process — 3 steps:**
1. Add tool schemas to the `RunConversation` function's tools list
2. Add conditional cases in the `RunTool` function to handle new tool names
3. Implement the actual tool functions

**Key components:**
- `RunConversation` function — contains a tools list that makes Claude aware of available tools
- `RunTool` function — routes tool calls to the appropriate functions based on tool name
- Tool schemas — define the tool structure for the AI model
- Tool functions — the actual implementation code

**Example tools added:**
- `AddDurationToDateTime` — calculates date/time with a duration offset
- `SetReminder` — creates a reminder (mock implementation that prints confirmation)

**Tool chaining** — the AI can use multiple tools sequentially in a single conversation (e.g., calculate a date first, then set a reminder with the result).

**Message structure** — assistant responses can contain multiple blocks: text blocks + tool use blocks in the same message.

**Scalability** — after initial framework setup, adding new tools becomes a simple pattern of schema + routing + implementation.

---

## The Batch Tool

**Batch Tool** — a tool that enables Claude to run multiple tools in parallel within a single assistant message instead of making separate sequential requests.

**Problem:** Claude can technically send multiple tool use blocks in one message but rarely does so in practice, leading to unnecessary sequential tool calls.

**Solution:** create a batch tool schema that takes a list of invocations (each containing a tool name + arguments). Instead of calling tools directly, Claude calls the batch tool with an array of desired tool executions.

**Implementation:**
- Add a batch tool to the schema with an `invocations` parameter
- Create a `run_batch` function that iterates through the invocations list
- Extract the tool name and JSON-parsed arguments from each invocation
- Call `run_tool` for each requested tool
- Return a `batch_output` list containing results from all tool executions

**Mechanism:** tricks Claude into parallel tool execution by providing a higher-level abstraction that manually handles what multiple tool use blocks would accomplish automatically.

**Result:** a single request-response cycle instead of multiple sequential rounds for parallel-executable tasks.

---

## Tools for Structured Data

**Tools for Structured Data** — alternative method to extract structured JSON from data sources using Claude's tool system instead of message pre-fill and stop sequences.

**Key differences from prompt-based extraction:**
- More reliable output
- More complex setup
- Requires a JSON schema specification

**Core process:**
1. Define a JSON schema for the tool where inputs = desired data structure
2. Send the prompt + schema to Claude
3. Claude calls the tool with structured arguments matching the schema
4. Extract JSON from the tool use block (no tool result needed)

**Critical requirement** — force tool calling using the `tool_choice` parameter:
- `tool_choice = {"type": "tool", "name": "your_tool_name"}`
- Ensures Claude always calls the specified tool

**Implementation steps:**
1. Create a schema definition for the extraction tool
2. Update the `chat` function to accept a `tool_choice` parameter
3. Pass `tool_choice` to `client.messages.create()`
4. Access structured data from `response.content[0].input`

**Use cases** — when reliability is more important than simplicity. Prompt-based methods are better for quick/simple extractions; tools are better for complex/reliable extractions.

---

## Fine Grained Tool Calling

*(Transcript)*

**Tool Streaming** — streaming API responses while using tools with Claude.

**Key components:**
- Standard streaming returns `content_block_delta` events
- Tool streaming adds `input_json_delta` events with `partial_json` (chunk) and `snapshot` (cumulative sum)
- Implementation requires handling an additional event type in the streaming pipeline

**Fine-Grained Tool Calling** — feature that disables JSON validation for faster streaming.

**Default behavior:**
- Claude generates JSON chunks for tool arguments
- The API buffers chunks until a complete top-level key-value pair is generated
- Validates JSON against the schema before sending chunks to the server
- Results in delays followed by a burst of chunks arriving simultaneously

**Fine-grained mode (`fine_grained: true`):**
- Disables API-side JSON validation
- Sends chunks immediately as they're generated
- Provides a traditional streaming experience
- Requires client-side error handling for invalid JSON

**Trade-offs:**
- Default — slower but validated JSON
- Fine-grained — faster streaming but potential invalid JSON (like `"undefined"` instead of `null`)
- Invalid JSON in default mode gets wrapped as a string rather than a proper object structure

**Use cases:**
- Fine-grained — useful for immediate UI updates or early processing of tool arguments
- Default — sufficient when validation delays are acceptable

---

## The Text Edit Tool

**Text Editor Tool** — built-in Claude tool for file/text operations (read, write, create, replace, undo files/directories).

**Key characteristics:**
- Only JSON schema built into Claude; implementation must be custom-coded
- Schema stub sent to Claude gets auto-expanded to the full schema
- Schema type string varies by Claude model version (3.5 vs 3.7 have different dates)
- Enables Claude to act as a software engineer out-of-the-box

**Required implementation:**
- Custom class/functions to handle Claude's tool use requests
- Functions for: view files, string replace, create files, etc.
- Actual file system operations are not provided by Claude

**Workflow:**
1. Send a minimal schema stub to Claude (name + type with version-specific date)
2. Claude expands it to the full schema internally
3. Claude sends tool use requests
4. Custom implementation executes actual file operations
5. Results sent back to Claude

**Use cases:**
- Replicate AI code editor functionality
- File system operations where native editors are unavailable
- Automated code generation/refactoring
- Multi-file project manipulation

**Benefits** — approximates fancy code editor capabilities through API calls rather than GUI interaction.

---

## The Web Search Tool

**Web Search Tool** — built-in Claude tool for searching the web to find up-to-date/specialized information for user questions.

**Implementation** — no custom code needed; Claude handles search execution automatically.

**Schema requirements:**
- `type`: `"web_search_20250305"`
- `name`: `"web_search"`
- `max_uses`: number (limits total searches, default 5)
- `allowed_domains`: optional list to restrict search to specific domains

**Response structure:**
- Text blocks — Claude's explanatory text
- Tool use blocks — search queries Claude executed
- Web search result blocks — found pages (title, URL)
- Citation blocks — specific text supporting Claude's statements

**Key features:**
- Multiple searches possible per request (up to the `max_uses` limit)
- Domain restriction available for quality control
- Citation system links statements to source material

**UI rendering pattern:**
- Display text blocks as normal text
- Show search results as a reference list
- Highlight citations with source attribution (domain, title, URL, quoted text)

**Use case example** — restricting search to NIH.gov for medical/exercise advice ensures scientifically-backed information vs. generic web content.

---

## Introducing Retrieval Augmented Generation

**RAG** — Retrieval Augmented Generation technique for querying large documents using language models.

**Problem:** how to extract specific information from large documents (100–1000+ pages) using Claude without hitting context limits.

**Option 1 (Direct approach):** place the entire document text directly into the prompt.
- Limitations — hard token limits, decreased effectiveness with longer prompts, higher costs, slower processing

**Option 2 (RAG approach):** two-step process
- Step 1 — break the document into small chunks
- Step 2 — for user questions, find the most relevant chunks and include only those in the prompt

**RAG benefits** — model focuses on relevant content, scales to large/multiple documents, smaller prompts, lower costs, faster processing.

**RAG downsides** — more complexity, requires preprocessing, needs a search mechanism to find relevant chunks, no guarantee chunks contain complete context, multiple chunking strategies possible (equal portions vs. header-based).

**Key challenge** — defining relevance and the optimal chunking strategy for a specific use case.

RAG trades simplicity for scalability and efficiency but requires careful implementation and evaluation.

---

## Text Chunking Strategies

**Text Chunking Strategies** — process of dividing documents into smaller pieces for RAG pipelines.

**Core problem:** chunking quality directly impacts RAG performance. Poor chunking leads to irrelevant context retrieval (e.g., medical "bug" text retrieved for a software engineering query about bugs).

**Three main strategies:**

1. **Size-Based Chunking** — dividing text into equal-length strings
   - Pros: easy to implement, most common in production
   - Cons: cut-off words, lacks context
   - Solution: overlap strategy — include characters from neighboring chunks to preserve context
   - Trade-off: creates text duplication but improves chunk meaning

2. **Structure-Based Chunking** — dividing based on document structure (headers, paragraphs, sections)
   - Best for structured documents (markdown, HTML)
   - Limitation: requires guaranteed document formatting
   - Example: split on markdown headers (`##`) to create section-based chunks

3. **Semantic-Based Chunking** — using NLP to group related sentences/sections
   - Most advanced technique
   - Groups consecutive sentences based on semantic similarity
   - Complex implementation

**Key implementation notes:**
- Chunk by character — most reliable fallback, works with any document type
- Chunk by sentence — good middle ground if sentence detection works reliably
- Chunk by section — optimal results but requires structured input
- Strategy choice depends on document type guarantees and use case requirements

**Rule:** no universal best chunking method — it depends on document structure guarantees and specific use case.

---

## Text Embeddings

**Text Embeddings** — numerical representation of text meaning generated by embedding models.

**Embedding Model** — takes text input, outputs a long list of numbers (range −1 to +1).

**Embedding numbers** — scores representing unknown qualities/features of the input text. Each number theoretically scores a different aspect (happiness, topic relevance, etc.) but the actual meaning is unknown to users.

**Semantic Search** — uses text embeddings to find text chunks related to user questions in RAG pipelines. Solves the search problem of matching user queries to relevant document chunks.

**RAG pipeline process** — extract text chunks → user submits query → find related chunks using semantic search → add relevant chunks as context to the prompt.

**Implementation** — Anthropic recommends Voyage AI for embedding generation. Requires a separate account/API key. Free to start, easy integration via SDK.

**Key insight** — embeddings enable semantic similarity matching rather than keyword matching, allowing better understanding of text relationships for retrieval tasks.

---

## The Full RAG Flow

**RAG Flow** — 7-step process combining text chunking, embeddings, and vector search to retrieve relevant context for LLM queries.

1. **Text Chunking** — split source documents into separate text pieces
2. **Generate Embeddings** — convert text chunks into numerical vectors using embedding models
3. **Normalization** — scale vector magnitudes to 1.0 (handled automatically by embedding APIs)
4. **Vector Database Storage** — store embeddings in a specialized database optimized for numerical vector operations
5. **Query Processing** — convert the user question into an embedding using the same model
6. **Similarity Search** — find the most similar stored embeddings using cosine similarity calculation
7. **Prompt Assembly** — combine the user question with retrieved relevant text chunks, send to the LLM

**Key math concepts:**
- **Cosine Similarity** — cosine of the angle between vectors, returns values −1 to 1; closer to 1 means more similar
- **Cosine Distance** — 1 minus cosine similarity; values closer to 0 mean higher similarity
- **Vector Database** — performs similarity calculations to find the closest matching embeddings

**Process flow:** pre-processing (steps 1–4) → user query → real-time retrieval (steps 5–7) → LLM response.

---

## Implementing the RAG Flow

**RAG Flow Implementation** — practical walkthrough of the 5-step retrieval-augmented generation process.

**Step 1: Text Chunking** — split the document into sections using a `chunk_by_section` function on a `report.md` file.

**Step 2: Embedding Generation** — create vector representations for each chunk using a `generate_embedding` function (supports single string or list of strings input).

**Step 3: Vector Store Population** — create a vector index instance, loop through chunk-embedding pairs using `zip()`, store each pair with `store.add_vector(embedding, {content: chunk})`. Store the original text with embeddings for meaningful retrieval results.

**Step 4: Query Processing** — user asks a question ("what did the software engineering department do last year"), generate an embedding for the user query.

**Step 5: Similarity Search** — use `store.search(user_embedding, 2)` to find the 2 most relevant chunks; returns results with cosine distances (0.71 for section two, 0.72 for the methodology section).

**Key components:**
- Vector Index Class — custom vector database implementation
- Cosine Distance — similarity metric between query and stored embeddings
- Metadata Storage — storing original text content alongside embeddings enables meaningful retrieval results

The workflow is complete but has limitations requiring further improvements.

---

## BM25 Lexical Search

**BM25** — Best Match 25, a lexical search algorithm commonly used in RAG pipelines to complement semantic search.

**Problem with semantic search alone** — can miss exact term matches, returning irrelevant results even when specific terms appear frequently in certain documents.

**Hybrid search approach** — combines semantic search (embeddings/vector database) with lexical search (BM25) in parallel, then merges results for better balance.

**BM25 algorithm steps:**
1. Tokenize the user query into separate terms (remove punctuation, split on spaces)
2. Count the frequency of each term across all text chunks/documents
3. Assign relative importance to terms based on usage frequency (rare terms = higher importance, common terms like "a" = lower importance)
4. Rank text chunks by how often they contain higher-weighted terms

**Key insight** — frequently used terms across the corpus are less important for search relevance than rare, specific terms.

**BM25 advantages** — better at finding exact term matches, prioritizes documents containing rare/specific search terms, complements semantic search weaknesses.

**Implementation** — both semantic and lexical search systems use similar APIs (`add_document`, `search` functions), making them easy to combine.

**Next step** — merge results from both search systems to get the benefits of semantic understanding plus exact term matching.

---

## A Multi-Index RAG Pipeline

**Multi-Index RAG Pipeline** — system combining semantic search (vector index) and lexical search (BM25 index) for improved retrieval accuracy.

**Key components:**
- Vector Index — semantic similarity search using embeddings
- BM25 Index — lexical/keyword-based search
- Retriever Class — wrapper that forwards queries to both indexes and merges results

**Reciprocal Rank Fusion** — technique for merging search results from different indexes. Formula: `RRF_score = sum of (1/(rank + 1))` across all search methods for each document. Documents ranked by highest combined score.

**Example:** vector search returns `[doc2, doc7, doc6]`, BM25 returns `[doc6, doc2, doc7]`. After RRF calculation, the final ranking becomes `[doc2, doc6, doc7]` because doc2 ranked high in both methods.

**Benefits:**
- Improved search accuracy by combining different search paradigms
- Modular design with a standardized API (`search()` and `add_document()` methods)
- Easy to extend with additional search indexes
- Better handling of edge cases where a single method fails

Implementation pattern allows multiple search methodologies to work together while maintaining separate, isolated index classes.

---

## Reranking Results

**Reranking** — post-processing step that uses an LLM to reorder search results by relevance after initial retrieval.

**Process:** run vector + BM25 search → merge results → pass to the LLM with a prompt asking to rank documents by relevance → get reordered results.

**Implementation details:** use document IDs instead of full text for efficiency. The LLM receives the user query + candidate documents + instructions to return the most relevant docs in decreasing order. Assistant message pre-fill + stop sequence ensures structured JSON output.

**Trade-offs:** increases search accuracy by leveraging the LLM's understanding of semantic relevance. Increases latency due to the additional LLM call. Particularly effective when initial retrieval methods miss nuanced query intent (e.g., "ENG team" vs. "engineering team").

**Example improvement:** the query "What did the engineering team do with incident 2023?" correctly prioritized the software engineering section over the cybersecurity section after reranking, despite hybrid search initially ranking it lower.

---

## Contextual Retrieval

**Contextual Retrieval** — technique to improve RAG pipeline accuracy by adding context to document chunks before embedding.

**Problem:** when documents are split into chunks, individual chunks lose context from the original document, reducing retrieval accuracy.

**Solution:** a pre-processing step that adds contextual information to each chunk before inserting it into the retriever database.

**Process:**
1. Take an individual chunk + the original source document
2. Send to an LLM (Claude) with a prompt asking it to generate situating context
3. The LLM generates brief context explaining the chunk's relationship to the larger document
4. Join the generated context with the original chunk = "contextualized chunk"
5. Use the contextualized chunk as input to the vector/BM25 indexes

**Large document handling:** if the source document is too large for a single prompt, use a selective context strategy:
- Include starter chunks (1–3) from the document beginning for summary/abstract
- Include chunks immediately before the target chunk for local context
- Skip middle chunks that provide less relevant context

**Implementation:** an `add_context` function takes a text chunk + source text, generates context via the LLM, concatenates the context with the original chunk, and returns the contextualized version.

**Benefit:** chunks retain ties to the larger document structure and cross-references, improving retrieval accuracy for complex documents with interconnected sections.

---

## Extended Thinking

**Extended Thinking** — Claude feature that allows reasoning time before generating a final response.

**Key mechanics:**
- Displays a separate thinking process visible to users
- Increases accuracy for complex tasks but adds cost (charged for thinking tokens) and latency
- Thinking budget — minimum 1024 tokens allocated for the thinking phase
- `max_tokens` must exceed the thinking budget (e.g., budget 1024 requires `max_tokens` ≥ 1025)

**When to use:**
- Enable after prompt optimization fails to achieve desired accuracy
- Use prompt evals to determine necessity

**Response structure:**
- Thinking block — contains reasoning text + cryptographic signature
- Text block — final response
- Signature — prevents tampering with thinking text (safety measure)

**Special cases:**
- Redacted thinking blocks — encrypted thinking text flagged by safety systems
- Provided for conversation continuity without losing context
- Can force redacted blocks using a special test string

**Implementation:**
- Set `thinking=true` and a `thinking_budget` parameter
- Ensure `max_tokens` > `thinking_budget` for adequate response generation capacity

---

## Image Support

**Claude Vision Capabilities** — ability to process images within user messages for analysis, comparison, counting, and description tasks.

**Image limitations:**
- Max 100 images per request
- Size/dimension restrictions apply
- Images consume tokens (charged based on pixel height/width calculation)

**Image block structure** — special block type within user messages that holds either raw image data (base64) or a URL reference to an online image. Multiple image blocks allowed per message.

**Critical success factor** — strong prompting techniques required for accurate results. Simple prompts often fail.

**Prompting techniques for images:**
- Step-by-step analysis instructions
- One-shot/multi-shot examples (alternating image and text pairs)
- Clear guidelines and verification steps
- Structured analysis frameworks

**Example use case** — automated fire risk assessment from satellite imagery analyzing tree density, property access, roof overhang, and assigning numerical risk scores.

**Implementation** — base64 encode image data, create a message with an image block (`type: image, source: base64, media_type, data`) followed by a text block containing detailed prompt instructions.

**Key takeaway** — image accuracy depends entirely on prompt sophistication, not just image quality.

---

## PDF Support

Claude can read PDF files directly using code similar to image processing.

**Key implementation changes:**
- File type = `"document"` instead of `"image"`
- Media type = `"application/pdf"` instead of `"image/png"`
- Variable naming = `file_bytes` instead of `image_bytes`

**Claude PDF capabilities** — read text + images + charts + tables + mixed content extraction.

**PDF processing** — one-stop solution for comprehensive document analysis.

**Usage pattern** — same as image input but with document-specific parameters.

---

## Citations

**Citations** — feature allowing Claude to reference source documents and show where information comes from.

**Citation types:**
- `citation_page_location` — for PDF documents; shows document index/title/start page/end page/cited text
- `citation_char_location` — for plain text; shows character position in the text block

**Implementation:**
- Add `"citations": {"enabled": true}` to the request
- Add a `"title"` field to identify the source document
- Works with both PDF files and plain text sources

**Response structure** — content becomes a list of text blocks, some containing `citations` arrays with location data.

**Purpose** — transparency for users to verify Claude's information sources and check the accuracy of interpretations.

**UI benefit** — enables citation popups/overlays showing the source document, page numbers, and exact cited text when users hover over referenced content.

**Key use case** — ensuring users can investigate how Claude builds responses from source materials rather than appearing to speak from memory alone.

---

## Prompt Caching

**Prompt Caching** — feature that speeds up Claude's responses and reduces text generation costs by reusing computational work from previous requests.

**Normal request flow:** user sends message → Claude processes input (creates internal data structures, performs calculations) → Claude generates output → Claude discards all processing work → ready for next request.

**Problem:** when follow-up requests contain identical input messages, Claude must repeat all the same computational work it just threw away, creating inefficiency.

**Solution:** prompt caching stores the results of input message processing in a temporary cache instead of discarding it. When identical input appears in subsequent requests, Claude retrieves cached work rather than reprocessing, dramatically speeding up response generation.

**Key benefit:** reuses previous computational work to avoid redundant processing of repeated content.

---

## Rules of Prompt Caching

**Core mechanism:** initial request → Claude processes + saves work to cache → follow-up requests with identical content → Claude retrieves cached work instead of reprocessing.

**Cache duration** — 1 hour maximum.

**Cache activation requires manual cache breakpoint addition to message blocks.**

**Text block formats:**
- Shorthand — `content = "text string"` (cannot add cache control)
- Longhand — `content = [{"type": "text", "text": "content", "cache_control": {...}}]` (required for caching)

**Cache scope** — all content up to and including the breakpoint gets cached.

**Cache invalidation** — any change in content before the breakpoint invalidates the entire cache.

**Content processing order** — tools → system prompt → messages (joined together).

**Cache breakpoint placement options:**
- Tool schemas
- System prompts
- Message blocks (text, image, tool use, tool result)

**Maximum breakpoints** — 4 per request.

**Multiple breakpoints** — create multiple cache layers; partial cache hits possible if only later content changes.

**Minimum cache threshold** — 1024 tokens required for content to be cached.

**Best use cases** — repeated identical content (system prompts, tool definitions, static message prefixes).

---

## Prompt Caching in Action

**Setup** — modify the `chat` function to enable caching by default for tools and system prompts.

**Tool Schema Caching** — add a `cache_control` field with `type: "ephemeral"` to the last tool in the list. Best practice: create a copy of the tools list, clone the last tool schema, add cache control, then overwrite to avoid modifying the original schemas.

**System Prompt Caching** — wrap the system prompt in a text block dictionary with `cache_control: {"type": "ephemeral"}`.

**Multiple cache breakpoints** — can set cache points for both tools and the system prompt in a single request.

**Cache order** — tools → system prompt → messages.

**Token usage patterns:**
- `cache_creation_input_tokens` — tokens written to cache on first use
- `cache_read_input_tokens` — tokens retrieved from cache on subsequent identical requests
- Partial cache reads possible when some content matches cached data

**Cache invalidation** — any change to cached content (tools or system prompt) invalidates the cache, forcing new cache creation.

**Use cases** — identical content across requests: same tool schemas, system prompts, or message sequences.

---

## Code Execution and the Files API

**Files API** — allows uploading files ahead of time and referencing them later via a file ID instead of including raw file data in each request. Upload file → get a file metadata object with an ID → use the ID in future requests.

**Code Execution** — server-based tool where Claude executes Python code in isolated Docker containers. No implementation needed, just include the predefined tool schema. Claude can run code multiple times, interpret results, and generate a final response.

**Key constraints:** Docker containers have no network access. Data input/output relies on Files API integration.

**Combined workflow:** upload a file via the Files API → get a file ID → include the ID in the container upload block → ask Claude to analyze → Claude writes/executes code with access to the uploaded file → returns analysis and results.

Claude can generate files (plots, reports) inside the container that can be downloaded using file IDs returned in the response.

**Use cases:** data analysis, file processing, automated code generation for complex tasks. The response contains code blocks, execution results, and final analysis.

**Implementation:** use a container upload block with a file ID, include an analysis prompt, and Claude handles code execution automatically.

---

## Introducing MCP

**MCP** — Model Context Protocol, communication layer providing Claude with context and tools without requiring developers to write tedious code.

**Architecture:** MCP client connects to an MCP server. The server contains tools, resources, and prompts as internal components.

**Problem solved:** eliminates the burden of authoring/maintaining numerous tool schemas and functions for service integrations. Example: a GitHub chatbot would require implementing tools for repositories, pull requests, issues, projects — significant developer effort.

**Solution:** MCP server handles tool definition and execution instead of your application server. MCP servers = interfaces to outside services, wrapping functionality into ready-to-use tools.

**Key benefits:** developers avoid writing tool schemas and function implementations themselves.

**Common questions:**
- Who creates MCP servers? Anyone — often service providers make official implementations (AWS, etc.)
- vs. direct API calls? MCP eliminates the need to author tool schemas/functions yourself
- vs. tool use? MCP and tool use are complementary — MCP handles *who* does the work (server vs. developer), both still involve tools

**Core value** — shifts the integration burden from application developers to MCP server maintainers.

---

## MCP Clients

**MCP Client** — communication interface between your server and an MCP server; provides access to the server's tools.

**Transport agnostic** — client/server can communicate via multiple protocols (stdio, HTTP, WebSockets).

**Common setup** — client and server on the same machine using standard input/output.

**Communication** — message exchange defined by the MCP spec.

**Key message types:**
- List tools request — client asks server for available tools
- List tools result — server responds with tool list
- Call tool request — client asks server to run a tool with arguments
- Call tool result — server responds with the tool execution result

**Typical flow:**
1. User queries the server
2. Server requests the tool list from the MCP client
3. MCP client sends a list-tools request to the MCP server
4. MCP server responds with the list-tools result
5. Server sends the query + tools to Claude
6. Claude requests tool execution
7. Server asks the MCP client to run the tool
8. MCP client sends a call-tool request to the MCP server
9. MCP server executes the tool (e.g., GitHub API call)
10. Results flow back through the chain: MCP server → MCP client → server → Claude → user

**Purpose** — enables servers to delegate tool execution to specialized MCP servers while maintaining Claude integration.

---

## Project Setup

**CLI-based chatbot project** — teaches MCP client-server interaction through hands-on implementation.

**Project components:**
- MCP client — connects to a custom MCP server
- MCP server — provides 2 tools (read document, update document)
- Document collection — fake documents stored in memory only

**Key distinction:** normal projects implement either the client OR the server, not both. This project implements both for educational purposes.

**Setup process:**
1. Download `CLI_project.zip` starter code
2. Extract and open in a code editor
3. Follow `readme.md` setup directions
4. Add API key to `.env` file
5. Install dependencies (with/without UV)
6. Run the project: `uv run main.py` or `python main.py`
7. Test with a chat prompt

**Expected outcome** — a working chat interface that responds to basic queries, ready for MCP feature additions.

---

## Defining Tools with MCP

**MCP Python SDK** — official package that auto-generates tool JSON schemas from Python function definitions using the `@mcp.tool` decorator.

**Tool definition syntax** — `@mcp.tool(name="tool_name", description="description")` + function with typed parameters using `Field()` for argument descriptions.

**Two tools implemented:**
1. `read_doc_contents` — takes a `doc_id` string, returns document content from an in-memory docs dictionary
2. `edit_document` — takes `doc_id`, `old_string`, `new_string` parameters, performs find/replace on the document content

**Error handling** — check if `doc_id` exists in the docs dictionary, raise `ValueError` if not found.

**Key advantage** — the SDK eliminates manual JSON schema writing, generating schemas automatically from Python function signatures and decorators.

**Required imports** — `Field` from `pydantic` for parameter descriptions, the `mcp` package for server and tool decorators.

**Implementation pattern** — decorator defines tool metadata, function parameters define tool arguments with types and descriptions, function body contains tool logic.

---

## The Server Inspector

**MCP Inspector** — in-browser debugger for testing MCP servers without connecting to applications.

**Access:** run `mcp dev [server_file.py]` in the terminal → opens server on a port → navigate to the provided URL in the browser.

**Interface:** left sidebar has a connect button → top menu shows resources/prompts/tools sections → tools section lists available tools → click a tool to open the right panel for manual testing.

**Testing workflow:** connect to server → navigate to tools → select a specific tool → input required parameters → click "run tool" → verify output.

**Key features:** live development testing, manual tool invocation, parameter input forms, success/failure feedback, no need for full application integration.

**Note:** UI actively changing during development; core functionality remains similar.

**Example usage:** test document tools by inputting document IDs, verify read operations, test edit operations, chain operations to verify changes.

**Primary benefit** — debug MCP server implementations efficiently during the development phase.

---

## Implementing a Client

**MCP Client** — wrapper class around the client session for resource cleanup and connection management to the MCP server.

**Client Session** — the actual connection to the MCP server from the MCP Python SDK; requires resource cleanup on close.

**Client purpose** — exposes MCP server functionality to the rest of the codebase, enables reaching out to the server for tool lists and tool execution.

**Key functions:**
- `list_tools()` — `await self.session.list_tools()`, return `result.tools`
- `call_tool()` — `await self.session.call_tool(tool_name, tool_input)`

**Usage flow** — client gets tool definitions to send to Claude, then executes tools when Claude requests them.

**Common pattern** — wrap the client session in a larger class for resource management rather than using the session directly.

**Testing** — can run the client file directly with a testing harness to verify server connection and tool retrieval.

**Integration** — other code in the project calls client functions to interact with the MCP server, enabling Claude to inspect/edit documents through defined tools.

---

## Defining Resources

**MCP Resources** — mechanism allowing MCP servers to expose data to clients for read operations.

**Resource types** — 2 types: direct (static URI like `docs://documents`) and templated (parameterized URI like `docs://documents/{doc_id}`).

**URI** — address/identifier for accessing a specific resource, defined when creating the resource.

**Resource flow** — client sends a read-resource request with the URI → server matches the URI to a function → server executes the function → returns data in the read-resource result.

**Implementation** — use the `@mcp.resource` decorator with URI and MIME type parameters.

**MIME types** — hint to the client about the returned data format (`application/json` for structured data, `text/plain` for plain text).

**Templated resources** — URI parameters automatically parsed by the SDK and passed as keyword arguments to the handler function.

**Resources vs. tools** — resources provide data proactively (fetch document contents when @ mentioned); tools perform actions reactively (when Claude decides to call them).

**Data return** — SDK automatically serializes returned data to strings; client is responsible for deserialization.

**Testing** — the MCP inspector can list direct resources separately from templated resources, allowing testing of individual resource calls.

---

## Accessing Resources

**Resource reading function** — client-side function to request and parse resources from the MCP server.

**Function parameters** — URI (resource identifier).

**Implementation steps:**
- Import the `json` module + `AnyUrl` from `pydantic`
- Call `await self.session.read_resource(AnyUrl(uri))`
- Extract the first element from `result.contents[0]`
- Check `resource.mime_type` for the parsing strategy

**Content parsing logic:**
- If `mime_type == "application/json"` → return `json.loads(resource.text)`
- Otherwise → return `resource.text` (plain text)

**Server response structure** — `result.contents` list, with the first element containing type/mime_type metadata.

**Resource integration** — MCP client functions called by other application components to fetch document contents for prompts.

**End result** — document contents automatically included in Claude prompts without requiring tool calls.

**Key point** — resources expose server information directly to clients through a structured request/response pattern.

---

## Defining Prompts

**MCP Prompts** — pre-defined, tested prompt templates that MCP servers expose to client applications for specialized tasks.

**Purpose** — instead of users writing ad-hoc prompts, server authors create high-quality, evaluated prompts tailored to their server's domain.

**Implementation** — use the `@mcpserver.prompt` decorator with name/description, define a function that returns a list of messages (user/assistant messages that can be sent directly to Claude).

**Example use case** — a document formatting prompt that takes a document ID, instructs Claude to read the document using tools, reformat it to markdown, and save the changes.

**Key benefits** — server-specific expertise, pre-tested quality, reusable across client applications, better results than user-generated prompts.

**Message structure** — returns `base.UserMessage` objects containing the formatted prompt text with interpolated parameters.

**Client integration** — prompts appear as autocomplete options (slash commands) in client applications, prompt the user for required parameters, then execute the pre-built prompt workflow.

---

## Prompts in the Client

- List prompts — `await self.session.list_prompts()`, return `result.prompts`
- Get prompt — `await self.session.get_prompt(prompt_name, arguments)`, return `result.messages`

**Prompt workflow:**
1. Define a prompt in the MCP server with expected arguments (e.g., `document_id`)
2. Client calls `get_prompt` with the prompt name + arguments dictionary
3. Arguments are passed as keyword arguments to the prompt function
4. The function interpolates the arguments into the prompt text
5. Returns a messages array for direct feeding to the LLM

**Key concept:** prompts are server-defined templates that clients can invoke with specific arguments to generate contextualized instructions for LLMs. Arguments flow from client call → prompt function → interpolated prompt text → LLM consumption.

---

## Anthropic Apps

**Anthropic Apps** — two deployed applications by Anthropic: Claude Code and Computer Use.

**Claude Code** — terminal-based coding assistant that serves as an example of agent architecture.

**Computer Use** — toolset that expands Claude's capabilities beyond text generation.

**Key purpose** — these apps demonstrate agent concepts and provide practical examples for understanding agent design and implementation.

**Setup process** — involves terminal configuration for Claude Code usage on sample projects.

**Agent connection** — both applications exemplify how agents work, serving as learning models for building effective agents.

---

## Claude Code Setup

**Claude Code** — terminal-based coding assistant program that helps with code-related tasks.

**Core capabilities** — search/read/edit files + advanced tools (web fetching, terminal access) + MCP client support for expanded functionality via MCP servers.

**Setup process:**
1. Install Node.js (check with `npm help` command)
2. Run `npm install` to install Claude Code
3. Execute the `claude` command in the terminal to log in to your Anthropic account

**Full setup guide** — docs.anthropic.com

**MCP client functionality** — can consume tools from MCP servers to extend capabilities beyond basic file operations.

---

## Claude Code in Action

**Claude Code** — AI coding assistant that functions as a collaborative engineer on projects, not just a code generator.

**Key capabilities:** project setup, feature design, code writing, testing, deployment, error fixing in production.

**Setup workflow:**
- Download the project, open it in an editor
- Run the `claude` command to launch
- Ask Claude to read the README and execute setup directions
- Run the `init` command — Claude scans the codebase for architecture/coding style, creates a `claude.md` file
- `claude.md` — automatically included context for future requests

**Memory types:** Project (shared), Local, User memory files.

**Context management:**
- Use `#` symbol to add specific notes to memory
- Can manually edit `claude.md` or rerun `init` to update
- Claude can handle Git operations (staging, committing)

**Effective prompting strategies:**

**Method 1 — Three-step workflow:**
1. Identify relevant files, ask Claude to analyze them
2. Describe the feature, ask Claude to plan a solution (no code yet)
3. Ask Claude to implement the plan

**Method 2 — Test-driven development:**
1. Provide relevant context
2. Ask Claude to suggest tests for the feature
3. Select and implement chosen tests
4. Ask Claude to write code until tests pass

**Core principle:** Claude Code = effort multiplier. More detailed instructions = significantly better results. Treat it as a collaborative engineer, not just a code generator.

---

## Enhancements with MCP Servers

**Claude Code** — AI assistant with an embedded MCP (Model Context Protocol) client that can connect to MCP servers to expand functionality.

**MCP server integration** — connect external tools/services to Claude Code via the command: `claude mcp add [server-name] [startup-command]`.

**Example implementation** — a document processing server exposing a "Document Path to Markdown" tool, allowing Claude Code to read PDF/Word documents by running `uv run main.py`.

**Dynamic capability expansion** — MCP servers add new functions to Claude Code in real time without core modifications.

**Common use cases** — production monitoring (Sentry), project management (Jira), communication (Slack), custom development workflow tools.

**Key benefit** — significant flexibility increase for development workflows through modular server connections.

**Setup process:**
1. Create an MCP server with tools
2. Add the server to Claude Code with a name and startup command
3. Restart Claude Code to access new capabilities

---

## Parallelizing Claude Code

**Parallelizing Claude Code** — running multiple Claude instances simultaneously to complete different tasks in parallel.

**Core problem** — multiple Claude instances modifying the same files simultaneously creates conflicts and invalid code.

**Solution** — Git work trees providing isolated workspaces per Claude instance.

**Git work trees** — feature creating complete project copies in separate directories, each corresponding to a different Git branch.

**Workflow** — create a work tree → assign a task to a Claude instance → work in isolation → commit changes → merge back to the main branch.

**Custom commands** — automating work tree creation/management through a `.claude/commands` directory with markdown files containing prompts.

**Command structure** — `.claude/commands/filename.md` with a `$ARGUMENTS` placeholder for dynamic values.

**Parallel execution benefits** — a single developer commanding a virtual team of software engineers; major productivity scaling limited only by the engineer's management capacity.

**Merge conflicts** — Claude automatically resolves conflicts during the branch-merging process.

**Cleanup** — Claude handles work tree removal after feature completion.

**Key advantage** — scales to unlimited parallel instances based on the developer's capacity to manage simultaneous tasks.

---

## Automated Debugging

**Automated Debugging** — using AI (Claude) to automatically detect, analyze, and fix production errors without manual intervention.

**Core workflow:**
1. GitHub Action runs daily to check the production environment
2. Fetches CloudWatch logs from the last 24 hours
3. Claude identifies errors, deduplicates them
4. Claude analyzes each error and generates fixes
5. Creates a pull request with proposed solutions

**Key components:**
- GitHub Actions for scheduling/automation
- AWS CLI for log retrieval
- Claude Code for error analysis and code fixes
- CloudWatch for production error monitoring

**Benefits:**
- Catches production-only errors (issues not present in development)
- Reduces manual log hunting and debugging time
- Provides context-aware fixes with explanations
- Creates reviewable pull requests for changes

**Common use case:** configuration errors between environments (invalid model IDs, API keys, etc. that work locally but fail in production).

**Implementation requirements:** repository access, cloud logging service, AI coding assistant, CI/CD pipeline integration.

---

## Computer Use

**Computer Use** — Claude's ability to interact with computer interfaces through visual observation and control actions.

**Key capabilities:**
- Takes screenshots of applications/browsers
- Clicks buttons, types text, navigates interfaces
- Follows multi-step instructions autonomously
- Performs QA testing and automation tasks

**How it works:**
- Runs in an isolated Docker container environment
- User provides instructions via the chat interface
- Claude observes the screen visually and executes actions
- Generates reports on task completion/results

**Primary use cases:**
- Automated QA testing of web applications
- UI interaction testing across different scenarios
- Time-saving for repetitive computer tasks
- Bug identification through systematic testing

**Setup requirement** — reference implementation available for local testing.

**Example workflow:** user describes testing requirements → Claude navigates to the application → executes test cases → reports pass/fail results with detailed findings.

---

## How Computer Use Works

**Tool use flow:** user sends message + tool schema → Claude responds with a tool use request (ID, name, input) → server executes code → result sent back to Claude as a tool result.

**Computer use follows an identical flow:**
- Special tool schema sent to Claude (small schema expands to a larger structure behind the scenes)
- Expanded schema includes an action function with arguments: mouse move, left click, screenshot, etc.
- Claude sends a tool use request
- Developers must fulfill the request via a computing environment (typically a Docker container)
- Container executes programmatic key presses/mouse movements
- Response sent back to Claude

**Key points:**
- Claude doesn't directly manipulate computers
- Computer use = tool system + developer-provided computing environment
- Anthropic provides a reference implementation (Docker container with pre-built mouse/keyboard execution code)
- Setup requires Docker + simple command execution
- Enables a direct chat interface for testing Claude's computer use functionality

**Computer use** = abstraction layer where the tool system handles Claude communication while the Docker container handles actual computer interactions.

---

## Agents and Workflows

**Workflows and agents** — strategies for handling user tasks that can't be completed by Claude in a single request.

**Decision rule:** use workflows when you have precise task understanding and know the exact sequence of steps. Use agents when task details are unclear.

**Workflow** — a series of calls to Claude for specific problems where steps are predetermined.

**Example workflow — Image to 3D model converter:**
1. Claude describes the uploaded image in detail
2. Claude uses the CADQuery Python library to model the object from the description
3. Create a rendering of the model
4. Claude compares the rendering to the original image
5. If inaccurate, repeat from step 2 with feedback

This follows the **evaluator-optimizer pattern:**
- Producer — generates output (Claude + CADQuery modeling)
- Evaluator — assesses output quality (comparison step)
- Loop continues until the evaluator accepts the output

**Key point:** workflows are implementation patterns that other engineers have successfully used. Identifying workflow patterns doesn't automatically implement them — you still need to write the actual code.

---

## Parallelization Workflows

**Parallelization Workflows** — breaking one complex task into multiple simultaneous subtasks, then aggregating results.

**Example — Material selection for parts:**
- Instead of: one large prompt asking Claude to choose between metal/polymer/ceramic/composite with all criteria
- Use: separate parallel requests, each evaluating one material's suitability, then a final aggregation step to compare results

**Structure:** input → multiple parallel subtasks → aggregator → final output.

**Benefits:**
- Focus — each subtask handles one specific analysis instead of juggling multiple considerations
- Modularity — individual prompts can be improved/evaluated separately
- Scalability — easy to add new subtasks without affecting existing ones
- Quality — reduces confusion from overly complex single prompts

**Key principle:** decompose complex decisions into specialized parallel analyses, then synthesize the results.

---

## Chaining Workflows

**Chaining Workflows** — breaking large tasks into a series of distinct sequential steps rather than a single complex prompt.

**Core concept:** instead of one massive prompt with multiple requirements, split into separate calls where each focuses on one specific subtask.

**Example workflow:** user enters a topic → search trending topics → Claude selects the most interesting → Claude researches the topic → Claude writes a script → generate video → post to social media.

**Key benefit:** allows the AI to focus on individual tasks rather than juggling multiple constraints simultaneously.

**Primary use case:** when Claude consistently ignores constraints in complex prompts despite repetition. Common with long prompts containing many "don't do X" requirements.

**Problem scenario:** long prompt with constraints (don't mention AI, no emojis, professional tone) → Claude violates some constraints regardless of repetition.

**Solution:**
- Step 1 — send the initial prompt, accept imperfect output
- Step 2 — follow-up prompt asking Claude to rewrite based on specific violations found

**Critical insight:** even a simple-seeming workflow becomes essential when dealing with constraint-heavy prompts that AI struggles to follow completely in a single pass.

---

## Routing Workflows

**Routing Workflows** — workflow pattern that categorizes user input to determine the appropriate processing pipeline.

**Key mechanism:** an initial request to Claude categorizes user input into predefined genres/categories. Based on the categorization response, the system routes to a specialized processing pipeline with customized prompts/tools.

**Example flow:**
1. User enters a topic (e.g., "Python functions")
2. Claude categorizes the topic (e.g., "educational")
3. System uses an educational-specific prompt template
4. Claude generates a script with an educational tone/structure

**Benefits:** ensures output matches the nature of the topic. Programming topics get educational treatment with definitions/explanations. Entertainment topics get trendy language/engaging hooks.

**Structure:** one routing step → multiple specialized processing pipelines → each pipeline has customized prompts/tools for its specific category.

**Use case:** social media video script generation where different topics require different tones and approaches.

---

## Agents and Tools

**Agents** — AI systems that create plans to complete tasks using provided tools; effective when exact steps are unknown. **Workflows** — better when precise steps are known.

**Key differences:** workflows require predetermined steps; agents dynamically plan using available tools.

**Agent advantages:** flexibility to solve a variety of tasks with the same toolset, can combine tools in unexpected ways.

**Tool abstraction principle:** provide generic/abstract tools rather than hyper-specialized ones. Example — Claude Code uses `bash`, `web_fetch`, `file_write` (abstract) rather than `refactor_tool`, `install_dependencies` (specialized).

**Tool combination examples:** `get_current_datetime` + `add_duration` + `set_reminder` can solve various time-related tasks through different combinations.

**Agent behavior:** can request additional information when needed, combines tools creatively to achieve goals, works best with a small set of flexible tools.

**Design approach:** give the agent abstract tools that can be pieced together rather than single-purpose specialized tools. This enables dynamic problem-solving and unexpected use cases.

---

## Environment Inspection

**Environment Inspection** — agents evaluating their environment and action results to understand progress and handle errors.

**Core concept:** after each action, agents need feedback mechanisms beyond basic tool returns to understand the new environment state.

**Computer use example:** Claude takes a screenshot after every action (typing, clicking) to see how the environment changed, since it cannot predict the exact results of actions like button clicks.

**Code editing example:** before modifying files, agents must read current file contents to understand the existing state.

**Social media video agent applications:**
- Use Whisper CPP via bash to generate timestamped captions, verify dialogue placement
- Use FFmpeg to extract video screenshots at intervals, inspect visual results
- Validate video creation meets expectations before posting

**Key benefit:** environment inspection enables agents to gauge task progress, detect errors, and adapt to unexpected results rather than operating blindly.

---

## Workflows vs Agents

**Workflows** — pre-defined series of calls to Claude with known exact steps. **Agents** — flexible approach using basic tools that Claude combines to complete unknown tasks.

**Key differences:**

- **Task division:** workflows break big tasks into smaller, specific subtasks, enabling higher focus and accuracy. Agents handle varied challenges creatively without predetermined steps.
- **Testing/evaluation:** workflows are easier to test due to a known execution sequence. Agents are harder to test since the execution path is unpredictable.
- **User experience:** workflows require specific inputs. Agents create their own inputs from user queries and can request additional input when needed.
- **Success rates:** workflows = higher task completion rates due to a structured approach. Agents = lower completion rates due to delegated complexity.

**Recommendation:** prioritize workflows for reliability. Use agents only when flexibility is truly required. Users want 100% working products over fancy agents.

**Core principle:** solve problems reliably first, innovation second.