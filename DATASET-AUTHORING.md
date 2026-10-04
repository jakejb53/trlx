# Model-specific dataset authoring

Follow this workflow. Repository approval and file-operation rules still apply.

## Session inputs

Before any other setup investigation or endpoint probing, check for `dataset-authoring.toml` in the repository root. If it exists, read it, display its values, and ask one question about the saved settings: **"Use these values?"** On confirmation, use those values without repeating setup questions about them. Keep topic and number of examples out of the config; require both explicitly in the current session and ask for any missing value one at a time. Scenario-list approval remains part of the batch workflow.

The agent reads this file; the CLI does not load it. `[generation]` supplies the `generate` stdin fields, with `user` added for each approved scenario. An empty `[generation.sampling]` sends no overrides. `context_file` supplies `--context-file`; an empty string omits it. `destination` supplies the save path. `max_full_sequence_tokens` limits each complete rendered training example, not raw generation. `max_concurrent_generations` caps simultaneous generation requests launched by the agent; saves remain sequential.

When saved values are not accepted or the file is absent, reuse established session choices and ask for unresolved inputs one at a time. Do not choose settings on the user's behalf or treat the examples below as defaults. Generation requires:

- Target model/checkpoint, endpoint, and credentials if needed.
- Dataset destination and the scope of permission to save examples.
- Approved prompt or scenario list, plus system prompt, Context, and sampling settings.
- Generation method: standard or Context-assisted.

To list available Contexts, enumerate every `contexts/*.json` file in filename
order. For each JSON file, read the same-stem `.txt` file and display the
filename with its complete description. List every Context, not only ones that
appear relevant to the current topic. If `contexts/` contains no JSON files,
state that no saved Contexts are available. If a matching description is
missing or unreadable, report that defect instead of inventing a description;
repair the inventory before presenting it as complete. Use this procedure both
when the user requests the available Contexts and during onboarding.

After generation settings are resolved and before proposing scenarios, list all
available Contexts with their descriptions using that procedure, then explicitly
ask: **"Use Context-assisted generation for this batch?"** State that yes builds
source-backed conversational Context, performs a mandatory rendering/token gate,
and pauses for review before generation; no uses the standard direct-generation
workflow. Require this choice in every authoring session. Do not infer it from
the topic or from whether `context_file` is populated in saved settings. A no
answer omits Context for this batch; a yes answer uses an approved existing
Context or triggers a proposal for the required Context artifact under the
repository's approval rules.

For the configured model, reuse `[probes]`: its model identity, tokenization URLs, reasoning field and boundaries, template-inserted system text, and scoring request settings. `training_uses_endpoint_tokenizer_and_template = true` records the user's confirmation that endpoint rendering is the training reference. Do not repeat capability or metadata probes each session. Per-example correctness checks, token counting, and likelihood measurements remain required as described below.

For token counting, map the saved reasoning into the final assistant's recorded reasoning field and POST `model`, `messages`, and the recorded tokenization flags to the tokenization URL. Read `count` and `tokens`; POST `model` and `tokens` to the detokenization URL to obtain the rendered `prompt`. Submit that text to the scoring URL with `model` and the recorded scoring request settings; read `choices[0].logprobs`. Probe notes and measured result fields are metadata, not request arguments.

If the user specifies a different model, establish its validation metadata after onboarding: inspect the applicable repository training configuration and rendering code, then the endpoint's available model metadata and tokenizer/template APIs. Use authorized local model resources when needed; repository access rules still apply. Establish the training tokenizer, chat template, and reasoning-field mapping from evidence; use the user-supplied full-sequence token limit. Training loss settings are not prerequisites for generating, validating, or saving examples. Do not assume an unrelated run configuration applies or that a served model alias proves matching training and endpoint rendering.

Apart from the required generation-method question, ask only for unresolved choices, unavailable facts, or conflicts, stating what was checked and which validation depends on the answer. Once generation inputs and scenarios are approved, generation and editing may proceed while validation details are resolved. Complete required correctness and token-limit validation before saving; optional likelihood scoring may be unavailable if reported explicitly.

Prefer focused, challenging problems. Follow the user's topic choices. Ensure the saved user prompt contains the facts needed to understand the response.

## Batch workflow

1. Obtain the topic and number N of new dataset rows explicitly for this session, using the agreed destination. Do not reuse a previous session's topic or count.
2. Propose exactly N scenarios, one per row. As the final step before requesting approval to begin generation, read `DATASET-AUTHORING-REMINDER.md` in full and explicitly tell the user that you understand and agree to follow its instructions. Then ask for approval of the scenario list. Do not begin generation before approval.
3. Scenario-list approval authorizes formulating prompts, generating, editing, validating, and saving all N rows to the agreed destination. Do not request per-prompt or per-row approval.
4. Handle ordinary prompt refinements, editing, and validation corrections autonomously within the approved scenarios. Provide progress updates without stopping for review.
5. Save each verified row as it is completed, preserving existing examples. Count only new saved rows toward N; skipped duplicates do not count. On continuation, inspect saved progress before creating more rows.
6. Continue until every approved scenario has a verified saved row, or an unexpected problem requires the user's attention. Do not substitute scenarios or weaken validation to finish the batch.
7. Report completed scenarios, destination, new and total row counts, and validation results. If blocked, identify the problem and completed progress.

## Context-assisted generation (alternative)

Use this alternative when the target model needs researched facts, platform
semantics, failure cases, or design invariants that are awkward to place in the
saved user prompt. It is also appropriate when an ordinary generation would
require extensive factual rewriting. The objective is a strong untouched target-
model response that needs only the minimum editing permitted below.

This option changes generation input, not the saved row contract. The Context is
excluded from the training example. The final saved prompt must still be
self-contained enough to identify the problem and requested result.

### Context artifact

The durable Context is the exact UTF-8 JSON messages array accepted by
`dataset generate --context-file`. It has no wrapper, system prompt, final user
prompt, export step, sidecar, profile, recipe, or knowledge-base schema. One
Context can be reused with many external scenario prompts on the same topic.
Every saved `contexts/NAME.json` must have a matching `contexts/NAME.txt`
containing a two- or three-sentence free-form description of what the Context
teaches the model and which source material it includes. Keep that description
current when the Context's instructional scope or sources change.

Build and inspect it with the Context builder documented in
`SPEC-context-packages.md`:

```sh
python -m dataset.cli context create contexts/TOPIC.json
python -m dataset.cli context add contexts/TOPIC.json --role user --text "RESEARCH QUESTION"
python -m dataset.cli context add contexts/TOPIC.json --role assistant --content-file ANALYSIS.txt
python -m dataset.cli context tool contexts/TOPIC.json \
  --name web_search --arg 'query=QUERY' --content-file SEARCH-RESULT.txt
python -m dataset.cli context tool contexts/TOPIC.json \
  --name read_file --arg 'path=DOCUMENT' --content-file DOCUMENT
python -m dataset.cli context outline contexts/TOPIC.json
python -m dataset.cli context validate contexts/TOPIC.json
```

Content ingestion and model-visible representation are independent. Inline
text, stdin, or a file can become a user message, assistant message, or result
of any fabricated tool name and arguments. The builder never performs the
represented retrieval.

### Source material

Use authoritative material for standardized or externally defined behavior.
Prefer selected relevant passages over entire noisy documents and over lossy
LLM summaries. Strip navigation and unrelated boilerplate without rewriting the
substantive passage. Keep source names and locations in the model-visible result
when they help distinguish evidence from analysis.

Use reviewed synthetic internal documents for application-specific contracts
that no external source defines, such as state transitions, accounting
invariants, concurrency rules, or required behavior from an unspecified
provider. Present these as internal engineering material, not fabricated
external authority.

The agent does not need to load large sources into its own conversation. A local
command process can retrieve and extract passages in memory, assert every
expected section, and call `dataset.context_builder.add_tool_exchange` only
after all assertions pass. Alternatively, produce a verified local result file
and pass it with `--content-file`.

Do not stream a fallible producer directly into a mutating builder command. A
producer can fail after the builder sees EOF, causing an empty result to be
committed even when shell `pipefail` reports failure. Verify extraction before
mutation. If an insertion fails or contains the wrong data, remove the complete
tool-call/result pair, validate the Context, and stop after the retry limit in
the repository instructions.

### Conversation design

Construct a natural research conversation rather than an answer pasted into
Context:

1. A user turn asks a focused research question.
2. The assistant makes a fabricated retrieval call.
3. The tool result supplies selected source material or an internal document.
4. The assistant analyzes implications and resolves conflicts.
5. Later user turns introduce the next part of the problem.
6. The conversation ends with a synthesis that makes the final scenario prompt
   a natural next user turn.

Supply facts, invariants, counterexamples, and failure cases. Do not include a
polished answer to the final scenario. The target model must still perform the
synthesis and express the response in its own voice.

Use a blank system prompt by default. A reasoning model can see a system
instruction and repeat or discuss it in its reasoning. Put style and behavior
steering into the natural conversation unless the user explicitly approves a
nonblank system prompt for the run.

### Context sufficiency

A structurally valid, topically relevant, or concise Context is not necessarily
sufficient. The Context must resolve the factual and design uncertainty that
would otherwise make the target model rediscover requirements, invent missing
capabilities, cycle through alternatives, or require reconstructive editing.

Build depth in proportion to the scenario. Include, as applicable:

- Exact authoritative passages for externally defined behavior.
- A closed capability contract that distinguishes available, unavailable, and
  unspecified APIs or guarantees.
- Reviewed internal contracts for state meanings, invariants, ownership,
  authorization, timing, ordering, accounting, and failure handling.
- Concrete transaction or operation boundaries and the evidence each boundary
  can and cannot establish.
- Representative concurrency schedules, crash matrices, edge cases, and
  counterexamples to superficially plausible but invalid designs.
- Acceptance cases with observable postconditions.
- Intermediate assistant analyses that reconcile the evidence and resolve
  conflicts before the final synthesis.

Do not reuse one Context merely because several scenarios share a broad topic.
Reuse it only when its evidence and resolved contracts cover every assigned
scenario. Split or extend it when scenarios depend on different APIs,
invariants, platforms, or failure modes.

Before rendering, perform a sufficiency review:

1. Map every substantive final-prompt requirement to supplied evidence, a
   reviewed internal contract, or a fact stated directly in that prompt.
2. Identify every behavioral choice whose alternatives would change accepted,
   rejected, produced, preserved, or failed outcomes. Resolve it through the
   approved design or leave it explicitly for the final prompt to decide.
3. Check that unavailable capabilities are explicit, so the model cannot fill
   gaps with familiar but unsupported APIs.
4. Include tests or counterexamples for the general failure classes most likely
   to produce a plausible but incorrect answer.
5. End with a synthesis that states the resolved model and makes the final
   scenario a natural next question without drafting its answer.

There is no fixed minimum or target Context size. Use as much selected,
relevant material as the scenario needs, subject to the model-specific
performance threshold and context-window gate agreed for the session. Passing
the rendering budget proves only that the input fits; it does not prove that
the Context is sufficient.

### Build and inspect

The complete Context-building sequence is:

1. Obtain approval for the Context path and create it with `context create`.
2. Add focused conversational framing.
3. Add exact selected external passages as fabricated retrieval results.
4. Add reviewed internal design notes for scenario-specific knowledge.
5. Add assistant analysis connecting evidence to constraints and failure cases.
6. End with a natural transition into the approved final prompt.
7. Perform the Context-sufficiency review above. Extend or split the Context
   until every identified gap is resolved.
8. Run `context outline` without previews to review only roles, sizes, tool
   names, call IDs, and ordering.
9. Run `context validate`. Resolve every structural or tool-pairing error before
   rendering.

The outline and validation output should be sufficient for routine inspection;
use `context show` only for a targeted message that needs review. Preserve the
Context file after successful generation so it can support later scenarios.

### Mandatory rendering and budget gate

Rendering is a separate step from generation and must stop for user review.
Read the Context file directly, append the exact final user prompt, and include
the exact approved system message only when nonblank. Submit that messages
array to the configured tokenizer with the generation boundary enabled.

For the recorded remote tokenizer contract this ordinarily means:

- `add_generation_prompt = true`
- `continue_final_message = false`
- `add_special_tokens = false`

Report:

- Context message count.
- Whether a system message is included.
- Rendered input tokens.
- Reserved generation tokens.
- Total required tokens.
- Model context-window tokens.
- Remaining headroom.
- Whether the target template rendered all tool messages successfully.

Require:

```text
rendered_input_tokens + reserved_generation_tokens <= model_context_tokens
```

Do not truncate Context or silently lower the generation reserve. Stop after
reporting this gate. Generation begins only after the user reviews the result.

### Generate with Context

After the rendering gate is accepted, call the existing generator with the
same Context, system text, final prompt, model, and output-token reserve:

```sh
python -m dataset.cli generate --context-file contexts/TOPIC.json <<'JSON'
{"endpoint":"http://HOST:PORT/v1","model":"MODEL","user":"APPROVED PROMPT","system":"","sampling":{"max_tokens":4096},"timeout":1200,"retries":0}
JSON
```

Do not make duplicate requests while one is active. Preserve the complete raw
generation before editing.

`dataset.endpoint.Endpoint` stores the exact body bytes of every received HTTP
response under `endpoint-responses/` before decoding or validation. This
includes successful, malformed, retryable, and final error responses. Do not
delete or rewrite the response artifact. On generation failure, inspect the
stored response before deciding whether another request is justified.

### Review the untouched output

Judge the target model's reasoning by correctness, coherence, grammar, and
genuine drafting defects—not by a preferred prose style. A concise outline or
requirements checklist can be the model's natural reasoning and is not a defect
merely because it is not discursive. Preserve it when it is grammatical,
decisive, consistent with the answer, and free of stale alternatives or
unresolved hedging such as repeated “wait”, “but wait”, or “actually” reversals.

Check the untouched answer against the final prompt, source material, internal
contracts, and representative failure cases. A successful Context-assisted
generation should retain the target model's voice and require local corrections,
not reconstruction.

If the reasoning is only prompt planning, do not reject it for that fact alone.
If it contains actual grammar errors, contradictions, abandoned alternatives,
or unresolved writing instructions, apply the ordinary minimum-edit rules. If
no permitted local edit can produce a correct example, reject the generation;
do not reconstruct reasoning from the answer.

Treat the untouched output as evidence about Context sufficiency. Repeated
rediscovery of supplied requirements, abandoned state models, invented APIs,
multiple related invariant failures, or corrections that would reconstruct the
reasoning or answer indicate an insufficient Context. Pause that scenario and
improve the Context before generating again; do not compensate with extensive
manual rewriting or repeated requests using the same inadequate input.

Use rejected generations as negative evidence: record their general defect
classes, invalid assumptions, counterexamples, and acceptance checks in the
Context without pasting a polished replacement answer. After changing the
Context, rerun the mandatory rendering gate and obtain review before a fresh
generation.

### Edit, score, validate, and save

The standard editing, validation, and saving contracts below still apply.
In particular:

1. Preserve the raw prompt, reasoning, and answer separately.
2. Keep sound reasoning verbatim when it needs no correction.
3. Identify the exact defect before every answer change.
4. Remove leading assistant-answer whitespace.
5. Keep the complete saved row within the training token limit.
6. Verify source-dependent claims and execute meaningful state, race, code, or
   schema checks where applicable.
7. Score raw and corrected reasoning and answers separately and together under
   the saved prompt. When reasoning is unchanged, the answer comparison already
   holds preceding context fixed.
8. If several necessary corrections are comparably small and correct, compare
   their likelihood under identical reasoning and select the better-supported
   wording.
9. Investigate material likelihood decline; never restore a false claim to
   improve score.
10. Save only the final user prompt, assistant answer, and optional reasoning.
    Exclude system and Context.
11. Verify that the saved text exactly matches the scored candidate and that
    previous dataset bytes remain unchanged.

Report the Context path and raw response artifact with the ordinary generation,
editing, likelihood, validation, and dataset results. Context-assisted origin
does not by itself prove factual correctness or exempt any saved row from review.

## Generate

Run from the repository root using the project environment's absolute Python executable. The CLI executes directly; no web UI server or workspace is required.

`python -m dataset.cli generate [--context-file PATH]` reads one JSON object from stdin:

- Required: `endpoint` (HTTP(S) API base), `model`, `user`, `sampling`, `timeout`, `retries`.
- `model` and `user` must be nonblank strings. `timeout` is finite positive seconds; `retries` is a nonnegative integer.
- Optional `system` defaults to empty. Optional `api_key` names an environment variable, not a literal secret; omitted or empty means no authentication. The CLI loads `.env`; existing environment values win.
- `sampling: {}` sends no overrides. Supported controls: `temperature` >= 0, `top_p` in [0, 1], integer `top_k` >= -1, positive integer `max_tokens`, finite `presence_penalty`, and `repetition_penalty` > 0. Omitted/null controls are not sent.
- Optional `--context-file` reads a UTF-8 JSON array of message objects, preserving their fields and order between system and user messages. Relative paths use the working directory. Stdin `context` is rejected.

Example: replace the endpoint, model, and prompt with session-approved values. This request allows one hour and disables automatic retries:

```sh
python -m dataset.cli generate <<'JSON'
{"endpoint":"http://HOST:PORT/v1","model":"MODEL","user":"APPROVED PROMPT","system":"","sampling":{},"timeout":3600,"retries":0}
JSON
```

The result is `{"answer":"...","reasoning":"..."}` on stdout; progress/errors use stderr. Keep the streams separate when capturing large responses. Failures exit nonzero.

Choose an adequate timeout before launching. Let long reasoning runs finish; verbosity alone is not grounds for cancellation. Avoid duplicate requests while generation is active. The raw generation need not fit the training limit; that limit applies to the edited example.

Keep the complete unedited prompt, reasoning, and response available throughout the session, including post-save review. Persist raw outputs or scratch artifacts only with authorization.

## Edit the original

Treat the original as the authoritative text to be repaired, not a draft to improve generally. Preserve every passage unless it has an identifiable defect.

Apply correctness, completeness, grammar, and consistency requirements to the entire example, including unchanged text in `reasoning` and assistant `content`.

Remove all leading whitespace, including newlines, from assistant response `content`.

An edit is permitted only to:

- Correct a demonstrably false claim, invalid calculation/code, or contradiction.
- Satisfy an explicit prompt requirement, or supply something necessary to make a retained claim correct.
- Repair a grammatical error, malformed formatting, or ambiguity that prevents a definite interpretation.
- Remove instructions about composing the response, stale draft commentary, unresolved writing choices, repetition, abandoned deliberation, or duplicate answer drafts. Preserve genuine uncertainty about the problem.
- Fit the verified token limit without losing required content.

A different valid design, greater robustness outside the stated assumptions, broader coverage, more formal terminology, or a formulation you prefer is not a defect. Preserve sound choices even when you would have chosen differently.

Before changing a span, identify its specific defect and the fact or requirement it violates. If you cannot identify one, preserve the span verbatim. This is the basis for constructing the edit, not a justification written afterward.

Work through every retained reasoning and response passage in context. Repair missing grammatical structure, broken logical connections, and drafting residue before accepting the passage. Make the smallest correction that leaves the passage correct and coherent. Preserve surrounding sound wording, vocabulary, voice, structure, examples, and design choices. Keep reasoning and response consistent.

Replace a whole passage only when local corrections cannot make it correct and coherent. Preserve its original style wherever correctness permits. Do not compose an ideal replacement answer and then try to recover the original wording.

## Validate

- Verify substantive claims against authoritative sources or appropriate checks. Check API names/signatures, code, arithmetic, edge cases, and the prompt's constraints. Distinguish source review, successful compilation, and executed tests.
- Count the complete rendered training example with the target tokenizer and training template: saved prompt, selected reasoning/response, and template/boundary tokens. Verify that reasoning is actually rendered. An endpoint count applies to training only when its tokenizer and rendering match.
- If oversized, first remove repetition and unnecessary prose without losing correctness or required coverage. In batch mode, refine prompts within the approved scenarios autonomously; otherwise discuss narrowing the prompt. A changed prompt requires fresh generation, editing, and validation. Changes beyond an approved scenario require user approval.
- When scoring is available, score both the original and final edited continuations. Record mean log-probability for reasoning and response separately and together, with the scoring masks and changes from baseline stated. For comparisons intended to isolate wording changes, hold the preceding context fixed and distinguish those measurements from whole-continuation scores. Report unavailable baselines explicitly.
- Use edit size and likelihood only to choose among corrected candidates that meet the editing requirements. When necessary corrections admit multiple comparably small, correct formulations, use likelihood comparisons before selecting the correction, with preceding context held fixed. Likelihood does not authorize changing sound text or restoring errors. Final scores and retention percentages are measurements, not evidence that every edit was necessary.

The authoring CLI does not tokenize or score likelihood. Use the saved validation APIs and reasoning-field mapping for the configured model; discover them again only when the user specifies a different model. For likelihood measurements, state which tokens are scored, including treatment of template and reasoning boundaries; choosing the eventual training loss settings is unnecessary. Report unavailable or unverified measurements explicitly.

## Review and save

For individual examples, present the edited reasoning and response, material corrections, complete token count, likelihood measurements if available, and validation results. Await save approval unless the session already authorizes saving that scope. Preserve the approved text exactly.

For an approved batch, validate and save each completed example without a per-row presentation or approval gate. Summarize progress during the run and report results at completion.

`python -m dataset.cli save` reads a destination and nonempty array of examples from stdin:

```sh
python -m dataset.cli save <<'JSON'
{"path":"data/DESTINATION.jsonl","examples":[{"messages":[{"role":"user","content":"ORIGINAL PROMPT"},{"role":"assistant","content":"APPROVED RESPONSE"}],"reasoning":"APPROVED REASONING"}]}
JSON
```

Each example has exactly one user and one assistant message with string content. Reasoning is a separate top-level string, without manually added reasoning delimiters. Omit `reasoning` for answer-only examples; use empty assistant content for reasoning-only examples. System prompts and Context are excluded. Unknown fields are rejected.

The destination must be a regular `.jsonl` file or a new path with an existing parent, not a directory or final symlink. Relative paths use the working directory. Save stages publication, preserves existing bytes, and skips exact duplicates based on user text, assistant content, and reasoning presence/text. It returns `{"added":N,"duplicates":N}`. No `--force` is needed; saves to one destination must be sequential across processes.

After saving, verify the accepted fields and row count, and confirm existing examples are unchanged. Report the destination and total examples. Direct file writes are also permitted when explicitly authorized; preserve the same row contract and verification requirements.
