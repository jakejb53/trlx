# Model-specific dataset authoring

Follow this workflow. Repository approval and file-operation rules still apply.

## Session inputs

Before any other setup investigation or endpoint probing, check for `dataset-authoring.toml` in the repository root. If it exists, read it, display its values, and ask one question about the saved settings: **"Use these values?"** On confirmation, use those values without repeating setup questions about them. Keep topic and number of examples out of the config; require both explicitly in the current session and ask for any missing value one at a time. Scenario-list approval remains part of the batch workflow.

The agent reads this file; the CLI does not load it. `[generation]` supplies the `generate` stdin fields, with `user` added for each approved scenario. An empty `[generation.sampling]` sends no overrides. `context_file` supplies `--context-file`; an empty string omits it. `destination` supplies the save path. `max_full_sequence_tokens` limits each complete rendered training example, not raw generation. `max_concurrent_generations` caps simultaneous generation requests launched by the agent; saves remain sequential.

When saved values are not accepted or the file is absent, reuse established session choices and ask for unresolved inputs one at a time. Do not choose settings on the user's behalf or treat the examples below as defaults. Generation requires:

- Target model/checkpoint, endpoint, and credentials if needed.
- Dataset destination and the scope of permission to save examples.
- Approved prompt or scenario list, plus system prompt, Context, and sampling settings.

For the configured model, reuse `[probes]`: its model identity, tokenization URLs, reasoning field and boundaries, template-inserted system text, and scoring request settings. `training_uses_endpoint_tokenizer_and_template = true` records the user's confirmation that endpoint rendering is the training reference. Do not repeat capability or metadata probes each session. Per-example correctness checks, token counting, and likelihood measurements remain required as described below.

For token counting, map the saved reasoning into the final assistant's recorded reasoning field and POST `model`, `messages`, and the recorded tokenization flags to the tokenization URL. Read `count` and `tokens`; POST `model` and `tokens` to the detokenization URL to obtain the rendered `prompt`. Submit that text to the scoring URL with `model` and the recorded scoring request settings; read `choices[0].logprobs`. Probe notes and measured result fields are metadata, not request arguments.

If the user specifies a different model, establish its validation metadata after onboarding: inspect the applicable repository training configuration and rendering code, then the endpoint's available model metadata and tokenizer/template APIs. Use authorized local model resources when needed; repository access rules still apply. Establish the training tokenizer, chat template, and reasoning-field mapping from evidence; use the user-supplied full-sequence token limit. Training loss settings are not prerequisites for generating, validating, or saving examples. Do not assume an unrelated run configuration applies or that a served model alias proves matching training and endpoint rendering.

Ask only for unresolved choices, unavailable facts, or conflicts, stating what was checked and which validation depends on the answer. Once generation inputs and scenarios are approved, generation and editing may proceed while validation details are resolved. Complete required correctness and token-limit validation before saving; optional likelihood scoring may be unavailable if reported explicitly.

Prefer focused, challenging problems whose corrected solutions fit the training limit. Follow the user's topic choices. Ensure the saved user prompt contains the facts needed to understand the response.

## Batch workflow

1. Obtain the topic and number N of new dataset rows explicitly for this session, using the agreed destination. Do not reuse a previous session's topic or count.
2. Propose exactly N scenarios, one per row. As the final step before requesting approval to begin generation, read `DATASET-AUTHORING-REMINDER.md` in full and explicitly tell the user that you understand and agree to follow its instructions. Then ask for approval of the scenario list. Do not begin generation before approval.
3. Scenario-list approval authorizes formulating prompts, generating, editing, validating, and saving all N rows to the agreed destination. Do not request per-prompt or per-row approval.
4. Handle ordinary prompt refinements, editing, and validation corrections autonomously within the approved scenarios. Provide progress updates without stopping for review.
5. Save each verified row as it is completed, preserving existing examples. Count only new saved rows toward N; skipped duplicates do not count. On continuation, inspect saved progress before creating more rows.
6. Continue until every approved scenario has a verified saved row, or an unexpected problem requires the user's attention. Do not substitute scenarios or weaken validation to finish the batch.
7. Report completed scenarios, destination, new and total row counts, and validation results. If blocked, identify the problem and completed progress.

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
