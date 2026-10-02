# Model-specific dataset authoring

Follow this workflow only when explicitly invoked. Repository approval and file-operation rules still apply.

## Session inputs

Reuse established session choices. Generation requires:

- Target model/checkpoint, endpoint, and credentials if needed.
- Dataset destination and the scope of permission to save examples.
- Approved prompt or scenario list, plus system prompt, Context, and sampling settings.

Before asking for validation inputs, inspect the applicable repository training configuration and rendering code, then the endpoint's available model metadata and tokenizer/template APIs. Use authorized local model resources when needed; repository access rules still apply. Establish the training tokenizer, chat template, reasoning-field mapping, and full-sequence token limit from evidence. Training loss settings are not prerequisites for generating, validating, or saving examples. Do not assume an unrelated run configuration applies or that a served model alias proves matching training and endpoint rendering.

Ask only for unresolved choices, unavailable facts, or conflicts, stating what was checked and which validation depends on the answer. Once generation inputs and scenarios are approved, generation and editing may proceed while validation details are resolved. Complete required correctness and token-limit validation before saving; optional likelihood scoring may be unavailable if reported explicitly.

Prefer focused, challenging problems whose corrected solutions fit the training limit. Follow the user's topic choices. Ensure the saved user prompt contains the facts needed to understand the response.

## Batch workflow

1. Agree on the topic and number N of new dataset rows, using established session inputs and destination.
2. Propose exactly N scenarios, one per row, and obtain approval of the list before starting generation.
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

Keep the original prompt, reasoning, and response available for comparison. Persist raw outputs or scratch artifacts only with authorization.

## Edit the original

Improve problem-solving with minimal changes to the model's actual text:

1. Correct grammar and formatting while preserving vocabulary, voice, and explanatory structure. For example, “Need answer user. Language English? Need final.” becomes “I need to answer the user in English.” Perform these edits directly; another model call is unnecessary.
2. Remove circular wandering, abandoned approaches, repeated deliberation, and duplicate answer drafts. Retain the final coherent path, including necessary justification and substantive uncertainty.
3. Make localized correctness fixes in both reasoning and response. Preserve sound code, arithmetic, examples, tests, and design choices. Keep the two fields consistent.

Edit in place rather than compose a replacement and then try to make it resemble the model. An already-correct answer is still useful when its reasoning becomes clearer and more direct.

## Validate

- Verify substantive claims against authoritative sources or appropriate checks. Check API names/signatures, code, arithmetic, edge cases, and the prompt's constraints. Distinguish source review, successful compilation, and executed tests.
- Count the complete rendered training example with the target tokenizer and training template: saved prompt, selected reasoning/response, and template/boundary tokens. Verify that reasoning is actually rendered. An endpoint count applies to training only when its tokenizer and rendering match.
- If oversized, first remove repetition and unnecessary prose without losing correctness or required coverage. In batch mode, refine prompts within the approved scenarios autonomously; otherwise discuss narrowing the prompt. A changed prompt requires fresh generation, editing, and validation. Changes beyond an approved scenario require user approval.
- When available, score the edited continuation under the target model, conditioned on the original saved prompt and the preceding continuation tokens. Exclude editing instructions from scoring. Record mean log-probability for reasoning and response separately and together, with the scoring mask stated.
- Use likelihood to support model-compatible editing. Prefer higher-likelihood wording when quality is preserved. Do not invent acceptance thresholds, restore mistakes, or add filler to improve scores. Edited examples are not strictly on-policy samples; likelihood alone does not establish training effectiveness.

The authoring CLI does not tokenize or score likelihood. Inspect the target endpoint's supported API or use authorized local model resources. Use the actual reasoning-field mapping. For likelihood measurements, state which tokens are scored, including treatment of template and reasoning boundaries; choosing the eventual training loss settings is unnecessary. Report unavailable or unverified measurements explicitly.

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
