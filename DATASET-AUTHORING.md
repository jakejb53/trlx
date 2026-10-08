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

Autonomous batches are the normal authoring mode. One-at-a-time review is a testing mode.

Before every autonomous run, ask how many scenarios the user wants in that
batch. Do not reuse the size of an earlier batch, including when the user
continues the same topic and settings.

Before starting an autonomous batch, obtain agreement on the logprob-difference
metric, adjustment/retry limits, and what happens when those limits are
exhausted. The acceptance threshold defaults to a mean-logprob difference of
`-0.30`; state that value before the run and use a different threshold when the
user chooses one. Do not infer the other settings or supply defaults for them.

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
source-backed conversational Context, prints a mandatory sufficiency challenge,
and performs a mandatory rendering/token check before generation; no uses the
standard direct-generation workflow. Neither check requires user approval under
normal procedure. Require this choice in every authoring session. Do not infer it
from the topic or from whether `context_file` is populated in saved settings. A no
answer omits Context for this batch; a yes answer uses an approved existing
Context or triggers a proposal for the required Context artifact under the
repository's approval rules.

Also ask in every authoring session: **"Do you want native subagents or
subagents via a tool call?"** The answer selects how the adversarial reasoning
reviewer and any other delegated review is dispatched. Native uses the harness
subagent mechanism. Tool call runs a one-shot CLI process from the repository
root; the default shape is `claude --model claude-opus-4-8 -p "PROMPT"`, with
the contract and the exact texts inline in PROMPT, no file reads, and the
ruling read from stdout. Use a different shape when the user supplies one. The
ruling is binding under either method. Do not infer the choice from saved
settings.

When the tool-call method is selected, write PROMPT to a scratch file and run
`claude --model claude-opus-4-8 -p "$(cat FILE)" < /dev/null` from the
repository root with a timeout, capturing stdout and stderr separately
(`authoring/make_review_prompt.py` assembles PROMPT and `authoring/review.sh`
runs it). PROMPT
must contain: the editing contract summary; the complete diff between the
untouched and edited texts, or both texts in full when the diff is not
self-explanatory; the complete retained reasoning and response whenever any
material was removed under "Fitting the token limit", so the reviewer can
verify duplication and flow claims against the whole retained text; the
editor's justification for each change; any retained passages the editor
wants adjudicated; and the instruction to rule exactly `ACCEPT` or `REVISE`
on the final line of the output. Treat any exit without a
final-line ruling as a failed call. On a failed call, retry once with the same
shape; if it fails again, stop that row, record the failure as the row's
outcome under the agreed exhausted-retry policy, and report it. A REVISE ruling
is resubmitted as a fresh call that states the prior ruling, confirms the
correction was applied, and gives the complete remaining diff.

Treat every existing Context as a candidate until its printed sufficiency
challenge returns `READY` for the exact final prompt or named scenario set. A
matching filename, description, prior use, or structural validity does not prove
readiness. Do not state that no Context modification is planned before that
adjudication.

For the configured model, reuse `[probes]`: its model identity, tokenization URLs, reasoning field and boundaries, template-inserted system text, and scoring request settings. `training_uses_endpoint_tokenizer_and_template = true` records the user's confirmation that endpoint rendering is the training reference. Do not repeat capability or metadata probes each session. Per-example correctness checks, token counting, and likelihood measurements remain required as described below.

For token counting, map the saved reasoning into the final assistant's recorded reasoning field and POST `model`, `messages`, and the recorded tokenization flags to the tokenization URL. Read `count` and `tokens`; POST `model` and `tokens` to the detokenization URL to obtain the rendered `prompt`. Submit that text to the scoring URL with `model` and the recorded scoring request settings; read `choices[0].logprobs`. Probe notes and measured result fields are metadata, not request arguments.

If the user specifies a different model, establish its validation metadata after onboarding: inspect the applicable repository training configuration and rendering code, then the endpoint's available model metadata and tokenizer/template APIs. Use authorized local model resources when needed; repository access rules still apply. Establish the training tokenizer, chat template, and reasoning-field mapping from evidence; use the user-supplied full-sequence token limit. Training loss settings are not prerequisites for generating, validating, or saving examples. Do not assume an unrelated run configuration applies or that a served model alias proves matching training and endpoint rendering.

Apart from the required generation-method question, ask only for unresolved choices, unavailable facts, or conflicts, stating what was checked and which validation depends on the answer. Once generation inputs and scenarios are approved, generation and editing may proceed while validation details are resolved. Complete required correctness and token-limit validation before saving. If likelihood scoring or a baseline is unavailable, report it explicitly; autonomous saves must still satisfy the agreed logprob acceptance policy.

Prefer focused, challenging problems. Follow the user's topic choices. Ensure the saved user prompt contains the facts needed to understand the response.

## Batch workflow

1. Obtain the topic explicitly for this session and the number N of new dataset rows for this autonomous run, using the agreed destination. Do not reuse a previous session's topic or an earlier run's count.
2. Draft exactly N scenarios, one per row, and assign any existing Contexts as candidates rather than assuming they are ready.
3. When an exact prompt or scenario set and an existing candidate Context are already known, print the sufficiency challenge before requesting scenario-list approval. Resolve a `NOT READY` result first and include the final adjudication in the scenario proposal. Do not promise that the Context will remain unchanged before this challenge.
4. If a required Context does not exist yet, name its exact artifact paths and intended scope in the proposal. After the required approval, build it and run the sufficiency challenge before rendering; a `READY` result needs no additional approval.
5. As the final step before requesting approval to begin generation, read `DATASET-AUTHORING-REMINDER.md` in full and explicitly tell the user that you understand and agree to follow its instructions. Then ask for approval of the scenario list. Do not begin generation before approval.
6. Scenario-list approval authorizes formulating prompts, generating, editing, validating, and saving rows that satisfy the agreed acceptance policy to the agreed destination. Do not request per-prompt or per-row approval in autonomous mode.
7. Handle ordinary prompt refinements, editing, and validation corrections autonomously within the approved scenarios. When a candidate fails the logprob threshold, adjust and retry within the editing contract and agreed limits. Threshold failures follow the agreed policy without introducing a per-row approval pause. Provide progress updates without stopping for review.
8. Save each completed row only when it satisfies the agreed logprob threshold and all editing, correctness, and token-limit requirements, preserving existing examples. Count only new saved rows toward N; skipped duplicates do not count. On continuation, inspect saved progress before creating more rows.
9. Continue until every approved scenario has a verified saved row or has reached the outcome specified by the agreed exhausted-retry policy, or an unexpected problem requires the user's attention. Do not substitute scenarios or weaken validation to finish the batch.
10. Report completed scenarios, destination, new and total row counts, and validation results. If blocked, identify the problem and completed progress.

## Context-assisted generation (alternative)

Use this alternative when the target model needs researched facts, platform
semantics, failure cases, or design invariants that are awkward to place in the
saved user prompt. It is also appropriate when an ordinary generation would
require extensive factual rewriting. The objective is a strong untouched target-
model response that needs only the minimum editing permitted below.

This option changes generation input, not the saved row contract. The Context is
excluded from the training example. The final saved prompt must still be
self-contained enough to identify the problem and requested result.

### Teach the model through Context

Build Context as a curriculum that develops the knowledge needed for the final
prompt. Work backward from that prompt to identify prerequisite facts, concepts,
reasoning methods, and likely failure modes. Keep the final prompt a natural next
question that requires applying and combining what the conversation establishes.

Construct a coherent investigation: user questions motivate retrieval; tool
results supply authoritative material; assistant analyses interpret the evidence;
follow-up questions expose uncertainty, test understanding, and connect the
pieces. Use actual retrieved material, with source locations and a clear
distinction between excerpts, paraphrases, and original analysis.

Provide depth through worked examples, complete derivations, exercises,
counterexamples, and representative operation sequences where relevant. Teach
why plausible approaches fail and how to evaluate alternatives. Short assertions
or summaries are not substitutes for the evidence and reasoning needed to
understand the subject.

Preserve the final task's synthesis. Context may resolve prerequisite
uncertainty, but should not contain the completed implementation or a polished
answer to the final prompt. Avoid fabricated authority or conveniently tailored
documents that merely deliver the solution.

Context must not read as feedback on an earlier generation. Never tell the
model that a prior response, draft, implementation, or reasoning attempt failed.
Do not include reviewer comments, defect lists, exact corrections, replacement
spans, retry instructions, scoring results, or scenario-specific acceptance
checks. This prohibition applies even when the failed generation identified a
real knowledge gap.

Use as much relevant material as needed within the agreed context and performance
limits. Neither brevity nor token volume establishes sufficiency.

#### Evidence provenance gate

“Fabricated” or “simulated” describes only the model-visible tool call. It
never authorizes fabricating the tool result. Every new or changed result must
come from an operation that actually occurred or from an existing artifact
whose exact contents and provenance were already verified.

Before adding or replacing a model-visible tool result:

1. Perform the represented retrieval or execution, or read the previously
   verified artifact.
2. Inspect the returned source material, output, status, and relevant metadata.
3. Verify every passage, result, and observation that will appear in Context.
4. Only then construct the model-visible tool exchange and mutate the Context.

A source name, URL, citation, search-result snippet, remembered fact, or
agent-written summary is not retrieved source material by itself. Do not label
content “retrieved,” “executed,” “observed,” or “verified” unless that operation
occurred and supports the claim. For external sources, identify locations and
distinguish direct excerpts, faithful paraphrases, and original analysis. Label
reviewed synthetic internal material as internal rather than external authority.

Do not draft a tool result and verify it afterward. Any unsupported retrieval
or execution claim invalidates the Context and requires correction before the
sufficiency challenge.

When any violation of these authoring instructions is found in an existing or
in-progress Context, stop using that Context. Remove the violating
model-visible material completely, including the complete tool-call/result pair
when either half is affected. Do not leave the violation in place and append a
correction, disclaimer, or compliant parallel account around it.

Replace removed material only with curriculum that independently satisfies the
current instructions: obtain and verify its evidence first, present it without
feedback framing or answer leakage, reconnect the surrounding conversation, and
update the description when its instructional scope or sources change. Then
rerun structural validation, the provenance audit, feedback and answer-leakage
checks, the complete sufficiency challenge, and rendering before generation.

#### Repair curriculum gaps from generation evidence

A rejected generation is private authoring evidence. Keep the generation, its
defects, and proposed corrections outside model-visible Context.

For each material defect:

1. Identify the general prerequisite fact, concept, reasoning method, or failure
   class that the curriculum failed to teach.
2. Retrieve authoritative material for that prerequisite.
3. Introduce it through a natural research question and evidence-bearing tool
   result.
4. Develop understanding through analysis, derivation, or an analogous exercise
   that uses different names, values, and circumstances from the final task.
5. Include general counterexamples where useful, without reproducing the final
   task's erroneous and corrected forms.
6. Reconnect the new material to the existing curriculum so the final prompt
   remains a natural request for new synthesis.

Do not append isolated corrective assertions. Do not expose the prior failure,
state the exact fix, provide corrected final-task code, or turn the Context into
a review checklist for the final answer.

Before accepting a repair, perform both checks:

- **Feedback check:** Could a model-visible reader infer that these turns are
  correcting or grading an earlier answer? If yes, rewrite them as curriculum.
- **Answer-leakage check:** Could the final answer be produced largely by copying
  the added material and substituting the final prompt's names? If yes, the
  repair tells rather than teaches.

A repaired Context is ready only when it teaches the missing prerequisite while
leaving the final prompt's implementation and synthesis unresolved.

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
`SPEC-context-packages.md`; `authoring/ctxbuild.py` wraps these commands for a
build script and composes tool results in the provenance header format used in
`contexts/`:

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

Install the `authoring` project extra when extracting HTML or XML. Use `lxml`
to parse the document and select unique structural elements such as section IDs;
do not use regular expressions, repeated plaintext headings, or expected prose
sentences as document boundaries or extraction-success checks. Use the native
parser and structural identifiers for other machine-readable formats. Validate
selector uniqueness, ordering, and a plausible nonempty result, then inspect the
actual selected text before treating it as retrieved evidence.

Use reviewed synthetic internal documents for application-specific contracts
that no external source defines, such as state transitions, accounting
invariants, concurrency rules, or required behavior from an unspecified
provider. Present these as internal engineering material, not fabricated
external authority.

The agent does not need to load large sources into its own conversation. A local
command process can retrieve and extract passages in memory, assert every
expected section, and call `dataset.context_builder.add_tool_exchange` only
after all assertions pass. Alternatively, produce a verified local result file
and pass it with `--content-file`. `authoring/extract.py` performs structural
extraction from htmlized and v3 RFC HTML, HTML by id, Markdown headings, and
PDF page ranges; inspect its output before use.

When rebuilding a Context from external sources, finish every fallible retrieval
and extraction first. Assemble and validate the complete replacement in memory,
then publish it once through the staged writer; do not leave a partially rebuilt
Context when a later source or insertion fails.

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
2. After the provenance gate passes, the assistant makes a simulated
   model-visible retrieval call backed by that verified evidence.
3. The tool result supplies the verified selected source material, execution
   result, or reviewed internal document.
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

Before declaring the Context ready, perform these coverage checks:

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
6. Map every model-visible retrieval and execution claim to the actual source,
   command output, or previously verified artifact that supports it. A missing
   or unverifiable mapping makes the Context `NOT READY`.

There is no fixed minimum or target Context size. Use as much selected,
relevant material as the scenario needs, subject to the model-specific
performance threshold and context-window gate agreed for the session. Passing
the rendering budget proves only that the input fits; it does not prove that
the Context is sufficient.

#### Printed sufficiency challenge

After the Context appears complete and before rendering, print this three-part
challenge:

1. **Case for sufficiency** — Give a concrete defense of why the Context contains
   enough facts, resolved decisions, failure cases, and guidance for the target
   model to answer correctly without reconstructive edits.
2. **Case against sufficiency** — Give the strongest good-faith argument that the
   Context remains inadequate. Identify plausible missing facts, unresolved
   choices, unsupported assumptions, insufficient depth, or work the model must
   still perform. Identify concrete model-visible material that could materially
   reduce the risk of incorrect reasoning, invented behavior, or reconstructive
   editing. A generic claim that more Context is always possible is not a valid
   case against.
3. **Adjudication** — Evaluate both arguments against the final prompt and actual
   Context. Return `READY` only when the case against does not expose a material
   risk that the model must research, invent, or redesign something essential.
   If the case against identifies useful missing material, return `NOT READY`
   unless the Context already contains it or the adjudication explains with
   specific evidence why it would be redundant. Otherwise improve or split the
   Context and repeat the complete challenge.

A `READY` adjudication must answer the strongest objection with concrete Context
evidence. Message count, token count, structural validation, and topical
relevance are not evidence of sufficiency by themselves.

Evaluate only whether the model-visible input is sufficient for a substantially
correct untouched response. Post-generation compilation, tests, scoring,
editing, formatting checks, and token or word limits may detect defects, but
they are not evidence that the Context is sufficient and must not be used to
justify `READY`.

Under normal procedure this challenge is not an approval request. Print it for
the user, but do not ask the user to approve a `READY` result. Continue to the
rendering check automatically. The user may explicitly override normal
procedure and require review. Other repository approval rules remain unchanged.
For a batch, one challenge may cover every named row that uses the same Context;
a `NOT READY` result pauses only those rows.

### Build and inspect

The complete Context-building sequence is:

1. Obtain approval for the Context path and create it with `context create`.
2. Add focused conversational framing.
3. Retrieve or execute outside the builder, inspect the result, and verify every
   passage or observation intended for Context.
4. Only after that verification, add selected external passages or execution
   evidence through simulated model-visible tool exchanges.
5. Add reviewed internal design notes for scenario-specific knowledge, clearly
   labelled as internal material.
6. Add assistant analysis connecting evidence to constraints and failure cases.
7. End with a natural transition into the approved final prompt.
8. Audit the provenance mapping for every retrieval and execution claim.
9. Perform the coverage checks and print the sufficiency challenge above.
   Extend or split the Context until the adjudication is `READY`.
10. Run `context outline` without previews to review only roles, sizes, tool
   names, call IDs, and ordering.
11. Run `context validate`. Resolve every structural or tool-pairing error before
   rendering.

The outline and validation output should be sufficient for routine inspection;
use `context show` only for a targeted message that needs review. Preserve the
Context file after successful generation so it can support later scenarios.

### Mandatory rendering and budget check

Rendering is a separate mandatory technical check before generation.
Read the Context file directly, append the exact final user prompt, and include
the exact approved system message only when nonblank. Submit that messages
array to the configured tokenizer with the generation boundary enabled.
`authoring/render_check.py` performs this check and prints the fields below.

For the recorded remote tokenizer contract this ordinarily means:

- `add_generation_prompt = true`
- `continue_final_message = false`
- `add_special_tokens = false`

Print:

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

Do not truncate Context or silently lower the generation reserve. Under normal
procedure user approval is not needed: if rendering succeeds and the budget
fits, proceed directly to generation. If the check fails, correct the affected
Context or prompt before generation. The user may explicitly override normal
procedure and require review.

### Generate with Context

After the rendering check passes, call the existing generator with the
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

Review the reasoning and answer as they will appear in the saved row, with the
system prompt and Context omitted. References such as “the research,” “the
internal contracts,” “the provided context,” prior tool calls, or earlier
authoring turns are drafting residue when their referent exists only in excluded
input. Remove or localize only the dangling reference; preserve the substantive
reasoning and any source attribution that is self-contained or required by the
saved prompt.

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

Use rejected generations as private diagnostic evidence. Do not copy their
defects, corrections, or acceptance checks into Context. Repair the curriculum
using “Repair curriculum gaps from generation evidence,” then repeat the
printed sufficiency challenge and mandatory rendering check before a fresh
generation. Under normal procedure, proceed without requesting user approval
when both pass.

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
11. Read that saved prompt, reasoning, and answer without the excluded inputs.
    Remove or localize any authoring-process reference that no longer has a
    self-contained referent.
12. Verify that the saved text exactly matches the scored candidate and that
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

Before beginning the complete reasoning edit, measure the complete raw
rendered example, read `REASONING-EDIT-REMINDER.md` in full, and reread the
untouched reasoning from the raw response artifact. Complete the reasoning
and answer edits, then run correctness checks, rendered-token validation, and
likelihood scoring. Do not dispatch the adversarial reasoning reviewer unless
the complete candidate passes those checks and the agreed likelihood threshold.

Use `jq` only to inspect or extract fields. Do not use `jq sub`, shell regular
expressions, or shell-embedded replacement expressions to edit generated text.
Inspect the raw response's actual message keys before accessing them. Perform
edits from the untouched response with one Python transformation using the
observed field names and exact literal replacements. Every replacement must
assert that its original span occurs exactly once. Do not apply transformations
to an intermediate candidate. `authoring/edit.py` applies a JSON list of exact
literal replacements this way and prints the diff.

If the transformation command fails, return to the untouched response, correct
the mechanical error, and rerun the same intended edit. Inspect the complete
resulting reasoning and answer before continuing.

Give the reviewer the exact untouched and edited reasoning in its initial
request. If it returns `ACCEPT`, continue to saving. If it returns `REVISE`,
apply only its identified local corrections, rerun correctness, rendering, and
likelihood checks, then resubmit the exact revised reasoning. Regenerate only
when the required correction cannot be made without violating the reasoning-edit
contract.

Treat the original as the authoritative text to be repaired, not a draft to improve generally. Preserve every passage unless it has an identifiable defect.

Apply correctness, completeness, grammar, and consistency requirements to the entire example, including unchanged text in `reasoning` and assistant `content`.

Remove all leading whitespace, including newlines, from assistant response `content`.

An edit is permitted only to:

- Correct a demonstrably false claim, invalid calculation/code, or contradiction.
- Satisfy an explicit prompt requirement, or supply something necessary to make a retained claim correct.
- Repair a grammatical error, malformed formatting, or ambiguity that prevents a definite interpretation.
- Remove instructions about composing the response, stale draft commentary, unresolved writing choices, repetition, abandoned deliberation, or duplicate answer drafts. Preserve genuine uncertainty about the problem.
- Remove dangling references to excluded system text, Context, tool calls,
  research steps, internal contracts, or prior authoring turns. Do not remove
  source references that remain meaningful and self-contained in the saved row.
- Fit the verified token limit without losing required content.

A different valid design, greater robustness outside the stated assumptions, broader coverage, more formal terminology, or a formulation you prefer is not a defect. Preserve sound choices even when you would have chosen differently.

Before changing a span, identify its specific defect and the fact or requirement it violates. If you cannot identify one, preserve the span verbatim. This is the basis for constructing the edit, not a justification written afterward.

Work through every retained reasoning and response passage in context. Repair missing grammatical structure, broken logical connections, and drafting residue before accepting the passage. Make the smallest correction that leaves the passage correct and coherent. Preserve surrounding sound wording, vocabulary, voice, structure, examples, and design choices. Keep reasoning and response consistent.

Replace a whole passage only when local corrections cannot make it correct and coherent. Preserve its original style wherever correctness permits. Do not compose an ideal replacement answer and then try to recover the original wording.

## Validate

- Verify substantive claims against authoritative sources or appropriate checks. Check API names/signatures, code, arithmetic, edge cases, and the prompt's constraints. Distinguish source review, successful compilation, and executed tests.
- Count the complete rendered training example with the target tokenizer and training template: saved prompt, selected reasoning/response, and template/boundary tokens. Verify that reasoning is actually rendered. An endpoint count applies to training only when its tokenizer and rendering match. `authoring/count_score.py` renders, counts, and with `--score` reports span mean log-probabilities under the saved probes.
- If oversized, first remove repetition and unnecessary prose without losing correctness or required coverage. In batch mode, refine prompts within the approved scenarios autonomously; otherwise discuss narrowing the prompt. A changed prompt requires fresh generation, editing, and validation. Changes beyond an approved scenario require user approval.
- When scoring is available, score both the original and final edited continuations. Record mean log-probability for reasoning and response separately and together, with the scoring masks and changes from baseline stated. For comparisons intended to isolate wording changes, hold the preceding context fixed and distinguish those measurements from whole-continuation scores. Report unavailable baselines explicitly.
- Use edit size and likelihood only to choose among corrected candidates that meet the editing requirements. When necessary corrections admit multiple comparably small, correct formulations, use likelihood comparisons before selecting the correction, with preceding context held fixed. Likelihood does not authorize changing sound text or restoring errors. Final scores and retention percentages are measurements, not evidence that every edit was necessary.

The authoring CLI does not tokenize or score likelihood. Use the saved validation APIs and reasoning-field mapping for the configured model; discover them again only when the user specifies a different model. For likelihood measurements, state which tokens are scored, including treatment of template and reasoning boundaries; choosing the eventual training loss settings is unnecessary. Report unavailable or unverified measurements explicitly.

## Fitting the token limit

A good generation is reasoning that demonstrates understanding of the topic and a response that follows from it. Editing serves two purposes: removing fluff (filler, hedging, drafting residue), especially from the reasoning, and minor correctness repairs (grammar, punctuation, spelling, formatting, errors fixable by changing a few tokens). Editing cannot turn a demonstration of non-understanding into one of understanding; that needs regeneration.

When the edited example exceeds the full-sequence limit:

1. If the reasoning does not demonstrate understanding, improve the Context that failed to teach the topic and regenerate.
2. If the prompt asked for more than the model can answer accurately within the limit, narrow the prompt and regenerate.
3. Otherwise, for a good generation that is slightly over, cut. Judge each cut by its effect on the demonstration of understanding and on the chain of thought that leads to the response. In the reasoning, remove only fluff: filler words, hedging, drafting residue, and repetition of a point already made. Any other reasoning cut is presumed harmful. It is permitted only when the editor can state, for that exact span, why removing it leaves both the demonstration of understanding and the chain of thought intact, and the reviewer must adjudicate that statement; when in doubt, do not cut reasoning. Cut the response instead: remove whole sections from the end, since the beginning matters more than the end for training. Remove earlier response material only when it is fluff or duplicates retained content. Do not rewrite retained text beyond a minimal fix to a reference to a removed section. The retained response must remain correct; reduced completeness is accepted.
4. Re-score the cut candidate under the agreed likelihood policy and name the removed material for the reviewer.

## Review and save

In testing mode, present the final prompt, edited reasoning and response, material corrections, complete rendered token count, and validation results. Show original and final mean logprobs and their differences for reasoning, response, and both combined, with scored token counts and scoring masks. State treatment of template and reasoning boundaries; report unavailable measurements explicitly. Report fixed-context comparisons separately from original-versus-final continuation scores. Obtain explicit approval of the exact final candidate after presenting these results and before writing it to the dataset. Prior scenario or destination approval does not replace this save approval. Preserve the approved text exactly.

In autonomous mode, score each original and final candidate's reasoning, response, and combined continuation. Save only candidates that satisfy the agreed logprob threshold and all editing, correctness, and token-limit requirements. When a candidate fails the threshold, adjust and retry within the editing contract and agreed limits; likelihood does not authorize rewriting sound text, restoring errors, or weakening validation. Follow the agreed exhausted-retry policy without introducing a per-row approval pause. Summarize progress during the run and report results at completion.

After an autonomous run completes, the agent presents the aggregate
scenario results, exhausted retries or failures, validation results,
destination, and new and total row counts. It then asks whether the user wants
another autonomous run. If the user continues, ask for a fresh batch size
before preparing scenarios; do not carry the completed run's count forward.

`python -m dataset.cli save` reads a destination and nonempty array of examples from stdin:

```sh
python -m dataset.cli save <<'JSON'
{"path":"data/DESTINATION.jsonl","examples":[{"messages":[{"role":"user","content":"ORIGINAL PROMPT"},{"role":"assistant","content":"APPROVED RESPONSE"}],"reasoning":"APPROVED REASONING"}]}
JSON
```

Each example has exactly one user and one assistant message with string content. Reasoning is a separate top-level string, without manually added reasoning delimiters. Omit `reasoning` for answer-only examples; use empty assistant content for reasoning-only examples. System prompts and Context are excluded. Unknown fields are rejected.

The destination must be a regular `.jsonl` file or a new path with an existing parent, not a directory or final symlink. Relative paths use the working directory. Save stages publication, preserves existing bytes, and skips exact duplicates based on user text, assistant content, and reasoning presence/text. It returns `{"added":N,"duplicates":N}`. No `--force` is needed; saves to one destination must be sequential across processes.

After saving, verify the accepted fields and row count, and confirm existing examples are unchanged. Report the destination and total examples. Direct file writes are also permitted when explicitly authorized; preserve the same row contract and verification requirements.
