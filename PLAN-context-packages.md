# Context builder implementation plan

## Objective

Implement the direct Context builder specified in
`SPEC-context-packages.md`. The durable artifact is the exact JSON messages
array consumed by `dataset generate --context-file`. The implementation edits
that array safely and constructs fabricated tool-call/result messages; it does
not generate responses, retrieve data, manage system prompts, compile context,
or save training rows.

Implementation is one phase. Research-context construction and regeneration of
the three rejected dataset scenarios are a separate validation phase after the
tool is complete.

## Approved design

- One Context file is one ordered JSON array of message objects.
- There is no wrapper, sidecar, hidden workspace, export, snapshot, profile,
  recipe, fragment graph, or knowledge-base schema.
- Content ingestion through inline text, a UTF-8 file, or stdin is independent
  from its model-visible representation.
- The builder can represent supplied content as an ordinary message, a raw
  message, or the result of a fabricated standard function-tool call.
- The builder never executes the represented tool.
- Writers to one Context file are serialized by the caller.
- Mutations validate before and after the change and publish through staged
  replacement.
- Existing generation, browser Context, and dataset-save behavior remain
  unchanged.

## Files

Create:

- `dataset/context_builder.py`
- `tests/test_context_builder.py`

Modify:

- `dataset/cli.py`
- `tests/test_dataset_cli.py`
- `README.md`
- `SPEC.md`

Do not modify:

- `dataset/endpoint.py`
- `dataset/authoring.py` behavior or request contracts
- `dataset/ui.py` or `dataset/ui/*`
- `pyproject.toml`
- `dataset-authoring.toml`
- Any dataset file

No dependency or configuration change is part of this implementation.

## Implementation sequence

### 1. Context storage and serialization

Add `dataset/context_builder.py` with the public operations named in the
specification:

- `create_context`
- `read_context`
- `validate_context`
- `add_message`
- `add_tool_exchange`
- `replace_message`
- `remove_messages`
- `move_messages`
- `outline_context`
- `show_messages`

Use the existing finite JSON reader from `dataset.authoring` rather than
creating a second JSON-number and diagnostic policy. Use
`dataset.io.write_text` for staged publication.

Before reading or replacing an existing Context, reject final symlinks and
non-regular files. Creation requires an absent destination with an existing
parent. Mutation reads and validates the complete current array before doing
work.

Serialize successful mutations with:

- UTF-8
- `ensure_ascii=False`
- `allow_nan=False`
- two-space indentation
- one final newline
- retained object-key order for loaded messages
- fixed field order for builder-created messages

Do not normalize message strings or strip content read from stdin or files.

Every function must have a useful preceding comment explaining its purpose or
invariant. Add boundary comments for validation, input interpretation, and
publication where behavior is not locally obvious.

### 2. Context and tool validation

Validate the outer contract first:

- The root is an array.
- Every item is a non-null object.
- Every JSON number is finite.

Preserve arbitrary provider-owned fields. Interpret standard tool fields only
when `tool_calls` or `tool_call_id` is present.

Implement one ordered validation pass that tracks declared, pending, and
completed tool-call IDs. Enforce the standard function-tool invariants from the
specification, including assistant ownership of `tool_calls`, unique IDs,
function shape, JSON-object argument strings, exactly one later result per
call, and completion of pending results before the next non-tool conversational
message.

Return compact counts for validation and inspection. Errors identify the file,
message index, field, or call ID without echoing complete message or result
content.

### 3. Mutations and fabricated tool exchanges

Implement zero-based insertion, replacement, range removal, and range movement
with all boundary checks in the core module.

For ordinary messages, create exactly `{role, content}` in that order. For raw
messages, preserve the supplied object fields and order.

For fabricated tool exchanges:

- Accept a nonempty function name.
- Accept either flat string arguments or one supplied JSON object.
- Reject duplicate flat argument keys.
- Serialize function arguments as compact JSON with sorted keys.
- Accept an explicit unique call ID or allocate the lowest unused positive
  `call_NNNN` value.
- Create the assistant call and matching tool result as adjacent messages.
- Insert both messages in one mutation and publish only after full validation.

Each mutation returns a compact result containing the new message count and
affected indices. Tool insertion also returns the call ID. Mutation results do
not include message content.

### 4. CLI integration

Add a nested `context` parser to `dataset/cli.py` with these subcommands:

```text
create
add
tool
replace
remove
move
outline
show
validate
```

CLI handlers own only argument parsing, content-source selection, calls into
`dataset.context_builder`, and stdout formatting.

Add shared parser/input helpers for:

- `--text`
- `--content-file`
- stdin fallback and `--content-file -`
- `--message-file`
- repeatable `--arg KEY=VALUE`
- `--arguments-file`
- optional `--call-id`
- optional zero-based `--at`
- positive `--count`

Enforce mutual exclusions before mutation. Split `--arg` on its first `=` and
require a nonempty key. Read argument files as one finite JSON object.

`outline` prints a content-free human table by default and a JSON array with
`--json`; content previews appear only with explicit `--preview`. `show` prints
one message when count is one and an array for larger ranges. Other command
results are compact JSON.

Nested command errors must retain the existing `dataset` progress, expected
failure, and traceback behavior.

### 5. Tests

Create `tests/test_context_builder.py`. Exercise the Python API and real CLI
dispatch rather than replacing handlers with mocks.

Cover:

- Empty creation and every filesystem rejection.
- Malformed UTF-8/JSON, non-finite numbers, wrong root shape, and non-object
  messages.
- Exact inline, file, and stdin content, including Unicode and trailing
  newlines.
- Ordinary messages and arbitrary raw provider fields.
- Append and every insertion boundary.
- Flat, nested, and empty tool arguments.
- Explicit and automatic call IDs.
- Tool result content from all three ingestion methods.
- Standard tool-call and result validation, including multiple calls in an
  existing raw Context.
- Content-only and complete-message replacement.
- Valid and invalid range removal and movement.
- Human/JSON outline and optional previews.
- Focused show output.
- Validation and publication failures preserving exact original bytes.
- Direct consumption of a builder-created Context by the existing generation
  command with a mocked endpoint.

Update `tests/test_dataset_cli.py` so command-discovery coverage recognizes the
nested `context` command without treating it as a row-transform command.

Keep existing authoring and UI Context tests unchanged; they are regression
coverage for the existing boundary.

### 6. Documentation

Update `README.md` with concise examples for:

- Creating a Context.
- Adding a message from stdin and from a file.
- Adding fabricated web-search and file-read exchanges.
- Inspecting and validating without printing the complete Context.
- Passing the resulting file directly to `dataset generate --context-file`.

Update `SPEC.md` with the command-group summary and reference
`SPEC-context-packages.md` for the full contract.

Do not add implementation commentary or abandoned alternatives to durable
documentation.

### 7. Verification and review

Run from `/home/goon/trl/trlx`:

```bash
/home/goon/trl/venv/bin/python -m unittest tests.test_context_builder
/home/goon/trl/venv/bin/python -m unittest tests.test_dataset_cli
/home/goon/trl/venv/bin/python -m unittest tests.test_authoring_cli
/home/goon/trl/venv/bin/python -m unittest tests.test_ui
/home/goon/trl/venv/bin/python -m unittest discover
```

Perform a CLI smoke test in a temporary directory:

1. Create one Context.
2. Add ordinary user and assistant messages through different ingestion paths.
3. Add fabricated web-search and file-read exchanges.
4. Replace one tool result.
5. Move one complete tool pair.
6. Inspect the outline and selected messages.
7. Validate the final Context.
8. Submit it unchanged through `dataset generate --context-file` using a mocked
   endpoint or the existing CLI integration test fixture.

Review the final diff against `PRINCIPLES.md`, the approved specification, and
the agreed file scope. In particular, confirm that no endpoint, generation, UI,
configuration, dependency, or dataset behavior changed.

## Completion criteria

Implementation is complete only when:

- Every command and invariant in `SPEC-context-packages.md` is implemented.
- Focused and full tests pass, except any pre-existing unrelated failure that is
  identified with evidence.
- The smoke test succeeds without hand-editing the Context JSON.
- Existing generation accepts the produced file directly.
- Failed mutations preserve exact original bytes.
- The implementation contains no generation, retrieval, compilation,
  knowledge-base, provenance, or training-row subsystem.
- No required implementation work remains.

## Separate validation phase

After implementation completion and separate authorization, use the tool to
build one reusable security-design Context and apply three external prompts for
support authorization, refund authorization, and backup capabilities. Evaluate
the untouched generations before editing them. Dataset additions remain a
separate explicitly approved action.
