# Context builder for dataset authoring

## Purpose

The context builder incrementally creates and edits the exact JSON messages
array accepted by:

```bash
dataset generate --context-file CONTEXT.json
```

The Context file itself is the durable source of truth. There is no package
wrapper, export format, generated snapshot, profile, recipe, fragment graph,
knowledge-base schema, or generation record. The same Context file can be
reused with any number of system prompts and final user prompts supplied through
the existing dataset-authoring workflow.

The tool exists to make a plain messages array safe and convenient to build over
many agent or human interactions. It provides atomic mutation, focused
inspection, validation, and construction of valid fabricated tool exchanges.

## Scope

The tool can:

- Create an empty Context messages array.
- Insert ordinary messages using inline, file, or stdin content.
- Insert arbitrary raw message objects.
- Construct paired assistant tool-call and tool-result messages.
- Replace message content or a complete message.
- Remove and move messages or contiguous ranges.
- Show selected messages without loading the complete Context.
- Summarize message positions, roles, sizes, and tool-call relationships.
- Validate the complete array, including standard function-tool pairing.

The tool does not:

- Generate model responses.
- Manage system prompts or final scenario prompts.
- Fetch URLs, query databases, execute commands, or invoke represented tools.
- Record source provenance outside the messages themselves.
- Compose fragments, profiles, recipes, or inherited contexts.
- Count tokens or enforce a model context window in the first implementation.
- Save training examples.
- Add private IDs or metadata to model-visible message objects.
- Maintain a parallel sidecar or hidden workspace.

Research and retrieval happen through other tools. The context builder records
their supplied results in the representation chosen by the caller.

## Durable file contract

A Context file is UTF-8 JSON containing one array. Every array element is a
non-null JSON object. This is exactly the current `--context-file` contract;
the builder does not introduce another persistent schema.

For example:

```json
[
  {
    "role": "user",
    "content": "Investigate the authorization boundary before answering."
  },
  {
    "role": "assistant",
    "content": null,
    "tool_calls": [
      {
        "id": "call_0001",
        "type": "function",
        "function": {
          "name": "web_search",
          "arguments": "{\"query\":\"JWT algorithm verification requirements\"}"
        }
      }
    ]
  },
  {
    "role": "tool",
    "tool_call_id": "call_0001",
    "content": "Supplied search result text."
  }
]
```

Arbitrary provider-owned message fields are preserved. The builder interprets
only fields needed by the requested operation and the standard tool validation
defined below.

JSON numbers must be finite. The builder never normalizes string content,
including Unicode, spaces, or leading and trailing newlines. File and stdin
content is inserted exactly after UTF-8 decoding.

A successful mutation rewrites the complete array using deterministic
two-space-indented UTF-8 JSON, `ensure_ascii=false`, and one final newline.
Object key order already present in loaded messages is retained. Builder-created
messages use the field order shown in this specification. Array order is always
authoritative.

## Independent input and representation choices

Content ingestion and message representation are independent.

### Content ingestion

An operation that needs content accepts exactly one source:

- `--text TEXT`: use the command-line string exactly.
- `--content-file PATH`: read the complete UTF-8 file exactly.
- Neither option: read complete UTF-8 content from stdin.

`--content-file -` is equivalent to stdin. Supplying both `--text` and
`--content-file` is an error. Supplying either option while stdin is ignored is
not an error; the selected explicit source is authoritative.

A file used as the content source does not determine the model-visible role,
tool name, tool arguments, path, URL, query, or any other representation. For
example, bytes read from `/tmp/page.txt` may be represented as a
`web_search` result for a caller-supplied query.

Content files are inputs only. Their paths are not stored unless the caller also
puts those paths in a message or tool argument.

### Message representation

The same ingested content can be represented as:

- An ordinary message with a caller-selected role.
- The result of a fabricated standard function-tool call.
- A complete raw message object supplied by the caller.

“Web search,” “file read,” “database query,” and similar operations are ordinary
caller-selected tool names and arguments. They are representations, not
executable adapters.

## CLI contract

The command group is:

```text
dataset context create PATH
dataset context add PATH --role ROLE [--at INDEX] [CONTENT SOURCE]
dataset context add PATH --message-file MESSAGE.json [--at INDEX]
dataset context tool PATH --name NAME [ARGUMENTS] [--call-id ID] [--at INDEX] [CONTENT SOURCE]
dataset context replace PATH INDEX [CONTENT SOURCE]
dataset context replace PATH INDEX --message-file MESSAGE.json
dataset context remove PATH INDEX [--count COUNT]
dataset context move PATH FROM TO [--count COUNT]
dataset context outline PATH [--json] [--preview CHARACTERS]
dataset context show PATH INDEX [--count COUNT]
dataset context validate PATH
```

Indices are zero-based. Negative indices are rejected. Commands never interpret
Python-style negative indexing.

All commands use existing dataset CLI conventions: progress and diagnostics go
to stderr, machine results go to stdout, expected input failures produce no
traceback, and secrets or complete content are not echoed in errors.

### Create

```bash
dataset context create context.json
```

`create` writes `[]` in deterministic formatting. The parent directory must
already exist. The destination must not exist, including as a symlink. There is
no `--force`; replacing a Context is an explicit series of mutations or a new
file.

### Add an ordinary message

Content from stdin:

```bash
dataset context add context.json --role user < question.txt
```

Content from a file:

```bash
dataset context add context.json \
  --role assistant \
  --content-file analysis.txt
```

Short inline content:

```bash
dataset context add context.json \
  --role user \
  --text "Now apply those findings to this design."
```

`--role` must be a nonempty string. The builder does not restrict roles to a
fixed enum because Context fields belong to the endpoint. It creates exactly:

```json
{"role": "ROLE", "content": "INGESTED CONTENT"}
```

Without `--at`, the message is appended. `--at` accepts an insertion index
from zero through the current array length, inclusive.

### Add a raw message

```bash
dataset context add context.json \
  --message-file provider-message.json \
  --at 3
```

The file must contain one non-null JSON object with finite values. It is inserted
without field removal, role conversion, content conversion, or normalization.
`--message-file` is mutually exclusive with `--role` and every content-source
option.

This is the escape hatch for rich content and provider-specific messages. The
ordinary role and tool helpers remain preferred when they can express the
intended message.

### Add a fabricated tool exchange

Result content from a file:

```bash
dataset context tool context.json \
  --name web_search \
  --arg 'query=JWT verification requirements' \
  --content-file search-results.txt
```

Result content from stdin:

```bash
curl --silent --show-error https://example.test/page |
  dataset context tool context.json \
    --name web_fetch \
    --arg 'url=https://example.test/page'
```

Actual file content represented as a fabricated file read:

```bash
dataset context tool context.json \
  --name read_file \
  --arg 'path=manual.md' \
  --content-file manual.md
```

The tool name is a nonempty string. Arguments use one of:

- Repeatable `--arg KEY=VALUE`, producing an object whose values are strings.
- `--arguments-file PATH`, reading one finite JSON object.

The two forms are mutually exclusive. Duplicate `--arg` keys are errors.
Omitting both forms produces an empty argument object. The builder serializes
the argument object as compact deterministic JSON with sorted keys; that string
becomes `function.arguments`.

The operation inserts two adjacent messages:

```json
[
  {
    "role": "assistant",
    "content": null,
    "tool_calls": [
      {
        "id": "CALL ID",
        "type": "function",
        "function": {
          "name": "NAME",
          "arguments": "COMPACT JSON OBJECT"
        }
      }
    ]
  },
  {
    "role": "tool",
    "tool_call_id": "CALL ID",
    "content": "INGESTED CONTENT"
  }
]
```

`--call-id` supplies a deliberate nonempty ID. Otherwise the builder scans all
standard tool calls and chooses the lowest unused positive
`call_NNNN` identifier, beginning with `call_0001`. Allocation is
deterministic for the current array. Any duplicate call ID is an error.

Without `--at`, the pair is appended. With `--at`, the assistant call is
inserted at that index and its result immediately after it. The insertion index
may range from zero through the current array length.

The builder does not claim that the represented tool was invoked. It only
constructs the conversation supplied by the author.

### Replace

Replace only one message’s `content` field:

```bash
dataset context replace context.json 4 --content-file corrected-result.txt
```

Content replacement preserves every other field and sets `content` to the
ingested string. The target must be a message object; an absent existing
`content` field is created.

Replace the complete message:

```bash
dataset context replace context.json 4 --message-file corrected-message.json
```

The raw file follows the same contract as raw insertion. `--message-file` is
mutually exclusive with content-source options.

Replacement validates the complete resulting Context before publication.
Changing a call or result in a way that breaks tool pairing therefore fails
without altering the file.

### Remove

```bash
dataset context remove context.json 4
dataset context remove context.json 4 --count 2
```

`--count` is a positive integer and defaults to one. The complete range must
exist. Removal validates the resulting Context; deleting only one half of a
standard tool exchange fails. Removing both paired messages succeeds.

### Move

```bash
dataset context move context.json 4 10
dataset context move context.json 4 10 --count 2
```

The command removes the contiguous source range and inserts it at `TO` in the
array that remains after removal. `--count` is positive and defaults to one.
The complete source range and destination must be valid. Moving one half of a
tool exchange to an invalid position fails final validation.

### Outline

```bash
dataset context outline context.json
dataset context outline context.json --json
dataset context outline context.json --preview 80
```

The default human-readable table contains:

- Zero-based index.
- Role or `<missing>`.
- Character count for string content, otherwise the content JSON type.
- Tool-call IDs and names declared by the message.
- Tool-call ID referenced by a tool result.

It does not print message content by default. `--preview` adds at most the
requested number of content characters with control characters escaped.
`--json` emits the same information as a JSON array. Preview text remains
excluded unless explicitly requested.

### Show

```bash
dataset context show context.json 20
dataset context show context.json 20 --count 3
```

`show` prints the selected message or array of messages as indented UTF-8 JSON.
It never mutates or normalizes the Context file. `--count` is positive and
defaults to one.

### Validate

```bash
dataset context validate context.json
```

Validation prints a compact JSON result containing the message count, standard
tool-call count, standard tool-result count, and `{"valid": true}`. Invalid
Context exits nonzero with an indexed error and does not echo complete content.

## Standard tool validation

The durable Context contract remains an array of arbitrary objects. When
standard function-tool fields appear, validation checks their structure rather
than passing malformed fabricated exchanges silently.

For each message:

- `tool_calls`, when present, is a nonempty array.
- Each standard tool call is an object with nonempty string `id`,
  `type = "function"`, and a `function` object.
- `function.name` is a nonempty string.
- `function.arguments` is a string containing one finite JSON object.
- Standard tool-call IDs are unique across the Context.
- A message with `role = "tool"` has a nonempty string `tool_call_id`.
- Every standard tool call has exactly one later tool result.
- Every standard tool result refers to one earlier unresolved call.
- A standard tool result precedes any later non-tool conversational message
  after its declaring assistant message.

Multiple tool calls in one raw assistant message are allowed, followed by one
result per call in any order. The `tool` helper creates one call and one result.

Provider-specific message objects that do not use `tool_calls` or
`tool_call_id` remain opaque. Once those standard field names are used, their
standard invariants apply.

Validation does not require strict user/assistant alternation and does not
require the Context to end in a particular role. The existing generation command
appends the final user prompt after the Context.

## Mutation and publication behavior

Every mutation:

1. Opens and parses the complete current Context.
2. Validates the current Context.
3. Applies the requested change in memory.
4. Validates the complete result.
5. Serializes deterministic JSON.
6. Publishes the replacement through the repository’s staged output mechanism.

A parse, validation, input-read, serialization, flush, publication, or cleanup
failure preserves the original Context bytes. Final symlinks and non-regular
files are rejected.

Writers to one Context file must be serialized by the caller, matching the
existing stateless dataset-authoring contract. The first implementation does not
add cross-process locks, revision fields, hidden lock files, or sidecars.

Read-only commands may run concurrently with mutation. Staged atomic replacement
ensures a reader observes either the complete previous file or the complete new
file.

## Error behavior

Expected failures use `DatasetError`, exit nonzero, and produce no traceback.
Messages identify the path, operation, index, field, tool-call ID, or argument
key involved and state how to correct the input when known.

Errors do not print complete message content, complete tool results, or data read
from content files. File decoding errors identify the input path. Shell
arguments remain the caller’s responsibility; the tool never evaluates them as
code.

## Python ownership

A dataset-owned module provides the reusable operations:

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

The module owns parsing, validation, deterministic serialization, mutation, and
publication. `dataset.cli` owns argument parsing, stdin selection, and output
formatting only. The browser and existing authoring module are unchanged.

The implementation imports nothing from `trl` or `trlx` and adds no
dependency.

## Compatibility

- Existing Context files that satisfy the current array-of-objects contract can
  be inspected and edited.
- Standard tool fields receive the additional structural validation described
  above.
- `dataset generate --context-file` remains unchanged and consumes builder
  output directly.
- The browser Context editor remains unchanged.
- Saved training examples remain unchanged and continue to exclude Context.
- No migration or export command is necessary.

## Tests

Tests use real CLI dispatch and temporary files. Coverage includes:

- Empty creation, existing paths, missing parents, symlinks, and non-regular
  files.
- UTF-8, malformed JSON, non-finite values, non-array roots, and non-object
  elements.
- stdin, inline, and file content with exact Unicode and newline preservation.
- Ordinary roles and raw provider-owned message fields.
- Append and every valid insertion boundary.
- Flat string arguments, nested arguments files, empty arguments, duplicate
  keys, invalid argument JSON, and deterministic argument serialization.
- Automatic and explicit call IDs.
- Tool creation from inline, stdin, and file result content.
- Multiple standard calls, result ordering, missing results, duplicate results,
  unknown results, duplicate IDs, and malformed function fields.
- Content-only replacement and complete-message replacement.
- Range removal and movement, including paired tool exchanges.
- Human and JSON outlines with and without previews.
- Focused show and all index/count boundary errors.
- Failed current-state validation and failed result validation preserving exact
  original bytes.
- Staged publication failure preserving the original.
- Existing `dataset generate`, browser Context, and dataset-save tests remain
  unchanged.

## Acceptance criteria

The first implementation is complete when:

1. An agent can build a substantial Context over many calls without resending or
   rereading the complete array.
2. Human CLI use requires no JSON for ordinary messages or ordinary single-call
   tool exchanges.
3. Inline, file, and stdin content can each be represented independently as a
   user message, assistant message, or fabricated tool result.
4. The resulting file is accepted directly by
   `dataset generate --context-file`.
5. All mutations are deterministic and preserve the original on failure.
6. The three rejected security scenarios can reuse one Context file with three
   external final prompts.
7. No implementation component performs generation, retrieval, context
   compilation, knowledge-base management, or training-row publication.
