# Interactive dataset authoring UI

## Purpose

A web interface for generating, comparing, and editing single-turn responses,
then collecting selected examples into a training dataset. Generation uses
configured OpenAI-compatible endpoints. The interface does not run training.

## Web application

- Use FastAPI and an established HTTP server implementation.
- Expose `--host` and `--port` launch arguments. Deployment and access controls
  are the operator's responsibility.
- Necessary additional dependencies are permitted; unnecessary dependencies
  are prohibited. This supersedes SPEC.md's blanket prohibition for this feature.

## Prompts and generation

- Accept a user prompt and an optional system prompt, shared across outputs.
- Keep both prompts populated after generation and adding examples.
- Provide separate buttons to clear the user prompt, system prompt, or both.
- Generate a configurable number of response options, with a minimum of two.
- Each output has independently configurable endpoint, model, credentials, and
  sampling settings, all changeable in the UI.
- Each output exposes temperature, top-p, top-k, maximum output tokens,
  presence penalty, and repetition penalty.
- Each sampling control has an enable checkbox. Enabled values are sent to
  the endpoint; disabled values are omitted and their inputs disabled.
- Endpoint support determines whether a supplied parameter takes effect.
  Unsupported parameters may be ignored or rejected by the endpoint. Returned
  errors remain visible; the application does not silently remove parameters.

## Response editing and selection

- Display reasoning and assistant answer separately for every response option.
  Both are editable before adding the example.
- Each output has independent toggles for including reasoning and the answer.
  At least one must be selected.
- Each output has an **Add to dataset** button. Any number of outputs from a
  submission may be added; adding does not write the destination file.
- Unselected responses are not added automatically.

## Saved row contract

Every example contains a `messages` list with exactly one user message followed
by one assistant message. The user message is always included. The system
prompt is never included.

| Selection | Assistant `content` | Top-level `reasoning` |
|---|---|---|
| Answer only | Edited answer | Omitted |
| Reasoning only | Empty string | Edited reasoning |
| Both | Edited answer | Edited reasoning |

Duplicate identity is the exact user text, assistant content, and presence and
text of the `reasoning` field. Different reasoning makes a distinct example.
Do not normalize text when comparing examples.

## Pending collection and saving

- Added examples accumulate in a pending collection until explicit **Save**.
- The destination JSONL dataset is selectable in the UI.
- Save appends unique pending examples, preserving existing dataset contents.
- Skip exact duplicates within the pending collection and already present in
  the destination dataset.
- After successful saving, remove saved and already-present examples from the
  pending collection. A failed save retains pending work.

## Browser persistence

- Persist the pending collection, prompts, per-output endpoint and model
  settings, sampling settings, and credentials in browser `localStorage`.
- Restore that state across refreshes and application restarts when accessed
  through the same browser origin.
- Credentials are entered directly in the UI and may be stored in
  `localStorage`; environment-variable references are not required.
- Backend persistence for the pending collection is not required.

## Implementation decisions still pending

The command name, browser asset organization, API routes, dependency versions,
generation transport and progress presentation, and file publication mechanism
remain subject to implementation-plan approval. Staged rewriting to implement
append semantics was proposed but has not been approved.
