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
  are prohibited.

## Prompts and generation

- Share the user prompt across outputs; each output has its own optional system prompt.
- Each output has an optional Context JSON editor. Blank means no context; otherwise
  require an array of objects. Send the system prompt, Context entries in their original
  order, then the user prompt. Preserve all Context message fields without interpretation.
- Keep prompts populated after generation and adding examples.
- Provide a button to clear the user prompt and a separate system-prompt clear button per output.
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
prompt and Context are never included.

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

- Persist the pending collection, shared user prompt, per-output system prompts, endpoint and model
  settings, sampling settings, credentials, and Context editor text in browser `localStorage`.
- Restore that state across refreshes and application restarts when accessed
  through the same browser origin.
- Credentials are entered directly in the UI and may be stored in
  `localStorage`; environment-variable references are not required.
- Backend persistence for the pending collection is not required.

## Runtime contract

- Launch with `dataset ui --host HOST --port PORT`; both arguments are required.
  Port is an integer in 1..65535. From an uninstalled checkout, use
  `python -m dataset.cli ui --host HOST --port PORT`.
- FastAPI serves the page and API through Uvicorn. Each output uses an independent
  non-streaming request; completed responses appear without waiting for other outputs.
  Failures retain the previous response, when present, and identify it as retained.
- Add to dataset captures the current user prompt and edited response. Later edits
  do not alter examples already in the pending collection.
- Per-output timeout and retries start at 120 seconds and 2 additional attempts,
  respectively, matching the dataset endpoint defaults. Both are editable and persisted.
- Saved paths refer to the server filesystem, relative to its working directory
  unless absolute. The parent must exist; destinations must be regular JSONL files
  or new paths, not directories or final symlinks.
- Save stages existing bytes plus unique appended rows before publication.
  Requests within one UI process serialize duplicate checking and publication.
  Unrelated processes must not concurrently write the destination.
- Browser storage also retains edited responses and the destination path. Storage
  errors are visible and never trigger a silent reset of saved browser state.
