const STORAGE_KEY = "trlx.dataset.workspace.v1";
const CONTROLS = [
  ["temperature", "Temperature", 1, "any", 0],
  ["top_p", "Top-p", 1, "any", 0, 1],
  ["top_k", "Top-k", 40, "1", -1],
  ["max_tokens", "Maximum output tokens", 1024, "1", 1],
  ["presence_penalty", "Presence penalty", 0, "any"],
  ["repetition_penalty", "Repetition penalty", 1, "any", 0.000001],
];

// New cards have no guessed endpoint or model; sampling values remain inactive.
export function newCard() {
  return {endpoint: "", model: "", api_key: "", timeout: 120, retries: 2,
    sampling: Object.fromEntries(CONTROLS.map(([name, , value]) => [name, {enabled: false, value}])),
    includeReasoning: true, includeAnswer: true, response: null};
}

// Reasoning presence is part of identity; field ordering and metadata are not.
export function identity(row) {
  return JSON.stringify([row.messages[0].content, row.messages[1].content,
    Object.hasOwn(row, "reasoning"), row.reasoning ?? null]);
}

// Capture the current prompt and edited response when adding, not when generating.
export function exampleFrom(card, user) {
  if (!card.response) throw new Error("Generate a response first.");
  if (!card.includeReasoning && !card.includeAnswer) throw new Error("Select reasoning, assistant answer, or both.");
  const row = {messages: [{role: "user", content: user},
    {role: "assistant", content: card.includeAnswer ? card.response.answer : ""}]};
  if (card.includeReasoning) row.reasoning = card.response.reasoning;
  return row;
}

// Disabled controls are omitted, including any stale value left in their input.
export function generationBody(card, user, system) {
  const sampling = {};
  for (const [name] of CONTROLS) {
    if (card.sampling[name].enabled) {
      const value = card.sampling[name].value;
      if (value === "" || !Number.isFinite(Number(value))) throw new Error(`Enter a numeric value for ${name}.`);
      sampling[name] = Number(value);
    }
  }
  if (card.timeout === "" || card.retries === "") throw new Error("Enter timeout and retries.");
  return {endpoint: card.endpoint, model: card.model, api_key: card.api_key,
    timeout: Number(card.timeout), retries: Number(card.retries), user, system, sampling};
}

// Validate stored structure before binding editors; an invalid workspace is never overwritten.
export function validateWorkspace(value) {
  if (!value || value.version !== 1 || !Array.isArray(value.cards) || value.cards.length < 2 ||
      !Array.isArray(value.pending) || ![value.user, value.system, value.destination].every(v => typeof v === "string")) {
    throw new Error("Stored workspace has an unsupported or damaged format.");
  }
  for (const card of value.cards) {
    if (![card.endpoint, card.model, card.api_key].every(v => typeof v === "string") ||
        typeof card.includeReasoning !== "boolean" || typeof card.includeAnswer !== "boolean" ||
        ![card.timeout, card.retries].every(v => typeof v === "number" || typeof v === "string")) {
      throw new Error("Stored output settings are invalid.");
    }
    for (const [name] of CONTROLS) {
      const control = card.sampling?.[name];
      if (!control || typeof control.enabled !== "boolean" || !["number", "string"].includes(typeof control.value)) {
        throw new Error("Stored sampling settings are invalid.");
      }
    }
    if (card.response !== null && (!card.response ||
        ![card.response.user, card.response.answer, card.response.reasoning].every(v => typeof v === "string"))) {
      throw new Error("Stored response is invalid.");
    }
  }
  for (const row of value.pending) {
    if (!Array.isArray(row.messages) || row.messages.length !== 2 ||
        row.messages[0]?.role !== "user" || row.messages[1]?.role !== "assistant" ||
        !row.messages.every(m => typeof m.content === "string") ||
        (Object.hasOwn(row, "reasoning") && typeof row.reasoning !== "string")) {
      throw new Error("Stored pending example is invalid.");
    }
  }
  return value;
}

// Browser initialization stays separate from pure row/request helpers for verification.
export function start() {
  const $ = selector => document.querySelector(selector);
  let workspace;
  let lastStored;
  let busy = false;
  let saving = false;
  const storageError = $("#storage-error");
  try {
    lastStored = localStorage.getItem(STORAGE_KEY);
    workspace = lastStored === null ? {version: 1, user: "", system: "", destination: "", cards: [newCard(), newCard()], pending: []}
      : validateWorkspace(JSON.parse(lastStored));
  } catch (error) {
    storageError.textContent = `Cannot restore workspace: ${error.message} Stored data was not changed.`;
    storageError.hidden = false;
    document.querySelectorAll("button, input, textarea").forEach(node => { node.disabled = true; });
    return;
  }

  // Detect another tab's changes rather than overwriting its pending examples.
  function persist() {
    try {
      if (localStorage.getItem(STORAGE_KEY) !== lastStored) {
        throw new Error("This workspace changed in another tab. Use that tab or reload before continuing.");
      }
      const serialized = JSON.stringify(workspace);
      localStorage.setItem(STORAGE_KEY, serialized);
      lastStored = serialized;
      storageError.hidden = true;
      return true;
    } catch (error) {
      storageError.textContent = `Browser storage failed: ${error.message} Current changes are only in memory; keep this page open.`;
      storageError.hidden = false;
      return false;
    }
  }

  // JSON-only requests keep credentials in the body, never URLs or access logs.
  async function post(path, body) {
    const response = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    let result;
    try { result = await response.json(); }
    catch { throw new Error(`Server returned an unreadable response (HTTP ${response.status}).`); }
    if (!response.ok) throw new Error(result.error || `Request failed (HTTP ${response.status}).`);
    return result;
  }

  // Text content is never interpreted as markup, including model output and prompts.
  function showMessage(node, text, error = false) {
    node.textContent = text;
    node.classList.toggle("error", error);
  }

  // Each add captures an immutable edited example; later edits do not mutate the collection.
  function addExample(card, message) {
    try {
      const row = exampleFrom(card, workspace.user);
      if (workspace.pending.some(existing => identity(existing) === identity(row))) {
        showMessage(message, "This exact example is already pending.");
        return;
      }
      workspace.pending.push(row);
      const stored = persist();
      renderPending();
      showMessage(message, stored ? "Added to pending dataset." : "Added in memory; browser persistence failed.", !stored);
    } catch (error) { showMessage(message, error.message, true); }
  }

  // Render settings from state only when the card list changes, preserving focused editors.
  function renderCards() {
    const outputs = $("#outputs");
    outputs.replaceChildren();
    workspace.cards.forEach((card, index) => {
      const article = $("#output-template").content.firstElementChild.cloneNode(true);
      article.querySelector("h3").textContent = `Output ${index + 1}`;
      for (const input of article.querySelectorAll("[data-setting]")) {
        const name = input.dataset.setting;
        input.value = card[name];
        input.addEventListener("input", () => { card[name] = input.value; persist(); });
      }
      for (const [name, label, , step, min, max] of CONTROLS) {
        const field = document.createElement("div");
        const caption = document.createElement("label");
        caption.className = "inline";
        const toggle = document.createElement("input");
        toggle.type = "checkbox";
        toggle.checked = card.sampling[name].enabled;
        const input = document.createElement("input");
        input.type = "number";
        input.step = step;
        if (min !== undefined) input.min = min;
        if (max !== undefined) input.max = max;
        input.value = card.sampling[name].value;
        input.disabled = !toggle.checked;
        input.setAttribute("aria-label", `${label} for output ${index + 1}`);
        caption.append(toggle, document.createTextNode(label));
        toggle.addEventListener("change", () => {
          card.sampling[name].enabled = toggle.checked;
          input.disabled = !toggle.checked;
          persist();
        });
        input.addEventListener("input", () => { card.sampling[name].value = input.value; persist(); });
        field.append(caption, input);
        article.querySelector(".sampling").append(field);
      }
      for (const input of article.querySelectorAll("[data-include]")) {
        const key = input.dataset.include === "reasoning" ? "includeReasoning" : "includeAnswer";
        input.checked = card[key];
        input.addEventListener("change", () => { card[key] = input.checked; persist(); });
      }
      for (const input of article.querySelectorAll("[data-response]")) {
        input.setAttribute("aria-label", `${input.dataset.response} for output ${index + 1}`);
        input.value = card.response?.[input.dataset.response] ?? "";
        input.disabled = !card.response;
        input.addEventListener("input", () => { card.response[input.dataset.response] = input.value; persist(); });
      }
      const add = article.querySelector(".add");
      add.disabled = !card.response;
      add.addEventListener("click", () => addExample(card, article.querySelector(".card-message")));
      if (card.response) article.querySelector(".card-status").textContent = "Ready to review";
      outputs.append(article);
    });
  }

  // Pending rows are inspectable and removable before the explicit file save.
  function renderPending() {
    $("#pending-count").textContent = workspace.pending.length;
    $("#collection-count").textContent = `${workspace.pending.length} example${workspace.pending.length === 1 ? "" : "s"}`;
    const container = $("#pending");
    container.replaceChildren();
    if (!workspace.pending.length) {
      const empty = document.createElement("p");
      empty.className = "empty";
      empty.textContent = "Add a response above to start your dataset.";
      container.append(empty);
    }
    workspace.pending.forEach((row, index) => {
      const item = document.createElement("details");
      item.className = "pending-item";
      const summary = document.createElement("summary");
      const title = document.createElement("span");
      title.textContent = `${index + 1}. ${row.messages[0].content.slice(0, 120)}`;
      const remove = document.createElement("button");
      remove.textContent = "Remove";
      remove.className = "quiet";
      remove.disabled = saving;
      remove.addEventListener("click", event => {
        event.preventDefault();
        workspace.pending.splice(index, 1);
        persist();
        renderPending();
      });
      summary.append(title, remove);
      item.append(summary);
      const fields = [["User", row.messages[0].content]];
      if (Object.hasOwn(row, "reasoning")) fields.push(["Reasoning", row.reasoning]);
      fields.push(["Assistant", row.messages[1].content]);
      for (const [name, text] of fields) {
        const heading = document.createElement("h4");
        heading.textContent = name;
        const content = document.createElement("pre");
        content.textContent = text || "(empty)";
        item.append(heading, content);
      }
      container.append(item);
    });
    $("#save").disabled = saving || !workspace.pending.length;
  }

  // A round captures prompts/settings before any network call. Previous candidates
  // remain recoverable on failure; cards finish independently without a shared batch failure.
  async function generate() {
    if (busy) return;
    if (!workspace.user.trim()) { showMessage($("#generation-status"), "Enter a user prompt.", true); return; }
    // Snapshot all inputs, but validate inside each task so one bad card cannot
    // prevent valid neighbors from issuing their own requests.
    const cards = structuredClone(workspace.cards);
    const user = workspace.user;
    const system = workspace.system;
    persist();
    busy = true;
    $("#generate").disabled = true;
    $("#output-count").disabled = true;
    let finished = 0;
    showMessage($("#generation-status"), `0 / ${cards.length} finished`);
    await Promise.all(cards.map(async (settings, index) => {
      const card = workspace.cards[index];
      const article = $("#outputs").children[index];
      const status = article.querySelector(".card-status");
      const message = article.querySelector(".card-message");
      article.querySelector(".add").disabled = true;
      article.querySelectorAll("[data-response]").forEach(input => { input.disabled = true; });
      const started = Date.now();
      status.textContent = "Waiting for endpoint…";
      showMessage(message, "");
      const clock = setInterval(() => { status.textContent = `Waiting · ${Math.floor((Date.now() - started) / 1000)}s`; }, 1000);
      try {
        const body = generationBody(settings, user, system);
        const result = await post("api/generate", body);
        if (typeof result.answer !== "string" || typeof result.reasoning !== "string") throw new Error("Server returned an invalid response.");
        card.response = {user: body.user, answer: result.answer, reasoning: result.reasoning};
        article.querySelectorAll("[data-response]").forEach(input => { input.value = card.response[input.dataset.response]; });
        persist();
        status.textContent = "Ready to review";
        article.querySelector("details").open = false;
      } catch (error) {
        status.textContent = "Generation failed";
        showMessage(message, error.message + (card.response ? " Previous response retained." : ""), true);
      } finally {
        clearInterval(clock);
        article.querySelector(".add").disabled = !card.response;
        article.querySelectorAll("[data-response]").forEach(input => { input.disabled = !card.response; });
        finished += 1;
        showMessage($("#generation-status"), `${finished} / ${cards.length} finished`);
      }
    }));
    busy = false;
    $("#generate").disabled = false;
    $("#output-count").disabled = false;
  }

  // Only the submitted snapshot is removed after success; examples added while
  // saving stay pending. A lost response retains everything for a deduplicated retry.
  async function save() {
    if (saving || !workspace.pending.length) return;
    if (!workspace.destination.trim()) { showMessage($("#save-status"), "Enter a .jsonl dataset path on the server.", true); return; }
    // Saving is the escape path when browser storage is full or unavailable.
    // The explicit in-memory snapshot remains saveable even if persistence fails.
    persist();
    const submitted = workspace.pending.slice();
    const destination = workspace.destination;
    saving = true;
    renderPending();
    showMessage($("#save-status"), `Saving to ${destination}…`);
    try {
      const result = await post("api/save", {path: destination, examples: submitted});
      if (!Number.isInteger(result.added) || !Number.isInteger(result.duplicates) || result.added < 0 ||
          result.duplicates < 0 || result.added + result.duplicates !== submitted.length) throw new Error("Server did not confirm the complete save; pending examples retained.");
      workspace.pending = workspace.pending.filter(row => !submitted.includes(row));
      persist();
      showMessage($("#save-status"), `Saved ${result.added}; skipped ${result.duplicates} duplicate${result.duplicates === 1 ? "" : "s"}. Dataset: ${destination}`);
    } catch (error) { showMessage($("#save-status"), `Save failed: ${error.message} Pending examples retained.`, true); }
    finally { saving = false; renderPending(); }
  }

  // Clearing prompts never discards pending data or changes a candidate's captured user text.
  function clearPrompts(names) {
    for (const name of names) { workspace[name] = ""; $(`#${name}`).value = ""; }
    persist();
  }

  for (const name of ["user", "system", "destination"]) {
    $(`#${name}`).value = workspace[name];
    $(`#${name}`).addEventListener("input", event => { workspace[name] = event.target.value; persist(); });
  }
  $("#clear-user").addEventListener("click", () => clearPrompts(["user"]));
  $("#clear-system").addEventListener("click", () => clearPrompts(["system"]));
  $("#clear-both").addEventListener("click", () => clearPrompts(["user", "system"]));
  $("#output-count").value = workspace.cards.length;
  $("#output-count").addEventListener("change", event => {
    const count = Number(event.target.value);
    if (!Number.isSafeInteger(count) || count < 2 || busy) { event.target.value = workspace.cards.length; return; }
    while (workspace.cards.length < count) workspace.cards.push(newCard());
    workspace.cards.length = count;
    persist();
    renderCards();
  });
  $("#generate").addEventListener("click", generate);
  $("#save").addEventListener("click", save);
  renderCards();
  renderPending();
}

if (typeof document !== "undefined") start();
