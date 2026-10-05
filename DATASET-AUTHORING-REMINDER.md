Before generating, reaffirm the editing contract below. It governs the entire batch, including reasoning, answers, validation, and saving.

Your task is to repair the target model’s actual output while preserving its voice and sound reasoning. You are not being asked to write the answer you would prefer.

PROCESS

1. Preserve the complete original prompt, reasoning, and answer separately throughout the session, including after saving.

2. Edit the original text directly. Before changing any span, identify its specific defect and the fact or explicit requirement it violates. If you cannot identify a defect, preserve that span verbatim.

3. Make the smallest correction that resolves the defect. Preserve surrounding sound wording, vocabulary, structure, explanations, examples, and design choices. Apply correctness and grammar checks to retained text as well as edited text.

4. Remove response-composition instructions, stale drafts, abandoned writing choices, and genuine repetition. Distinguish these from substantive reasoning, useful intermediate steps, and genuine uncertainty about the problem. Do not classify sound analysis as drafting residue merely because it is long or informal.

5. Count the complete rendered training example using the approved tokenizer and template. The token limit is a ceiling, not a target for aggressive shortening. If further reduction is necessary, remove only what is needed to fit without losing required content. If the scenario cannot fit, use the approved prompt-refinement procedure and generate afresh. Never change the saved prompt to fit an answer generated for a different prompt.

6. Use available likelihood measurements during editing, not merely for the completion report. Score original and edited reasoning and answers separately and together. Compare alternative necessary corrections under identical preceding context. For answer comparisons intended to isolate wording changes, hold the preceding reasoning fixed. Clearly distinguish those comparisons from whole-continuation scores using different reasoning.

7. Before saving, review the final diff for authorization as well as correctness:
   - Every change must address an allowed defect or necessary token-limit reduction.
   - Sound original passages must remain intact.
   - Reasoning and answer must remain consistent.
   - The saved text must be exactly the text validated and scored.
   - Any substantial likelihood deterioration must be investigated. Do not save while an editing or compatibility concern remains unresolved.

8. When using Context-assisted generation, treat reconstruction-level defects as a Context failure. If the untouched output repeatedly rediscovers requirements, cycles through abandoned designs, invents capabilities, or violates related invariants, stop the scenario and improve the Context. Add the defect classes, counterexamples, and acceptance checks, repeat the printed sufficiency challenge and rendering check, and generate afresh; do not compensate with extensive manual rewriting.

9. When using Context-assisted generation, favor improving the Context until the target model produces a response whose core reasoning and design are correct and require only local repair. Use direct edits for minor correctness, grammar, formatting, repetition, and token-limit cleanup after that threshold is met.

10. Review the final reasoning and answer as they will be saved, with system text and Context omitted. Remove or localize dangling references to “the research,” internal contracts, provided Context, tool calls, or prior authoring turns while preserving the substantive reasoning and any self-contained source attribution.

11. When using Context-assisted generation, before rendering print the concrete case that the Context is sufficient, the strongest good-faith case that it is not, and an evidence-based `READY` or `NOT READY` adjudication. If `NOT READY`, improve the Context and repeat. If `READY` and rendering passes, proceed without requesting user approval unless the user explicitly overrides normal procedure.

12. Treat an existing Context as a candidate until that challenge is `READY` for the exact prompt or named scenario set. When they are already known, complete the challenge before requesting scenario approval and include its adjudication in the proposal. Do not state that no Context changes are planned beforehand.

WHAT NOT TO DO

- Do not write an ideal replacement answer and then recover fragments of the original.
- Do not summarize or broadly rewrite sound passages because your version seems clearer, more formal, more robust, or better organized.
- Do not replace the model’s design choices with your preferred architecture.
- Do not discard substantive reasoning to make room for a rewritten answer.
- Do not reconstruct reasoning from the final answer.
- Do not compress toward an arbitrary length below the approved token limit.
- Do not treat a few local wording comparisons as validation of extensive rewriting elsewhere.
- Do not attribute all likelihood changes to wording when the preceding context also changed.
- Do not call edited examples “on-policy” merely because they originated from the target model or were scored by it.
- Do not use passing correctness checks as permission for unauthorized edits.
- Do not save questionable rows to finish the batch and explain the departures afterward.
- When using Context-assisted generation, do not keep generating from or reconstruct outputs produced by a Context whose sufficiency adjudication was `NOT READY`; improve the Context and repeat its challenge and rendering check first.
- When using Context-assisted generation, do not use direct editing to turn a fundamentally incorrect response into a different answer or complete rewrite; improve the Context and generate afresh.
- Do not weaken the case against Context sufficiency, treat size or structural validity as proof, or request user approval for a `READY` result under normal procedure.
- Do not justify a `READY` Context using post-generation compilation, tests, scoring, editing, formatting checks, or token or word limits; those validate output and cannot compensate for missing model-visible guidance.
- Do not treat a matching Context filename, description, prior use, or structural validity as a readiness decision.
- Do not save reasoning or answers that depend on excluded authoring scaffolding for their meaning or cite that hidden scaffolding as provenance.

Batch approval removes per-row approval requirements; it does not relax this contract. Continue autonomously within it. If an unexpected problem prevents compliance, stop the affected work before saving and explain the specific conflict.
