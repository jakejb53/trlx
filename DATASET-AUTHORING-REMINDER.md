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

8. When Context-assisted output has reconstruction-level defects, stop that scenario and treat the generation as private evidence of a curriculum gap. Identify the missing general prerequisite, retrieve authoritative evidence, and teach it through natural investigation and an analogous exercise using different names and circumstances. Never mention the failed attempt or place its defect list, exact correction, corrected final-task code, reviewer feedback, or tailored acceptance checks in model-visible Context. Repeat the sufficiency challenge and rendering check before generating again.

9. Improve Context until the target model has the general knowledge and reasoning methods needed to produce a substantially correct response requiring only local repair. Test every Context repair for feedback framing and answer leakage. A shorter postmortem or correction list is not a curriculum.

10. Review the final reasoning and answer as they will be saved, with system text and Context omitted. Remove or localize dangling references to “the research,” internal contracts, provided Context, tool calls, or prior authoring turns while preserving the substantive reasoning and any self-contained source attribution.

11. When using Context-assisted generation, before rendering print the concrete case that the Context is sufficient, the strongest good-faith case that it is not, and an evidence-based `READY` or `NOT READY` adjudication. If `NOT READY`, improve the Context and repeat. If `READY` and rendering passes, proceed without requesting user approval unless the user explicitly overrides normal procedure.

12. Treat an existing Context as a candidate until that challenge is `READY` for the exact prompt or named scenario set. When they are already known, complete the challenge before requesting scenario approval and include its adjudication in the proposal. Do not state that no Context changes are planned beforehand.

13. Before every new or changed model-visible tool result, apply the evidence provenance gate: actually retrieve, execute, or read the previously verified artifact; inspect and verify the evidence; then construct and add the tool exchange. “Fabricated” or “simulated” applies only to the model-visible call, never to its result. A citation, URL, search snippet, remembered fact, or agent-written summary is not retrieved evidence by itself. Audit every retrieval and execution claim before declaring the Context `READY`.

14. If any existing or newly added Context material violates these instructions, stop using the Context, remove the violating material completely, and replace it only with curriculum that independently complies. Remove a complete tool-call/result pair when either half is affected. Do not preserve the violation beside a correction or disclaimer. After replacement, reconnect the conversation, update its description when needed, and repeat structural validation, provenance, feedback, answer-leakage, sufficiency, and rendering checks.

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
- Do not expose a rejected generation, its defects, its score, or its correction history in model-visible Context.
- Do not phrase Context as review feedback, retry guidance, or instructions to avoid mistakes made by a prior response.
- Do not provide the final task's exact correction, corrected implementation, or scenario-specific acceptance checklist through a tool result, internal document, worked example, or assistant analysis.
- Do not append isolated corrective facts. Integrate authoritative evidence, analysis, and an analogous exercise into the curriculum.
- Do not accept a Context repair when the final answer can be obtained mainly by copying it and substituting the final prompt's names.
- Do not save reasoning or answers that depend on excluded authoring scaffolding for their meaning or cite that hidden scaffolding as provenance.
- Do not draft a tool result and verify it afterward, or claim that material was retrieved, executed, observed, or verified when the supporting operation did not occur.
- Do not treat naming an authoritative source as using it. Read and verify the relevant material, preserve its location, and distinguish excerpts, faithful paraphrases, and original analysis.
- Do not leave instruction-violating material in Context and append a correction, disclaimer, or compliant alternative around it; remove the violation and replace it with compliant curriculum.

Batch approval removes per-row approval requirements; it does not relax this contract. Continue autonomously within it. If an unexpected problem prevents compliance, stop the affected work before saving and explain the specific conflict.
