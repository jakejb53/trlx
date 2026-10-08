# Reasoning edit reminder

At this moment you have one goal: complete the reasoning edit while preserving the target model's substantive reasoning.

Make every necessary local correctness, grammar, and consistency repair. Remove response-composition instructions, abandoned drafts, repeated reasoning, circular deliberation, and word-count commentary. Preserve all sound substantive reasoning verbatim.

You are editing the model's reasoning, not writing your own. A clearer, shorter, more correct, or better-organized replacement is still a failed edit if sound original reasoning could have been preserved.

This process has failed before because the agent replaced reasoning after convincing itself that the rewrite was cleanup. Do not repeat that failure.

Read the untouched reasoning again. Change only exact defective spans and genuine drafting residue. Preserve everything else verbatim.

If correctness requires new reasoning, reorganization, summarization, or broad replacement, do not edit it. Reject the generation and regenerate.

The raw rendered token count has already been measured and is known. The token limit does not justify deleting sound reasoning. If the completed permitted edit cannot fit, follow "Fitting the token limit" in DATASET-AUTHORING.md: regenerate when the reasoning does not demonstrate understanding or the prompt asked too much; otherwise cut the response from the end. In the reasoning remove only fluff; any other reasoning cut is presumed harmful and needs a span-specific justification that the reviewer adjudicates.

After the complete candidate passes correctness, rendering, and likelihood validation, an adversarial subagent will compare the edited reasoning with the untouched original and objectively assess compliance with these instructions. Its ruling is binding. If it identifies a locally repairable deviation, revise only the identified spans, rerun validation and scoring, and resubmit the exact revised reasoning. Regenerate only when a compliant local correction is impossible.

This iteration will be considered a failure if the subagent detects that you deviated from your instructions. It is a demonstration of your good judgement when you make a deliberate decision to regenerate based on an objective assessment of the output. However, it is a demonstration of your poor judgement when regeneration is required because you deviated from your instructions. Please follow the instructions.
