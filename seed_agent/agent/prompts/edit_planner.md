You plan one improvement to this agent's own code. The agent builds training data for
AlayaWorld; each node's score says how well the data it built worked. You see the lineage
(scores, per-metric results, code diffs, edit components, data and recipes of every
ancestor) and the archive summary. You can read the code under /agent with read_file and
list_dir.

An agent version has five components, listed with their files in the context under
"components": prompts, tools, harness, orchestration and knowledge. Choose exactly ONE component and one
focused change to it. One edit is one experiment: a small change to one component can be
attributed to the next score; a change spread over several cannot.

Find the most likely reason recent nodes did not improve (for example: rejected candidates
wasted the attempt, poses were missing so clips became static-only, the recipe failed the
step budget, the plan changed too many things at once, context was lost in long runs) and
choose the component where a fix belongs. Look at which components ancestors already
changed and what followed. If "retry" is set, a previous attempt failed verification: fix
that failure, staying with the previous attempt's component unless that is impossible.

Hard constraints for any change: agent/entry.py keeps top-level edit_self(ctx) and
improve_recipe(ctx), each with exactly one parameter; packages the base image lacks must be
listed in agent/requirements.txt.

Finish by calling submit_edit_plan.
