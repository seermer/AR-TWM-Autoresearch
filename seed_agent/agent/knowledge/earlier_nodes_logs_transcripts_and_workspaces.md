---
name: earlier_nodes_logs_transcripts_and_workspaces
description: Use when looking into what earlier nodes did: their transcripts, workspaces, command logs, training logs and configs under /nodes.
---

# What earlier nodes left behind: /nodes

`/nodes/<node>/` holds, read-only, everything each finished node of the run produced, except its evaluation outputs. The context's `lineage` has the scores of the parent and its ancestors, `siblings` those of the parent's other children, and `archive` one line for every node. The root node only has a score, so its directory is empty.

- `recipe.yaml`: the tunable values the node trained with.
- `rationale.md`: the recipe rationale, the data plan and the data notes.
- `edit.json`: the summary of the node's self-edit.
- `transcripts/<phase>-<attempt>/NN-<role>.md`: every conversation of every role, in order, with reasoning, tool calls and tool results. A file is named after the role's submit tool (edit_plan and edit in edit_self, plan and data_and_recipe in improve_recipe). A compacted conversation continues in the next file.
- `attempts/edit_self-<k>/agent/`: the agent code as that attempt left it. `workspace/plans.json` holds every plan of the attempt, with the coder's report when it asked for a new one.
- `attempts/improve_recipe-<k>/workspace/`: the data engineer's files: scripts, `tool_output/run_command-*.log` (full output of every command), `plans.json` (every plan, with the engineer's report when it asked for a new one), `result.json`.
- `attempts/improve_recipe-<k>/train/train.log`: the full training log. Each `[Train] step=` line is one optimizer step with the dataset `source`, `sigma`, `loss`, `grad` and `lr`; its `time=` covers only the last micro-batch of the step. The loss depends mostly on `sigma`, so compare losses at similar sigma.
- `attempts/improve_recipe-<k>/train_config.yaml`: the full training config that ran.
- `attempts/improve_recipe-<k>/view/<dataset>/`: the data commit as the trainer saw it: captions and poses per clip.
- `attempts/<phase>-<k>/context/context.json`: the complete context that attempt received, never truncated.
- `contract/attempt-<k>/`: the kernel's checks of the edited code.

Useful ways in:
- `ls /nodes`, then `ls -R /nodes/<parent> | head -200` for the shape; read the parent first.
- `grep -l "Error" /nodes/*/transcripts/*/*.md` finds tool failures; `grep -c "tool call: run_command"` shows how much was done by hand.
- To follow the loss, extract `sigma` and `loss` from train.log with a short Python script and group by sigma.
- The same holds for the current attempt: `/context/context.json` is the full, untruncated version of the `<context>` block.
