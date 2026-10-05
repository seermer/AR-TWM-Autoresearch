---
name: node_files
description: Use when looking into what earlier nodes did: their transcripts, workspaces, command logs, training logs and configs under /nodes.
---

# What earlier nodes left behind: /nodes

`/nodes/<node>/` holds, read-only, what each finished node of the run produced. The root node only has an evaluation, so its directory is empty.

- `recipe.yaml`: the tunable values the node trained with.
- `rationale.md`: the recipe rationale and what the node's data role recorded with it.
- `edit.json`: the summary of the node's self-edit.
- `transcripts/<phase>-<attempt>/NN-<role>.md`: every conversation of every role, in order, with reasoning, tool calls and tool results. A file is named after the role's submit tool. A compacted conversation continues in the next file.
- `transcripts/<phase>-<attempt>/NN-<role>.json`: the same conversation as a JSON list of messages (`role`, `content`, `reasoning`, `tool_calls`, `tool_call_id`). Count turns, calls and errors from this file with a script: in the `.md`, a tool result that printed another transcript looks like more turns.
- `attempts/edit_self-<k>/agent/`: the agent code as that attempt left it.
- `attempts/<phase>-<k>/workspace/`: the files the roles wrote, including `tool_output/` (the full output of commands) and `plans.json` (every plan of the attempt, with the reports that went back to the planner).
- `attempts/improve_recipe-<k>/train/train.log`: the full training log. Each `[Train] step=` line is one optimizer step with the dataset `source`, `sigma`, `loss`, `grad` and `lr`; its `time=` covers only the last micro-batch of the step. The loss depends mostly on `sigma`, so compare losses at similar sigma.
- `attempts/improve_recipe-<k>/train_config.yaml`: the full training config that ran.
- `attempts/improve_recipe-<k>/view/<dataset>/`: the data commit as the trainer saw it: captions and poses per clip.
- `attempts/<phase>-<k>/context/context.json`: the complete context that attempt received.
- `contract/attempt-<k>/`: the kernel's checks of the edited code.

Useful ways in:
- `ls /nodes`, then `ls -R /nodes/<parent> | head -200` for the shape; read the parent first.
- `grep -l "Error" /nodes/*/transcripts/*/*.md` finds tool failures; `grep -c "tool call: run_command"` shows how much was done by hand.
- To follow the loss, extract `sigma` and `loss` from train.log with a short Python script and group by sigma.
