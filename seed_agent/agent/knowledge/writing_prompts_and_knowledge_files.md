---
name: writing_prompts_and_knowledge_files
description: Use when an edit adds or changes a role prompt (agent/prompts) or a knowledge file (agent/knowledge).
---

# Writing prompts and knowledge files

- A fact that is always true goes in a knowledge file; a behaviour rule goes in a prompt.
- A prompt rule states a general behaviour; the evidence for it goes in the plan's rationale.
- A knowledge file covers one topic and starts with front matter: its `name` and a `description` of when it is needed. Roles see only the description until they read the file.
