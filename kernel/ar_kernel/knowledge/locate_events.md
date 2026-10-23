---
name: locate_events
tool: locate_events
description: Use when a clip's events need times: what locate_events takes and gives.
---

# What locate_events gives

- Give each event as the sentence you would write in the annotation: short and simple. Do not add how clearly or how close it is seen. The model is asked about each event on its own: whether it is happening or has already happened with its result visible.
- A span is where that holds: `start_s` is when the event starts, `end_s` when it last holds. An event whose result stays visible has a span that runs to the end of the clip.
- Both are null when the event was not found. Leave such an event out of the annotation, or look at the clip yourself.
- `disagreeing_s` counts the seconds of the clip that contradict the span. A count that is large for the clip's length means the model was unsure: look at the clip before you rely on the span.
- The tool returns spans only; you write the annotation. A turn runs until the next turn starts, and the last runs to the end of the clip.
- The first turn lasts at least 2.4 s and every later turn at least 1.4 s. Where to start each turn so its text holds: knowledge file `training_text`.
- A turn has at most one event. When two spans overlap, decide which turn each event belongs to.
