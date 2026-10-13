---
name: training_text
description: Use when writing a clip's annotation: what text the model is trained with.
---

# What text the model is trained with

- A clip is trained turn by turn. A turn is one stretch of the clip with one text.
- The kernel writes each turn's text from the clip's annotation: the scene (`scene_prompt`), then the subject (`character_prompt`, when the clip has one), then the turn's `event`, when it has one.
- Nothing is carried over from earlier turns: a turn's text describes that turn only. A turn without an event is trained with the scene and the subject alone.
- A turn whose `perspective` differs from the turn before it also gets a sentence that says the view switches.
- The text is the model's whole text condition for the frames of the turn. It describes what is on screen during those frames; it is not an instruction relative to earlier text.
- Write the scene and the subject as prose: what is there. Write an event as one sentence: what happens during that turn.
- The text says nothing about the camera or how it moves: the camera is trained from the clip's poses.
- An event belongs to the turn in which it happens on screen. Set that turn's `start_s` to when the event starts in the clip (locate_events finds it). A turn runs until the next turn starts; the last runs to the end of the clip.
- A turn has at most one event. When two events overlap in time, decide which turn each belongs to, or leave one out.
- The first turn lasts at least 2.4 s and every later turn at least 1.4 s.
