---
name: training_text
description: Use when writing a clip's annotation: what text the model is trained with.
---

# What text the model is trained with

- A clip is trained turn by turn. A turn is one stretch of the clip with one text.
- The kernel writes each turn's text from the clip's annotation: the scene (`scene_prompt`), then the subject (`character_prompt`, when the clip has one), then the turn's `event`, when it has one. On a turn whose perspective differs from the turn before, the kernel adds a sentence that the view switches; on every third-person turn it adds a sentence that the subject stays centered and moves naturally with the camera. Do not write either sentence yourself.
- Nothing is carried over from earlier turns: a turn's text describes that turn only.
- The text describes what is on screen during the frames of the turn; it is not an instruction relative to earlier text.
- Write the scene and the subject as prose: what is there. Write an event as one sentence: what happens during that turn.
- Do not describe the camera or how it moves: the camera is learned from the clip's poses.
- An event belongs to the turn in which it happens on screen. Set that turn's `start_s` to when the event starts in the clip (locate_events finds it). A turn runs until the next turn starts; the last runs to the end of the clip.
- A turn's text holds for the whole turn. Start the turn when what its text describes first shows on screen, not before, and start the next turn only after it is complete.
- If what a turn describes first shows less than 2.4 s into the clip, the first turn's text would claim it before it shows: trim the clip's start to that moment, or leave the clip out.
- A turn has at most one event. When two events overlap in time, decide which turn each belongs to.
- The first turn lasts at least 2.4 s and every later turn at least 1.4 s.
