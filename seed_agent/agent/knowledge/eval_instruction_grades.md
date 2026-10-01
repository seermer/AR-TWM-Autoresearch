---
name: eval_instruction_grades
description: Use when you need to know what the judge checks for event_edit_adherence, subject_action_adherence, perspective_switch_adherence and causal_fidelity.
---

# What the instruction and causal grades ask

A vision-language judge answers questions about frames of the generated video. The video is split evenly into its turns, and each turn is judged on its own frames with that turn's instruction text.

- `event_edit_adherence`, per event turn, frames at 3 per second, five yes/no questions, score = correct answers / 5:
  1. Does the scene stay largely unchanged, with no sign of the event? (expected: no)
  2. Does the video show something resembling the event, even partially? (yes)
  3. By the end of the turn, has the event reached a clear conclusion or outcome? Still ongoing or only starting counts as no. (yes)
  4. Are the key details right: objects, agents, directions, quantities? (yes)
  5. Does an unrelated object, entity or visual anomaly appear? (no)
- `subject_action_adherence`, per subject-action turn, the same five about the subject: idle with no attempt (no); performing something resembling the action (yes); the action reached a clear conclusion by the end of the turn (yes); the right subject, objects and manner (yes); unnatural movement, impossible pose or implausible interaction (no).
- `perspective_switch_adherence`, per switch turn, pass (1) or fail (0). The judge sees 3 frames from the start and 3 from the end of the turn. All three must hold:
  1. The view changes between start and end: to the other perspective type, or, when the type stays the same, to a clearly different camera position, followed subject or angle.
  2. The end frames show the target type: first-person (through the character's eyes; at most hands or a held tool visible), third-person (an external camera showing the character's body) or scoped (a magnified view through an optic).
  3. The end view is a valid one of its type. Third-person: the character roughly centred, the camera behind it, upper or full body visible. First-person: the character's own face, back or full body not visible. Scoped: a scope ring, reticle, magnification or overlay visible.
- `causal_fidelity`, for cases annotated for it, frames at 3 per second over the whole video. The judge rates 0 to 3 the physics and cause-and-effect of objects and characters overall (clipping, gravity, teleporting, objects appearing or vanishing, effects without a cause; an instructed action and its consequences are expected, and camera motion is not judged), and 0 to 3 each physics aspect annotated for the case. The score is the mean of the overall rating and the average aspect rating, divided by 3.
- A case's grade is the mean over its turns of that kind.
