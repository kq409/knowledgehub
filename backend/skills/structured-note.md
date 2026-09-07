---
name: structured-note
description: The app's note field schema for organising loose text into a research note.
---

# Structured note fields

When organising text into a research note (via `preview_note_extraction` or when advising the researcher), use this exact shape:

- `title` — short descriptive title
- `summary` — 1–3 sentence summary of the idea
- `observations` — factual observations from the speaker
- `hypotheses` — guesses or proposed explanations
- `questions` — open questions to investigate
- `next_steps` — concrete follow-up actions
- `tags` — short topic tags

Rules:

- Use empty arrays when a field has no content.
- Do not invent facts that are not in the source text.
- Keep items concise.
- `preview_note_extraction` saves nothing; the researcher saves from the Notes tab.
