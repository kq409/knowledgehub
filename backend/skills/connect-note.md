---
name: connect-note
description: When and how to link a researcher's note to papers in the library.
---

# Connect a note to literature

Use `link_note` / `unlink_note` when the researcher wants a note attached to one or more papers. These tools only write the link. They do not change note text or paper text.

How to decide:

- If the researcher names a paper, `list_papers` or `search_library` to get its id, then `link_note`.
- If several papers could fit, ask which one. Do not guess.
- A note may link to more than one paper. Call `link_note` once per paper.
- `unlink_note` removes one pair. The note and the paper stay in the library.

Rules:

- Notes are the researcher's commentary, hypotheses, and questions. Never present a linked note as something the paper claims.
- Search and `read_paper` may pull linked notes or paper chunks along the link. Those extra snippets are still the same source types: paper text vs researcher notes.
- Do not invent links for "probably related" papers. If you are not sure, say so and wait.
