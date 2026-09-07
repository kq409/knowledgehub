---
name: compare-papers
description: When and how to use compare_papers, including default dimensions.
---

# Compare papers

Use `compare_papers` when the question is about how 2–4 specific papers relate: how they differ, what they agree on, which one to build on.

Default dimensions (use unless the researcher asks for others):

- problem
- method
- dataset
- evaluation
- key_results
- strengths
- limitations

Rules:

- It costs one model call per paper, so you get one comparison per turn. Spend it on the papers that matter.
- Prefer `search_library` / `read_paper` when the question is about a single paper, needs more than four papers, or you still need to work out which papers are relevant. Narrow the field first, then compare.
- After `compare_papers` the Compare workspace opens with the table. Answer their actual question from it. Do not read the table back row by row.
- Check researcher memory for preferred dimensions before choosing your own defaults.
