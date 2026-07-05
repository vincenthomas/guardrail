# Golden set protocol

**Task:** structured extraction from customer-support emails into
`{intent, urgency, product, requested_action}`.

**v1: 40 cases**, authored and labeled by hand for this repo (fictional
products, no real customer data). Authored rather than scraped because no
suitable public support-email corpus with field-level labels exists, and the
eval's value is in label quality, not corpus size. 40 is small; it is sized
to the labeling budget of one build week and is a stated limitation, not a
hidden one.

**Design of the cases:** roughly half are straightforward; the rest each
encode one deliberate difficulty — mixed intents, urgency implied but not
stated, misspelled product names, multiple products where one is primary,
anger without urgency, politeness masking urgency, no explicit ask.

**Leak policy:** prompts are iterated against `dev_cases.jsonl` (10 separate
cases) only. `cases.jsonl` is scored, never read while editing prompts. Any
prompt change that follows an error analysis of golden-set outputs requires
new golden cases for the error pattern studied (they leak by definition once
studied).

**Versioning:** the set is append-only; cases are never edited to make a
config pass. Version bumps in this file; results always cite the version.

Current version: **v1** (40 cases + 10 dev cases).
