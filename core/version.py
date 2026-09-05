"""Single source of truth for the engine version.

Semantic versioning, from the point of view of run results:

- MAJOR: results change for the same spec + config + data (execution rules,
  cost accounting, fill logic). Requires a golden-reference update with a
  changelog note.
- MINOR: new capabilities or new metrics; existing results are unchanged.
- PATCH: fixes with no observable change in any output.

Every bump gets an entry in ENGINE_CHANGELOG.md.
"""

ENGINE_VERSION = "4.2.0"
