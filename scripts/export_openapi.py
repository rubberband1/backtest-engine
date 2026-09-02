"""Exports the OpenAPI schema for TypeScript type generation.

The frontend types are not written by hand: they are generated from here, so
a field renamed in the API becomes a compile error in the UI instead of an
`undefined` at runtime.

    python -m scripts.export_openapi
    cd ui && npm run gen:api
"""
from __future__ import annotations

import json
from pathlib import Path

from api.main import app

TARGET = Path("ui/src/api/openapi.json")


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    schema = app.openapi()
    TARGET.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{TARGET}: {len(schema['paths'])} paths, "
          f"{len(schema['components']['schemas'])} models")


if __name__ == "__main__":
    main()
