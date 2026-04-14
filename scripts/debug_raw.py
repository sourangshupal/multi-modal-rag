"""Debug script — prints the raw SDK response so we can fix the parser."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from glmocr import GlmOcr  # type: ignore[import]

from doc_parser.config import get_settings

if len(sys.argv) < 2:
    print("Usage: python scripts/debug_raw.py <path-to-pdf>")
    sys.exit(1)

settings = get_settings()
parser = GlmOcr(
    config_path=settings.config_yaml_path,
    api_key=settings.z_ai_api_key.get_secret_value(),
)

file_path = sys.argv[1]
print(f"Parsing: {file_path}\n")
raw = parser.parse(file_path)

print("=== TYPE ===")
print(type(raw))

print("\n=== DIR (non-dunder attributes) ===")
attrs = [a for a in dir(raw) if not a.startswith("_")]
print(attrs)

print("\n=== REPR ===")
print(repr(raw)[:2000])

# Try to serialize as dict/JSON
print("\n=== AS DICT (if possible) ===")
if hasattr(raw, "__dict__"):
    try:
        print(json.dumps(raw.__dict__, indent=2, default=str)[:3000])
    except Exception as e:
        print(f"Cannot JSON-serialize __dict__: {e}")
        print(raw.__dict__)
elif isinstance(raw, dict):
    print(json.dumps(raw, indent=2, default=str)[:3000])
else:
    print("Not a dict-like object")

# Try common attribute names
print("\n=== PROBING COMMON KEYS ===")
for key in ["pages", "results", "data", "content", "items", "blocks", "layout", "text", "markdown"]:
    val = getattr(raw, key, "<<NOT FOUND>>")
    if val != "<<NOT FOUND>>":
        print(f"  raw.{key} = {repr(val)[:200]}")
