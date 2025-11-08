#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 [major|minor|patch]" >&2
  exit 1
fi

PART="$1"
if [[ "$PART" != "major" && "$PART" != "minor" && "$PART" != "patch" ]]; then
  echo "PART must be one of: major, minor, patch" >&2
  exit 1
fi

PYPROJECT="pyproject.toml"
if [[ ! -f "$PYPROJECT" ]]; then
  echo "Cannot find $PYPROJECT" >&2
  exit 1
fi

python - "$PYPROJECT" "$PART" <<'PY'
import pathlib
import sys

pyproject = pathlib.Path(sys.argv[1])
part = sys.argv[2]

text = pyproject.read_text()
prefix = "version = \""
start = text.index(prefix) + len(prefix)
end = text.index("\"", start)
version = text[start:end]

major, minor, patch = map(int, version.split("."))
if part == "major":
    major += 1
    minor = 0
    patch = 0
elif part == "minor":
    minor += 1
    patch = 0
else:
    patch += 1

new_version = f"{major}.{minor}.{patch}"
new_text = text[:start] + new_version + text[end:]
pyproject.write_text(new_text)
print(f"Bumped version: {version} -> {new_version}")
PY
