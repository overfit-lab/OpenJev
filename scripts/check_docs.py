#!/usr/bin/env python3
"""Check repository Markdown links, fenced examples, and SVG syntax.

Uses only Python's standard library. Remote URLs and fragment anchors are
not fetched or validated. This is a documentation check, not a model test.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    "data", "datasets", "checkpoints", "models", "outputs", "runs",
    "wandb", "artifacts", "build", "dist",
}
MARKDOWN_LINK = re.compile(r"!?\[[^\]\n]*\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)")
HTML_LINK = re.compile(r"\b(?:href|src)\s*=\s*['\"]([^'\"]+)['\"]", re.I)
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def check_target(path: Path, line: int, target: str, errors: list[str]) -> None:
    target = html.unescape(target.strip("<>"))
    try:
        parsed = urlsplit(target)
    except ValueError:
        errors.append(f"{path.relative_to(ROOT)}:{line}: invalid link {target!r}")
        return
    if parsed.scheme or parsed.netloc or not parsed.path:
        return
    decoded = unquote(parsed.path)
    candidate = ROOT / decoded.lstrip("/") if decoded.startswith("/") else path.parent / decoded
    resolved = candidate.resolve()
    if not resolved.is_relative_to(ROOT):
        errors.append(f"{path.relative_to(ROOT)}:{line}: link leaves repository: {target}")
    elif not resolved.exists():
        errors.append(f"{path.relative_to(ROOT)}:{line}: missing local target: {target}")


def check_markdown(path: Path, errors: list[str]) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    fence_char = ""
    fence_size = 0
    fence_start = 0
    language = ""
    body: list[str] = []
    json_count = 0
    for number, line in enumerate(lines, 1):
        match = FENCE.match(line)
        if fence_char:
            if (match and match[1][0] == fence_char
                    and len(match[1]) >= fence_size and not match[2].strip()):
                if language == "json":
                    try:
                        json.loads("\n".join(body))
                        json_count += 1
                    except json.JSONDecodeError as exc:
                        errors.append(
                            f"{path.relative_to(ROOT)}:{fence_start}: invalid JSON: {exc}"
                        )
                fence_char = ""
                body = []
            else:
                body.append(line)
            continue
        if match:
            fence_char = match[1][0]
            fence_size = len(match[1])
            fence_start = number
            language = match[2].strip().lower()
            continue
        # Inline code can contain example paths or Markdown syntax.
        prose = re.sub(r"`+[^`]*`+", "", line)
        for target in MARKDOWN_LINK.findall(prose) + HTML_LINK.findall(prose):
            check_target(path, number, target, errors)
    if fence_char:
        errors.append(f"{path.relative_to(ROOT)}:{fence_start}: unclosed code fence")
    return json_count


def main() -> int:
    errors: list[str] = []
    files: list[Path] = []
    for directory, subdirectories, filenames in os.walk(ROOT):
        subdirectories[:] = sorted(
            name for name in subdirectories if name not in SKIP_DIRS
        )
        files.extend(Path(directory) / name for name in filenames)
    markdown = sorted(path for path in files if path.suffix == ".md")
    svg_files = sorted(path for path in files if path.suffix == ".svg")
    examples = 0
    for path in markdown:
        try:
            examples += check_markdown(path, errors)
        except (OSError, UnicodeError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    for path in svg_files:
        try:
            root = ET.parse(path).getroot()
            if root.tag != "{http://www.w3.org/2000/svg}svg":
                errors.append(f"{path.relative_to(ROOT)}: missing SVG root namespace")
        except (OSError, ET.ParseError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    if errors:
        print("Documentation checks failed:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    print(
        f"Documentation checks passed: {len(markdown)} Markdown files, "
        f"{examples} JSON examples, {len(svg_files)} SVG assets."
    )
    print("Local inline links checked; remote URLs and fragment anchors were not checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
