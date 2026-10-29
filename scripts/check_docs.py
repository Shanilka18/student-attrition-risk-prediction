"""Check artifact-backed documentation, local links, and exported figures."""

from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path
from urllib.parse import unquote

from build_docs_numbers import (
    DOCS,
    FINAL_FIGURES,
    NUMBER_FILE,
    SOURCE_FIGURES,
    SUMMARY_FILES,
    active_run,
    as_rows,
    json_safe,
    read_json,
    report_stats,
    results_markdown,
)

PACK_FILES = (
    "00_README_DOCS.md",
    "18_END_TO_END_WALKTHROUGH.md",
    "RESULTS_SUMMARY.md",
    "19_RESULTS_SUMMARY.md",
    "20_FIGURE_CATALOG.md",
    "21_CONCEPTS_FOR_BEGINNERS.md",
    "22_VIVA_QA.md",
    "23_PRESENTATION_GUIDE.md",
    "24_LIMITATIONS_ETHICS.md",
    "25_PROPOSAL_UPDATE_NOTES.md",
    "26_REPRODUCIBILITY_AND_RUN_GUIDE.md",
    "27_GLOSSARY.md",
)
LINK_ONLY_FILES = ("DECISIONS.md", "phase_reports/phase_17.md")
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
FIGURE_PNG_LINK = re.compile(r"\[([^]]+\.png)\]\(([^)]+)\)")
FORBIDDEN_PLACEHOLDER = re.compile(r"\b(?:TODO|lorem)\b", re.IGNORECASE)
WEBSITE_CONTEXT = re.compile(
    r"\b(?:website|web\s+site|frontend|front-end)\b", re.IGNORECASE
)
FORBIDDEN_WEBSITE_WORDS = re.compile(
    r"\b(?:purple|indigo|violet|gradients?|glow|glassmorphism|emojis?|sparkles?|"
    r"robot\s+icons?|prototype|demo|beta|mock|placeholder|AI-powered|smart|magic)\b",
    re.IGNORECASE,
)
FORBIDDEN_WEBSITE_CONTENT = re.compile(
    r"\b(?:viva|presentation|thesis|proposal)\b", re.IGNORECASE
)
NEGATION = re.compile(
    r"\b(?:not|no|never|without|does not|do not|must not)\b", re.IGNORECASE
)


def markdown_target(link: str) -> str:
    target = link.strip().split()[0].strip("<>")
    return unquote(target)


def check_links(path: Path, text: str, errors: list[str]) -> int:
    targets = MARKDOWN_LINK.findall(text) + MARKDOWN_IMAGE.findall(text)
    checked = 0
    for raw_target in targets:
        target = markdown_target(raw_target)
        if target.startswith(("#", "http://", "https://", "mailto:", "data:")):
            continue
        local_path = target.split("#", 1)[0]
        if not local_path:
            continue
        resolved = (path.parent / Path(local_path)).resolve()
        if not resolved.exists():
            errors.append(f"{path.relative_to(DOCS)}: broken local link: {target}")
        checked += 1
    return checked


def check_png_dpi(path: Path) -> bool:
    with path.open("rb") as handle:
        if handle.read(8) != b"\x89PNG\r\n\x1a\n":
            return False
        while True:
            length_bytes = handle.read(4)
            if not length_bytes:
                return False
            length = struct.unpack(">I", length_bytes)[0]
            kind = handle.read(4)
            payload = handle.read(length)
            handle.read(4)
            if kind == b"pHYs":
                x_ppm, y_ppm, unit = struct.unpack(">IIB", payload)
                return x_ppm == 11811 and y_ppm == 11811 and unit == 1
            if kind == b"IEND":
                return False


def main() -> int:
    errors: list[str] = []
    documents: dict[Path, str] = {}
    links_checked = 0

    for relative in PACK_FILES:
        path = DOCS / relative
        if not path.is_file():
            errors.append(f"Missing documentation file: {relative}")
            continue
        text = path.read_text(encoding="utf-8")
        documents[path] = text
        lines = text.splitlines()
        if len(lines) < 5 or any(not line.startswith("> ") for line in lines[:5]):
            errors.append(
                f"{relative}: the opening five-line summary is missing or malformed"
            )
        headings = [
            match.start()
            for match in re.finditer(
                r"^## (?:Common mistakes|If the examiner asks)\s*$", text, re.M
            )
        ]
        common_position = text.find("## Common mistakes")
        examiner_position = text.find("## If the examiner asks")
        has_later_section = bool(
            examiner_position >= 0
            and re.search(
                r"^## ",
                text[examiner_position + len("## If the examiner asks") :],
                re.M,
            )
        )
        if (
            common_position < 0
            or examiner_position < 0
            or common_position > examiner_position
            or has_later_section
            or len(headings) != 2
        ):
            errors.append(f"{relative}: required closing sections are missing")
        if FORBIDDEN_PLACEHOLDER.search(text):
            errors.append(f"{relative}: contains a forbidden TODO/lorem placeholder")
        for line_number, line in enumerate(lines, start=1):
            if WEBSITE_CONTEXT.search(line) and FORBIDDEN_WEBSITE_WORDS.search(line):
                errors.append(
                    f"{relative}:{line_number}: forbidden website description term"
                )
            content_term = FORBIDDEN_WEBSITE_CONTENT.search(line)
            if (
                WEBSITE_CONTEXT.search(line)
                and content_term
                and not NEGATION.search(line[: content_term.start()])
            ):
                errors.append(
                    f"{relative}:{line_number}: prohibited research-document content is "
                    "described as website content"
                )
        links_checked += check_links(path, text, errors)

    for relative in LINK_ONLY_FILES:
        path = DOCS / relative
        if path.is_file():
            links_checked += check_links(path, path.read_text(encoding="utf-8"), errors)

    for summary_path in SUMMARY_FILES:
        text = documents.get(summary_path)
        if text is None:
            continue
        numbers = read_json(NUMBER_FILE)
        if text != results_markdown(numbers):
            errors.append(
                f"{summary_path.name}: content differs from the artifact-backed renderer"
            )

    if NUMBER_FILE.is_file():
        run_id, run_dir = active_run()
        if (
            run_id
            != json.loads(NUMBER_FILE.read_text(encoding="utf-8"))["active_run_id"]
        ):
            errors.append("_numbers.json does not identify the active run")
        fresh = json_safe(
            report_stats(
                run_id,
                run_dir,
                as_rows(read_json(run_dir / "leaderboard_classification.json")),
                as_rows(read_json(run_dir / "leaderboard_regression.json")),
            )
        )
        source_figure_count = len(list(SOURCE_FIGURES.glob("*.png")))
        fresh["figure_count"] = source_figure_count
        saved = read_json(NUMBER_FILE)
        if saved != fresh:
            errors.append("_numbers.json differs from the active run artifacts")
    else:
        errors.append("Missing docs/_numbers.json")

    catalog_path = DOCS / "20_FIGURE_CATALOG.md"
    catalog = documents.get(catalog_path, "")
    catalog_figures = FIGURE_PNG_LINK.findall(catalog)
    source_figure_names = {path.name for path in SOURCE_FIGURES.glob("*.png")}
    catalog_figure_names = {name for name, _ in catalog_figures}
    if catalog_figure_names != source_figure_names:
        errors.append("Figure catalog entries do not match the source PNG inventory")
    for filename, target in catalog_figures:
        if not (DOCS / Path(markdown_target(target))).is_file():
            errors.append(f"Figure catalog references a missing figure: {filename}")
    for figure_path in sorted(FINAL_FIGURES.glob("*.png")):
        if not check_png_dpi(figure_path):
            errors.append(f"Figure is not tagged at 300 dpi: {figure_path.name}")
        if not (FINAL_FIGURES / f"{figure_path.stem}.svg").is_file():
            errors.append(f"Figure has no matching SVG export: {figure_path.name}")

    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print(
        f"PASS: {len(documents)} documentation files, {links_checked} local links, "
        f"{len(catalog_figures)} figure entries, and artifact-backed results."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
