from __future__ import annotations

import re
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
TOC_DIR = BASE_DIR / "data" / "tocs"


def get_toc_path_for_pdf(pdf_path: str | Path, toc_path: str | Path | None = None) -> Path:
    """
    Resolve the TOC file for a PDF.

    If `toc_path` is provided, use it directly.
    Otherwise, look for a file in `data/tocs/` with the same stem as the PDF.
    Example:
    - PDF: data/documents/linear_algebra.pdf
    - TOC: data/tocs/linear_algebra.txt
    """
    if toc_path is not None:
        resolved = Path(toc_path)
        if not resolved.is_absolute():
            resolved = (Path.cwd() / resolved).resolve()
        return resolved

    pdf = Path(pdf_path)
    return TOC_DIR / f"{pdf.stem}.txt"


def load_toc_text(pdf_path: str | Path, toc_path: str | Path | None = None) -> str:
    resolved_toc_path = get_toc_path_for_pdf(pdf_path, toc_path)
    if not resolved_toc_path.exists() or not resolved_toc_path.is_file():
        raise FileNotFoundError(f"TOC file not found: {resolved_toc_path}")
    return resolved_toc_path.read_text(encoding="utf-8").strip()


def parse_toc(toc_text: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    for line in toc_text.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("contents"):
            continue

        # Expected format: `2.2.1 Matrix Product 21`
        match = re.match(r"^([\d\.]+)\s+(.+?)\s+(\d+)$", line)
        if not match:
            continue

        number = match.group(1)
        title = match.group(2).strip()
        page = int(match.group(3))

        if "." not in number:
            level = "chapter"
        elif number.count(".") == 1:
            level = "section"
        else:
            level = "subsection"

        entries.append(
            {
                "level": level,
                "title": title,
                "page": page,
                "index": page - 1,
            }
        )

    return entries


def build_page_metadata(
    toc_entries: list[dict[str, Any]],
    total_pages: int,
) -> list[dict[str, Any]]:
    # Walk page-by-page and keep track of the most recent chapter/section/subsection
    # so every extracted page inherits the right semantic labels.
    page_meta: list[dict[str, Any]] = []
    current_chapter = None
    current_section = None
    current_subsection = None

    toc_entries = sorted(toc_entries, key=lambda x: x["index"])
    pointer = 0

    for i in range(total_pages):
        while pointer < len(toc_entries) and toc_entries[pointer]["index"] <= i:
            entry = toc_entries[pointer]
            if entry["level"] == "chapter":
                current_chapter = entry["title"]
                current_section = None
                current_subsection = None
            elif entry["level"] == "section":
                current_section = entry["title"]
                current_subsection = None
            else:
                current_subsection = entry["title"]
            pointer += 1

        page_meta.append(
            {
                "page": i,
                "chapter": current_chapter,
                "section": current_section,
                "subsection": current_subsection,
            }
        )

    return page_meta
