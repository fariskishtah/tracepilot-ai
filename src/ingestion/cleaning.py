"""Text cleaning: whitespace, hyphenation, repeated page headers/footers."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter


def clean_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", t).replace("\x00", " ")
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)          # de-hyphenate line breaks
    t = re.sub(r"[ \t\f\v]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def strip_repeated_lines(pages: list[str], min_frac: float = 0.5) -> list[str]:
    """Remove header/footer lines that repeat on >= min_frac of pages (page numbers included)."""
    if len(pages) < 3:
        return pages
    norm = lambda l: re.sub(r"\d+", "#", l.strip())
    counts = Counter(norm(l) for p in pages for l in set(p.splitlines()) if l.strip())
    bad = {k for k, v in counts.items() if v / len(pages) >= min_frac}
    return ["\n".join(l for l in p.splitlines() if norm(l) not in bad) for p in pages]
