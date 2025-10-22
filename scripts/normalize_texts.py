import argparse
import os
import re
import json
from typing import Tuple

# --- Tail / structure cleanup ---
DROP_SECTION_RE = re.compile(
    r"(?im)^(?:##|###|####)\s*(?:См\.?\s*также|Примечания|Литература|Ссылки|Внешние\s+ссылки|Источники)\b.*\Z",
    re.DOTALL,
)
REFNUM_RE = re.compile(r"\s*\[\d{1,3}\]")
WS_RE = re.compile(r"[ \t]+")
BLANKS_RE = re.compile(r"\n{3,}")
PUNCT_SPACE = re.compile(r"\s+([,.;:!?])")
EDITBOX_RE = re.compile(r"\[\s*править\s*\|\s*править\s*код\s*\]", re.IGNORECASE)

# --- Cosmetic-only cleanup ---
# Lines like "↑ ..." (optionally with a bullet before)
UP_ARROW_LINE_RE = re.compile(r"(?m)^\s*(?:[-–—]\s*)?↑.*(?:\n|$)")
# Service brackets like [источник не указан ...], [Архив ...], [http...]
BRACKET_SERVICE_RE = re.compile(r"\[\s*(?:источник не указан|архив[^]]*|https?://[^\] ]+[^]]*)\s*\]", re.IGNORECASE)
# Tighten quotes and parentheses spacing
OPEN_QUOTE_SPACE_RE = re.compile(r"«\s+")
CLOSE_QUOTE_SPACE_RE = re.compile(r"\s+»")
OPEN_PAREN_SPACE_RE = re.compile(r"\(\s+")
CLOSE_PAREN_SPACE_RE = re.compile(r"\s+\)")
CLOSE_PUNCT_SPACE_RE = re.compile(r"\s+([\)\]\»])")

# --- English gloss removal ---
# (1) Parenthesized glosses: "(англ. Hogsmeade)" -> remove entirely
EN_GLOSS_PAREN_RE = re.compile(r"\(\s*англ\.\s*[^)]*\)", re.IGNORECASE)
# (2) Inline glosses without parentheses: ", англ. Hogsmeade," or " — англ. Elder Wand — "
# remove the 'англ. <...>' chunk; surrounding punctuation/spacing will be compacted later
EN_GLOSS_INLINE_RE = re.compile(r"(?:(?<=\s)|^)(англ\.\s*[^,;\.\)\]\n]+)", re.IGNORECASE)

AGGREGATE_TITLES = {
    "Локации_мира_Гарри_Поттера",
    "Волшебные_предметы_мира_Гарри_Поттера",
    "Волшебные_существа_мира_Гарри_Поттера",
}


def is_aggregate_filename(name: str) -> bool:
    """Return True for filenames known to represent aggregator-style articles."""
    base = os.path.splitext(name)[0]
    if base.startswith("Список_"):
        return True
    if base in AGGREGATE_TITLES:
        return True
    return False


def strip_empty_headings(text: str) -> str:
    """Drop headings that are followed only by blank lines."""
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"^(##|###|####)\s+\S", line):
            j = i + 1
            buf = []
            while j < len(lines) and not re.match(r"^(##|###|####)\s+\S", lines[j]):
                buf.append(lines[j])
                j += 1
            if any(x.strip() for x in buf):
                out.append(line)
                out.extend(buf)
            i = j
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def normalize_text(text: str) -> str:
    """Apply typography fixes, strip references and compact whitespace."""
    # typographic fixes
    text = text.replace("\u00a0", " ")  # NBSP -> space
    text = text.replace("➤", "")  # nav arrows
    text = EDITBOX_RE.sub("", text)  # [править | править код]
    text = PUNCT_SPACE.sub(r"\1", text)  # tighten spaces before punctuation

    # content cleanup
    text = REFNUM_RE.sub("", text)  # drop [1] style refs

    # remove parenthesized English glosses before cutting tails, so leftover "()"
    # won't be preserved by later spacing rules
    text = EN_GLOSS_PAREN_RE.sub("", text)
    # remove inline 'англ. ...' fragments (non-parenthesized)
    text = EN_GLOSS_INLINE_RE.sub("", text)

    # cut tail: "См. также/Ссылки/Источники/..."
    m = DROP_SECTION_RE.search(text)
    if m:
        text = text[: m.start()].rstrip()

    # cosmetic cleanup
    text = UP_ARROW_LINE_RE.sub("", text)
    text = BRACKET_SERVICE_RE.sub("", text)
    text = OPEN_QUOTE_SPACE_RE.sub("«", text)
    text = CLOSE_QUOTE_SPACE_RE.sub("»", text)
    text = OPEN_PAREN_SPACE_RE.sub("(", text)
    text = CLOSE_PAREN_SPACE_RE.sub(")", text)
    text = CLOSE_PUNCT_SPACE_RE.sub(r"\1", text)

    # whitespace compaction
    text = WS_RE.sub(" ", text)
    text = BLANKS_RE.sub("\n\n", text)
    text = text.strip()
    text = strip_empty_headings(text)
    if text and not text.endswith("\n"):
        text += "\n"
    return text


def looks_like_aggregator_text(text: str) -> bool:
    """Heuristic: detect list-heavy texts that look like indexes or portals."""
    headings = len(re.findall(r"^##+\s", text, flags=re.M))
    bullets = len(re.findall(r"^-\s", text, flags=re.M))
    paras = len([p for p in re.split(r"\n\s*\n", text) if p.strip() and not p.strip().startswith("- ")])
    return (headings == 0 and bullets >= 120) or (paras < 2 and bullets >= 80)


def process_file(in_path: str) -> Tuple[str, str]:
    """Normalize a single file and return (text, status)."""
    try:
        with open(in_path, "r", encoding="utf-8", errors="ignore") as source:
            raw = source.read()
    except OSError:
        return "", "io_error"
    clean = normalize_text(raw)
    if looks_like_aggregator_text(clean):
        return "", "aggregator_like"
    if len(clean.split()) < 80:
        return "", "too_short"
    return clean, "ok"


def main():
    """Normalize wiki texts in bulk and emit a JSON summary."""
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    stats = {
        "normalized": 0,
        "skipped_aggregate_title": 0,
        "skipped_aggregate_like": 0,
        "too_short": 0,
        "io_error": 0,
    }
    for name in os.listdir(args.in_dir):
        if not name.endswith(".txt") or name.startswith("_"):
            continue
        if is_aggregate_filename(name):
            stats["skipped_aggregate_title"] += 1
            continue
        in_path = os.path.join(args.in_dir, name)
        clean, status = process_file(in_path)
        if status == "ok":
            out_path = os.path.join(args.out_dir, name)
            with open(out_path, "w", encoding="utf-8") as dest:
                dest.write(clean)
            stats["normalized"] += 1
        elif status == "aggregator_like":
            stats["skipped_aggregate_like"] += 1
        elif status == "too_short":
            stats["too_short"] += 1
        else:
            stats["io_error"] += 1

    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
