import asyncio
import aiohttp
import argparse
import async_timeout
import json
import os
import re
import logging
from urllib.parse import urlparse, unquote
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("fetch_hp_ru")

API = "https://ru.wikipedia.org/w/api.php"
OK_HOSTS = {"ru.wikipedia.org", "ru.m.wikipedia.org"}

AGGREGATORS = {
    "Список персонажей серии романов о Гарри Поттере",
    "Локации мира Гарри Поттера",
    "Волшебные существа мира Гарри Поттера",
    "Волшебные предметы мира Гарри Поттера",
    "Хогвартс",
}

HP_CATEGORY_KEYWORDS = [
    "гарри поттер",
    "мир гарри поттера",
    "персонажи цикла «гарри поттер»",
    "хогвартс",
    "пожиратели смерти",
    "орден феникса",
    "волшебные предметы мира гарри поттера",
    "локации мира гарри поттера",
]
HP_MARKERS = [
    "гарри поттер",
    "хогвартс",
    "дамблдор",
    "волан-де-морт",
    "волдеморт",
    "северус снейп",
    "северус снегг",
    "рон уизли",
    "гермиона грейнджер",
    "хогсмид",
    "азкабан",
    "пожиратели смерти",
    "орден феникса",
    "квиддич",
    "крестраж",
    "дементор",
    "слизерин",
    "гриффиндор",
    "когтевран",
    "пуффендуй",
    "платформа девять и три четверти",
]
MIN_HP_MARKERS = 2

CUTOFF_HEADINGS = {"см. также", "примечания", "литература", "ссылки", "внешние ссылки", "источники", "галерея"}

ALIASES = {
    "больница святого мунго": ["больница святого мунго", "больница святого мунго для магических болезней и травм"],
}


def make_headers(ua):
    """Build HTTP headers for Wikipedia requests."""
    return {"User-Agent": ua, "Accept-Language": "ru"}


def seed_to_title(seed_url: str) -> str:
    """Extract and decode the Wikipedia page title from a seed URL."""
    p = urlparse(seed_url.strip())
    if p.netloc not in OK_HOSTS or not p.path.startswith("/wiki/"):
        raise ValueError(f"Not a ru.wikipedia wiki URL: {seed_url}")
    title = p.path[len("/wiki/") :].split("#", 1)[0]
    return unquote(title)


def slugify(title: str) -> str:
    """Create a filesystem-safe slug from a Wikipedia title."""
    name = title.replace("/", "／")
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^\w\-\u0400-\u04FF_]+", "", name)
    return name


def normkey(s: str) -> str:
    """Normalize strings so they can be compared independent of punctuation and case."""
    s = s.replace("_", " ").strip().lower()
    s = s.replace("ё", "е")
    s = s.replace("—", "-").replace("–", "-").replace("−", "-")
    s = re.sub(r"[^\w\s\-]", " ", s, flags=re.U)
    s = re.sub(r"\s+", " ", s)
    return s


async def api(session, **params):
    """Call the Wikipedia API and decode the JSON payload."""
    async with session.get(API, params=params) as r:
        txt = await r.text()
        if r.status != 200:
            raise aiohttp.ClientResponseError(r.request_info, r.history, status=r.status, message=txt[:200])
        try:
            return json.loads(txt)
        except Exception:
            raise RuntimeError(f"Non-JSON response for params={params}: {txt[:200]}")


async def resolve_page(session, title: str):
    """Resolve a title to a main namespace page, returning basic metadata or None."""
    params = dict(
        action="query", format="json", formatversion=2, redirects=1, titles=title, prop="info|pageprops", inprop="url"
    )
    data = await api(session, **params)
    pages = (data.get("query") or {}).get("pages") or []
    if not pages:
        return None
    page = pages[0]
    if page.get("missing") or page.get("invalid"):
        return None
    if page.get("ns") != 0:
        return None
    return {"pageid": page.get("pageid"), "title": page.get("title")}


async def fetch_sections(session, page_title: str):
    """Fetch section metadata so specific subsections can be extracted."""
    data = await api(
        session, action="parse", page=page_title, prop="sections", format="json", formatversion=2, redirects=1
    )
    parse = data.get("parse") or {}
    return parse.get("sections") or []


def pick_section(sections, desired_title: str):
    """Return the best matching section dict for the requested title."""
    want = normkey(desired_title)
    candidates_of_want = [want]
    for alt in ALIASES.get(want, []):
        candidates_of_want.append(normkey(alt))

    def s_norm(s):
        return normkey(s.get("line", ""))

    def s_level(s):
        try:
            return int(s.get("level") or s.get("toclevel") or 2)
        except Exception:
            return 2

    exact = [s for s in sections if s_norm(s) in candidates_of_want]
    if exact:
        return max(exact, key=s_level)

    sw = [s for s in sections if any(s_norm(s).startswith(c) for c in candidates_of_want)]
    if sw:
        return max(sw, key=s_level)
    ct = [s for s in sections if any(c in s_norm(s) for c in candidates_of_want)]
    if ct:
        return max(ct, key=s_level)

    want_tokens = [t for t in want.split() if len(t) >= 3]
    best = None
    best_score = 0
    best_level = 0
    for s in sections:
        line = s_norm(s)
        line_tokens = [t for t in line.split() if len(t) >= 3]
        score = len(set(want_tokens) & set(line_tokens))
        if score > best_score or (score == best_score and s_level(s) > best_level):
            best, best_score, best_level = s, score, s_level(s)
    if best_score >= 2:
        return best

    return None


def strip_toc_and_chrome(soup: BeautifulSoup):
    """Remove navigation, infoboxes and other Wikipedia-specific chrome."""
    for sel in [
        "#toc",
        "div.toc",
        "table.infobox",
        "table.navbox",
        "table.metadata",
        "table.vertical-navbox",
        "div.hatnote",
        "div.navbox",
        "div.sidebar",
        "div.reflist",
        "div.thumb",
        "aside",
        "sup.reference",
    ]:
        for tag in soup.select(sel):
            tag.decompose()
    for tag in soup.find_all("table"):
        tag.decompose()


def html_to_text(html: str):
    """Convert article HTML into plain text with lightweight heading markup."""
    soup = BeautifulSoup(html, "lxml")
    strip_toc_and_chrome(soup)

    def norm(t: str) -> str:
        return re.sub(r"[ \t]+", " ", t).strip()

    lines = []
    paras = 0
    lists = 0

    # lead
    lead = []
    for p in soup.find_all("p", recursive=True):
        prev = p.find_previous(["h2", "h3", "h4"])
        if prev:
            continue
        txt = norm(p.get_text(" "))
        if txt:
            lead.append(txt)
            paras += 1
    if lead:
        lines.append("\n".join(lead).strip())

    current_heading = None
    current_block = []

    def flush_block():
        nonlocal current_heading, current_block, lines
        if current_heading is not None:
            body = "\n".join([b for b in current_block if b.strip()]).strip()
            if body:
                lines.append(current_heading)
                lines.append(body)
        current_heading, current_block = None, []

    for tag in soup.find_all(["h2", "h3", "h4", "p", "ul", "ol"], recursive=True):
        if tag.name in ("h2", "h3", "h4"):
            htxt = norm(tag.get_text(" "))
            hclean = re.sub(r"\[.*?\]", "", htxt).strip().lower().replace("—", "-")
            if hclean in CUTOFF_HEADINGS:
                break
            flush_block()
            level = {"h2": "##", "h3": "###", "h4": "####"}[tag.name]
            current_heading = f"{level} {htxt.strip()}"
        elif tag.name == "p":
            txt = norm(tag.get_text(" "))
            if txt:
                current_block.append(txt)
                paras += 1
        elif tag.name in ("ul", "ol"):
            items = [norm(li.get_text(" ")) for li in tag.find_all("li", recursive=False)]
            items = [f"- {it}" for it in items if it]
            if items:
                current_block.append("\n".join(items))
                lists += len(items)
    flush_block()

    text = "\n\n".join([ln for ln in lines if ln.strip()])
    text = re.sub(r"\s*\[\d{1,3}\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    stats = {"paras": paras, "list_items": lists, "chars": len(text)}
    return text, stats


def looks_like_aggregator(stats) -> bool:
    """Heuristic to detect list-only aggregator pages."""
    return stats.get("paras", 0) < 1 and stats.get("list_items", 0) > 100


async def fetch_html(session, title: str, section: int | None = None):
    """Fetch raw HTML (whole page or specific section) for a Wikipedia article."""
    params = dict(action="parse", page=title, prop="text", format="json", formatversion=2, redirects=1)
    if section is not None:
        params["section"] = section
    data = await api(session, **params)
    parse = data.get("parse") or {}
    html = parse.get("text") or ""
    if isinstance(html, dict):
        html = html.get("*", "")
    if not html:
        raise RuntimeError("Empty HTML from parse API")
    return html


async def fetch_one(session, seed_url, out_dir):
    """Process a single seed URL and persist the cleaned text when it matches criteria."""
    seed_title = None
    try:
        seed_title = seed_to_title(seed_url)
    except Exception as e:
        return {"seed": seed_url, "status": f"bad_seed:{e}"}

    page = await resolve_page(session, seed_title)
    if not page:
        return {"seed": seed_url, "status": "resolve_failed"}
    canonical = page["title"]
    pageid = page["pageid"]

    slice_needed = canonical in AGGREGATORS and (canonical != seed_title)
    section_index = None
    if slice_needed:
        try:
            sections = await fetch_sections(session, canonical)
            sec = pick_section(sections, seed_title)
            if sec:
                try:
                    section_index = int(sec.get("index"))
                except Exception:
                    section_index = None
        except Exception as e:
            log.exception("Failed to locate section for %s", seed_title)

    try:
        async with async_timeout.timeout(40):
            html = await fetch_html(session, canonical, section=section_index)
    except Exception as e:
        return {"seed": seed_url, "canonical": canonical, "pageid": pageid, "status": f"fetch_html_error:{e}"}

    text, stats = html_to_text(html)
    if looks_like_aggregator(stats) and section_index is None:
        return {"seed": seed_url, "canonical": canonical, "pageid": pageid, "status": "aggregator_like"}

    if len(text.split()) < 80 or stats["chars"] < 400:
        return {"seed": seed_url, "canonical": canonical, "pageid": pageid, "status": "too_short"}

    # Category/marker relevance (categories only for whole page; for sliced rely on markers)
    cat_ok = False
    if section_index is None:
        try:
            data = await api(
                session,
                action="query",
                format="json",
                formatversion=2,
                redirects=1,
                prop="categories",
                cllimit=500,
                titles=canonical,
            )
            pages = (data.get("query") or {}).get("pages") or []
            cats = [c.get("title", "") for c in (pages[0].get("categories") or [])] if pages else []
            s = " || ".join(cats).lower()
            cat_ok = any(kw in s for kw in HP_CATEGORY_KEYWORDS)
        except Exception:
            cat_ok = False

    # Markers fallback
    if not cat_ok:
        low = text.lower()
        if sum(1 for m in HP_MARKERS if m in low) < MIN_HP_MARKERS:
            return {"seed": seed_url, "canonical": canonical, "pageid": pageid, "status": "category_miss"}

    out_name = slugify(seed_title if section_index is not None else canonical) + ".txt"
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    res = {"seed": seed_url, "canonical": canonical, "pageid": pageid, "status": "ok", "path": out_path, "stats": stats}
    if section_index is not None:
        res["sectioned"] = True
        res["section_index"] = section_index
    return res


async def run(seed_path, out_dir, concurrency, user_agent):
    """Entry point for fetching texts for multiple seeds."""
    try:
        with open(seed_path, "r", encoding="utf-8") as f:
            seeds = [s.strip() for s in f.read().splitlines() if s.strip()]
    except OSError as e:
        log.error("Unable to read seeds from %s: %s", seed_path, e)
        return 1
    os.makedirs(out_dir, exist_ok=True)

    timeout = aiohttp.ClientTimeout(total=60)
    headers = make_headers(user_agent)
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0, ssl=False)
    sem = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession(headers=headers, timeout=timeout, connector=connector, trust_env=True) as session:

        async def bounded(u):
            async with sem:
                try:
                    return await fetch_one(session, u, out_dir)
                except Exception as e:
                    return {"seed": u, "status": f"fatal:{e}"}

        tasks = [asyncio.create_task(bounded(u)) for u in seeds]
        results = []
        for fut in asyncio.as_completed(tasks):
            results.append(await fut)

    with open(os.path.join(out_dir, "_fetch_report.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    ok = len([r for r in results if r.get("status") == "ok"])
    log.info("ok=%d total=%d", ok, len(results))
    return 0 if ok > 0 else 1


def main():
    """Parse CLI arguments and launch the fetch pipeline."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--ua", type=str, default=None)
    args = ap.parse_args()
    ua = args.ua or os.environ.get("WIKI_UA") or "HP-RU-Corpus/3.1 (+contact: your-email@example.com)"
    return asyncio.run(run(args.seed, args.out, args.concurrency, ua))


if __name__ == "__main__":
    raise SystemExit(main())
