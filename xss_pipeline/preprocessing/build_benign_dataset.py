"""Build the reproducible 4,000-sample benign dataset for Project T02.

Workflow:
1. Load ``oracle_results_2.csv``, filter ``valid == 1``, and deduplicate via
   :func:`preprocessing.decoder.deduplicate_malicious_df` (preferring ``Tainted``
   ahead of ``Obfuscated``).
2. Measure the decoded length distribution (min, p10, p25, p50, p75, p90, p95,
   p99, max) of the unique malicious payloads.
3. Generate exactly 4,000 unique benign samples (2,000 ``Clean`` across 40
   independent families + 2,000 ``Unusual`` across 40 independent families,
   50 samples per family) whose ``payload_decoded`` length distribution closely
   tracks the malicious distribution.
4. Embed harmless ``https://attacker.test/...`` and ``.test``/``.example`` URLs
   across 8 ``Unusual`` families (360 samples = 18.0% of ``Unusual``) to counter
   domain-based shortcut learning.
5. Verify zero internal duplicates and zero decoded overlap with malicious rows,
   then write ``benign_dataset_4000.csv``.

Run from ``xss_pipeline/``:
    ./venv/bin/python -m preprocessing.build_benign_dataset
"""

from __future__ import annotations

import argparse
import math
import random
import re
from pathlib import Path
from typing import Callable

import pandas as pd

from preprocessing.decoder import decode_payload, deduplicate_malicious_df

SEED = 42
SAMPLES_PER_FAMILY = 50
NUM_CLEAN_FAMILIES = 40
NUM_UNUSUAL_FAMILIES = 40
TARGET_CLEAN_COUNT = NUM_CLEAN_FAMILIES * SAMPLES_PER_FAMILY      # 2000
TARGET_UNUSUAL_COUNT = NUM_UNUSUAL_FAMILIES * SAMPLES_PER_FAMILY  # 2000
TOTAL_BENIGN_COUNT = TARGET_CLEAN_COUNT + TARGET_UNUSUAL_COUNT    # 4000

# 8 Unusual families * 45 attacker.test samples = 360 samples (18.0% of Unusual).
ATTACKER_TEST_PER_FAMILY = 45

# 8 high-frequency XSS keywords that must be spread across >= 3 Unusual families
# so group-based splitting never drops any keyword completely from train.
KEY_SPREAD_KEYWORDS: tuple[str, ...] = (
    "fetch",
    "script",
    "alert",
    "document.cookie",
    "onerror",
    "javascript:",
    "onfocus",
    "onload",
)

PIPELINE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ORACLE_CSV = PIPELINE_DIR / "oracle_results_2.csv"
DEFAULT_OUTPUT_CSV = PIPELINE_DIR / "benign_dataset_4000.csv"

BENIGN_COLUMNS = [
    "id",
    "payload",
    "payload_decoded",
    "label",
    "group",
    "context",
    "family",
    "obfuscation",
    "source",
]

# Guard pattern to keep Clean samples free of XSS-associated tokens.
_CLEAN_FORBIDDEN_RE = re.compile(
    r"(?:attacker\.test|javascript:|data:|document\.cookie|window\.location|"
    r"\b(?:alert|confirm|prompt|onerror|onload|onclick|onfocus|onmouseover|"
    r"ontoggle|script|iframe|svg|img|fetch)\b|[<>])",
    re.IGNORECASE,
)

# Guard against accidentally producing an executable XSS-like payload in Unusual.
_UNUSUAL_FORBIDDEN_RE = re.compile(
    r"(?:<\s*script\b|<\s*(?:img|svg|iframe|body|video|audio|input|details)\b[^>]*\bon[a-z]+\s*=|"
    r"^\s*javascript:|href\s*=\s*['\"]?\s*javascript:|src\s*=\s*['\"]?\s*javascript:)",
    re.IGNORECASE,
)

_PANDAS_NA_STRINGS = {
    "", "#n/a", "#n/a n/a", "#na", "-1.#ind", "-1.#qnan", "-nan",
    "1.#ind", "1.#qnan", "<na>", "n/a", "na", "null", "nan", "none",
}

# ============================================================================
# Shared lexicons (rich, realistic English & technical vocabularies)
# ============================================================================

FIRST_NAMES = [
    "Alice", "Benjamin", "Clara", "Daniel", "Elena", "Marcus", "Hannah",
    "David", "Priya", "Lucas", "Nadia", "Samuel", "Olivia", "Ethan",
    "Rachel", "Victor", "Maya", "Owen", "Sophia", "Liam", "Tessa",
    "Noah", "Grace", "Julian", "Chloe", "Adrian", "Kieran", "Mei",
    "Carlos", "Freya", "Isaac", "Amina", "Leo", "Naomi",
]

LAST_NAMES = [
    "Carter", "Jenkins", "Patel", "O'Connor", "Nguyen", "Brooks", "Kim",
    "Rivera", "Mitchell", "Bennett", "Sato", "Holloway", "Alvarez", "Rossi",
    "Kowalski", "Foster", "Chen", "Wallace", "Morales", "Park", "Sullivan",
    "Reeves", "Tanaka", "Bauer", "Hayes", "Fletcher", "Silva", "Grant",
    "Keller", "Walsh", "Sharma", "Vance",
]

CITIES = [
    "Seattle, WA", "Austin, TX", "Boston, MA", "Denver, CO", "Chicago, IL",
    "Portland, OR", "Toronto, ON", "Dublin, IE", "London, UK", "Berlin, DE",
    "Zurich, CH", "Singapore, SG", "Sydney, AU", "Vancouver, BC",
    "San Diego, CA", "Raleigh, NC", "Madison, WI", "Atlanta, GA",
]

STREETS = [
    "Maple Ave", "Oakwood Dr", "Cedar Lane", "Pinecrest Blvd", "Harbor Rd",
    "Summit Way", "Riverview St", "Elmhurst Ct", "Beacon Hill Rd",
    "Parkside Ter", "Willow Creek Dr", "Sycamore Ave", "Meadowbrook Ln",
    "Lakeview Blvd", "Highland St", "Birchwood Pl",
]

COMPANIES = [
    "Northwind", "Bluepeak", "Cedarline", "Brightleaf", "Harborview",
    "Clearwater", "Sunridge", "Silveroak", "Redstone", "Cloudbridge",
    "Evergreen", "Maplecrest", "Ironwood", "Veridian", "Summitworks",
    "Oakridge", "Crestline", "Starlight",
]

COMPANY_SUFFIXES = ["Labs", "Logistics", "Health", "Systems", "Retail", "Studio", "Foods", "Partners"]

CLEAN_DOMAINS = [
    "northwind.io", "bluepeak.org", "cedarline.com", "brightleaf.co",
    "harborview.net", "clearwater.org", "sunridge.com", "silveroak.io",
    "evergreen-shop.com", "maplecrest.org", "cloudbridge.dev", "veridian.co",
]

PRODUCTS = [
    "wireless keyboard", "insulated mug", "desk lamp", "linen backpack",
    "coffee grinder", "running shoes", "ceramic teapot", "usb-c hub",
    "standing mat", "bamboo cutting board", "travel umbrella", "wool blanket",
    "bluetooth speaker", "ergonomic mouse", "water bottle", "reading light",
    "cast iron skillet", "canvas tote", "notebook set", "charging dock",
]

DEPARTMENTS = [
    "Operations", "Finance", "Design", "Support", "Supply Chain",
    "People Ops", "Research", "Analytics", "Marketing", "Product",
    "Facilities", "Customer Success", "Legal", "Quality",
]

ROLES = [
    "Product Manager", "Operations Lead", "Data Analyst", "UX Designer",
    "Account Coordinator", "Finance Specialist", "Project Planner",
    "Support Lead", "Editorial Assistant", "Research Associate",
    "Logistics Coordinator", "Quality Reviewer",
]

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TIMEZONES = ["UTC", "EST", "PST", "CET", "GMT", "CST"]

COURSES = [
    "Linear Algebra", "Macroeconomics", "Organic Chemistry", "World History",
    "Technical Writing", "Statistical Inference", "Environmental Policy",
    "Microbiology", "Urban Planning", "Art Conservation", "Cognitive Science",
    "Supply Chain Strategy", "Applied Ethics", "Digital Photography",
]

FOODS = [
    "sourdough bread", "roasted carrots", "lemon pasta", "chickpea curry",
    "oatmeal cookies", "mushroom risotto", "tomato soup", "blueberry muffins",
    "grilled salmon", "basil pesto", "apple crumble", "lentil stew",
    "garlic focaccia", "ginger tea", "spinach quiche", "pumpkin bread",
]

SPORTS_TEAMS = [
    "Harbor Hawks", "Metro Lions", "Valley Wolves", "North Stars",
    "Coastal Rovers", "Summit Bears", "River City FC", "Pinecrest United",
    "Lakeside Athletics", "Cedar Bay Owls",
]

BOOKS = [
    "The Quiet Harbor", "Fields of Glass", "Northern Bridges", "Letters from Kyoto",
    "The Cedar Atlas", "Winter on the Ridge", "Clockwork Seasons", "Salt and Timber",
    "Maps of Tomorrow", "The Stone Orchard", "Voices in the Rain", "Lantern Street",
]

PETS = [
    "golden retriever", "tabby cat", "beagle puppy", "corgi", "calico cat",
    "border collie", "cockatiel", "holland lop rabbit", "siamese cat", "terrier mix",
]

PLANTS = [
    "monstera deliciosa", "snake plant", "boston fern", "rosemary", "lavender",
    "peace lily", "pothos vine", "cherry tomato", "basil", "jade plant",
    "spider plant", "hydrangea", "dwarf lemon tree", "thyme",
]

VEHICLES = [
    "2023 Honda Civic", "2022 Toyota Corolla", "2024 Subaru Outback",
    "2021 Mazda CX-5", "2023 Hyundai Ioniq 5", "2022 Ford Maverick",
    "2024 Volvo XC40", "2021 Kia Niro", "2023 Trek Domane AL", "2022 Giant Escape",
]

# Technical lexicons for Unusual families
SERVICES = [
    "auth-api", "billing-worker", "edge-proxy", "search-idx", "media-cdn",
    "orders-svc", "notify-bus", "metrics-gw", "session-store", "catalog-api",
    "webhook-relay", "cache-node", "audit-sink", "scheduler-01",
]

TEST_PATHS = [
    "/healthz", "/api/v1/echo", "/api/v2/status", "/sandbox/ping",
    "/mock/callback", "/fixtures/sample.json", "/v1/metrics", "/qa/probe",
    "/hooks/delivery", "/static/spec.yaml", "/ci/check", "/test/redirect",
    "/docs/example", "/v2/events", "/internal/ready", "/stubs/profile",
]

ALT_TEST_DOMAINS = [
    "sandbox.test", "fixture.test", "qa-target.test", "ci-mock.test",
    "staging.example", "mock-api.example", "demo-app.example", "docs.example",
]

HTTP_METHODS = ["GET", "POST", "HEAD", "PUT", "DELETE", "OPTIONS"]
HTTP_CODES = [200, 201, 204, 301, 304, 400, 404, 429, 502, 503]
ENV_NAMES = ["staging", "ci", "sandbox", "local", "qa", "dev", "preview"]
LOG_LEVELS = ["INFO", "WARN", "DEBUG", "NOTICE", "ERROR"]
FRAMEWORKS = ["React 19", "Vue 3", "Svelte 5", "Angular 18", "Next.js 15", "Django 5", "FastAPI", "Flask 3"]
BROWSERS_LIST = ["Chrome 130", "Firefox 132", "Safari 18", "Edge 130"]


# ============================================================================
# Helpers
# ============================================================================

def compute_length_percentiles(lengths: list[int]) -> dict[str, float]:
    """Compute exact linear-interpolated percentiles for a non-empty length list."""
    if not lengths:
        raise ValueError("Cannot compute percentiles on empty sequence.")
    arr = sorted(lengths)
    n = len(arr)

    def _pct(p: float) -> float:
        idx = (n - 1) * (p / 100.0)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        return arr[lo] + (arr[hi] - arr[lo]) * (idx - lo)

    gt_200 = sum(1 for x in arr if x > 200)
    return {
        "count": float(n),
        "min": float(arr[0]),
        "p10": _pct(10),
        "p25": _pct(25),
        "p50": _pct(50),
        "p75": _pct(75),
        "p90": _pct(90),
        "p95": _pct(95),
        "p99": _pct(99),
        "max": float(arr[-1]),
        "mean": sum(arr) / n,
        "gt200_count": float(gt_200),
        "gt200_pct": 100.0 * gt_200 / n,
    }


def load_dedup_malicious(oracle_csv: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read oracle_results_2.csv, filter valid == 1, and deduplicate via decoder."""
    raw_df = pd.read_csv(oracle_csv, keep_default_na=False)
    valid_mask = pd.to_numeric(raw_df["valid"], errors="coerce") == 1
    valid_df = raw_df.loc[valid_mask].copy().reset_index(drop=True)
    dedup_df, dedup_stats = deduplicate_malicious_df(valid_df)
    dedup_stats["raw_total"] = int(len(raw_df))
    dedup_stats["valid_count"] = int(len(valid_df))
    return dedup_df, dedup_stats


def resample_target_lengths(malicious_lengths: list[int], count: int) -> list[int]:
    """Deterministically resample ``count`` target lengths matching the empirical quantiles."""
    arr = sorted(malicious_lengths)
    m = len(arr)
    if m == 0:
        raise ValueError("malicious_lengths must not be empty.")
    if count <= 1:
        return [arr[m // 2]]
    return [arr[int(round(i * (m - 1) / (count - 1)))] for i in range(count)]


def _rand_date(rng: random.Random) -> str:
    yr = rng.choice([2024, 2025, 2026])
    mo = rng.randint(1, 12)
    da = rng.randint(1, 28)
    return f"{yr:04d}-{mo:02d}-{da:02d}"


def _rand_time(rng: random.Random) -> str:
    return f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}"


def _rand_hex(rng: random.Random, n: int = 6) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


def _rand_ip(rng: random.Random) -> str:
    return f"10.{rng.randint(0, 250)}.{rng.randint(1, 250)}.{rng.randint(2, 254)}"


def _rand_person(rng: random.Random) -> tuple[str, str]:
    return rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)


def _generate_pool(
    rng: random.Random,
    templates: list[Callable[[random.Random], str]],
    pool_size: int = 900,
) -> list[str]:
    """Generate a deduplicated candidate pool evenly distributed across templates."""
    out: list[str] = []
    seen_raw: set[str] = set()
    attempts = 0
    max_attempts = pool_size * 12
    t_count = len(templates)
    while len(out) < pool_size and attempts < max_attempts:
        fn = templates[attempts % t_count]
        attempts += 1
        cand = " ".join(fn(rng).split()).strip()
        if not cand or cand in seen_raw:
            continue
        seen_raw.add(cand)
        out.append(cand)
    return out


# ============================================================================
# 40 CLEAN FAMILY BUILDERS (everyday English & normal web input)
# ============================================================================

def _build_clean_families() -> dict[str, Callable[[random.Random], list[str]]]:
    def fam(templates: list[Callable[[random.Random], str]]) -> Callable[[random.Random], list[str]]:
        return lambda rng: _generate_pool(rng, templates, pool_size=900)

    def _p(rng: random.Random) -> str:
        f, l = _rand_person(rng)
        return f"{f} {l}"

    def _comp(rng: random.Random) -> str:
        return f"{rng.choice(COMPANIES)} {rng.choice(COMPANY_SUFFIXES)}"

    return {
        # 1
        "clean_person_name": fam([
            lambda r: _p(r),
            lambda r: f"Dr. {_p(r)}" if r.random() < 0.5 else f"{_p(r)} Jr.",
            lambda r: f"{_p(r)} - {r.choice(ROLES)}",
            lambda r: f"Contact: {_p(r)} ({r.choice(CITIES)})",
            lambda r: f"{_p(r)}, {r.choice(ROLES)} at {_comp(r)}, {r.choice(CITIES)}",
            lambda r: f"Directory entry: {_p(r)}, {r.choice(DEPARTMENTS)} team in {r.choice(CITIES)}, floor {r.randint(2, 12)}, ext {r.randint(1000, 9999)}",
            lambda r: f"Staff member {_p(r)} joined the {r.choice(DEPARTMENTS)} group at {_comp(r)} in {r.choice(MONTHS)} {r.randint(2019, 2026)} ({r.choice(CITIES)})",
        ]),
        # 2
        "clean_email_contact": fam([
            lambda r: f"{r.choice(FIRST_NAMES).lower()}.{r.choice(LAST_NAMES).lower().replace(chr(39), '')}@{r.choice(CLEAN_DOMAINS)}",
            lambda r: f"{r.choice(['billing', 'help', 'sales', 'hello', 'orders', 'team'])}@{r.choice(CLEAN_DOMAINS)}",
            lambda r: f"From: {_p(r)} ({r.choice(FIRST_NAMES).lower()}{r.randint(10, 99)}@{r.choice(CLEAN_DOMAINS)})",
            lambda r: f"Reply-To: support@{r.choice(CLEAN_DOMAINS)} regarding ticket #{r.randint(10000, 99999)}",
            lambda r: f"Email {r.choice(FIRST_NAMES).lower()}.{r.choice(LAST_NAMES).lower().replace(chr(39), '')}@{r.choice(CLEAN_DOMAINS)} for updated invoice copies and account questions",
            lambda r: f"Forwarded message from {_p(r)} ({r.choice(FIRST_NAMES).lower()}@{r.choice(CLEAN_DOMAINS)}) about the {r.choice(MONTHS)} schedule review in {r.choice(CITIES)}",
        ]),
        # 3
        "clean_street_address": fam([
            lambda r: f"{r.randint(12, 998)} {r.choice(STREETS)}",
            lambda r: f"{r.randint(100, 9800)} {r.choice(STREETS)}, Apt {r.randint(1, 45)}{r.choice('ABCD')}",
            lambda r: f"{r.randint(100, 8900)} {r.choice(STREETS)}, {r.choice(CITIES)} {r.randint(10001, 99500)}",
            lambda r: f"Suite {r.randint(100, 950)}, {r.randint(10, 999)} {r.choice(STREETS)}, {r.choice(CITIES)} {r.randint(10001, 98999)}",
            lambda r: f"Deliver to {_p(r)}, {r.randint(100, 7999)} {r.choice(STREETS)}, Unit {r.randint(2, 40)}, {r.choice(CITIES)} {r.randint(10001, 98999)}",
            lambda r: f"Billing address: {_comp(r)}, Attn: {r.choice(DEPARTMENTS)}, {r.randint(100, 4500)} {r.choice(STREETS)}, {r.choice(CITIES)} {r.randint(10001, 98999)}",
        ]),
        # 4
        "clean_search_query": fam([
            lambda r: f"best {r.choice(PRODUCTS)}",
            lambda r: f"{r.choice(PRODUCTS)} under ${r.randint(25, 190)}",
            lambda r: f"how to clean a {r.choice(PRODUCTS)} at home",
            lambda r: f"top rated {r.choice(PRODUCTS)} for small apartments {r.randint(2024, 2026)}",
            lambda r: f"comparison of {r.choice(PRODUCTS)} brands with two year warranty and free returns",
            lambda r: f"step by step beginner guide to choosing a durable {r.choice(PRODUCTS)} for daily commute and weekend travel in rainy weather",
        ]),
        # 5
        "clean_short_comment": fam([
            lambda r: f"Thanks {_p(r).split()[0]}!",
            lambda r: f"Loved this {r.choice(FOODS)} idea!",
            lambda r: f"Really helpful overview, worked smoothly on my first try.",
            lambda r: f"Appreciate the clear summary from the {r.choice(DEPARTMENTS).lower()} team on this update.",
            lambda r: f"Thanks for sharing the notes from {r.choice(WEEKDAYS)}'s workshop in {r.choice(CITIES)}, super clear and practical.",
            lambda r: f"We tried this approach last {r.choice(MONTHS)} with our small team of {r.randint(4, 15)} people and it saved us at least three hours every week.",
        ]),
        # 6
        "clean_product_review": fam([
            lambda r: f"5/5 great {r.choice(PRODUCTS)}",
            lambda r: f"4 stars - sturdy {r.choice(PRODUCTS)}, fast delivery",
            lambda r: f"Bought this {r.choice(PRODUCTS)} last {r.choice(MONTHS)} and it still works like new every morning.",
            lambda r: f"Solid build quality on the {r.choice(PRODUCTS)}, though the box arrived slightly dented on one corner.",
            lambda r: f"Five stars for this {r.choice(PRODUCTS)}: easy setup, quiet operation, compact footprint, and helpful customer support when I had a sizing question.",
            lambda r: f"After using the {r.choice(PRODUCTS)} daily for {r.randint(2, 11)} months, I can honestly recommend it for anyone wanting reliable quality without overspending.",
        ]),
        # 7
        "clean_url_path": fam([
            lambda r: f"https://{r.choice(CLEAN_DOMAINS)}/help",
            lambda r: f"https://{r.choice(CLEAN_DOMAINS)}/shop/{r.choice(PRODUCTS).replace(' ', '-')}",
            lambda r: f"https://www.{r.choice(CLEAN_DOMAINS)}/blog/{r.randint(2024, 2026)}/{r.choice(MONTHS).lower()}-guide",
            lambda r: f"https://docs.{r.choice(CLEAN_DOMAINS)}/manuals/{r.choice(PRODUCTS).replace(' ', '-')}-v{r.randint(1, 4)}?lang=en",
            lambda r: f"https://store.{r.choice(CLEAN_DOMAINS)}/catalog/items/{r.randint(1000, 9999)}?color={r.choice(['navy', 'sage', 'slate', 'sand'])}&size={r.choice(['S', 'M', 'L'])}&page={r.randint(1, 5)}",
            lambda r: f"https://support.{r.choice(CLEAN_DOMAINS)}/articles/{r.randint(10000, 99999)}/returns-and-warranty-coverage-for-online-orders?region={r.choice(['us', 'eu', 'uk', 'ca'])}",
        ]),
        # 8
        "clean_file_path": fam([
            lambda r: f"docs/notes_{r.randint(1, 99)}.txt",
            lambda r: f"/home/user/Documents/report_{_rand_date(r)}.pdf",
            lambda r: f"C:\\Users\\{r.choice(FIRST_NAMES)}\\Downloads\\invoice_{r.randint(1000, 9999)}.xlsx",
            lambda r: f"/shared/{r.choice(DEPARTMENTS).lower().replace(' ', '_')}/Q{r.randint(1, 4)}_{r.randint(2024, 2026)}/budget_summary_v{r.randint(1, 9)}.csv",
            lambda r: f"/mnt/archive/projects/{r.choice(COMPANIES).lower()}/{r.randint(2024, 2026)}/{r.choice(MONTHS).lower()}/quarterly_presentation_final_rev{r.randint(2, 8)}.pptx",
            lambda r: f"Users/{r.choice(FIRST_NAMES).lower()}/Photos/{r.randint(2023, 2026)}/{r.choice(CITIES).split(',')[0].lower().replace(' ', '_')}_trip/album_{r.randint(10, 99)}/photo_{r.randint(100, 999)}.jpg",
        ]),
        # 9
        "clean_datetime_entry": fam([
            lambda r: _rand_date(r),
            lambda r: f"{_rand_date(r)} {_rand_time(r)} {r.choice(TIMEZONES)}",
            lambda r: f"{r.choice(WEEKDAYS)}, {r.choice(MONTHS)} {r.randint(1, 28)}, {r.randint(2025, 2026)} at {r.randint(8, 18):02d}:{r.choice(['00', '15', '30', '45'])} {r.choice(TIMEZONES)}",
            lambda r: f"Meeting booked: {r.choice(WEEKDAYS)} {r.choice(MONTHS)} {r.randint(1, 28)} {r.randint(9, 16):02d}:00-{r.randint(17, 19):02d}:00 {r.choice(TIMEZONES)} (Room {r.randint(101, 410)})",
            lambda r: f"Calendar reminder: Quarterly {r.choice(DEPARTMENTS)} sync on {r.choice(WEEKDAYS)}, {r.choice(MONTHS)} {r.randint(1, 28)}, {r.randint(2025, 2026)} at {r.randint(9, 17):02d}:30 {r.choice(TIMEZONES)}",
            lambda r: f"Planned office hours window from {_rand_date(r)} 09:00 {r.choice(TIMEZONES)} through {_rand_date(r)} 17:30 {r.choice(TIMEZONES)} for {r.choice(CITIES)} staff",
        ]),
        # 10
        "clean_order_code": fam([
            lambda r: f"ORD-{r.randint(2024, 2026)}-{r.randint(1000, 9999)}",
            lambda r: f"Order #{r.randint(100000, 999999)}: packed",
            lambda r: f"Order ORD-{r.randint(2025, 2026)}-{r.randint(10000, 99999)} ({r.randint(1, 4)}x {r.choice(PRODUCTS)}) shipped via ground",
            lambda r: f"Tracking #{r.choice(['1Z', '94', 'JD'])}{r.randint(10000000, 99999999)} for ORD-{r.randint(10000, 99999)} - estimated arrival {r.choice(MONTHS)} {r.randint(1, 28)}",
            lambda r: f"Order ORD-{r.randint(2025, 2026)}-{r.randint(10000, 99999)} for {_p(r)}: {r.randint(1, 3)}x {r.choice(PRODUCTS)}, status=out for delivery in {r.choice(CITIES)}",
            lambda r: f"Shipment update for ORD-{r.randint(2025, 2026)}-{r.randint(10000, 99999)}: parcel left regional hub in {r.choice(CITIES)} on {_rand_date(r)} at {r.randint(6, 22):02d}:15",
        ]),
        # 11
        "clean_product_spec": fam([
            lambda r: f"{r.randint(120, 950)}g, {r.randint(10, 65)}W USB-C",
            lambda r: f"Model {r.choice(['MX', 'KL', 'TR', 'VX'])}-{r.randint(100, 990)}: {r.randint(12, 48)}cm x {r.randint(8, 30)}cm, matte gray",
            lambda r: f"Specs: {r.choice(PRODUCTS)} - weight {r.randint(200, 1800)}g, battery {r.randint(8, 36)}h, warranty {r.randint(1, 3)} years",
            lambda r: f"Dimensions {r.randint(15, 55)} x {r.randint(10, 40)} x {r.randint(3, 18)} cm; anodized aluminum frame; operating temp 0C to 40C",
            lambda r: f"Technical sheet: {r.choice(PRODUCTS)} series {r.randint(2, 7)}, input 100-240V AC 50/60Hz, standby power 0.{r.randint(1, 9)}W, recyclable box",
            lambda r: f"Hardware details for {r.choice(PRODUCTS)} ({r.choice(['black', 'silver', 'navy'])}): {r.randint(300, 1400)}g net weight, IPX{r.randint(4, 7)} splash rating, includes braided 1.5m cable",
        ]),
        # 12
        "clean_customer_support": fam([
            lambda r: f"Ticket #{r.randint(10000, 99999)} open",
            lambda r: f"Need receipt for order #{r.randint(100000, 999999)}",
            lambda r: f"Hi team, could you update the shipping city on my order #{r.randint(10000, 99999)} to {r.choice(CITIES)}?",
            lambda r: f"Case #{r.randint(20000, 99999)}: customer asked to exchange the {r.choice(PRODUCTS)} for a larger size before {r.choice(WEEKDAYS)}.",
            lambda r: f"Support follow-up for {_p(r)}: replacement {r.choice(PRODUCTS)} was dispatched today and tracking details were emailed to the account on file.",
            lambda r: f"Hello, I returned my {r.choice(PRODUCTS)} at the local drop-off counter on {_rand_date(r)} and wanted to check when the refund posts back to my card.",
        ]),
        # 13
        "clean_shipping_note": fam([
            lambda r: f"Leave at side porch",
            lambda r: f"Ring buzzer #{r.randint(101, 415)} upon arrival",
            lambda r: f"Please leave parcel with the front desk attendant before {r.randint(5, 8)} PM.",
            lambda r: f"Gate code is {r.randint(1000, 9999)}#; place box behind the planters near the garage door.",
            lambda r: f"Delivery instructions for {r.choice(CITIES)}: signature required on weekdays, otherwise hold safely at the neighborhood locker.",
            lambda r: f"Courier note: loading dock on {r.choice(STREETS)} closes at 16:30 sharp; call reception ext {r.randint(100, 899)} if arriving earlier.",
        ]),
        # 14
        "clean_invoice_line": fam([
            lambda r: f"INV-{r.randint(10000, 99999)}: ${r.randint(15, 450)}.00",
            lambda r: f"Paid ${r.randint(20, 890)}.{r.randint(10, 99)} via Visa ending {r.randint(1000, 9999)}",
            lambda r: f"Invoice #{r.randint(40000, 99999)} - Annual Workspace Plan ({r.randint(3, 25)} seats) - Net 30",
            lambda r: f"Line item: {r.randint(2, 12)}x {r.choice(PRODUCTS)} @ ${r.randint(14, 95)}.00 each, subtotal ${r.randint(60, 950)}.00",
            lambda r: f"Receipt #{r.randint(100000, 999999)} issued {_rand_date(r)} to {_comp(r)} for {r.choice(MONTHS)} subscription renewal: ${r.randint(99, 1490)}.00 paid",
            lambda r: f"Billing statement {_rand_date(r)}: consulting retainer for {r.choice(DEPARTMENTS)} advisory ({r.randint(4, 18)} hours at ${r.randint(85, 160)}/hr), payment settled in full",
        ]),
        # 15
        "clean_company_profile": fam([
            lambda r: _comp(r),
            lambda r: f"{_comp(r)} - {r.choice(CITIES)}",
            lambda r: f"{_comp(r)} builds everyday tools for independent teams across {r.choice(CITIES)}.",
            lambda r: f"Founded in {r.randint(2005, 2023)}, {_comp(r)} employs {r.randint(18, 420)} people in {r.choice(CITIES)}.",
            lambda r: f"{_comp(r)} partners with regional suppliers to deliver sustainable packaging and reliable fulfillment across North America.",
            lambda r: f"About {_comp(r)}: a family-owned business headquartered in {r.choice(CITIES)} focused on thoughtful industrial design and transparent customer care.",
        ]),
        # 16
        "clean_job_listing": fam([
            lambda r: f"Hiring: {r.choice(ROLES)}",
            lambda r: f"{r.choice(ROLES)} (Hybrid, {r.choice(CITIES)})",
            lambda r: f"{_comp(r)} is hiring a {r.choice(ROLES)} to join our {r.choice(DEPARTMENTS)} team.",
            lambda r: f"Open role: Senior {r.choice(ROLES)}, {r.choice(CITIES)}, ${r.randint(85, 155)}k-${r.randint(160, 195)}k + benefits",
            lambda r: f"Join {_comp(r)} as a {r.choice(ROLES)} in {r.choice(CITIES)}: 4 days flexible schedule, health coverage, and annual learning stipend.",
            lambda r: f"We are seeking a detail-oriented {r.choice(ROLES)} with {r.randint(2, 7)}+ years experience to support quarterly planning across our {r.choice(DEPARTMENTS)} group.",
        ]),
        # 17
        "clean_course_title": fam([
            lambda r: f"{r.choice(['CS', 'ECON', 'HIST', 'STAT', 'BIO', 'WRIT'])} {r.randint(101, 490)}: {r.choice(COURSES)}",
            lambda r: f"Intro to {r.choice(COURSES)} (Fall {r.randint(2024, 2026)})",
            lambda r: f"Workshop: Practical {r.choice(COURSES)} with Prof. {_p(r)} on {r.choice(WEEKDAYS)} mornings",
            lambda r: f"Syllabus for {r.choice(COURSES)} {r.randint(201, 399)}: weekly readings, two lab reports, and final seminar presentation.",
            lambda r: f"Online certificate in {r.choice(COURSES)}: {r.randint(4, 10)} self-paced modules, graded exercises, and weekly live Q&A sessions with instructors.",
            lambda r: f"Graduate seminar in {r.choice(COURSES)} meets {r.choice(WEEKDAYS)} 14:00-16:30 in Hall {r.randint(2, 9)}, Room {r.randint(104, 320)}; open to audited enrollment.",
        ]),
        # 18
        "clean_book_citation": fam([
            lambda r: f"{r.choice(BOOKS)} ({r.randint(1998, 2026)})",
            lambda r: f"{r.choice(BOOKS)} by {_p(r)}",
            lambda r: f"{_p(r)}, \"{r.choice(BOOKS)}\", 2nd Edition, {r.choice(COMPANIES)} Press ({r.randint(2015, 2026)}).",
            lambda r: f"Reading list: {r.choice(BOOKS)} by {_p(r)} - ISBN 978-0-{r.randint(100, 999)}-{r.randint(10000, 99999)}-{r.randint(0, 9)}",
            lambda r: f"Library hold ready: \"{r.choice(BOOKS)}\" by {_p(r)}, hardcover ({r.randint(210, 480)} pages), pickup at {r.choice(CITIES).split(',')[0]} Central Branch.",
            lambda r: f"Book club pick for {r.choice(MONTHS)}: \"{r.choice(BOOKS)}\" by {_p(r)} - discussing chapters 1 through {r.randint(5, 10)} next {r.choice(WEEKDAYS)} evening.",
        ]),
        # 19
        "clean_recipe_note": fam([
            lambda r: f"Bake at {r.randint(175, 220)}C for {r.randint(15, 45)}m",
            lambda r: f"Homemade {r.choice(FOODS)} in {r.randint(20, 50)} minutes",
            lambda r: f"Whisk 2 tbsp olive oil, lemon zest, and sea salt before tossing with warm {r.choice(FOODS)}.",
            lambda r: f"Recipe tip for {r.choice(FOODS)}: let the mixture rest {r.randint(10, 30)} minutes at room temperature before baking.",
            lambda r: f"Weekend prep: {r.choice(FOODS)} serves {r.randint(4, 6)}, keeps well in an airtight glass container in the fridge for up to four days.",
            lambda r: f"For extra flavor in {r.choice(FOODS)}, toast the spices in a dry pan for 60 seconds over medium heat before adding broth and simmering gently.",
        ]),
        # 20
        "clean_travel_itinerary": fam([
            lambda r: f"Flight {r.choice(['DL', 'UA', 'BA', 'LH', 'AC'])}{r.randint(100, 999)} on time",
            lambda r: f"Train #{r.randint(12, 88)} to {r.choice(CITIES).split(',')[0]} departs {r.randint(6, 21):02d}:{r.choice(['05', '20', '35', '50'])}",
            lambda r: f"Hotel check-in: {r.choice(COMPANIES)} Inn, {r.choice(CITIES)}, {r.randint(2, 6)} nights from {_rand_date(r)}",
            lambda r: f"Itinerary ref {r.choice(['BK', 'TR', 'FL'])}{r.randint(10000, 99999)}: depart {r.choice(CITIES).split(',')[0]} 08:40, arrive {r.choice(CITIES).split(',')[0]} 11:25",
            lambda r: f"Confirmed booking #{r.randint(100000, 999999)} for {_p(r)}: non-stop economy seat {r.randint(8, 32)}{r.choice('ABCDEF')}, one carry-on included.",
            lambda r: f"Travel reminder for {r.choice(MONTHS)} trip to {r.choice(CITIES)}: airport shuttle picks up at 06:15 outside lobby entrance; bring printed boarding pass.",
        ]),
        # 21
        "clean_weather_update": fam([
            lambda r: f"{r.choice(CITIES).split(',')[0]}: {r.randint(4, 29)}C sunny",
            lambda r: f"Light rain expected in {r.choice(CITIES).split(',')[0]} after {r.randint(2, 8)} PM",
            lambda r: f"Forecast for {r.choice(CITIES)}: high {r.randint(14, 30)}C, low {r.randint(3, 13)}C, gentle breeze at {r.randint(5, 22)} km/h.",
            lambda r: f"{r.choice(WEEKDAYS)} weather in {r.choice(CITIES).split(',')[0]}: partly cloudy skies, {r.randint(20, 75)}% humidity, sunset at 18:{r.randint(10, 55):02d}.",
            lambda r: f"Weekend outlook for {r.choice(CITIES)}: crisp morning air around {r.randint(5, 12)}C warming to {r.randint(16, 24)}C by afternoon with clear visibility.",
            lambda r: f"Regional weather note for {r.choice(CITIES)}: scattered showers tapering off by midday {r.choice(WEEKDAYS)}, followed by mild sunshine through the weekend.",
        ]),
        # 22
        "clean_sports_summary": fam([
            lambda r: f"{r.choice(SPORTS_TEAMS)} won {r.randint(1, 5)}-{r.randint(0, 4)}",
            lambda r: f"Final score: {r.choice(SPORTS_TEAMS)} {r.randint(60, 112)}, {r.choice(SPORTS_TEAMS)} {r.randint(58, 109)}",
            lambda r: f"{r.choice(SPORTS_TEAMS)} clinched a {r.randint(2, 4)}-{r.randint(0, 2)} home victory in front of {r.randint(4, 28)},000 fans.",
            lambda r: f"Match report ({r.choice(WEEKDAYS)}): {r.choice(SPORTS_TEAMS)} held a strong second-half lead with {r.randint(12, 22)} shots on target.",
            lambda r: f"Weekend standings update: {r.choice(SPORTS_TEAMS)} move into second place with {r.randint(38, 72)} points after three consecutive away wins.",
            lambda r: f"Postgame highlight: captain {_p(r)} scored twice in the final ten minutes to seal a comeback win for {r.choice(SPORTS_TEAMS)} on {_rand_date(r)}.",
        ]),
        # 23
        "clean_movie_review": fam([
            lambda r: f"Great film, 8.{r.randint(1, 9)}/10",
            lambda r: f"Showtime: {r.choice(BOOKS)} at {r.randint(14, 21)}:{r.choice(['00', '15', '30', '45'])}",
            lambda r: f"Enjoyed \"{r.choice(BOOKS)}\" last night - sharp pacing, warm cinematography, and strong lead performances.",
            lambda r: f"Three and a half stars: the second act slows down slightly, yet the ending ties every subplot together nicely.",
            lambda r: f"Cinema notes from {r.choice(CITIES).split(',')[0]} Festival: \"{r.choice(BOOKS)}\" stood out for its understated score and grounded dialogue.",
            lambda r: f"Watched all {r.randint(6, 10)} episodes over the weekend; thoughtful character arcs, crisp editing, and zero filler scenes from start to finish.",
        ]),
        # 24
        "clean_music_playlist": fam([
            lambda r: f"Track {r.randint(1, 14)}: Morning Coffee",
            lambda r: f"Now playing: {_p(r)} - {r.choice(BOOKS)} ({r.randint(2, 6)}:{r.randint(10, 59)})",
            lambda r: f"Acoustic focus playlist: {r.randint(18, 45)} instrumental tracks for rainy {r.choice(WEEKDAYS)} afternoons.",
            lambda r: f"Podcast episode #{r.randint(12, 240)}: conversations on urban gardens and neighborhood bakeries ({r.randint(28, 58)} min).",
            lambda r: f"Added \"{r.choice(BOOKS)}\" by {_p(r)} to the Sunday Morning Jazz playlist - mellow piano trio recorded live in {r.choice(CITIES).split(',')[0]}.",
            lambda r: f"Album recommendation for {r.choice(MONTHS)}: gentle folk arrangements with cello and acoustic guitar, ideal background listening while reading or cooking.",
        ]),
        # 25
        "clean_health_fitness": fam([
            lambda r: f"Walked {r.randint(5200, 14800)} steps today",
            lambda r: f"5K morning jog in {r.randint(22, 34)}:{r.randint(10, 59)} along the park trail",
            lambda r: f"Workout log ({_rand_date(r)}): 30 min cycling, 15 min mobility stretching, 2.2L water.",
            lambda r: f"Sleep summary: {r.randint(7, 9)}h {r.randint(5, 50)}m total rest, steady resting heart rate at {r.randint(52, 68)} bpm.",
            lambda r: f"Weekly fitness check-in: completed {r.randint(3, 5)} strength sessions and {r.randint(14, 32)} km of easy outdoor running without knee soreness.",
            lambda r: f"Training plan for {r.choice(MONTHS)}: alternate brisk 40-minute hill walks with core conditioning on {r.choice(WEEKDAYS)} and rest fully on Sunday.",
        ]),
        # 26
        "clean_finance_memo": fam([
            lambda r: f"Groceries: ${r.randint(42, 185)}.{r.randint(10, 99)}",
            lambda r: f"{r.choice(MONTHS)} savings goal reached ({r.randint(85, 100)}%)",
            lambda r: f"Monthly budget note: utilities came in ${r.randint(12, 48)} lower than {r.choice(MONTHS)} estimate.",
            lambda r: f"Expense summary Q{r.randint(1, 4)}: office supplies ${r.randint(180, 640)}, transit passes ${r.randint(90, 280)}, books ${r.randint(45, 160)}.",
            lambda r: f"Household memo ({_rand_date(r)}): transferred ${r.randint(250, 900)} into the emergency reserve account and reviewed annual insurance renewals.",
            lambda r: f"Quarterly review for {r.choice(DEPARTMENTS)}: travel spending remained {r.randint(6, 19)}% under budget thanks to consolidated regional meetings.",
        ]),
        # 27
        "clean_real_estate": fam([
            lambda r: f"2BR / 1BA, {r.randint(680, 1250)} sq ft",
            lambda r: f"Bright studio on {r.choice(STREETS)} - ${r.randint(1350, 2400)}/mo",
            lambda r: f"Open house {r.choice(WEEKDAYS)} 13:00-15:00 at {r.randint(100, 950)} {r.choice(STREETS)}, {r.choice(CITIES)}.",
            lambda r: f"Sunny corner apartment in {r.choice(CITIES)}: hardwood floors, balcony, in-unit laundry, bike storage included.",
            lambda r: f"Rental listing #{r.randint(1000, 9999)}: 2-bedroom flat ({r.randint(780, 1180)} sq ft) two blocks from transit, available {r.choice(MONTHS)} 1, pets welcome.",
            lambda r: f"Renovated townhouse near {r.choice(STREETS)} with south-facing garden patio, updated kitchen counters, and dedicated off-street parking.",
        ]),
        # 28
        "clean_event_rsvp": fam([
            lambda r: f"RSVP: Yes (+{r.randint(1, 3)} guests)",
            lambda r: f"Registered for {r.choice(CITIES).split(',')[0]} Design Meetup",
            lambda r: f"Count me in for the {r.choice(MONTHS)} community book swap on {r.choice(WEEKDAYS)} at 18:30!",
            lambda r: f"RSVP from {_p(r)}: attending the {r.choice(DEPARTMENTS)} town hall in person; vegetarian lunch preferred.",
            lambda r: f"Event pass #{r.randint(10000, 99999)} for Annual Makers Fair in {r.choice(CITIES)} on {_rand_date(r)} - doors open at 09:30.",
            lambda r: f"Thank you for registering for the {r.choice(MONTHS)} neighborhood workshop; light refreshments and printed handouts will be available at check-in.",
        ]),
        # 29
        "clean_vehicle_listing": fam([
            lambda r: f"{r.choice(VEHICLES)}, {r.randint(12, 68)}k mi",
            lambda r: f"Oil change due at {r.randint(35, 95)},000 miles",
            lambda r: f"Service record ({_rand_date(r)}): {r.choice(VEHICLES)} - tire rotation, cabin air filter, brake pad inspection.",
            lambda r: f"For sale in {r.choice(CITIES)}: {r.choice(VEHICLES)}, one owner, full maintenance log, clean title.",
            lambda r: f"Annual tune-up completed on {r.choice(VEHICLES)} at {r.randint(18, 74)},{r.randint(100, 999)} miles; battery health and alignment checked out fine.",
            lambda r: f"Commuter review of {r.choice(VEHICLES)}: averages {r.randint(31, 48)} mpg on highway drives, quiet cabin, and roomy cargo space for weekend gear.",
        ]),
        # 30
        "clean_faq_question": fam([
            lambda r: f"Ship to {r.choice(CITIES).split(',')[0]}?",
            lambda r: f"Warranty on {r.choice(PRODUCTS)}?",
            lambda r: f"Do you offer express delivery to {r.choice(CITIES)} on {r.choice(PRODUCTS)}?",
            lambda r: f"Can I pause my {r.choice(['monthly', 'quarterly', 'annual'])} plan for {r.randint(1, 6)} weeks while traveling?",
            lambda r: f"Where do I download past {r.choice(['VAT', 'tax', 'billing'])} invoices for {r.choice(MONTHS)} {r.randint(2023, 2026)} orders?",
            lambda r: f"Is it possible to combine a {r.choice(MONTHS)} voucher with free shipping on {r.choice(PRODUCTS)} over ${r.randint(40, 150)}?",
            lambda r: f"What steps transfer an existing {r.choice(DEPARTMENTS).lower()} workspace from {_p(r)} to another team administrator?",
        ]),
        # 31
        "clean_status_update": fam([
            lambda r: f"Sprint {r.randint(12, 48)} on track",
            lambda r: f"Milestone {r.randint(1, 6)} wrapped up ahead of {r.choice(WEEKDAYS)}",
            lambda r: f"Standup note: finalized {r.choice(DEPARTMENTS).lower()} slides and shared draft with {_p(r)}.",
            lambda r: f"Weekly update ({_rand_date(r)}): onboarding handbook refreshed and Q{r.randint(1, 4)} vendor checklist signed off.",
            lambda r: f"Project status green: {r.choice(CITIES).split(',')[0]} warehouse inventory count finished at 100% accuracy with zero backordered items.",
            lambda r: f"Team announcement: our {r.choice(MONTHS)} customer advisory roundtable moves to {r.choice(WEEKDAYS)} afternoon so European colleagues can join live.",
        ]),
        # 32
        "clean_inventory_sku": fam([
            lambda r: f"SKU-{r.randint(1000, 9999)}: {r.randint(5, 240)} in stock",
            lambda r: f"Bin {r.choice('ABCDEF')}{r.randint(1, 18)}-{r.randint(10, 99)}: {r.choice(PRODUCTS)}",
            lambda r: f"Warehouse {r.choice(CITIES).split(',')[0]}: restocked {r.randint(20, 150)} units of {r.choice(PRODUCTS)} (SKU-{r.randint(1000, 9999)}).",
            lambda r: f"Cycle count {_rand_date(r)}: aisle {r.randint(1, 14)}, shelf {r.randint(1, 6)}, quantity {r.randint(12, 320)} units verified.",
            lambda r: f"Low stock reminder: {r.choice(PRODUCTS)} in {r.choice(['slate', 'olive', 'ivory', 'navy'])} down to {r.randint(3, 14)} units; purchase order queued.",
            lambda r: f"Receiving log #{r.randint(10000, 99999)}: pallet of {r.randint(40, 120)} cartons from {_comp(r)} inspected and checked into zone {r.choice('ABCD')}{r.randint(1, 9)}.",
        ]),
        # 33
        "clean_phone_directory": fam([
            lambda r: f"+1-555-{r.randint(200, 999)}-{r.randint(1000, 9999)}",
            lambda r: f"Reception: (555) {r.randint(200, 899)}-{r.randint(1000, 9999)} ext {r.randint(10, 99)}",
            lambda r: f"{r.choice(CITIES).split(',')[0]} office desk: +1 (555) {r.randint(200, 899)}-{r.randint(1000, 9999)}, Mon-Fri 08:30-17:30",
            lambda r: f"Call {_p(r)} ({r.choice(DEPARTMENTS)}) at 555-{r.randint(200, 899)}-{r.randint(1000, 9999)} during business hours.",
            lambda r: f"Customer care hotline: +1-800-555-{r.randint(1000, 9999)} (English and Spanish support available daily 08:00-20:00 {r.choice(TIMEZONES)}).",
            lambda r: f"After-hours facilities contact for {_comp(r)} {r.choice(CITIES).split(',')[0]} campus: call security desk at (555) {r.randint(200, 899)}-{r.randint(1000, 9999)}.",
        ]),
        # 34
        "clean_feedback_survey": fam([
            lambda r: f"NPS {r.randint(6, 10)}/10 ({r.choice(CITIES).split(',')[0]})",
            lambda r: f"Rated {r.randint(3, 5)}/5 for {r.choice(PRODUCTS)}",
            lambda r: f"Checkout took {r.randint(20, 90)}s on {r.choice(WEEKDAYS)} - smooth experience overall.",
            lambda r: f"Survey #{r.randint(100, 999)}: friendly staff at {r.choice(CITIES).split(',')[0]} branch and clear signage.",
            lambda r: f"Suggestion: please add a filter by {r.choice(['delivery date', 'color', 'material', 'weight'])} when browsing {r.choice(PRODUCTS)}.",
            lambda r: f"Satisfaction {r.randint(4, 5)}/5 on order #{r.randint(1000, 9999)}: recyclable box and easy setup card for the {r.choice(PRODUCTS)}.",
            lambda r: f"Thanks for asking ({_rand_date(r)}): the {r.choice(DEPARTMENTS).lower()} agent answered my question in {r.randint(1, 8)} minutes with a clear summary.",
        ]),
        # 35
        "clean_newsletter_subject": fam([
            lambda r: f"{r.choice(MONTHS)} Digest #{r.randint(10, 99)}",
            lambda r: f"5 ideas for organizing your desk this {r.choice(MONTHS)}",
            lambda r: f"Weekend Reads: sustainable travel tips and seasonal {r.choice(FOODS)} recipes",
            lambda r: f"{_comp(r)} Weekly: highlights from our {r.choice(CITIES).split(',')[0]} workshop and new spring arrivals",
            lambda r: f"Inside Issue #{r.randint(40, 180)}: how small studios plan calmer Mondays, plus our favorite {r.choice(PRODUCTS)} picks",
            lambda r: f"Community Newsletter ({r.choice(MONTHS)} {r.randint(2025, 2026)}): upcoming neighborhood classes, member stories, and volunteer sign-ups",
        ]),
        # 36
        "clean_academic_note": fam([
            lambda r: f"Ch. {r.randint(1, 14)} summary notes",
            lambda r: f"Lab {r.randint(1, 9)}: sample mass {r.randint(12, 88)}.{r.randint(1, 9)}g, pH 6.{r.randint(2, 9)}",
            lambda r: f"Study note ({r.choice(COURSES)}): compare primary sources from {r.randint(1960, 2015)} before seminar.",
            lambda r: f"Abstract draft: longitudinal survey of {r.randint(120, 850)} households examining transit habits in {r.choice(CITIES)}.",
            lambda r: f"Bibliography entry: {_p(r)} ({r.randint(2018, 2025)}), Journal of {r.choice(COURSES)}, Vol. {r.randint(12, 48)}, pp. {r.randint(10, 90)}-{r.randint(95, 140)}.",
            lambda r: f"Revision checklist for {r.choice(COURSES)} midterm: review lecture slides 4-7, work through problem set {r.randint(2, 6)}, and verify citation formatting.",
        ]),
        # 37
        "clean_pet_care_note": fam([
            lambda r: f"Vet visit {_rand_date(r)}",
            lambda r: f"Adopted a {r.randint(1, 5)}-year-old {r.choice(PETS)}!",
            lambda r: f"Feeding note: half cup dry kibble at 07:30 and 18:00 for our {r.choice(PETS)}.",
            lambda r: f"Grooming appointment for our {r.choice(PETS)} on {r.choice(WEEKDAYS)} at {r.randint(9, 16):02d}:00 in {r.choice(CITIES).split(',')[0]}.",
            lambda r: f"Pet sitter instructions: the {r.choice(PETS)} loves a 20-minute morning walk around {r.choice(STREETS)} and fresh water after playtime.",
            lambda r: f"Annual checkup complete for our {r.choice(PETS)}: healthy weight, teeth look clean, and routine vaccinations are up to date through {r.choice(MONTHS)} {r.randint(2026, 2027)}.",
        ]),
        # 38
        "clean_gardening_tip": fam([
            lambda r: f"Water {r.choice(PLANTS)} weekly",
            lambda r: f"Prune {r.choice(PLANTS)} in early {r.choice(['spring', 'autumn'])}",
            lambda r: f"Move potted {r.choice(PLANTS)} to bright indirect sunlight near an east-facing window.",
            lambda r: f"Garden log ({_rand_date(r)}): repotted {r.choice(PLANTS)} into a {r.randint(15, 30)}cm terracotta pot with perlite mix.",
            lambda r: f"Balcony care tip: let the top two inches of soil dry between waterings so {r.choice(PLANTS)} roots stay aerated and healthy.",
            lambda r: f"Seasonal planting note for {r.choice(CITIES).split(',')[0]}: start {r.choice(PLANTS)} indoors {r.randint(4, 8)} weeks before the last spring frost for sturdier stems.",
        ]),
        # 39
        "clean_restaurant_menu": fam([
            lambda r: f"Table for {r.randint(2, 6)} at {r.randint(18, 21)}:00",
            lambda r: f"Daily special: {r.choice(FOODS)} (${r.randint(11, 26)})",
            lambda r: f"Reservation under {_p(r)} for {r.randint(2, 5)} guests on {r.choice(WEEKDAYS)} evening, patio seating.",
            lambda r: f"Cafe menu note: oat milk latte, seasonal {r.choice(FOODS)}, and gluten-free almond cake available daily.",
            lambda r: f"Bakery pre-order #{r.randint(100, 999)}: two loaves of {r.choice(FOODS)} and six cinnamon rolls for pickup {r.choice(WEEKDAYS)} at 09:30.",
            lambda r: f"Dinner review: cozy neighborhood bistro on {r.choice(STREETS)} with attentive servers, crisp salads, and freshly baked {r.choice(FOODS)}.",
        ]),
        # 40
        "clean_software_changelog": fam([
            lambda r: f"Release v{r.randint(1, 5)}.{r.randint(0, 14)}.{r.randint(0, 9)}",
            lambda r: f"Added dark theme and keyboard shortcuts",
            lambda r: f"Version {r.randint(2, 6)}.{r.randint(0, 9)}: improved offline sync speed and reduced battery usage by {r.randint(8, 24)}%.",
            lambda r: f"Changelog ({_rand_date(r)}): fixed calendar timezone display when switching between {r.choice(TIMEZONES)} and {r.choice(TIMEZONES)}.",
            lambda r: f"App update v{r.randint(3, 8)}.{r.randint(1, 12)}: smoother PDF export, clearer search highlights, and faster startup on older laptops.",
            lambda r: f"Release notes for {r.choice(MONTHS)} {r.randint(2025, 2026)}: added weekly digest preferences, refined table column widths, and resolved a rare printer crash.",
        ]),
    }


# ============================================================================
# 40 UNUSUAL FAMILY BUILDERS (benign samples with misleading keywords/symbols)
# ============================================================================

ATTACKER_TEST_FAMILIES: set[str] = {
    "unusual_testdomain_http_log",
    "unusual_testdomain_json_config",
    "unusual_testdomain_qa_doc",
    "unusual_testdomain_url_discussion",
    "unusual_testdomain_curl_cmd",
    "unusual_testdomain_firewall_rule",
    "unusual_testdomain_pytest_fixture",
    "unusual_testdomain_openapi_spec",
}


def _build_unusual_families() -> dict[str, Callable[[random.Random], list[str]]]:
    def fam(templates: list[Callable[[random.Random], str]], pool_size: int = 900) -> Callable[[random.Random], list[str]]:
        return lambda rng: _generate_pool(rng, templates, pool_size=pool_size)

    def _td_fam(
        templates_for_domain: Callable[[str], list[Callable[[random.Random], str]]],
    ) -> Callable[[random.Random], list[str]]:
        """Generate rich pools for BOTH https://attacker.test and sibling .test/.example domains."""
        def _builder(rng: random.Random) -> list[str]:
            atk_pool = _generate_pool(rng, templates_for_domain("attacker.test"), pool_size=750)
            alt_templates: list[Callable[[random.Random], str]] = []
            for dom in ALT_TEST_DOMAINS:
                alt_templates.extend(templates_for_domain(dom))
            alt_pool = _generate_pool(rng, alt_templates, pool_size=250)
            return atk_pool + alt_pool
        return _builder

    return {
        # --- 8 Anti-shortcut families with https://attacker.test/... & .test/.example ---
        # 1
        "unusual_testdomain_http_log": _td_fam(lambda dom: [
            lambda r, d=dom: f"GET https://{d}{r.choice(TEST_PATHS)} 200",
            lambda r, d=dom: f"{_rand_time(r)} {r.choice(HTTP_METHODS)} https://{d}{r.choice(TEST_PATHS)} {r.choice(HTTP_CODES)} {r.randint(4, 240)}ms",
            lambda r, d=dom: f"{_rand_date(r)} {_rand_time(r)} INFO {r.choice(HTTP_METHODS)} https://{d}{r.choice(TEST_PATHS)} HTTP/1.1 {r.choice(HTTP_CODES)} {r.randint(120, 4800)}B",
            lambda r, d=dom: f"[nginx] {_rand_ip(r)} {r.choice(HTTP_METHODS)} https://{d}{r.choice(TEST_PATHS)}?env={r.choice(ENV_NAMES)} status={r.choice(HTTP_CODES)} rt=0.0{r.randint(10, 99)}s",
            lambda r, d=dom: f"access_log svc={r.choice(SERVICES)} method={r.choice(HTTP_METHODS)} url=https://{d}{r.choice(TEST_PATHS)}?rid={_rand_hex(r)} code={r.choice(HTTP_CODES)} ms={r.randint(5, 310)}",
            lambda r, d=dom: f"{_rand_date(r)}T{_rand_time(r)}Z proxy={r.choice(SERVICES)} upstream=https://{d}{r.choice(TEST_PATHS)}?retry={r.randint(0, 3)} status={r.choice(HTTP_CODES)} trace=tr_{_rand_hex(r, 8)}",
        ]),
        # 2
        "unusual_testdomain_json_config": _td_fam(lambda dom: [
            lambda r, d=dom: f'{{"url":"https://{d}{r.choice(TEST_PATHS)}"}}',
            lambda r, d=dom: f'{{"mockHost":"https://{d}{r.choice(TEST_PATHS)}","timeout":{r.randint(500, 5000)}}}',
            lambda r, d=dom: f'{{"service":"{r.choice(SERVICES)}","endpoint":"https://{d}{r.choice(TEST_PATHS)}","retries":{r.randint(1, 5)}}}',
            lambda r, d=dom: f'{{"env":"{r.choice(ENV_NAMES)}","webhook":"https://{d}{r.choice(TEST_PATHS)}","verify_tls":true,"code":{r.choice(HTTP_CODES)}}}',
            lambda r, d=dom: f'{{"fixture_id":"fx_{_rand_hex(r)}","base_url":"https://{d}{r.choice(TEST_PATHS)}","allow_redirects":false,"budget_ms":{r.randint(200, 3000)}}}',
            lambda r, d=dom: f'{{"suite":"integration","target":"https://{d}{r.choice(TEST_PATHS)}","headers":{{"Accept":"application/json"}},"env":"{r.choice(ENV_NAMES)}","ok":true}}',
        ]),
        # 3
        "unusual_testdomain_qa_doc": _td_fam(lambda dom: [
            lambda r, d=dom: f"Mock URL: https://{d}{r.choice(TEST_PATHS)}",
            lambda r, d=dom: f"Use https://{d}{r.choice(TEST_PATHS)} in offline unit tests.",
            lambda r, d=dom: f"QA note: RFC 6761 reserves https://{d}{r.choice(TEST_PATHS)} for local sandbox fixtures.",
            lambda r, d=dom: f"Testing guide: point staging webhooks to https://{d}{r.choice(TEST_PATHS)} so CI never hits production.",
            lambda r, d=dom: f"Documentation ({r.choice(SERVICES)}): replace live callback URLs with https://{d}{r.choice(TEST_PATHS)} during {r.choice(ENV_NAMES)} verification runs.",
            lambda r, d=dom: f"Sandbox policy: integration specs mock outbound requests to https://{d}{r.choice(TEST_PATHS)} with deterministic status {r.choice(HTTP_CODES)} responses.",
        ]),
        # 4
        "unusual_testdomain_url_discussion": _td_fam(lambda dom: [
            lambda r, d=dom: f"Try https://{d}{r.choice(TEST_PATHS)} in CI",
            lambda r, d=dom: f"Does https://{d}{r.choice(TEST_PATHS)} return {r.choice(HTTP_CODES)} locally?",
            lambda r, d=dom: f"PR comment: switched the sample host to https://{d}{r.choice(TEST_PATHS)} per RFC 2606.",
            lambda r, d=dom: f"Slack thread: our local DNS stub maps https://{d}{r.choice(TEST_PATHS)} to 127.0.0.1:{r.randint(8000, 9090)} cleanly.",
            lambda r, d=dom: f"Issue #{r.randint(100, 999)}: link validator should accept reserved test URI https://{d}{r.choice(TEST_PATHS)} when {r.choice(ENV_NAMES)} mode is active.",
            lambda r, d=dom: f"Code review note from {_rand_person(r)[0]}: keep https://{d}{r.choice(TEST_PATHS)} in README examples instead of any real third-party domain name.",
        ]),
        # 5
        "unusual_testdomain_curl_cmd": _td_fam(lambda dom: [
            lambda r, d=dom: f"curl -I https://{d}{r.choice(TEST_PATHS)}",
            lambda r, d=dom: f"curl -sS -X {r.choice(HTTP_METHODS)} https://{d}{r.choice(TEST_PATHS)}",
            lambda r, d=dom: f"curl -fsS --max-time {r.randint(3, 15)} https://{d}{r.choice(TEST_PATHS)} -H 'Accept: application/json'",
            lambda r, d=dom: f"http {r.choice(HTTP_METHODS)} https://{d}{r.choice(TEST_PATHS)} env={r.choice(ENV_NAMES)} trace={_rand_hex(r)}",
            lambda r, d=dom: f"curl -s -o /dev/null -w '%{{http_code}}' --resolve {d}:443:127.0.0.1 https://{d}{r.choice(TEST_PATHS)}",
            lambda r, d=dom: f"wget --spider --tries={r.randint(1, 3)} --timeout={r.randint(5, 20)} https://{d}{r.choice(TEST_PATHS)}?check={_rand_hex(r)} --no-verbose",
        ]),
        # 6
        "unusual_testdomain_firewall_rule": _td_fam(lambda dom: [
            lambda r, d=dom: f"allowlist: https://{d}{r.choice(TEST_PATHS)}",
            lambda r, d=dom: f"egress_policy action=ALLOW dst=https://{d}{r.choice(TEST_PATHS)}",
            lambda r, d=dom: f"dns_sinkhole zone={d} url=https://{d}{r.choice(TEST_PATHS)} env={r.choice(ENV_NAMES)} ok=true",
            lambda r, d=dom: f"proxy_rule id=PR-{r.randint(100, 999)} match=https://{d}{r.choice(TEST_PATHS)} target=mock_backend_{r.randint(1, 9)}",
            lambda r, d=dom: f"waf_audit rule_id={r.randint(1000, 9999)} note='benign fixture check' uri=https://{d}{r.choice(TEST_PATHS)} result=pass",
            lambda r, d=dom: f"egress_filter [{_rand_date(r)}] service={r.choice(SERVICES)} sandbox_url=https://{d}{r.choice(TEST_PATHS)} verdict=mock_routed",
        ]),
        # 7
        "unusual_testdomain_pytest_fixture": _td_fam(lambda dom: [
            lambda r, d=dom: f"BASE_URL = 'https://{d}{r.choice(TEST_PATHS)}'",
            lambda r, d=dom: f"assert resp.url == 'https://{d}{r.choice(TEST_PATHS)}'",
            lambda r, d=dom: f"responses.add(responses.GET, 'https://{d}{r.choice(TEST_PATHS)}', status={r.choice(HTTP_CODES)})",
            lambda r, d=dom: f"test_case('parses https://{d}{r.choice(TEST_PATHS)} with query id={r.randint(1, 999)}')",
            lambda r, d=dom: f"httpx_mock.add_response(url='https://{d}{r.choice(TEST_PATHS)}', json={{'status': 'ok', 'id': {r.randint(1, 999)}}})",
            lambda r, d=dom: f"def test_{r.choice(SERVICES).replace('-', '_')}_stub(): return parse_url('https://{d}{r.choice(TEST_PATHS)}?v={r.randint(1, 9)}')",
        ]),
        # 8
        "unusual_testdomain_openapi_spec": _td_fam(lambda dom: [
            lambda r, d=dom: f"MOCK_URL=https://{d}{r.choice(TEST_PATHS)}",
            lambda r, d=dom: f"servers: - url: https://{d}{r.choice(TEST_PATHS)} # {r.choice(ENV_NAMES)}",
            lambda r, d=dom: f"CALLBACK_URI=https://{d}{r.choice(TEST_PATHS)} # local docker compose override",
            lambda r, d=dom: f"openapi.servers[0].url = https://{d}{r.choice(TEST_PATHS)} (description: {r.choice(ENV_NAMES)} stub)",
            lambda r, d=dom: f"healthcheck: endpoint=https://{d}{r.choice(TEST_PATHS)} interval={r.randint(10, 60)}s expected_status={r.choice(HTTP_CODES)}",
            lambda r, d=dom: f"webhook_destination: name={r.choice(SERVICES)}_test target=https://{d}{r.choice(TEST_PATHS)} format=json retries={r.randint(1, 4)}",
        ]),

        # --- 32 Non-attacker.test Unusual families (misleading keywords & technical contexts) ---
        # 9: alert
        "unusual_alert_monitoring": fam([
            lambda r: f"alert: cpu {r.randint(70, 99)}%",
            lambda r: f"Alert resolved on {r.choice(SERVICES)}",
            lambda r: f"PagerDuty alert #{r.randint(1000, 9999)}: {r.choice(SERVICES)} latency > {r.randint(250, 900)}ms",
            lambda r: f"Prometheus alert rule HighDiskUsage fired for node-{r.randint(1, 16)} ({r.randint(82, 96)}% full).",
            lambda r: f"Notification preference: send email alert when monthly {r.choice(SERVICES)} spend exceeds ${r.randint(200, 2500)}.",
            lambda r: f"Runbook note: mute duplicate alert notifications during scheduled {r.choice(WEEKDAYS)} database maintenance window in {r.choice(ENV_NAMES)}.",
        ]),
        # 10: confirm
        "unusual_confirm_dialog_doc": fam([
            lambda r: f"confirm order #{r.randint(1000, 9999)}",
            lambda r: f"Please confirm your email address",
            lambda r: f"UX guide: replace browser confirm() popups with an accessible modal dialog.",
            lambda r: f"Check the box to confirm deletion of archived workspace backup #{r.randint(10, 999)}.",
            lambda r: f"Frontend review: custom confirm dialog in {r.choice(FRAMEWORKS)} should trap keyboard focus and support Escape.",
            lambda r: f"Support note: ask the user to confirm the last four digits of their invoice #{r.randint(10000, 99999)} before updating billing details.",
        ]),
        # 11: prompt
        "unusual_prompt_engineering": fam([
            lambda r: f"prompt: {r.choice(ENV_NAMES)}-{r.randint(1, 99)}$",
            lambda r: f"PS1 prompt for {r.choice(SERVICES)} shell",
            lambda r: f"CLI tip: pass --yes to skip the {r.choice(SERVICES)} confirmation prompt in {r.choice(ENV_NAMES)}.",
            lambda r: f"How to customize the {r.choice(['zsh', 'bash', 'fish'])} prompt theme to display {r.choice(ENV_NAMES)} git branch names?",
            lambda r: f"Docs ({r.choice(FRAMEWORKS)}): avoid calling window prompt() in {r.choice(['login', 'profile', 'billing'])} forms; use inline validation hints.",
            lambda r: f"System prompt template v{r.randint(1, 24)} ({r.choice(SERVICES)}): summarize customer tickets into {r.randint(2, 5)} concise bullet points.",
        ]),
        # 12: onerror
        "unusual_onerror_doc": fam([
            lambda r: f"onerror fallback handler",
            lambda r: f"Why is onerror firing twice in {r.choice(FRAMEWORKS)}?",
            lambda r: f"MDN note: attach addEventListener('error', cb) instead of inline onerror attributes.",
            lambda r: f"Bug #{r.randint(100, 999)}: websocket onerror callback logged timeout before reconnect backoff triggered.",
            lambda r: f"StackOverflow: how to render a default avatar placeholder when an image fails without inline onerror markup?",
            lambda r: f"Telemetry guide: global window.onerror captures unhandled runtime exceptions and reports line numbers to {r.choice(SERVICES)}.",
        ]),
        # 13: onload
        "unusual_onload_doc": fam([
            lambda r: f"onload timing: {r.randint(120, 980)}ms",
            lambda r: f"DOMContentLoaded vs window onload",
            lambda r: f"Perf audit: defer heavy analytics setup until after window.onload completes on mobile.",
            lambda r: f"Question: why does FileReader.onload return an empty buffer on zero-byte CSV uploads?",
            lambda r: f"Browser guide ({r.choice(BROWSERS_LIST)}): onload waits for stylesheets and subresources, whereas DOMContentLoaded fires earlier.",
            lambda r: f"Migration note for {r.choice(FRAMEWORKS)}: move legacy body onload hooks into component mount lifecycle methods for cleaner testing.",
        ]),
        # 14: onclick
        "unusual_onclick_doc": fam([
            lambda r: f"onClick #{r.randint(1, 999)} ok",
            lambda r: f"Refactor {r.choice(SERVICES)} onclick handler",
            lambda r: f"A11y tip ({r.choice(FRAMEWORKS)}): use a native button element instead of adding onclick to a {r.choice(['div', 'span', 'li'])}.",
            lambda r: f"Question ({r.choice(FRAMEWORKS)}): why does onClick={{handle{r.choice(['Save', 'Reset', 'Filter', 'Close', 'Export'])}}} fire on render?",
            lambda r: f"Code review ({r.choice(FRAMEWORKS)}): debounce the {r.choice(['submit', 'export', 'retry', 'sync'])} button onClick handler by {r.randint(150, 600)}ms.",
            lambda r: f"CSP checklist ({r.choice(SERVICES)}): removed {r.randint(3, 45)} legacy inline onclick attributes from {r.choice(ENV_NAMES)} templates in favor of listeners.",
        ]),
        # 15: onfocus
        "unusual_onfocus_doc": fam([
            lambda r: f"onfocus ring #{r.randint(1, 999)}",
            lambda r: f"Keyboard onfocus in {r.choice(BROWSERS_LIST)}",
            lambda r: f"A11y check ({r.choice(FRAMEWORKS)}): show a {r.randint(2, 4)}px contrast outline onfocus for keyboard navigation.",
            lambda r: f"Question: how to clear {r.choice(['email', 'search', 'zip', 'phone'])} errors onfocus without layout shift in {r.choice(BROWSERS_LIST)}?",
            lambda r: f"Form UX ({r.choice(SERVICES)}): pair onfocus and onblur handlers with aria-describedby for {r.choice(['checkout', 'signup', 'settings'])} fields.",
            lambda r: f"Design spec v{r.randint(2, 19)}: avoid trapping cursor focus with autofocus or aggressive onfocus scrolling on {r.choice(BROWSERS_LIST)} mobile.",
        ]),
        # 16: document.cookie
        "unusual_document_cookie_doc": fam([
            lambda r: f"document.cookie ({r.choice(ENV_NAMES)})",
            lambda r: f"document.cookie empty in {r.choice(BROWSERS_LIST)}",
            lambda r: f"Security note ({r.choice(SERVICES)}): tokens marked HttpOnly cannot be read via document.cookie.",
            lambda r: f"MDN guide: document.cookie only lists non-HttpOnly cookies matching path /{r.choice(['app', 'admin', 'api', 'docs'])} and domain.",
            lambda r: f"Audit #{r.randint(100, 999)} ({r.choice(SERVICES)}): verify sid_{r.choice(ENV_NAMES)} has Secure, SameSite=Lax, and HttpOnly off document.cookie.",
            lambda r: f"StackOverflow: why does SameSite={r.choice(['Strict', 'Lax'])} affect reading {r.choice(['prefs', 'theme', 'locale'])} from document.cookie in {r.choice(BROWSERS_LIST)}?",
        ]),
        # 17: window.location
        "unusual_window_location_doc": fam([
            lambda r: f"window.location.pathname",
            lambda r: f"Mock window.location in Jest tests",
            lambda r: f"SPA routing tip: prefer router.push() over assigning window.location.href directly.",
            lambda r: f"How to read URL query parameters from window.location.search using URLSearchParams in JS?",
            lambda r: f"Debugging note ({r.choice(FRAMEWORKS)}): window.location.origin differs between localhost:{r.randint(3000, 8080)} and staging proxy.",
            lambda r: f"Unit test helper: stub window.location.pathname to '/settings/profile' so breadcrumb rendering assertions stay deterministic.",
        ]),
        # 18: fetch
        "unusual_fetch_api_doc": fam([
            lambda r: f"fetch() retry={r.randint(1, 99)}",
            lambda r: f"AbortController for {r.choice(SERVICES)} fetch",
            lambda r: f"Remember fetch() only rejects on network failure, not HTTP {r.choice(HTTP_CODES)} from {r.choice(SERVICES)}.",
            lambda r: f"How to set a {r.randint(2, 30)}s timeout on fetch() calls in {r.choice(BROWSERS_LIST)} using AbortSignal.timeout?",
            lambda r: f"API guide ({r.choice(SERVICES)}): pass credentials: 'same-origin' to fetch('/api/v{r.randint(1, 3)}/{r.choice(['profile', 'orders', 'items', 'status'])}') and check res.ok.",
            lambda r: f"Perf tip ({r.choice(FRAMEWORKS)}): batch {r.randint(2, 8)} parallel fetch calls with Promise.allSettled so one slow endpoint never stalls the page.",
        ]),
        # 19: script
        "unusual_script_automation": fam([
            lambda r: f"backup script ok ({r.randint(2, 45)}s)",
            lambda r: f"Run migration script in dry-run mode",
            lambda r: f"Python script scripts/export_csv.py processed {r.randint(500, 9500)} rows with zero errors.",
            lambda r: f"Bash deployment script checks disk space before restarting {r.choice(SERVICES)} containers.",
            lambda r: f"Podcast script draft v{r.randint(2, 9)}: intro music 10 seconds, host welcome, and two listener Q&A segments.",
            lambda r: f"Build docs: add 'lint:fix' to package.json scripts so every contributor runs the exact same formatting script before opening a PR.",
        ]),
        # 20: img
        "unusual_img_asset_doc": fam([
            lambda r: f"img alt #{r.randint(1, 999)}",
            lambda r: f"img onload budget {r.randint(120, 650)}ms",
            lambda r: f"Compress {r.choice(['hero', 'thumb', 'avatar', 'banner'])} img to WebP ({r.randint(15, 240)}KB)",
            lambda r: f"Lighthouse ({r.choice(SERVICES)}): track hero img onload timing via PerformanceObserver ({r.choice(ENV_NAMES)}).",
            lambda r: f"CDN note: attach addEventListener('error') instead of inline onerror for fallback img placeholders.",
            lambda r: f"A11y checklist ({r.choice(FRAMEWORKS)}): decorative img icons in {r.choice(['nav', 'footer', 'card', 'modal'])} should use an empty alt attribute.",
            lambda r: f"Asset pipeline ({r.choice(SERVICES)}): lazy-load below-the-fold img tags with loading='lazy' and monitor onerror retry counts.",
        ]),
        # 21: svg
        "unusual_svg_design_doc": fam([
            lambda r: f"svg viewBox 0 0 {r.choice([16, 20, 24, 32, 48])} {r.choice([16, 20, 24, 32, 48])}",
            lambda r: f"Export {r.choice(['logo', 'badge', 'arrow', 'chart', 'menu'])} icon as SVG ({r.randint(1, 25)}KB)",
            lambda r: f"Design note: run SVGO v{r.randint(2, 4)} to strip metadata from {r.randint(5, 90)} exported svg icons before committing.",
            lambda r: f"Why does my inline svg icon look blurry at {r.choice([14, 16, 18, 22])}px with stroke-width 1.5 in {r.choice(BROWSERS_LIST)}?",
            lambda r: f"Icon library v{r.randint(2, 18)}: standardized {r.randint(40, 320)} monochrome svg glyphs onto a {r.choice([20, 24, 32])}px grid with currentColor.",
            lambda r: f"Security hygiene ({r.choice(SERVICES)}): sanitize user-uploaded svg assets or rasterize to PNG before serving on {r.choice(ENV_NAMES)} cards.",
        ]),
        # 22: iframe
        "unusual_iframe_embed_doc": fam([
            lambda r: f"iframe ratio {r.choice(['16/9', '4/3', '1/1', '21/9'])} #{r.randint(1, 99)}",
            lambda r: f"Sandbox iframe embed in {r.choice(SERVICES)}",
            lambda r: f"CSP frame-ancestors 'self' prevents third-party sites from framing {r.choice(SERVICES)} in an iframe.",
            lambda r: f"How to make an embedded {r.choice(['map', 'chart', 'video', 'calendar'])} iframe resize cleanly in {r.choice(BROWSERS_LIST)}?",
            lambda r: f"Docs ({r.choice(FRAMEWORKS)}): add title='{r.choice(['Chart', 'Demo', 'Preview', 'Widget'])} #{r.randint(1, 99)}' and loading='lazy' to every iframe.",
            lambda r: f"Integration guide ({r.choice(SERVICES)}): restrict payment iframe permissions via allow='payment' and verify postMessage origin ({r.choice(ENV_NAMES)}).",
        ]),
        # 23: javascript:
        "unusual_js_protocol_discussion": fam([
            lambda r: f"lint: ban javascript: #{r.randint(1, 999)}",
            lambda r: f"Block javascript: URLs in {r.choice(SERVICES)}",
            lambda r: f"ESLint rule no-script-url flags javascript: URIs in {r.choice(FRAMEWORKS)} anchor components.",
            lambda r: f"URL validator test #{r.randint(10, 999)}: reject javascript: and vbscript: schemes; permit only http: and https:.",
            lambda r: f"{r.choice(FRAMEWORKS)} migration note: disallow javascript: links across {r.choice(['nav', 'table', 'card', 'sidebar'])} templates in {r.choice(ENV_NAMES)}.",
            lambda r: f"Security FAQ ({r.choice(SERVICES)}): why replace legacy javascript:void(0) anchors with native button elements and delegated event listeners?",
        ]),
        # 24: data:
        "unusual_data_uri_discussion": fam([
            lambda r: f"data:image/{r.choice(['png', 'webp', 'gif'])} {r.randint(1, 8)}KB",
            lambda r: f"Cap inline data: URIs at {r.choice([2, 4, 8, 16])}KB ({r.choice(SERVICES)})",
            lambda r: f"Webpack asset/inline embeds tiny {r.choice(['icon', 'badge', 'dot', 'cursor'])} PNGs as data:image/png;base64 under {r.randint(1, 8)}KB.",
            lambda r: f"Email note ({r.choice(SERVICES)}): {r.choice(['Outlook', 'Gmail', 'Apple Mail'])} clients block data: URI images, so host logos over HTTPS.",
            lambda r: f"CSP audit #{r.randint(100, 999)}: allow data: only in {r.choice(['img-src', 'font-src'])} directives, never inside script-src ({r.choice(ENV_NAMES)}).",
            lambda r: f"Perf guide ({r.choice(FRAMEWORKS)}): overusing data: URIs inside CSS stylesheets inflates bundle size by {r.randint(10, 85)}KB and hurts caching.",
        ]),
        # 25: safe HTML prose
        "unusual_safe_html_prose": fam([
            lambda r: f"<p>Order #{r.randint(100, 999)} ready.</p>",
            lambda r: f"<h3>Chapter {r.randint(1, 12)}: Overview</h3>",
            lambda r: f"<p>Welcome back, <strong>{r.choice(FIRST_NAMES)}</strong>! Your summary is ready.</p>",
            lambda r: f"<div class=\"notice\"><p>Scheduled maintenance on {r.choice(WEEKDAYS)} at 02:00 UTC.</p></div>",
            lambda r: f"<ul><li>Backup completed ({r.randint(12, 98)} MB)</li><li>All {r.randint(4, 24)} checks passed</li></ul>",
            lambda r: f"<article class=\"card\"><h4>{r.choice(COURSES)}</h4><p>Updated study notes for {r.choice(MONTHS)} {r.randint(2025, 2026)}.</p></article>",
        ]),
        # 26: safe HTML anchor/nav
        "unusual_safe_html_anchor": fam([
            lambda r: f"<a href=\"/p/{r.randint(1, 999)}\">Page</a>",
            lambda r: f"<a href=\"https://{r.choice(CLEAN_DOMAINS)}/{r.choice(['docs', 'faq', 'blog', 'help', 'pricing'])}\">Docs</a>",
            lambda r: f"<a href=\"https://{r.choice(CLEAN_DOMAINS)}/guide/{r.randint(1, 99)}\" rel=\"noopener\">Read guide</a>",
            lambda r: f"<nav class=\"crumbs\"><a href=\"/\">Home</a> / <a href=\"/{r.choice(['shop', 'blog', 'docs', 'news'])}\">{r.choice(DEPARTMENTS)}</a></nav>",
            lambda r: f"<p>Download the <a href=\"/files/{r.choice(['report', 'summary', 'audit'])}_{r.randint(2022, 2026)}_{r.randint(1, 12)}.pdf\" class=\"btn\">annual PDF</a>.</p>",
            lambda r: f"<footer class=\"links\"><a href=\"/privacy\">Privacy</a> | <a href=\"/terms\">Terms</a> | <a href=\"/status/{r.choice(ENV_NAMES)}\">Status</a></footer>",
        ]),
        # 27: safe HTML table/code
        "unusual_safe_html_table_code": fam([
            lambda r: f"<code>v{r.randint(1, 9)}.{r.randint(0, 9)}.{r.randint(0, 9)}</code>",
            lambda r: f"<tr><td>Total</td><td>${r.randint(12, 480)}.00</td></tr>",
            lambda r: f"<span class=\"badge badge-ok\">Healthy ({r.randint(98, 100)}%)</span>",
            lambda r: f"<pre><code class=\"lang-py\">total = sum(row.qty for row in items)</code></pre>",
            lambda r: f"<table><thead><tr><th>Service</th><th>Latency</th></tr></thead><tbody><tr><td>{r.choice(SERVICES)}</td><td>{r.randint(8, 95)}ms</td></tr></tbody></table>",
            lambda r: f"<dl class=\"meta\"><dt>Region</dt><dd>{r.choice(CITIES).split(',')[0]}</dd><dt>Status</dt><dd>Active</dd></dl>",
        ]),
        # 28: JSON config
        "unusual_json_config": fam([
            lambda r: f'{{"ok":true,"id":{r.randint(1, 999)}}}',
            lambda r: f'{{"service":"{r.choice(SERVICES)}","alert":false}}',
            lambda r: f'{{"env":"{r.choice(ENV_NAMES)}","onload_ms":{r.randint(120, 900)},"gzip":true}}',
            lambda r: f'{{"service":"{r.choice(SERVICES)}","log_onerror":true,"timeout_ms":{r.randint(500, 5000)},"ssl":true}}',
            lambda r: f'{{"service":"{r.choice(SERVICES)}","alert_channel":"#ops-{r.choice(ENV_NAMES)}","max_retries":{r.randint(2, 6)}}}',
            lambda r: f'{{"telemetry":{{"track_onload":true,"sample_rate":0.{r.randint(1, 9)}}},"owner":"{r.choice(DEPARTMENTS).lower()}"}}',
        ]),
        # 29: system logs
        "unusual_system_log": fam([
            lambda r: f"[{r.choice(LOG_LEVELS)}] worker ready",
            lambda r: f"{_rand_time(r)} {r.choice(LOG_LEVELS)} {r.choice(SERVICES)}: alert cleared",
            lambda r: f"{_rand_date(r)} {_rand_time(r)} [{r.choice(LOG_LEVELS)}] {r.choice(SERVICES)} fetch_ms={r.randint(8, 220)} ok=true",
            lambda r: f"systemd[1]: Started {r.choice(SERVICES)}.service (memory={r.randint(32, 512)}MB, cpu=0.{r.randint(1, 9)}s).",
            lambda r: f"{_rand_date(r)}T{_rand_time(r)}Z level={r.choice(LOG_LEVELS).lower()} ws={r.choice(SERVICES)} onerror=timeout retry={r.randint(1, 5)}",
            lambda r: f"[{r.choice(LOG_LEVELS)}] monitor node-{r.randint(1, 12)}: fetch health check passed for {r.choice(SERVICES)} (alert=none, code={r.choice(HTTP_CODES)})",
        ]),
        # 30: StackOverflow Q&A
        "unusual_stackoverflow_qa": fam([
            lambda r: f"Cancel fetch in {r.choice(FRAMEWORKS)} #{r.randint(1, 99)}?",
            lambda r: f"Why is document.cookie empty in {r.choice(BROWSERS_LIST)}?",
            lambda r: f"How to log image onerror retries cleanly in {r.choice(FRAMEWORKS)} without an infinite loop?",
            lambda r: f"Difference between DOMContentLoaded and window onload in {r.choice(BROWSERS_LIST)} ({r.choice(ENV_NAMES)})?",
            lambda r: f"How to position a tooltip onfocus for keyboard users in {r.choice(FRAMEWORKS)} without layout shift?",
            lambda r: f"Why does fetch() keep returning HTTP {r.choice(HTTP_CODES)} when document.cookie lacks SameSite=None in {r.choice(BROWSERS_LIST)}?",
        ]),
        # 31: Python / SQL snippets
        "unusual_code_snippet_py_sql": fam([
            lambda r: f"len(rows) == {r.randint(10, 500)}",
            lambda r: f"cur.fetchmany({r.randint(10, 200)})",
            lambda r: f"counts = {{k: v for k, v in items.items() if v > {r.randint(1, 20)}}}",
            lambda r: f"SELECT status, COUNT(*) FROM orders WHERE created_at >= '{_rand_date(r)}' GROUP BY status;",
            lambda r: f"def fetch_batch(limit: int = {r.randint(10, 100)}): return cur.fetchall()[:limit]",
            lambda r: f"UPDATE inventory SET qty = qty - {r.randint(1, 5)} WHERE sku = 'SKU-{r.randint(1000, 9999)}' AND qty >= {r.randint(1, 5)};",
        ]),
        # 32: Regex / cron / glob
        "unusual_regex_expression": fam([
            lambda r: f"^[a-z0-9_-]{{{r.randint(3, 8)},{r.randint(12, 32)}}}$",
            lambda r: f"cron: {r.randint(0, 55)} */{r.randint(2, 6)} * * 1-5",
            lambda r: f"regex: ^ORD-202[4-6]-\\d{{{r.randint(4, 6)}}}$ (strict order code match)",
            lambda r: f"re.compile(r'^(?P<year>\\d{{4}})-(?P<month>\\d{{2}})-(?P<day>\\d{{2}})$')",
            lambda r: f"url_guard regex: ^(?!javascript:|data:)https?://[a-z0-9.-]+/ ({r.choice(SERVICES)})",
            lambda r: f"Validation pattern #{r.randint(10, 99)}: reject javascript: prefix and enforce ^[a-z0-9-]+$ slug under {r.randint(32, 96)} chars.",
        ]),
        # 33: Markdown docs
        "unusual_markdown_doc": fam([
            lambda r: f"Run `pytest -q` first",
            lambda r: f"Check `document.cookie` flags ({r.choice(ENV_NAMES)})",
            lambda r: f"### A11y Note: Pair `onfocus` outlines with `aria-describedby` in `{r.choice(SERVICES)}`.",
            lambda r: f"> **Lint:** Disallow `javascript:` URLs in Markdown links and `{r.choice(ENV_NAMES)}` templates.",
            lambda r: f"- [x] Defer analytics setup until after `window.onload` finishes on `{r.choice(SERVICES)}` v{r.randint(1, 5)}",
            lambda r: f"Security checklist: verify auth tokens never leak into `document.cookie` and replace `javascript:` stubs before release {r.randint(1, 9)}.",
        ]),
        # 34: CSS rule snippets
        "unusual_css_rule_snippet": fam([
            lambda r: f".btn-{r.randint(1, 99)} {{ padding: {r.randint(4, 18)}px; }}",
            lambda r: f"/* onfocus ring #{r.randint(1, 99)} */ .inp {{ outline: 2px solid #2563eb; }}",
            lambda r: f".{r.choice(['card', 'panel', 'tile', 'box'])} > .title:hover {{ color: #{_rand_hex(r, 6)}; text-decoration: underline; }}",
            lambda r: f"@media (max-width: {r.choice([480, 640, 768, 1024, 1280])}px) {{ .{r.choice(['sidebar', 'drawer', 'aside', 'rail'])} {{ display: none; }} }}",
            lambda r: f".grid-{r.randint(1, 99)} {{ display: grid; grid-template-columns: repeat({r.randint(2, 6)}, minmax(0, 1fr)); gap: {r.randint(1, 3)}rem; }}",
            lambda r: f"/* sync :focus-visible with onfocus state */ input[type='{r.choice(['search', 'text', 'email'])}'] {{ outline: 2px solid #{_rand_hex(r, 6)}; }}",
        ]),
        # 35: HTTP headers & CSP
        "unusual_http_header_csp": fam([
            lambda r: f"Cache-Control: max-age={r.randint(60, 86400)}",
            lambda r: f"X-Content-Type-Options: nosniff",
            lambda r: f"Set-Cookie: sid={_rand_hex(r, 8)}; HttpOnly; Secure # hidden from document.cookie",
            lambda r: f"Content-Security-Policy: default-src 'self'; object-src 'none'; base-uri 'self';",
            lambda r: f"URL-Sanitizer-Policy: block javascript: and vbscript: schemes on {r.choice(SERVICES)} ({r.choice(ENV_NAMES)})",
            lambda r: f"Content-Security-Policy: default-src 'self'; img-src 'self' https: data:; frame-ancestors 'none'; # blocks javascript: nav",
        ]),
        # 36: Git diff / commit messages
        "unusual_git_diff_commit": fam([
            lambda r: f"fix: trim whitespace #{r.randint(10, 999)}",
            lambda r: f"fix(a11y): clear hint onfocus #{r.randint(10, 999)}",
            lambda r: f"fix(auth): mark cookie HttpOnly so document.cookie excludes it (#{r.randint(100, 999)})",
            lambda r: f"refactor({r.choice(SERVICES)}): replace inline onerror handlers with delegated listeners",
            lambda r: f"fix(sanitizer): reject javascript: scheme in profile URLs and silence noisy alert emails (#{r.randint(100, 999)})",
            lambda r: f"feat(csp): enforce nonce-based script policy across {r.choice(ENV_NAMES)} templates and log violations to /csp-report",
        ]),
        # 37: Math / comparison expressions
        "unusual_math_formula_text": fam([
            lambda r: f"0 < x <= {r.randint(10, 999)}",
            lambda r: f"if {r.choice(['a', 'x', 'n', 'k'])} != {r.choice(['b', 'y', 'm'])} and c >= {r.randint(1, 99)}: ok",
            lambda r: f"Check #{r.randint(1, 99)}: ({r.randint(10, 89)} <= score <= 100) && (errors <= {r.randint(0, 3)})",
            lambda r: f"Formula ({r.choice(SERVICES)}): ratio = ({r.choice(['ok', 'hits', 'done'])} / total) * 100 where total > {r.randint(0, 10)}",
            lambda r: f"Inequality #{r.randint(1, 99)}: for all n >= {r.randint(2, 64)}, sum(1/k for k in 1..n) < ln(n) + 1",
            lambda r: f"Constraint ({r.choice(ENV_NAMES)}): min_v < val <= max_v and abs(delta) <= 0.0{r.randint(1, 99):02d} before commit #{r.randint(10, 99)}",
        ]),
        # 38: XML / YAML config
        "unusual_xml_yaml_config": fam([
            lambda r: f"retries: {r.randint(1, 5)} # default",
            lambda r: f"alert_enabled: true # {r.choice(ENV_NAMES)}",
            lambda r: f"<dependency><artifactId>{r.choice(SERVICES)}</artifactId><scope>test</scope></dependency>",
            lambda r: f"logging: level: {r.choice(LOG_LEVELS)} alert_webhook: slack format: json # {r.choice(ENV_NAMES)}",
            lambda r: f"<setting name=\"timeoutMs\" value=\"{r.randint(500, 8000)}\" enabled=\"true\" />",
            lambda r: f"deploy: service={r.choice(SERVICES)} replicas={r.randint(2, 8)} alert_cpu='{r.randint(75, 95)}%' mem='{r.randint(256, 1024)}Mi'",
        ]),
        # 39: Shell / DevOps CLI commands
        "unusual_shell_cli_command": fam([
            lambda r: f"df -h /var/log",
            lambda r: f"grep -rn 'timeout' /var/log/{r.choice(SERVICES)}/",
            lambda r: f"docker compose -f compose.{r.choice(ENV_NAMES)}.yml ps --format table",
            lambda r: f"tar -czf backup_{_rand_date(r)}.tar.gz ./config ./migrations --exclude='*.tmp'",
            lambda r: f"kubectl -n {r.choice(ENV_NAMES)} logs deploy/{r.choice(SERVICES)} --since={r.randint(5, 60)}m | tail -n {r.randint(20, 100)}",
            lambda r: f"psql -U readonly -d analytics -c \"SELECT count(*) FROM events WHERE day = '{_rand_date(r)}';\" --csv",
        ]),
        # 40: GraphQL / REST API query snippets
        "unusual_graphql_api_query": fam([
            lambda r: f"GET /api/v{r.randint(1, 3)}/items/{r.randint(1, 999)}",
            lambda r: f"query {{ viewer {{ id name role }} }}",
            lambda r: f"fetch('/api/v2/orders?status=shipped&limit={r.randint(10, 50)}') // {r.choice(ENV_NAMES)}",
            lambda r: f"POST /api/v1/{r.choice(SERVICES)}/jobs -> 202 Accepted (job_id=jb_{_rand_hex(r)})",
            lambda r: f"query GetInvoice($id: ID!) {{ invoice(id: $id) {{ number totalCents paidAt }} }} # fetchPolicy: cache-first",
            lambda r: f"PATCH /api/v2/accounts/{r.randint(1000, 9999)}/preferences?notify=true (If-Match: \"etag-{_rand_hex(r)}\")",
        ]),
    }


# ============================================================================
# Selection & validation
# ============================================================================

def _is_valid_candidate(
    raw: str,
    decoded: str,
    group: str,
    malicious_decoded_set: set[str],
    used_decoded_set: set[str],
) -> bool:
    if not raw or not decoded:
        return False
    if raw.strip().lower() in _PANDAS_NA_STRINGS or decoded in _PANDAS_NA_STRINGS:
        return False
    if decoded in malicious_decoded_set or decoded in used_decoded_set:
        return False
    if group == "Clean" and _CLEAN_FORBIDDEN_RE.search(raw):
        return False
    if group == "Unusual" and _UNUSUAL_FORBIDDEN_RE.search(decoded):
        return False
    return True


def _pick_closest_length(
    candidates: list[tuple[str, str, int]],
    target_len: int,
    used_decoded_set: set[str],
    rng: random.Random,
) -> tuple[str, str]:
    """Select an unused candidate whose decoded length is closest to ``target_len``."""
    best_diff: int | None = None
    best_indices: list[int] = []
    for idx, (_raw, decoded, dlen) in enumerate(candidates):
        if decoded in used_decoded_set:
            continue
        diff = abs(dlen - target_len)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_indices = [idx]
        elif diff == best_diff:
            best_indices.append(idx)

    if not best_indices:
        raise RuntimeError(f"Exhausted candidate pool while matching target_len={target_len}.")

    chosen_idx = rng.choice(best_indices)
    raw, decoded, _dlen = candidates[chosen_idx]
    used_decoded_set.add(decoded)
    return raw, decoded


def generate_benign_dataframe(
    malicious_df: pd.DataFrame,
    seed: int = SEED,
) -> pd.DataFrame:
    """Generate 4,000 unique benign rows matching the malicious decoded length distribution."""
    rng = random.Random(seed)

    malicious_decoded_list = malicious_df["payload_decoded"].astype(str).tolist()
    malicious_decoded_set = set(malicious_decoded_list)
    malicious_lengths = [len(s) for s in malicious_decoded_list]

    clean_builders = _build_clean_families()
    unusual_builders = _build_unusual_families()

    assert len(clean_builders) == NUM_CLEAN_FAMILIES, (
        f"Expected {NUM_CLEAN_FAMILIES} Clean families, got {len(clean_builders)}"
    )
    assert len(unusual_builders) == NUM_UNUSUAL_FAMILIES, (
        f"Expected {NUM_UNUSUAL_FAMILIES} Unusual families, got {len(unusual_builders)}"
    )
    assert ATTACKER_TEST_FAMILIES.issubset(unusual_builders.keys()), (
        "ATTACKER_TEST_FAMILIES must be a subset of unusual_builders"
    )

    used_decoded_set: set[str] = set()
    rows: list[dict[str, object]] = []

    # ------------------------------------------------------------------
    # 1. Clean group (2,000 samples across 40 families, 50 per family)
    # ------------------------------------------------------------------
    clean_targets = resample_target_lengths(malicious_lengths, TARGET_CLEAN_COUNT)
    rng.shuffle(clean_targets)

    clean_family_names = sorted(clean_builders.keys())
    clean_idx = 0
    for f_i, family in enumerate(clean_family_names):
        fam_targets = clean_targets[f_i * SAMPLES_PER_FAMILY : (f_i + 1) * SAMPLES_PER_FAMILY]
        raw_pool = clean_builders[family](rng)
        candidates: list[tuple[str, str, int]] = []
        pool_seen: set[str] = set()
        for raw in raw_pool:
            decoded = decode_payload(raw)
            if decoded in pool_seen:
                continue
            if not _is_valid_candidate(raw, decoded, "Clean", malicious_decoded_set, used_decoded_set):
                continue
            pool_seen.add(decoded)
            candidates.append((raw, decoded, len(decoded)))

        if len(candidates) < SAMPLES_PER_FAMILY:
            raise RuntimeError(
                f"Clean family '{family}' only produced {len(candidates)} valid candidates."
            )

        for target_len in fam_targets:
            raw, decoded = _pick_closest_length(candidates, target_len, used_decoded_set, rng)
            clean_idx += 1
            rows.append(
                {
                    "id": f"BEN_CLN_{clean_idx:04d}",
                    "payload": raw,
                    "payload_decoded": decoded,
                    "label": 0,
                    "group": "Clean",
                    "context": "benign",
                    "family": family,
                    "obfuscation": "none",
                    "source": "script_generated",
                }
            )

    # ------------------------------------------------------------------
    # 2. Unusual group (2,000 samples across 40 families, 50 per family)
    # ------------------------------------------------------------------
    unusual_targets = resample_target_lengths(malicious_lengths, TARGET_UNUSUAL_COUNT)

    # Allocate target lengths >= 34 for the 360 attacker.test slots so URLs
    # containing "https://attacker.test/..." stay completely natural without truncation.
    num_atk_slots = len(ATTACKER_TEST_FAMILIES) * ATTACKER_TEST_PER_FAMILY  # 360
    long_pool = [L for L in unusual_targets if L >= 34]
    short_pool = [L for L in unusual_targets if L < 34]
    rng.shuffle(long_pool)
    if len(long_pool) < num_atk_slots:
        raise RuntimeError("Not enough target lengths >= 34 for attacker.test slots.")

    atk_target_pool = long_pool[:num_atk_slots]
    rem_target_pool = long_pool[num_atk_slots:] + short_pool
    rng.shuffle(atk_target_pool)
    rng.shuffle(rem_target_pool)

    atk_cursor = 0
    rem_cursor = 0
    unusual_family_names = sorted(unusual_builders.keys())
    unusual_idx = 0

    for family in unusual_family_names:
        raw_pool = unusual_builders[family](rng)
        candidates_all: list[tuple[str, str, int]] = []
        pool_seen = set()
        for raw in raw_pool:
            decoded = decode_payload(raw)
            if decoded in pool_seen:
                continue
            if not _is_valid_candidate(raw, decoded, "Unusual", malicious_decoded_set, used_decoded_set):
                continue
            pool_seen.add(decoded)
            candidates_all.append((raw, decoded, len(decoded)))

        if family in ATTACKER_TEST_FAMILIES:
            atk_cands = [c for c in candidates_all if "attacker.test" in c[1]]
            alt_cands = [c for c in candidates_all if "attacker.test" not in c[1]]
            alt_needed = SAMPLES_PER_FAMILY - ATTACKER_TEST_PER_FAMILY

            if len(atk_cands) < ATTACKER_TEST_PER_FAMILY or len(alt_cands) < alt_needed:
                raise RuntimeError(
                    f"Unusual test-domain family '{family}' has insufficient candidates "
                    f"(atk={len(atk_cands)}, alt={len(alt_cands)})."
                )

            fam_atk_targets = atk_target_pool[atk_cursor : atk_cursor + ATTACKER_TEST_PER_FAMILY]
            atk_cursor += ATTACKER_TEST_PER_FAMILY
            fam_alt_targets = rem_target_pool[rem_cursor : rem_cursor + alt_needed]
            rem_cursor += alt_needed

            picked_pairs: list[tuple[str, str]] = []
            for target_len in fam_atk_targets:
                picked_pairs.append(_pick_closest_length(atk_cands, target_len, used_decoded_set, rng))
            for target_len in fam_alt_targets:
                picked_pairs.append(_pick_closest_length(alt_cands, target_len, used_decoded_set, rng))
            rng.shuffle(picked_pairs)
        else:
            non_atk_cands = [c for c in candidates_all if "attacker.test" not in c[1]]
            if len(non_atk_cands) < SAMPLES_PER_FAMILY:
                raise RuntimeError(
                    f"Unusual family '{family}' only produced {len(non_atk_cands)} valid candidates."
                )
            fam_targets = rem_target_pool[rem_cursor : rem_cursor + SAMPLES_PER_FAMILY]
            rem_cursor += SAMPLES_PER_FAMILY

            picked_pairs = [
                _pick_closest_length(non_atk_cands, target_len, used_decoded_set, rng)
                for target_len in fam_targets
            ]

        for raw, decoded in picked_pairs:
            unusual_idx += 1
            rows.append(
                {
                    "id": f"BEN_UNU_{unusual_idx:04d}",
                    "payload": raw,
                    "payload_decoded": decoded,
                    "label": 0,
                    "group": "Unusual",
                    "context": "benign",
                    "family": family,
                    "obfuscation": "none",
                    "source": "script_generated",
                }
            )

    benign_df = pd.DataFrame(rows, columns=BENIGN_COLUMNS)

    # ------------------------------------------------------------------
    # 3. Strict Verification Assertions
    # ------------------------------------------------------------------
    assert len(benign_df) == TOTAL_BENIGN_COUNT, (
        f"Expected {TOTAL_BENIGN_COUNT} benign rows, got {len(benign_df)}"
    )
    assert (benign_df["group"] == "Clean").sum() == TARGET_CLEAN_COUNT
    assert (benign_df["group"] == "Unusual").sum() == TARGET_UNUSUAL_COUNT
    assert benign_df["payload_decoded"].nunique() == TOTAL_BENIGN_COUNT, (
        "Duplicate payload_decoded detected inside benign dataset!"
    )
    overlap = set(benign_df["payload_decoded"]) & malicious_decoded_set
    assert not overlap, (
        f"Found {len(overlap)} benign payload_decoded colliding with malicious dataset!"
    )
    assert not benign_df[BENIGN_COLUMNS].isna().any().any(), (
        "NaN detected in benign dataset!"
    )
    assert not benign_df["payload"].str.contains("[attacker.test]", regex=False).any(), (
        "Unexpected '[attacker.test]' prefix detected in benign payloads!"
    )
    assert not benign_df["payload"].str.contains("env==", regex=False).any(), (
        "Unexpected 'env==' syntax detected in benign payloads!"
    )

    unusual_df = benign_df.loc[benign_df["group"] == "Unusual"]
    atk_mask = unusual_df["payload_decoded"].str.contains("attacker.test", regex=False)
    atk_count = int(atk_mask.sum())
    atk_fams = int(unusual_df.loc[atk_mask, "family"].nunique())
    assert 300 <= atk_count <= 400, f"Expected 300-400 attacker.test samples in Unusual, got {atk_count}"
    assert atk_fams >= 6, f"Expected attacker.test across >= 6 Unusual families, got {atk_fams}"

    # Verify each of the 8 high-frequency XSS keywords spans >= 3 distinct Unusual families.
    for kw in KEY_SPREAD_KEYWORDS:
        kw_mask = unusual_df["payload_decoded"].str.contains(kw, regex=False)
        kw_fams = int(unusual_df.loc[kw_mask, "family"].nunique())
        assert kw_fams >= 3, (
            f"Keyword '{kw}' only appears in {kw_fams} Unusual families (expected >= 3)."
        )

    return benign_df


def _print_inventory_and_summary(
    malicious_df: pd.DataFrame,
    mal_stats: dict[str, object],
    benign_df: pd.DataFrame,
    output_csv: Path,
) -> None:
    mal_lens = [len(s) for s in malicious_df["payload_decoded"].astype(str)]
    cln_lens = [len(s) for s in benign_df.loc[benign_df["group"] == "Clean", "payload_decoded"].astype(str)]
    unu_lens = [len(s) for s in benign_df.loc[benign_df["group"] == "Unusual", "payload_decoded"].astype(str)]

    mal_p = compute_length_percentiles(mal_lens)
    cln_p = compute_length_percentiles(cln_lens)
    unu_p = compute_length_percentiles(unu_lens)

    print("=" * 88)
    print("BƯỚC 1: KIỂM KÊ TẬP ĐỘC HẠI (oracle_results_2.csv)")
    print("=" * 88)
    print(f"  Tổng dòng gốc          : {mal_stats['raw_total']}")
    print(f"  Dòng hợp lệ (valid==1) : {mal_stats['valid_count']}")
    print(f"  Trùng lặp đã loại bỏ   : {mal_stats['duplicates_removed']}")
    print(f"  Tập độc hại duy nhất   : {mal_stats['rows_after']}")
    print(f"  Theo group             : {malicious_df['group'].value_counts().to_dict()}")
    print(f"  Theo context           : {malicious_df['context'].value_counts().to_dict()}")
    print(f"  Số family độc hại      : {malicious_df['family'].nunique()}")
    print()
    print("Phân bố độ dài payload_decoded (so sánh Malicious vs Clean vs Unusual):")
    header = (
        f"  {'Group':<10} {'N':>5} {'min':>5} {'p10':>5} {'p25':>5} {'p50':>5} "
        f"{'p75':>5} {'p90':>5} {'p95':>5} {'p99':>5} {'max':>5} {'>200(%)':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, st in [("Malicious", mal_p), ("Clean", cln_p), ("Unusual", unu_p)]:
        print(
            f"  {name:<10} {int(st['count']):>5} {st['min']:>5.0f} {st['p10']:>5.1f} "
            f"{st['p25']:>5.1f} {st['p50']:>5.1f} {st['p75']:>5.1f} {st['p90']:>5.1f} "
            f"{st['p95']:>5.1f} {st['p99']:>5.1f} {st['max']:>5.0f} {st['gt200_pct']:>7.2f}%"
        )

    unusual_df = benign_df.loc[benign_df["group"] == "Unusual"]
    clean_df = benign_df.loc[benign_df["group"] == "Clean"]
    unu_atk = int(unusual_df["payload_decoded"].str.contains("attacker.test", regex=False).sum())
    unu_atk_fams = int(
        unusual_df.loc[
            unusual_df["payload_decoded"].str.contains("attacker.test", regex=False), "family"
        ].nunique()
    )
    cln_atk = int(clean_df["payload_decoded"].str.contains("attacker.test", regex=False).sum())

    print()
    print("=" * 88)
    print("TỔNG KẾT TẬP BÌNH THƯỜNG ĐÃ SINH (benign_dataset_4000.csv)")
    print("=" * 88)
    print(f"  Tổng mẫu bình thường   : {len(benign_df)} (unique payload_decoded: {benign_df['payload_decoded'].nunique()})")
    print(f"  Clean                  : {len(clean_df)} mẫu | {clean_df['family'].nunique()} family | attacker.test: {cln_atk}")
    print(f"  Unusual                : {len(unusual_df)} mẫu | {unusual_df['family'].nunique()} family | attacker.test: {unu_atk} ({100.0 * unu_atk / len(unusual_df):.2f}%, trên {unu_atk_fams} family)")
    print(f"  Trùng với tập độc hại  : 0 (đã assert)")
    print(f"  Đã xuất file           : {output_csv}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build the 4,000-sample benign dataset aligned with oracle_results_2.csv."
    )
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE_CSV, help="Path to oracle_results_2.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV, help="Path to benign_dataset_4000.csv")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42)")
    args = parser.parse_args(argv)

    malicious_df, mal_stats = load_dedup_malicious(args.oracle)
    benign_df = generate_benign_dataframe(malicious_df, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    benign_df.to_csv(args.output, index=False, encoding="utf-8")

    _print_inventory_and_summary(malicious_df, mal_stats, benign_df, args.output)


if __name__ == "__main__":
    main()
