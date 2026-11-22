"""Automated Phase 4 evaluation of GA-bypassed XSS payloads against DVWA.

Runs the 380 valid=1 payloads from ``ga_bypassed_payloads.csv`` across four
defense configurations (C1-C4) on ``/vulnerabilities/xss_t02/index.php``, plus a
60-sample ``html_body`` cross-check on DVWA's native ``/vulnerabilities/xss_r/``.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import datetime, timezone
import http.cookiejar
import json
from pathlib import Path
import re
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
import urllib.request

XSS_PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(XSS_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(XSS_PIPELINE_DIR))

from event_actions import action_for, browser_compatibility_note, get_event_plan

try:
    import oracle as _oracle
    from playwright.async_api import async_playwright
except ModuleNotFoundError:  # Allows pure-Python unit tests without Playwright
    _oracle = None
    async_playwright = None

NAV_TIMEOUT_MS = _oracle.NAV_TIMEOUT_MS if _oracle else 4_000
ACTION_TIMEOUT_MS = _oracle.ACTION_TIMEOUT_MS if _oracle else 2_000
WAIT_MS = _oracle.WAIT_MS if _oracle else 300
LONG_WAIT_MS = _oracle.LONG_WAIT_MS if _oracle else 1_500
OVERALL_TIMEOUT_S = _oracle.OVERALL_TIMEOUT_S if _oracle else 8.0
MAX_ATTEMPTS = _oracle.MAX_ATTEMPTS if _oracle else 2

INIT_SCRIPT = (
    _oracle.INIT_SCRIPT
    if _oracle
    else ""
)

CSP_INIT_SCRIPT = """
(() => {
  window.__csp_violations = [];
  document.addEventListener('securitypolicyviolation', (e) => {
    const entry = {
      source: 'securitypolicyviolation',
      violatedDirective: e.violatedDirective || '',
      effectiveDirective: e.effectiveDirective || '',
      blockedURI: e.blockedURI || '',
      sourceFile: e.sourceFile || '',
      lineNumber: e.lineNumber || 0,
      disposition: e.disposition || '',
    };
    window.__csp_violations.push(entry);
    try {
      if (typeof window.__oracle_csp_report === 'function') {
        window.__oracle_csp_report(entry);
      }
    } catch (_) {}
  });
})();
"""

CONFIG_ORDER = ("C1", "C2", "C3", "C4")
CONFIG_SPECS: dict[str, dict[str, str]] = {
    "C1": {"csp": "0", "enc": "0", "label": "Baseline (No CSP, No Encoding)"},
    "C2": {"csp": "1", "enc": "0", "label": "CSP Only (script-src 'self'; object-src 'none')"},
    "C3": {"csp": "0", "enc": "1", "label": "Context-Aware Encoding Only"},
    "C4": {"csp": "1", "enc": "1", "label": "CSP + Context-Aware Encoding"},
}

CONTEXT_ORDER = ("html_body", "attribute", "js_string", "url")
MODEL_ORDER = ("rf", "svm", "cnn_lstm")

DEFAULT_AUDIT_PATHS = (
    "/vulnerabilities/xss_r/?name=test",
    "/vulnerabilities/xss_s/",
    "/vulnerabilities/csp/",
    "/__oracle_missing_resource__",
)

_NON_NETWORK_PREFIXES = ("about:", "data:", "blob:", "javascript:")
_TOKEN_RE_1 = re.compile(
    r"""<input\b[^>]*\bname\s*=\s*['"]user_token['"][^>]*\bvalue\s*=\s*['"]([^'"]+)['"]""",
    re.IGNORECASE,
)
_TOKEN_RE_2 = re.compile(
    r"""<input\b[^>]*\bvalue\s*=\s*['"]([^'"]+)['"][^>]*\bname\s*=\s*['"]user_token['"]""",
    re.IGNORECASE,
)


class DVWASetupRequiredError(RuntimeError):
    """Raised when DVWA requires initial database setup or login failed."""


def _effective_port(parsed) -> int | None:
    port = parsed.port
    if port is not None:
        return port
    scheme = (parsed.scheme or "").lower()
    if scheme == "http":
        return 80
    if scheme == "https":
        return 443
    return None


def is_external_url(url: str, base_url: str = "http://127.0.0.1:8080") -> bool:
    """Return True only when *url* targets a network origin outside *base_url*.

    ``about:``, ``data:``, ``blob:``, ``javascript:``, relative URLs, and URLs
    sharing scheme/hostname/port with *base_url* are treated as non-external so
    C2-C4 never record false-positive ``external-request`` signals.
    """
    if not url:
        return False
    lower = url.strip().lower()
    if lower.startswith(_NON_NETWORK_PREFIXES):
        return False
    try:
        parsed_url = urlparse(url)
        parsed_base = urlparse(base_url)
        if not parsed_url.scheme and not parsed_url.netloc:
            return False
        same_scheme = parsed_url.scheme.lower() == parsed_base.scheme.lower()
        same_host = (parsed_url.hostname or "").lower() == (parsed_base.hostname or "").lower()
        same_port = _effective_port(parsed_url) == _effective_port(parsed_base)
        return not (same_scheme and same_host and same_port)
    except ValueError:
        return True


def _extract_user_token(html: str) -> str | None:
    for pattern in (_TOKEN_RE_1, _TOKEN_RE_2):
        match = pattern.search(html)
        if match:
            return match.group(1)
    return None


def dvwa_login(
    base_url: str = "http://127.0.0.1:8080",
    username: str = "admin",
    password: str = "password",
    timeout_s: float = 10.0,
) -> tuple[urllib.request.OpenerDirector, dict[str, str]]:
    """Log into DVWA using urllib + CookieJar and return (opener, cookies)."""
    base = base_url.rstrip("/")
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

    login_url = f"{base}/login.php"
    try:
        with opener.open(login_url, timeout=timeout_s) as resp:
            final_url = resp.geturl()
            html = resp.read().decode("utf-8", errors="replace")
    except URLError as exc:
        raise RuntimeError(f"Cannot reach DVWA at {login_url}: {exc}") from exc

    if "setup.php" in final_url or "Create / Reset Database" in html:
        raise DVWASetupRequiredError(
            f"DVWA database is not initialized. Open {base}/setup.php in your browser, "
            "click 'Create / Reset Database', and re-run this script."
        )

    token = _extract_user_token(html)
    if not token:
        raise DVWASetupRequiredError(
            f"Could not extract CSRF user_token from {login_url} (final_url={final_url}). "
            f"Verify DVWA is initialized at {base}/setup.php."
        )

    form_data = urlencode({
        "username": username,
        "password": password,
        "Login": "Login",
        "user_token": token,
    }).encode("utf-8")
    req = urllib.request.Request(login_url, data=form_data, method="POST")
    with opener.open(req, timeout=timeout_s) as resp:
        post_url = resp.geturl()
        post_html = resp.read().decode("utf-8", errors="replace")

    if "setup.php" in post_url or "Create / Reset Database" in post_html:
        raise DVWASetupRequiredError(
            f"DVWA redirected to setup.php during login. Open {base}/setup.php, "
            "click 'Create / Reset Database', and re-run."
        )
    if "login.php" in post_url:
        raise DVWASetupRequiredError(
            f"DVWA login failed for user {username!r} at {login_url}. "
            f"Check credentials or initialize the database at {base}/setup.php."
        )

    parsed = urlparse(base)
    domain = parsed.hostname or "127.0.0.1"
    jar.set_cookie(
        http.cookiejar.Cookie(
            version=0,
            name="security",
            value="low",
            port=None,
            port_specified=False,
            domain=domain,
            domain_specified=False,
            domain_initial_dot=False,
            path="/",
            path_specified=True,
            secure=False,
            expires=None,
            discard=True,
            comment=None,
            comment_url=None,
            rest={},
        )
    )

    phpsessid = None
    for c in jar:
        if c.name == "PHPSESSID":
            phpsessid = c.value
            break
    if not phpsessid:
        raise RuntimeError("DVWA login succeeded without issuing PHPSESSID cookie.")

    return opener, {"PHPSESSID": phpsessid, "security": "low"}


def audit_default_dvwa_headers(
    opener: urllib.request.OpenerDirector,
    base_url: str = "http://127.0.0.1:8080",
    timeout_s: float = 10.0,
) -> list[dict[str, Any]]:
    """Fetch default DVWA endpoints to record HTTP status and CSP headers."""
    base = base_url.rstrip("/")
    results: list[dict[str, Any]] = []
    for path in DEFAULT_AUDIT_PATHS:
        full_url = f"{base}{path}"
        status = None
        headers_map: dict[str, str] = {}
        final_url = full_url
        try:
            with opener.open(full_url, timeout=timeout_s) as resp:
                status = resp.status
                final_url = resp.geturl()
                headers_map = {k.lower(): v for k, v in resp.headers.items()}
                resp.read()
        except HTTPError as err:
            status = err.code
            final_url = err.geturl()
            headers_map = {k.lower(): v for k, v in err.headers.items()} if err.headers else {}
        results.append({
            "path": path,
            "final_url": final_url,
            "http_status": status,
            "content_security_policy": headers_map.get("content-security-policy"),
            "x_xss_protection": headers_map.get("x-xss-protection"),
            "x_frame_options": headers_map.get("x-frame-options"),
            "content_type": headers_map.get("content-type"),
        })
    return results


def _resolve_carrier(row: dict[str, Any]) -> str:
    carrier = (row.get("carrier") or "").strip()
    if carrier:
        return carrier
    target_tag = (row.get("target_tag") or "").strip()
    if target_tag in {"iframe", "object"}:
        return target_tag
    family_lower = (row.get("family") or "").strip().lower()
    if family_lower.startswith("iframe_srcdoc"):
        return "iframe"
    if family_lower.startswith("object_data"):
        return "object"
    return "a"


def load_csv_cases(csv_path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load and validate GA-bypassed payload cases from CSV."""
    path = Path(csv_path)
    if not path.exists():
        alt = XSS_PIPELINE_DIR / path
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"id", "context", "family", "event", "payload"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")
        cases: list[dict[str, Any]] = []
        for row in reader:
            event = str(row["event"])
            get_event_plan(event)
            context = str(row["context"])
            if context not in CONTEXT_ORDER:
                raise ValueError(f"Unsupported context in CSV: {context!r}")
            cases.append({
                "id": str(row["id"]),
                "context": context,
                "family": str(row["family"]),
                "event": event,
                "carrier": _resolve_carrier(row),
                "target_tag": str(row.get("target_tag") or ""),
                "payload": str(row["payload"]),
                "models_bypassed": str(row.get("models_bypassed") or ""),
                "bypass_rf": str(row.get("bypass_rf") or "0"),
                "bypass_svm": str(row.get("bypass_svm") or "0"),
                "bypass_cnn_lstm": str(row.get("bypass_cnn_lstm") or "0"),
            })
            if limit is not None and len(cases) >= limit:
                break
    return cases


def build_t02_url(base_url: str, case: dict[str, Any], config_name: str) -> str:
    """Build URL for /vulnerabilities/xss_t02/index.php for a given config."""
    cfg = CONFIG_SPECS[config_name]
    params = {
        "payload": case["payload"],
        "ctx": case["context"],
        "family": case["family"],
        "event": case["event"],
        "carrier": case.get("carrier") or "a",
        "csp": cfg["csp"],
        "enc": cfg["enc"],
    }
    return f"{base_url.rstrip('/')}/vulnerabilities/xss_t02/index.php?{urlencode(params)}"


def build_xss_r_url(base_url: str, case: dict[str, Any]) -> str:
    """Build URL for DVWA native reflected XSS endpoint."""
    params = {"name": case["payload"]}
    return f"{base_url.rstrip('/')}/vulnerabilities/xss_r/?{urlencode(params)}"


async def _find_in_containers(page, selector_suffix: str, containers: tuple[str, ...]):
    for container in containers:
        if await page.locator(container).count():
            loc = page.locator(f"{container} {selector_suffix}")
            if await loc.count():
                return loc
    return page.locator(selector_suffix)


async def _event_target_scoped(page, event: str, containers: tuple[str, ...]):
    """Locate the injected element carrying *event* inside payload containers."""
    for container in containers:
        if await page.locator(container).count():
            handles = await page.locator(f"{container} *").element_handles()
            for handle in handles:
                if await handle.evaluate("(el, ev) => el.hasAttribute(ev)", event):
                    return handle
    handles = await page.locator("body *").element_handles()
    for handle in handles:
        if await handle.evaluate("(el, ev) => el.hasAttribute(ev)", event):
            return handle
    return None


async def _perform_action_dvwa(
    page,
    case: dict[str, Any],
    containers: tuple[str, ...] = ("#oracle-host", "div.vulnerable_code_area"),
) -> str | None:
    event = case["event"]
    action = action_for(event, case["context"], case["family"])
    if action in {"none", "wait"}:
        return None
    if action == "wait_long":
        await page.wait_for_timeout(LONG_WAIT_MS)
        return None
    if action == "open_url":
        await page.locator("#payload-target").click(timeout=ACTION_TIMEOUT_MS)
        note = browser_compatibility_note(case["payload"], case["context"], case["family"])
        return note or None
    if action == "click_link":
        for container in containers:
            if await page.locator(container).count():
                link = page.locator(f"{container} a[href], {container} svg a")
                if await link.count():
                    await link.first.click(force=True, timeout=ACTION_TIMEOUT_MS)
                    return None
        return f"no link element in {'/'.join(containers)}"
    if action == "transition":
        target = await _event_target_scoped(page, event, containers)
        if target is None:
            return f"no element with {event}"
        await target.evaluate("""
            async (target) => {
              if (!target.id) throw new Error('transition target has no id');
              getComputedStyle(target).color;
              await new Promise(requestAnimationFrame);
              location.hash = target.id;
              await new Promise(requestAnimationFrame);
            }
        """)
        return None
    if action == "popover":
        button = await _find_in_containers(page, "button[popovertarget]", containers)
        if await button.count():
            await button.first.click(timeout=ACTION_TIMEOUT_MS)
        return None

    target = await _event_target_scoped(page, event, containers)
    if target is None:
        return f"no element with {event}"
    if action == "hover":
        await target.hover(force=True, timeout=ACTION_TIMEOUT_MS)
    elif action == "focus":
        await target.focus()
    elif action == "click":
        await target.evaluate("el => el.click()")
    elif action == "dblclick":
        await target.dblclick(force=True, timeout=ACTION_TIMEOUT_MS)
    elif action == "mousedown":
        await target.hover(force=True, timeout=ACTION_TIMEOUT_MS)
        await page.mouse.down()
        await page.mouse.up()
    elif action == "keyup":
        await target.focus()
        await page.keyboard.press("A")
    elif action == "input":
        await target.fill("oracle-input", timeout=ACTION_TIMEOUT_MS)
    elif action == "change":
        initial_value = await target.input_value()
        next_value = "oracle-change" if initial_value != "oracle-change" else "oracle-change-2"
        await target.fill(next_value, timeout=ACTION_TIMEOUT_MS)
        await target.evaluate("el => el.blur()")
    elif action == "toggle":
        await target.click(force=True, timeout=ACTION_TIMEOUT_MS)
    else:
        raise AssertionError(f"Unknown Oracle action: {action}")
    return None


def _is_dvwa_auth_redirect(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith("/login.php") or path.endswith("/setup.php")


async def _check_once_dvwa(
    browser_context,
    case: dict[str, Any],
    target_url: str,
    route_prefix: str,
    base_url: str,
    containers: tuple[str, ...],
) -> dict[str, Any]:
    page = await browser_context.new_page()
    signals: list[str] = []
    csp_violations: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    fired_event = asyncio.Event()
    watch_navigation = False
    http_status: int | None = None
    csp_header: str | None = None

    def record_signal(sig: str) -> None:
        signals.append(sig)
        fired_event.set()

    async def report(_source, name):
        record_signal(str(name))

    async def report_csp(_source, entry):
        if isinstance(entry, dict):
            csp_violations.append(entry)
        else:
            csp_violations.append({"source": "securitypolicyviolation", "raw": str(entry)})

    def on_dialog(dialog):
        record_signal(f"dialog:{dialog.type}")
        asyncio.create_task(dialog.dismiss())

    def on_pageerror(error):
        diagnostics.append(f"pageerror: {error}")

    def on_console(msg):
        text = msg.text or ""
        if "Content Security Policy" in text or "Refused to" in text:
            csp_violations.append({
                "source": "console",
                "type": msg.type,
                "text": text,
            })

    def on_response(response):
        nonlocal http_status, csp_header
        try:
            if (
                response.request.is_navigation_request()
                and response.frame == page.main_frame
                and response.url.startswith(route_prefix)
            ):
                http_status = response.status
                csp_header = response.headers.get("content-security-policy")
        except Exception:
            pass

    def on_framenavigated(frame):
        url = frame.url
        if not url or url in {"about:blank", "about:srcdoc"} or url.startswith(route_prefix):
            return
        if _is_dvwa_auth_redirect(url):
            diagnostics.append(f"auth_redirect: {url}")
            return
        phase = "navigation" if watch_navigation else "navigation-during-load"
        if frame == page.main_frame:
            record_signal(f"{phase}:{url}")
        elif is_external_url(url, base_url):
            record_signal(f"subframe-{phase}:{url}")

    def on_request(request):
        if not request.is_navigation_request():
            return
        url = request.url
        if not url or url in {"about:blank", "about:srcdoc"} or url.startswith(route_prefix):
            return
        if _is_dvwa_auth_redirect(url):
            return
        if request.frame == page.main_frame:
            record_signal(f"navigation-request:{url}")
        elif is_external_url(url, base_url):
            record_signal(f"subframe-navigation-request:{url}")

    async def handle_route(route):
        url = route.request.url
        if not is_external_url(url, base_url):
            try:
                await route.continue_()
            except Exception:
                pass
            return
        record_signal(f"external-request:{url}")
        try:
            await route.abort("aborted")
        except Exception:
            pass

    page.on("dialog", on_dialog)
    page.on("pageerror", on_pageerror)
    page.on("console", on_console)
    page.on("response", on_response)
    page.on("framenavigated", on_framenavigated)
    page.on("request", on_request)
    await browser_context.route("**/*", handle_route)
    await page.expose_binding("__oracle_report", report)
    await page.expose_binding("__oracle_csp_report", report_csp)
    await page.add_init_script(INIT_SCRIPT)
    await page.add_init_script(CSP_INIT_SCRIPT)

    try:
        goto_task = asyncio.create_task(
            page.goto(target_url, timeout=NAV_TIMEOUT_MS, wait_until="load")
        )
        fired_task = asyncio.create_task(fired_event.wait())
        _, pending = await asyncio.wait(
            {goto_task, fired_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        if fired_event.is_set() or signals:
            if goto_task.done() and not goto_task.cancelled():
                resp = goto_task.result() if not goto_task.exception() else None
                if resp is not None and http_status is None:
                    http_status = resp.status
                    csp_header = resp.headers.get("content-security-policy")
            await asyncio.sleep(0.05)
            return {
                "executed": 1,
                "signals": signals,
                "csp_header": csp_header,
                "csp_violations": csp_violations,
                "http_status": http_status,
                "error": None,
            }

        try:
            resp = goto_task.result()
            if resp is not None and http_status is None:
                http_status = resp.status
                csp_header = resp.headers.get("content-security-policy")
        except Exception:
            if not signals:
                raise

        if _is_dvwa_auth_redirect(page.url):
            raise DVWASetupRequiredError(f"DVWA session redirected to {page.url}")

        if not page.url.startswith(route_prefix):
            record_signal(f"navigation-during-load:{page.url}")
        if signals:
            return {
                "executed": 1,
                "signals": signals,
                "csp_header": csp_header,
                "csp_violations": csp_violations,
                "http_status": http_status,
                "error": None,
            }

        watch_navigation = True
        action_error = None
        try:
            action_error = await _perform_action_dvwa(page, case, containers=containers)
        except Exception as exc:
            if signals:
                return {
                    "executed": 1,
                    "signals": signals,
                    "csp_header": csp_header,
                    "csp_violations": csp_violations,
                    "http_status": http_status,
                    "error": f"action interrupted after side effect: {exc}",
                }
            raise

        if signals:
            return {
                "executed": 1,
                "signals": signals,
                "csp_header": csp_header,
                "csp_violations": csp_violations,
                "http_status": http_status,
                "error": None,
            }

        await page.wait_for_timeout(WAIT_MS)
        fired = bool(signals)
        if not page.is_closed():
            try:
                if not fired:
                    fired = bool(await page.evaluate("Boolean(window.__xss_fired)"))
                    if fired and not signals:
                        signals.append("window.__xss_fired")
                if not csp_violations:
                    dom_csp = await page.evaluate("window.__csp_violations || []")
                    if isinstance(dom_csp, list) and dom_csp:
                        csp_violations.extend(dom_csp)
            except Exception as exc:
                if signals:
                    return {
                        "executed": 1,
                        "signals": signals,
                        "csp_header": csp_header,
                        "csp_violations": csp_violations,
                        "http_status": http_status,
                        "error": f"evaluate interrupted after side effect: {exc}",
                    }
                raise

        err = action_error
        if not fired and err is None and diagnostics:
            err = "; ".join(diagnostics[:3])
        return {
            "executed": 1 if fired else 0,
            "signals": signals,
            "csp_header": csp_header,
            "csp_violations": csp_violations,
            "http_status": http_status,
            "error": err,
        }
    finally:
        await page.close()


async def _check_with_retry_dvwa(
    browser,
    playwright_cookies: list[dict[str, Any]],
    case: dict[str, Any],
    target_url: str,
    route_prefix: str,
    base_url: str,
    containers: tuple[str, ...],
) -> dict[str, Any]:
    attempts = 0
    last_error = None
    t0 = time.perf_counter()
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        browser_context = await browser.new_context()
        try:
            await browser_context.add_cookies(playwright_cookies)
            outcome = await asyncio.wait_for(
                _check_once_dvwa(
                    browser_context,
                    case=case,
                    target_url=target_url,
                    route_prefix=route_prefix,
                    base_url=base_url,
                    containers=containers,
                ),
                timeout=OVERALL_TIMEOUT_S,
            )
            outcome["elapsed_ms"] = round((time.perf_counter() - t0) * 1000)
            outcome["attempts"] = attempts
            return outcome
        except DVWASetupRequiredError:
            raise
        except Exception as exc:
            last_error = str(exc)
        finally:
            await browser_context.close()
    return {
        "executed": 0,
        "signals": [],
        "csp_header": None,
        "csp_violations": [],
        "http_status": None,
        "error": last_error,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000),
        "attempts": attempts,
    }


def _make_log_record(case: dict[str, Any], config_name: str, outcome: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "id": case["id"],
        "config": config_name,
        "context": case["context"],
        "family": case["family"],
        "event": case["event"],
        "carrier": case.get("carrier") or "",
        "models_bypassed": case.get("models_bypassed") or "",
        "bypass_rf": case.get("bypass_rf", "0"),
        "bypass_svm": case.get("bypass_svm", "0"),
        "bypass_cnn_lstm": case.get("bypass_cnn_lstm", "0"),
        "payload": case["payload"],
        "executed": int(outcome["executed"]),
        "signals": outcome.get("signals") or [],
        "csp_header": outcome.get("csp_header"),
        "csp_violations": outcome.get("csp_violations") or [],
        "http_status": outcome.get("http_status"),
        "elapsed_ms": outcome.get("elapsed_ms", 0),
        "error": outcome.get("error"),
    }


async def run_evaluation(
    cases: list[dict[str, Any]],
    cookies_dict: dict[str, str],
    base_url: str,
    configs: list[str],
    concurrency: int,
    eval_log_path: Path,
    crosscheck_log_path: Path,
    skip_crosscheck: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute Playwright evaluation across configs and native xss_r crosscheck."""
    if _oracle is None or async_playwright is None:
        raise RuntimeError(
            "Playwright and oracle.py are required for browser evaluation. "
            "Run with the project virtualenv (e.g. ./venv/bin/python)."
        )
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    parsed = urlparse(base_url)
    cookie_domain = parsed.hostname or "127.0.0.1"
    playwright_cookies = [
        {"name": name, "value": value, "domain": cookie_domain, "path": "/"}
        for name, value in cookies_dict.items()
    ]

    eval_log_path.parent.mkdir(parents=True, exist_ok=True)
    eval_log_path.write_text("", encoding="utf-8")
    if not skip_crosscheck:
        crosscheck_log_path.write_text("", encoding="utf-8")

    t02_route_prefix = f"{base_url.rstrip('/')}/vulnerabilities/xss_t02/index.php"
    xss_r_route_prefix = f"{base_url.rstrip('/')}/vulnerabilities/xss_r/"

    eval_tasks_spec = [(cfg, case) for cfg in configs for case in cases]
    eval_records: list[dict[str, Any] | None] = [None] * len(eval_tasks_spec)
    crosscheck_cases = [c for c in cases if c["context"] == "html_body"] if not skip_crosscheck else []
    crosscheck_records: list[dict[str, Any] | None] = [None] * len(crosscheck_cases)

    sem = asyncio.Semaphore(concurrency)
    eval_lock = asyncio.Lock()
    cross_lock = asyncio.Lock()

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        try:
            async def eval_worker(idx: int, cfg: str, case: dict[str, Any]):
                async with sem:
                    outcome = await _check_with_retry_dvwa(
                        browser=browser,
                        playwright_cookies=playwright_cookies,
                        case=case,
                        target_url=build_t02_url(base_url, case, cfg),
                        route_prefix=t02_route_prefix,
                        base_url=base_url,
                        containers=("#oracle-host",),
                    )
                    rec = _make_log_record(case, cfg, outcome)
                    eval_records[idx] = rec
                    async with eval_lock:
                        with eval_log_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            await asyncio.gather(*(
                eval_worker(i, cfg, case)
                for i, (cfg, case) in enumerate(eval_tasks_spec)
            ))

            if crosscheck_cases:
                async def cross_worker(idx: int, case: dict[str, Any]):
                    async with sem:
                        outcome = await _check_with_retry_dvwa(
                            browser=browser,
                            playwright_cookies=playwright_cookies,
                            case=case,
                            target_url=build_xss_r_url(base_url, case),
                            route_prefix=xss_r_route_prefix,
                            base_url=base_url,
                            containers=("div.vulnerable_code_area", "#oracle-host"),
                        )
                        rec = _make_log_record(case, "DVWA_XSS_R_LOW", outcome)
                        crosscheck_records[idx] = rec
                        async with cross_lock:
                            with crosscheck_log_path.open("a", encoding="utf-8") as f:
                                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                await asyncio.gather(*(
                    cross_worker(i, case) for i, case in enumerate(crosscheck_cases)
                ))
        finally:
            await browser.close()

    return (
        [r for r in eval_records if r is not None],
        [r for r in crosscheck_records if r is not None],
    )


def parse_models_bypassed(record: dict[str, Any]) -> list[str]:
    """Return list of bypassed model names for a case or log record."""
    found: list[str] = []
    for model in MODEL_ORDER:
        flag = str(record.get(f"bypass_{model}", "")).strip().lower()
        if flag in {"1", "true"}:
            found.append(model)
    if found:
        return found
    raw = str(record.get("models_bypassed") or "").strip()
    if not raw:
        return []
    tokens = [tok.strip().lower() for tok in re.split(r"[,;+|]+", raw) if tok.strip()]
    return [tok for tok in tokens if tok in MODEL_ORDER]


def _block_stats(executed: int, total: int, c1_executed: int) -> dict[str, Any]:
    blocked_vs_c1 = c1_executed - executed
    block_rate = (blocked_vs_c1 / c1_executed * 100.0) if c1_executed > 0 else 0.0
    return {
        "executed": executed,
        "total": total,
        "c1_executed": c1_executed,
        "blocked_vs_c1": blocked_vs_c1,
        "block_rate_vs_c1_pct": round(block_rate, 2),
    }


def summarize_results(
    eval_records: list[dict[str, Any]],
    crosscheck_records: list[dict[str, Any]] | None = None,
    default_dvwa_headers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate JSONL evaluation records into the Phase 4 report structure."""
    crosscheck_records = crosscheck_records or []
    present_configs = [cfg for cfg in CONFIG_ORDER if any(r["config"] == cfg for r in eval_records)]

    by_cfg_ctx: dict[str, dict[str, list[dict[str, Any]]]] = {
        cfg: {ctx: [] for ctx in (*CONTEXT_ORDER, "ALL")} for cfg in present_configs
    }
    by_cfg_model: dict[str, dict[str, list[dict[str, Any]]]] = {
        cfg: {m: [] for m in MODEL_ORDER} for cfg in present_configs
    }

    for rec in eval_records:
        cfg = rec["config"]
        if cfg not in by_cfg_ctx:
            continue
        ctx = rec["context"]
        if ctx in by_cfg_ctx[cfg]:
            by_cfg_ctx[cfg][ctx].append(rec)
        by_cfg_ctx[cfg]["ALL"].append(rec)
        for model in parse_models_bypassed(rec):
            by_cfg_model[cfg][model].append(rec)

    c1_ctx_executed = {
        ctx: sum(int(r["executed"]) for r in by_cfg_ctx.get("C1", {}).get(ctx, []))
        for ctx in (*CONTEXT_ORDER, "ALL")
    }
    config_by_context: dict[str, dict[str, dict[str, Any]]] = {}
    for cfg in present_configs:
        config_by_context[cfg] = {}
        for ctx in (*CONTEXT_ORDER, "ALL"):
            recs = by_cfg_ctx[cfg][ctx]
            executed = sum(int(r["executed"]) for r in recs)
            c1_exec = c1_ctx_executed[ctx] if "C1" in present_configs else len(recs)
            config_by_context[cfg][ctx] = _block_stats(executed, len(recs), c1_exec)

    c1_model_executed = {
        m: sum(int(r["executed"]) for r in by_cfg_model.get("C1", {}).get(m, []))
        for m in MODEL_ORDER
    }
    config_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for cfg in present_configs:
        config_by_model[cfg] = {}
        for m in MODEL_ORDER:
            recs = by_cfg_model[cfg][m]
            executed = sum(int(r["executed"]) for r in recs)
            c1_exec = c1_model_executed[m] if "C1" in present_configs else len(recs)
            config_by_model[cfg][m] = _block_stats(executed, len(recs), c1_exec)

    c1_html = {r["id"]: int(r["executed"]) for r in by_cfg_ctx.get("C1", {}).get("html_body", [])}
    xss_r_map = {r["id"]: int(r["executed"]) for r in crosscheck_records}
    shared_ids = sorted(set(c1_html) | set(xss_r_map))
    both_exec = [i for i in shared_ids if c1_html.get(i) == 1 and xss_r_map.get(i) == 1]
    c1_only = [i for i in shared_ids if c1_html.get(i) == 1 and xss_r_map.get(i, 0) == 0]
    xss_r_only = [i for i in shared_ids if c1_html.get(i, 0) == 0 and xss_r_map.get(i) == 1]
    neither = [i for i in shared_ids if c1_html.get(i, 0) == 0 and xss_r_map.get(i, 0) == 0]

    crosscheck_summary = {
        "c1_html_body": {
            "executed": sum(c1_html.values()),
            "total": len(c1_html),
        },
        "dvwa_xss_r": {
            "executed": sum(xss_r_map.values()),
            "total": len(xss_r_map),
        },
        "both_executed_count": len(both_exec),
        "c1_only_ids": c1_only,
        "xss_r_only_ids": xss_r_only,
        "neither_executed_ids": neither,
    }

    c2_iframe_cases = [
        r for r in eval_records
        if r["config"] == "C2"
        and r["context"] == "url"
        and (
            str(r.get("family", "")).lower().startswith("iframe_srcdoc")
            or "data:text/html" in str(r.get("payload", "")).lower()
        )
    ]
    c2_iframe_exec = sum(int(r["executed"]) for r in c2_iframe_cases)
    c2_iframe_observation = {
        "total": len(c2_iframe_cases),
        "executed": c2_iframe_exec,
        "blocked": len(c2_iframe_cases) - c2_iframe_exec,
        "cases": [
            {
                "id": r["id"],
                "family": r["family"],
                "carrier": r.get("carrier", ""),
                "payload": r["payload"],
                "executed": int(r["executed"]),
                "signals": r.get("signals", []),
                "csp_header": r.get("csp_header"),
                "csp_violations": r.get("csp_violations", []),
            }
            for r in c2_iframe_cases
        ],
    }

    c1_not_executed = [
        {
            "id": r["id"],
            "context": r["context"],
            "family": r["family"],
            "event": r["event"],
            "carrier": r.get("carrier", ""),
            "models_bypassed": r.get("models_bypassed", ""),
            "payload": r["payload"],
            "signals": r.get("signals", []),
            "error": r.get("error"),
        }
        for r in eval_records
        if r["config"] == "C1" and int(r["executed"]) == 0
    ]

    bypassed_defenses: dict[str, list[dict[str, Any]]] = {}
    for cfg in ("C2", "C3", "C4"):
        if cfg not in present_configs:
            continue
        bypassed_defenses[cfg] = [
            {
                "id": r["id"],
                "context": r["context"],
                "family": r["family"],
                "event": r["event"],
                "carrier": r.get("carrier", ""),
                "models_bypassed": r.get("models_bypassed", ""),
                "payload": r["payload"],
                "signals": r.get("signals", []),
                "csp_header": r.get("csp_header"),
                "csp_violations": r.get("csp_violations", []),
            }
            for r in eval_records
            if r["config"] == cfg and int(r["executed"]) == 1
        ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "default_dvwa_headers": default_dvwa_headers or [],
        "config_by_context": config_by_context,
        "config_by_model": config_by_model,
        "crosscheck_html_body": crosscheck_summary,
        "c2_iframe_data_observation": c2_iframe_observation,
        "audit": {
            "c1_not_executed": c1_not_executed,
            "bypassed_defenses": bypassed_defenses,
        },
    }


def format_report_tables(report: dict[str, Any]) -> str:
    """Format the aggregated Phase 4 report into readable stdout tables."""
    lines: list[str] = []

    default_headers = report.get("default_dvwa_headers") or []
    if default_headers:
        lines.append("=== 0. AUDIT HEADER MAC DINH CUA DVWA (BUOC 4.1) ===")
        lines.append(f"{'Path':<38} | {'Status':<6} | {'Content-Security-Policy'}")
        lines.append("-" * 96)
        for item in default_headers:
            csp = item.get("content_security_policy") or "<none>"
            lines.append(f"{item['path']:<38} | {str(item.get('http_status')):<6} | {csp}")
        lines.append("")

    lines.append("=== 1. BANG 4 CAU HINH (C1-C4) x 4 NGU CANH ===")
    lines.append(f"{'Config':<8} | {'Context':<12} | {'Executed/Total':<15} | {'Blocked vs C1'}")
    lines.append("-" * 64)
    cfg_ctx = report.get("config_by_context") or {}
    for cfg, ctx_map in cfg_ctx.items():
        for ctx in (*CONTEXT_ORDER, "ALL"):
            if ctx not in ctx_map:
                continue
            s = ctx_map[ctx]
            frac = f"{s['executed']}/{s['total']}"
            blk = f"{s['blocked_vs_c1']} ({s['block_rate_vs_c1_pct']:.2f}%)"
            lines.append(f"{cfg:<8} | {ctx:<12} | {frac:<15} | {blk}")
    lines.append("")

    lines.append("=== 2. BANG 4 CAU HINH (C1-C4) x 3 MO HINH BI BYPASS ===")
    lines.append(f"{'Config':<8} | {'Model':<10} | {'Executed/Total':<15} | {'Blocked vs C1'}")
    lines.append("-" * 62)
    cfg_mod = report.get("config_by_model") or {}
    for cfg, mod_map in cfg_mod.items():
        for model in MODEL_ORDER:
            if model not in mod_map:
                continue
            s = mod_map[model]
            frac = f"{s['executed']}/{s['total']}"
            blk = f"{s['blocked_vs_c1']} ({s['block_rate_vs_c1_pct']:.2f}%)"
            lines.append(f"{cfg:<8} | {model:<10} | {frac:<15} | {blk}")
    lines.append("")

    cross = report.get("crosscheck_html_body") or {}
    lines.append("=== 3. DOI CHIEU CHEO 60 MAU html_body (/vulnerabilities/xss_r/ vs C1) ===")
    c1_hb = cross.get("c1_html_body", {})
    xr_hb = cross.get("dvwa_xss_r", {})
    lines.append(
        f"C1 html_body: {c1_hb.get('executed', 0)}/{c1_hb.get('total', 0)} | "
        f"DVWA xss_r (low): {xr_hb.get('executed', 0)}/{xr_hb.get('total', 0)} | "
        f"Both executed: {cross.get('both_executed_count', 0)}"
    )
    if cross.get("c1_only_ids"):
        lines.append(f"  C1 only: {cross['c1_only_ids']}")
    if cross.get("xss_r_only_ids"):
        lines.append(f"  xss_r only: {cross['xss_r_only_ids']}")
    lines.append("")

    obs = report.get("c2_iframe_data_observation") or {}
    lines.append("=== 4. QUAN SAT DAC BIET: <iframe src=\"data:text/html,...\"> O CAU HINH C2 ===")
    lines.append(
        f"Total: {obs.get('total', 0)} | Executed: {obs.get('executed', 0)} | "
        f"Blocked: {obs.get('blocked', 0)}"
    )
    for item in (obs.get("cases") or [])[:5]:
        lines.append(
            f"  - {item['id']}: executed={item['executed']}, signals={item['signals']}, "
            f"csp_violations={item['csp_violations']}"
        )
    lines.append("")

    audit = report.get("audit") or {}
    lines.append("=== 5. KIEM TOAN MAU BAT THUONG ===")
    c1_miss = audit.get("c1_not_executed") or []
    lines.append(f"C1 not executed ({len(c1_miss)}):")
    for item in c1_miss:
        lines.append(
            f"  - {item['id']} [{item['context']}/{item['family']}/{item['event']}]: "
            f"error={item.get('error')} | payload={item['payload'][:80]!r}"
        )
    for cfg in ("C2", "C3", "C4"):
        byp = (audit.get("bypassed_defenses") or {}).get(cfg)
        if byp is None:
            continue
        lines.append(f"{cfg} bypassed ({len(byp)}):")
        for item in byp:
            lines.append(
                f"  - {item['id']} [{item['context']}/{item['family']}/{item['event']}]: "
                f"signals={item['signals']} | payload={item['payload'][:80]!r}"
            )
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    default_csv = XSS_PIPELINE_DIR / "ga_bypassed_payloads.csv"
    default_out = Path(__file__).resolve().parent / "results"
    parser = argparse.ArgumentParser(description="Phase 4 DVWA XSS Evaluation")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--csv", default=str(default_csv))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(CONFIG_ORDER),
        choices=list(CONFIG_ORDER),
    )
    parser.add_argument("--skip-crosscheck", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", default=str(default_out))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    cases = load_csv_cases(args.csv, limit=args.limit)
    print(f"Loaded {len(cases)} cases from {args.csv}")

    try:
        opener, cookies = dvwa_login(base_url=args.base_url)
    except DVWASetupRequiredError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    default_headers = audit_default_dvwa_headers(opener, base_url=args.base_url)
    missing_audit = next((h for h in default_headers if h["path"] == "/__oracle_missing_resource__"), None)
    if missing_audit and missing_audit.get("http_status") != 404:
        print(
            f"[WARNING] Expected HTTP 404 for /__oracle_missing_resource__, got {missing_audit.get('http_status')}",
            file=sys.stderr,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_log_path = out_dir / "dvwa_eval_log.jsonl"
    crosscheck_log_path = out_dir / "dvwa_xss_r_crosscheck.jsonl"
    report_path = out_dir / "phase4_dvwa_report.json"

    eval_records, crosscheck_records = asyncio.run(
        run_evaluation(
            cases=cases,
            cookies_dict=cookies,
            base_url=args.base_url,
            configs=list(args.configs),
            concurrency=args.concurrency,
            eval_log_path=eval_log_path,
            crosscheck_log_path=crosscheck_log_path,
            skip_crosscheck=args.skip_crosscheck,
        )
    )

    report = summarize_results(
        eval_records=eval_records,
        crosscheck_records=crosscheck_records,
        default_dvwa_headers=default_headers,
    )
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(format_report_tables(report))
    print(f"\nSaved logs: {eval_log_path}")
    if not args.skip_crosscheck:
        print(f"Saved crosscheck: {crosscheck_log_path}")
    print(f"Saved report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
