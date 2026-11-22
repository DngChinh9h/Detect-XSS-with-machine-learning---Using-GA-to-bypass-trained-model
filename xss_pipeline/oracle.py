"""Playwright oracle dung chung cho filter dataset va GA.

Moi case co ``id, payload, context, event, family``.  Oracle khong gia DOM
event: no tim element da duoc payload tao ra va goi thao tac Playwright that
(hover/focus/click/type/...).
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

from playwright.async_api import async_playwright

from event_actions import action_for, browser_compatibility_note, get_event_plan

BASE_URL = "http://127.0.0.1:5000"
CONTEXT_ROUTES = {
    "html_body": "/html_body", "attribute": "/attribute",
    "js_string": "/js_string", "url": "/url",
}
NAV_TIMEOUT_MS = 4_000
ACTION_TIMEOUT_MS = 2_000
WAIT_MS = 300
LONG_WAIT_MS = 1_500
OVERALL_TIMEOUT_S = 8.0
MAX_ATTEMPTS = 2

# Bao phu cac side effect thuong gap trong CSV, khong chi alert/confirm.
# Binding giu bang chung o Python ngay ca khi payload dieu huong document di.
INIT_SCRIPT = """
(() => {
  const mark = (name) => { try { window.__oracle_report(name); } catch (_) {} };
  window.__xss_fired = false;
  const fired = (name) => { window.__xss_fired = true; mark(name); };
  for (const name of ['alert', 'confirm', 'prompt', 'print']) {
    window[name] = function () { fired(name); return name === 'confirm' ? true : undefined; };
  }
  window.__markXSS = () => fired('__markXSS');
  const oldFetch = window.fetch;
  if (oldFetch) window.fetch = function (...args) {
    fired('fetch'); return Promise.resolve(new Response('', {status: 204}));
  };
  const image = window.HTMLImageElement && window.HTMLImageElement.prototype;
  if (image) {
    const src = Object.getOwnPropertyDescriptor(image, 'src');
    if (src && src.set) Object.defineProperty(image, 'src', {
      configurable: true, enumerable: src.enumerable,
      get: src.get, set(value) { fired('image.src'); return src.set.call(this, value); }
    });
  }
  const doc = Object.getOwnPropertyDescriptor(Document.prototype, 'cookie');
  if (doc && doc.set) Object.defineProperty(Document.prototype, 'cookie', {
    configurable: true, enumerable: doc.enumerable,
    get: doc.get, set(value) { fired('document.cookie'); return doc.set.call(this, value); }
  });
})();
"""


def _normalise_case(item, index: int) -> dict:
    """Nhan dict moi hoac tuple cu (payload, context) cho GA/backward compat."""
    if isinstance(item, dict):
        required = {"payload", "context", "event", "family"}
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Oracle case thieu cot: {sorted(missing)}")
        case = dict(item)
    elif isinstance(item, (tuple, list)) and len(item) == 2:
        payload, context = item
        case = {"payload": payload, "context": context, "event": "none", "family": ""}
    else:
        raise TypeError("Case phai la dict hoac tuple (payload, context)")
    case.setdefault("id", f"adhoc-{index}")
    case["id"] = str(case["id"])
    case["payload"] = str(case["payload"])
    case["context"] = str(case["context"])
    case["event"] = str(case["event"])
    case["family"] = str(case["family"])
    if case["context"] not in CONTEXT_ROUTES:
        raise ValueError(f"Khong ho tro ngu canh: {case['context']}")
    get_event_plan(case["event"])
    return case


def _is_lab_url(url: str) -> bool:
    return url == BASE_URL or url.startswith(f"{BASE_URL}/") or url.startswith(f"{BASE_URL}?")


def _is_non_network_url(url: str) -> bool:
    return url.startswith(("about:", "data:", "javascript:"))


def _is_external_url(url: str) -> bool:
    return bool(url) and not _is_lab_url(url) and not _is_non_network_url(url)


def _url_for(case: dict) -> str:
    params = {"payload": case["payload"], "event": case["event"], "family": case["family"]}
    # target_tag do GA gui sang de URL carrier khop ca the duoc sinh.
    if case.get("target_tag") in {"iframe", "object"}:
        params["carrier"] = case["target_tag"]
    # Dataset vectors trong html_body tu tao iframe/object cua chung.  Quy tac
    # nay danh cho case URL do GA tao, de target_tag va sink khong bi lech.
    elif case["family"].lower().startswith("iframe_srcdoc"):
        params["carrier"] = "iframe"
    elif case["family"].lower().startswith("object_data"):
        params["carrier"] = "object"
    return f"{BASE_URL}{CONTEXT_ROUTES[case['context']]}?{urlencode(params)}"


async def _event_target(page, event: str):
    """Element co inline event handler do payload chen ra; khong co => None."""
    handles = await page.locator("*").element_handles()
    for handle in handles:
        if await handle.evaluate("(el, event) => el.hasAttribute(event)", event):
            return handle
    return None


async def _perform_action(page, case: dict) -> str | None:
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
        # html_body boc payload trong <main id="oracle-host">; khong co
        # #payload-target. Ho tro ca HTML <a href=...> va SVG <a xlink:href=...>.
        link = page.locator("#oracle-host a[href], #oracle-host svg a")
        if not await link.count():
            return "no link element in #oracle-host"
        await link.first.click(force=True, timeout=ACTION_TIMEOUT_MS)
        return None
    if action == "transition":
        # ``:target`` transition chi bat dau neu browser da resolve style ban
        # dau (khong co target) truoc luc fragment thay doi. Chi gan hash ngay
        # sau page.goto bi race voi style/layout va lam ontransitionrun luc
        # co luc co luc khong. Doc computed style ep layout, sau do doi hash
        # va cho mot frame render that su; van la event CSS cua payload, khong
        # phai dispatch_event gia.
        target = await _event_target(page, event)
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
        button = page.locator("button[popovertarget]")
        if await button.count():
            await button.first.click(timeout=ACTION_TIMEOUT_MS)
        return None

    target = await _event_target(page, event)
    if target is None:
        return f"no element with {event}"
    if action == "hover":
        await target.hover(force=True, timeout=ACTION_TIMEOUT_MS)
    elif action == "focus":
        # ElementHandle.focus() khong nhan timeout trong Playwright Python.
        # Truyen timeout lam Oracle fail truoc khi DOM focus event duoc goi.
        await target.focus()
    elif action == "click":
        # Dataset co input type=hidden co onclick. Playwright click khong the
        # target hidden control va accesskey cung khong kich hoat hidden input.
        # HTMLElement.click() goi DOM activation behavior va inline handler,
        # khong phu thuoc visibility cua Playwright.
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
        # change chi fire khi value thuc su khac ban dau VA control blur. Body
        # click tai toa do co the van trung input, nen blur truc tiep la ro rang.
        initial_value = await target.input_value()
        next_value = "oracle-change" if initial_value != "oracle-change" else "oracle-change-2"
        await target.fill(next_value, timeout=ACTION_TIMEOUT_MS)
        await target.evaluate("el => el.blur()")
    elif action == "toggle":
        await target.click(force=True, timeout=ACTION_TIMEOUT_MS)
    else:
        raise AssertionError(f"Unknown Oracle action: {action}")
    return None


async def _check_once(browser_context, case: dict) -> tuple[bool, str | None]:
    page = await browser_context.new_page()
    signals: list[str] = []
    fired_event = asyncio.Event()
    diagnostics: list[str] = []
    watch_navigation = False
    route_prefix = f"{BASE_URL}{CONTEXT_ROUTES[case['context']]}"

    def record_signal(sig: str) -> None:
        signals.append(sig)
        fired_event.set()

    async def report(_source, name):
        record_signal(str(name))

    # Fallback neu payload goi dialog truoc khi wrapper init-script duoc ap
    # dung (hoac wrapper bi payload ghi de). Dismiss de khong block action.
    def on_dialog(dialog):
        record_signal(f"dialog:{dialog.type}")
        asyncio.create_task(dialog.dismiss())

    def on_pageerror(error):
        diagnostics.append(f"pageerror: {error}")

    def on_framenavigated(frame):
        # document.location la side-effect XSS hop le (ca main frame lan child
        # iframe/srcdoc). Bo qua URL noi bo/blank/srcdoc va route lab hien tai.
        url = frame.url
        if not url or url in {"about:blank", "about:srcdoc"} or url.startswith(route_prefix):
            return
        phase = "navigation" if watch_navigation else "navigation-during-load"
        if frame == page.main_frame:
            record_signal(f"{phase}:{url}")
        elif _is_external_url(url):
            record_signal(f"subframe-{phase}:{url}")

    def on_request(request):
        # Bat navigation request di ra ngoai route lab cho ca main frame va
        # child iframe (vi du iframe srcdoc goi document.location).
        if not request.is_navigation_request():
            return
        url = request.url
        if not url or url in {"about:blank", "about:srcdoc"} or url.startswith(route_prefix):
            return
        if request.frame == page.main_frame:
            record_signal(f"navigation-request:{url}")
        elif _is_external_url(url):
            record_signal(f"subframe-navigation-request:{url}")

    async def handle_route(route):
        url = route.request.url
        if _is_lab_url(url) or _is_non_network_url(url):
            try:
                await route.continue_()
            except Exception:
                pass
            return
        # Chan moi request ra domain ngoai (ca main frame lan child iframe):
        # ghi nhan side-effect roi abort ngay lap tuc de khong treo load va
        # khong commit document moi lam huy execution context.
        record_signal(f"external-request:{url}")
        try:
            await route.abort("aborted")
        except Exception:
            pass

    page.on("dialog", on_dialog)
    page.on("pageerror", on_pageerror)
    page.on("framenavigated", on_framenavigated)
    page.on("request", on_request)
    await browser_context.route("**/*", handle_route)
    await page.expose_binding("__oracle_report", report)
    await page.add_init_script(INIT_SCRIPT)
    try:
        case_url = _url_for(case)
        goto_task = asyncio.create_task(
            page.goto(case_url, timeout=NAV_TIMEOUT_MS, wait_until="load")
        )
        fired_task = asyncio.create_task(fired_event.wait())
        done, pending = await asyncio.wait(
            {goto_task, fired_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        if fired_event.is_set() or signals:
            if goto_task.done() and not goto_task.cancelled():
                goto_task.exception()  # consume exception to prevent asyncio warning
            return True, None

        try:
            goto_task.result()
        except Exception:
            # Neu payload da dieu huong main frame khoi lab, navigation event
            # la bang chung side-effect; khong bien timeout load thanh false
            # negative. Loi goto khac van duoc retry/xu ly nhu truoc.
            if not signals:
                raise
        # Autofocus/onpageshow co the dieu huong ngay trong page.goto, truoc
        # khi watch_navigation duoc bat. Day van la side-effect can ghi nhan.
        if not page.url.startswith(route_prefix):
            record_signal(f"navigation-during-load:{page.url}")
        if signals:
            return True, None
        watch_navigation = True
        try:
            action_error = await _perform_action(page, case)
        except Exception as exc:
            # Navigation cua payload co the huy context giua locator/focus.
            # Neu listener da thay side effect thi do la XSS hop le, khong
            # retry/ghi de thanh valid=0 chi vi Playwright mat context.
            if signals:
                return True, f"action interrupted after side effect: {exc}"
            raise
        if signals:
            return True, None
        await page.wait_for_timeout(WAIT_MS)
        fired = bool(signals)
        if not fired and not page.is_closed():
            try:
                fired = bool(await page.evaluate("Boolean(window.__xss_fired)"))
            except Exception as exc:
                if signals:
                    return True, f"evaluate interrupted after side effect: {exc}"
                raise
        if not fired and action_error is None and diagnostics:
            action_error = "; ".join(diagnostics[:3])
        return fired, action_error
    finally:
        await page.close()


async def _check_with_retry(browser, case: dict):
    attempts, last_error = 0, None
    t0 = time.perf_counter()
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        browser_context = await browser.new_context()
        try:
            fired, action_error = await asyncio.wait_for(
                _check_once(browser_context, case), timeout=OVERALL_TIMEOUT_S
            )
            return (1 if fired else 0), time.perf_counter() - t0, attempts, action_error
        except Exception as exc:
            last_error = str(exc)
        finally:
            await browser_context.close()
    return 0, time.perf_counter() - t0, attempts, last_error


def _append_log(log_path, record):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def check_batch(cases, concurrency: int = 15, log_path=None):
    """Kiem tra nhieu case va tra valid theo dung thu tu dau vao.

    Moi worker dung browser context/page rieng; log checkpoint khoa bang ``id``.
    """
    if concurrency < 1:
        raise ValueError("concurrency phai >= 1")
    cases = [_normalise_case(item, i) for i, item in enumerate(cases)]
    results = [0] * len(cases)
    sem, log_lock = asyncio.Semaphore(concurrency), asyncio.Lock()
    async with async_playwright() as p:
        browser = await p.chromium.launch()

        async def worker(i, case):
            async with sem:
                valid, elapsed, attempts, error = await _check_with_retry(browser, case)
                results[i] = valid
                if log_path:
                    record = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "id": case["id"], "payload": case["payload"],
                        "context": case["context"], "event": case["event"],
                        "family": case["family"], "valid": valid,
                        "action": action_for(case["event"], case["context"], case["family"]),
                        "elapsed_ms": round(elapsed * 1000), "attempts": attempts,
                        "error": error,
                    }
                    async with log_lock:
                        _append_log(log_path, record)

        await asyncio.gather(*(worker(i, case) for i, case in enumerate(cases)))
        await browser.close()
    return results


def load_done_ids(log_path) -> set[str]:
    """Chi chap nhan log moi co id; log cu khong du an toan de resume."""
    done, path = set(), Path(log_path)
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                if "id" in record:
                    done.add(str(record["id"]))
            except json.JSONDecodeError:
                continue
    return done


async def rendered_html(case: dict) -> str:
    """Diagnostic: HTML sau parse de kiem tra payload raw/khong double escape."""
    case = _normalise_case(case, 0)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto(_url_for(case), timeout=NAV_TIMEOUT_MS)
        html = await page.content()
        await context.close()
        await browser.close()
    return html


def check_payload(payload: str, context: str, event="none", family="") -> int:
    return asyncio.run(check_batch([{"id": "manual", "payload": payload, "context": context,
                                     "event": event, "family": family}]))[0]


if __name__ == "__main__":
    print(check_payload("<img src=x onerror=alert(1)>", "html_body", "onerror"))
