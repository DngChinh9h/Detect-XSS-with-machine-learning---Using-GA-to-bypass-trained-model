"""Chinh sach kich hoat XSS dung chung cho dataset filter va GA.

Mot event chi co mot dinh nghia tai day.  ``oracle.py`` thuc thi cac action
Playwright tu bang nay; cac caller khong tu viet lai if/else theo event.
"""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class EventPlan:
    action: str
    needs_element: bool = True
    supported: bool = True
    note: str = ""


# action la ten thao tac thuc su trong Oracle, khong phai DOM dispatch_event.
EVENT_ACTIONS = {
    "none": EventPlan("context"),
    "onerror": EventPlan("wait", note="Tu phat khi tai tai nguyen loi."),
    "onload": EventPlan("wait", note="Tu phat khi tai tai nguyen hop le."),
    "onloadstart": EventPlan("wait", note="Media payload tu bat dau tai."),
    "ontoggle": EventPlan("toggle"),
    # Chromium khong con co implementation marquee onstart dang tin cay. Day
    # la vector duoc bao cao rieng, khong duoc tinh la false-negative Oracle.
    "onstart": EventPlan("wait", supported=False,
                         note="Khong trong pham vi Chromium: marquee/onstart deprecated."),
    "onpageshow": EventPlan("wait"),
    "onanimationend": EventPlan("wait_long"),
    "oncontentvisibilityautostatechange": EventPlan("wait"),
    "ondurationchange": EventPlan("wait_long"),
    "onend": EventPlan("wait_long"),
    "ontransitionrun": EventPlan("transition"),
    "onslotchange": EventPlan("wait"),
    "onloadedmetadata": EventPlan("wait_long"),
    "onbeforetoggle": EventPlan("popover"),
    "onbegin": EventPlan("wait_long"),
    "onmouseover": EventPlan("hover"),
    "onpointerover": EventPlan("hover"),
    "onpointerenter": EventPlan("hover"),
    "onfocus": EventPlan("focus"),
    "onfocusin": EventPlan("focus"),
    "onclick": EventPlan("click"),
    "ondblclick": EventPlan("dblclick"),
    "onmousedown": EventPlan("mousedown"),
    "onkeyup": EventPlan("keyup"),
    "oninput": EventPlan("input"),
    "onchange": EventPlan("change"),
}


def get_event_plan(event: str) -> EventPlan:
    """Tra ve plan ro rang; event la du lieu dau vao nen khong duoc am tham bo qua."""
    try:
        return EVENT_ACTIONS[event]
    except KeyError as exc:
        raise ValueError(f"Event chua co oracle plan: {event!r}") from exc


def context_action(context: str, family: str) -> str:
    """Action cho event=none, phu thuoc vector thay vi ep click moi URL."""
    family = (family or "").lower()
    if context == "html_body":
        if family.startswith(("anchor_js", "svg_xlink")):
            return "click_link"
        return "wait"
    if context != "url":
        return "wait"
    if family.startswith(("url_js", "url_data", "anchor_js", "svg_xlink")):
        return "open_url"
    if family.startswith(("iframe_srcdoc", "object_data", "meta_refresh")):
        return "wait"
    # URL payload la href nen khong tu dong click; unknown family duoc ghi ro.
    return "none"


def action_for(event: str, context: str, family: str) -> str:
    plan = get_event_plan(event)
    url_action = context_action(context, family)
    # URL la mot navigation boundary.  Vi du data:image/svg+xml co onload,
    # hay data:image/svg+xml co <animate onbegin>, chi co the tu bat sau khi
    # carrier duoc mo.  Khong duoc de event "tu bat" bien thanh "chi wait"
    # tren trang carrier chua he duoc click.
    if context == "url" and url_action == "open_url":
        return "open_url"
    return url_action if plan.action == "context" else plan.action


_ENCODED_JS_SCHEME_BYPASS = re.compile(
    r"^\s*java(?:%(?:0a|0d|09|0c|20))+script:", re.IGNORECASE
)


def browser_compatibility_note(payload: str, context: str, family: str) -> str:
    """Danh dau vector biet truoc da bi Chromium hien dai tu choi.

    ``java%0ascript:`` khong con duoc normalise thanh javascript:; Chromium
    coi no la relative URL. Gan co nay giu ket qua valid=0 co y nghia thay vi
    danh dong no la loi cua Oracle.
    """
    family = (family or "").lower()
    if (context == "url" and family.startswith("url_js")
            and _ENCODED_JS_SCHEME_BYPASS.match(payload or "")):
        return "known_browser_patched: encoded-whitespace javascript scheme"
    # Chromium tu choi top-level navigation tu HTTP(S) sang data: URL. Cac
    # vector URL data cua dataset deu duoc Oracle mo qua <a>, nen valid=0 la
    # chinh sach trinh duyet, khong phai action Oracle bi thieu.
    if context == "url" and family.startswith("url_data"):
        return "known_browser_patched: Chromium blocks top-level data: navigation"
    # <scr<script>ipt> khong tao Script element trong HTML parser Chromium.
    # Day la filter-bypass cu, khong phai XSS chay duoc tren engine hien tai.
    if family.startswith("nested_tag_filter_bypass"):
        return "known_browser_patched: Chromium parser does not create script from nested-tag bypass"
    return ""
