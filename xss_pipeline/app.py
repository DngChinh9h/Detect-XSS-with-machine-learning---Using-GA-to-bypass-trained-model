"""Ung dung lab co chu y de payload duoc chen *raw* cho Oracle.

Khong dung render_template/Jinja: payload lay tu query string duoc noi truc
tiep vao HTML. Day la mo hinh sink co lo hong, chi bind 127.0.0.1.
"""

from flask import Flask, Response, request

app = Flask(__name__)


def page(body: str, *, body_is_payload: bool = False) -> Response:
    """Tra ve HTML raw.

    ``<body onload/...>`` khong co hieu luc neu no bi long trong mot ``main``
    cua body da co san.  Voi hai event body-level nay, payload phai nam tai
    vi tri body-root; cac route khac van dung body wrapper binh thuong.
    """
    prefix = "<!doctype html><html><head><meta charset=utf-8>"
    style = "<style>#oracle-host{min-height:80px} .oracle-focus{min-width:20px;min-height:20px}</style>"
    if body_is_payload:
        return Response(prefix + style + "</head>" + body + "</html>", mimetype="text/html")
    return Response(
        prefix + style +
        "</head><body>" + body + "</body></html>",
        mimetype="text/html",
    )


@app.route("/html_body")
def html_body():
    payload = request.args.get("payload", "")
    # Tag (video/details/svg/...) trong payload chinh la phan tu can thiet.
    event = request.args.get("event", "")
    if event in {"onload", "onpageshow"} and "<body" in payload.lower():
        return page(payload, body_is_payload=True)
    return page(f'<main id="oracle-host">{payload}</main>')


@app.route("/attribute")
def attribute():
    payload = request.args.get("payload", "")
    event = request.args.get("event", "")
    family = request.args.get("family", "").lower()
    # Dataset co ca injection point value='...' va value="...". Neu luon
    # dung nhay kep thi cac attr_break_squote_* khong breakout duoc: toan bo
    # <script>/onfocus theo sau chi con la text trong attribute value.
    quote = "'" if family.startswith("attr_break_squote") else '"'
    # onerror can resource loi that su; keyboard/input/change can form control.
    if event in {"onkeyup", "oninput", "onchange", "onfocus", "onfocusin"}:
        # Ghi ro type=text: hidden control khong the nhan focus/click/key.
        sink = f'<input id="payload-target" type="text" class="oracle-focus" value={quote}{payload}{quote}>'
    else:
        sink = (
            '<img id="payload-target" class="oracle-focus" tabindex="0" '
            f'src="/__oracle_missing_resource__" value={quote}{payload}{quote}>'
        )
    return page(sink)


@app.route("/js_string")
def js_string():
    payload = request.args.get("payload", "")
    family = request.args.get("family", "").lower()

    # ``context=js_string`` khong phai chi co mot sink.  Dataset ghi ro loai
    # lexical context ma payload phai breakout: nhay don, nhay kep, template
    # literal, block comment, hay JS code truc tiep.  Boc tat ca trong "..."
    # se bien '...; //, ${...}, va */ ... /* thanh text/syntax error, tao
    # false-negative hang loat cho event=none.
    if family.startswith(("js_direct_code_no_quote", "js_break_bracket_constructor")):
        script = payload
    elif family.startswith("js_break_comment"):
        script = f"/*{payload}*/"
    elif family.startswith("js_break_template"):
        script = f"var q = `{payload}`;"
    elif family.startswith("js_break_dquote"):
        script = f'var q = "{payload}";'
    elif family.startswith("js_"):
        # Phan lon family js_string (js_break_squote va cac vector bo sung)
        # duoc ghi nhan la chen vao chuoi nhay don.
        #
        # js_break_backslash_bypass mo phong ung dung escape apostrophe mot
        # lop truoc khi reflect. Payload bat dau bang \\' lam cap backslash
        # chan, de apostrophe tiep theo dong chuoi. Cac family khac giu raw.
        if family.startswith("js_break_backslash_bypass"):
            payload = payload.replace("'", r"\'")
        script = f"var q = '{payload}';"
    else:
        # Giu tuong thich cho check_payload()/case thu cong khong co family.
        script = f'var q = "{payload}";'

    return page(f'<div id="payload-target" tabindex="0"></div><script>{script}</script>')


@app.route("/url")
def url():
    payload = request.args.get("payload", "")
    carrier = request.args.get("carrier", "a")
    # Khong auto-click: Oracle quyet dinh theo family/event.
    if carrier == "iframe":
        return page(f'<iframe id="payload-target" src="{payload}"></iframe>')
    if carrier == "object":
        return page(f'<object id="payload-target" data="{payload}"></object>')
    return page(f'<a id="payload-target" href="{payload}">open payload</a>')


@app.route("/__oracle_missing_resource__")
def missing_resource():
    return Response("missing", status=404, mimetype="text/plain")


@app.route("/validimage.gif")
def valid_image():
    """GIF 1x1 hop le cho gene/payload onload cua img."""
    return Response(
        b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,"
        b"\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;",
        mimetype="image/gif",
    )


@app.route("/validaudio.wav")
@app.route("/validaudio.mp3")
def valid_audio():
    # WAV im lang cuc nho, du de Chromium phat loadmetadata/durationchange.
    wav = (
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
        b"\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    )
    return Response(wav, mimetype="audio/wav")


@app.route("/xss-lab-probe")
def probe():
    return Response("ok", mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
