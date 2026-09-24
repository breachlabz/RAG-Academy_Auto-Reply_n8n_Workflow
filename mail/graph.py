"""Send a reply through Microsoft Graph with app-only credentials.

The n8n workflow's Outlook connection is *delegated* OAuth2 -- it acts as a
signed-in user, which is why standing it up needs an interactive sign-in that
has not happened yet (see the project handoff notes). The review queue cannot
wait on that: it needs to send mail from a background process with nobody
watching.

App-only credentials are the other shape Graph supports: an Azure AD app
registration with the application permission `Mail.Send`, admin-consented
once, authenticating itself with a client id/secret rather than a user token.
No interactive sign-in, ever -- which is the whole reason this exists as its
own path instead of waiting for the n8n credential to land.

Deliberately narrow: this module can only reply to an existing message
(`/reply`, which sends immediately -- there is no draft step here, unlike the
n8n Outlook Draft node this replaces). It cannot read mail, list messages, or
do anything else Mail.Send doesn't require.
"""

from __future__ import annotations

import base64
import dataclasses
import html
import os
import re
import time

import requests

TENANT_ID = os.environ.get("GRAPH_TENANT_ID", "")
CLIENT_ID = os.environ.get("GRAPH_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("GRAPH_CLIENT_SECRET", "")
# The mailbox to send as -- a UPN or object id. App-only calls address
# `/users/{mailbox}/...`; there is no `/me` without a signed-in user.
MAILBOX = os.environ.get("GRAPH_MAILBOX", "")

_TOKEN_URL_TMPL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
_GRAPH_BASE = "https://graph.microsoft.com/v1.0"
_SCOPE = "https://graph.microsoft.com/.default"

# Cached in module state, not per-call: a client-credentials token is valid
# for about an hour and every send in the review queue would otherwise pay a
# round trip to Entra for one it already holds.
_token: str | None = None
_token_expires_at: float = 0.0


class GraphNotConfigured(RuntimeError):
    """Raised instead of silently no-op'ing. See configured()."""


class GraphSendError(RuntimeError):
    """A configured send that Graph itself rejected or that failed on the wire."""


def configured() -> bool:
    return bool(TENANT_ID and CLIENT_ID and CLIENT_SECRET and MAILBOX)


def _fetch_token(*, timeout: int = 20) -> str:
    resp = requests.post(
        _TOKEN_URL_TMPL.format(tenant=TENANT_ID),
        data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "scope": _SCOPE,
            "grant_type": "client_credentials",
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    body = resp.json()
    global _token, _token_expires_at
    _token = body["access_token"]
    # 60s margin so a token that is about to expire mid-call is refreshed
    # early rather than failing the send it was fetched for.
    _token_expires_at = time.monotonic() + int(body.get("expires_in", 3600)) - 60
    return _token


def _access_token(*, timeout: int = 20) -> str:
    if _token and time.monotonic() < _token_expires_at:
        return _token
    return _fetch_token(timeout=timeout)


# The review edit box's Bold/Italic buttons wrap the selection in **/*. The
# lookarounds keep stray asterisks ("5 * 3", "a*b*c", a lone "*") as literal
# text: a marker only counts when it hugs non-space text on the inside and is
# not glued to a word character on the outside (italic) -- and nothing here
# crosses a line break.
_BOLD_RE = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
_ITALIC_RE = re.compile(r"(?<![*\w])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![*\w])")


_BULLET_RE = re.compile(r"^- (.*)$")
_NUMBER_RE = re.compile(r"^\d+\. (.*)$")


def _block_html(block: str) -> str:
    """One blank-line-separated block: consecutive "- " lines become a <ul>,
    "1. " lines an <ol>, anything else a <p> with <br> between its lines."""
    out: list[str] = []
    run_tag: str | None = None
    run: list[str] = []

    def flush() -> None:
        nonlocal run_tag, run
        if not run:
            return
        if run_tag == "p":
            out.append("<p>" + "<br>".join(run) + "</p>")
        else:
            items = "".join(f"<li>{item}</li>" for item in run)
            out.append(f"<{run_tag}>{items}</{run_tag}>")
        run_tag, run = None, []

    for line in block.split("\n"):
        if m := _BULLET_RE.match(line):
            tag, value = "ul", m.group(1)
        elif m := _NUMBER_RE.match(line):
            tag, value = "ol", m.group(1)
        else:
            tag, value = "p", line
        if tag != run_tag:
            flush()
            run_tag = tag
        run.append(value)
    flush()
    return "".join(out)


def reply_html(text: str) -> str | None:
    """Render the reviewer's **bold** / *italic* markers as an HTML body.

    "- " / "1. " lines render as real lists in the same pass. Returns None
    when the text has no bold/italic, so the caller keeps sending the plain
    `comment` (which Graph places above the quoted original thread): a list
    on its own already reads fine as plain text, so it is not worth dropping
    the quoted thread for.
    """
    escaped = html.escape(text, quote=False)
    formatted = _ITALIC_RE.sub(r"<em>\1</em>", _BOLD_RE.sub(r"<strong>\1</strong>", escaped))
    if formatted == escaped:
        return None
    return "".join(_block_html(b) for b in re.split(r"\n\s*\n", formatted.strip()))


# Graph rejects a fileAttachment's contentBytes over ~3MB on this same-call
# path (anything bigger needs a chunked upload session, which is a different,
# much bigger feature). Checked against the decoded size, not the base64
# text, which runs about a third larger than the bytes it encodes.
MAX_ATTACHMENT_BYTES = 3 * 1024 * 1024


class AttachmentTooLarge(RuntimeError):
    """A supplied attachment exceeds MAX_ATTACHMENT_BYTES."""


@dataclasses.dataclass(frozen=True)
class Attachment:
    """One reviewer-supplied file, decoded and ready for Graph.

    Deliberately just a name and bytes passed straight through in one
    request -- see attachment_name's comment in threads/store.py for what
    this app does and does not persist.
    """

    name: str
    content_type: str
    content_bytes: bytes

    def __post_init__(self) -> None:
        if len(self.content_bytes) > MAX_ATTACHMENT_BYTES:
            raise AttachmentTooLarge(
                f"attachment is {len(self.content_bytes)} bytes, "
                f"over the {MAX_ATTACHMENT_BYTES} byte limit"
            )

    def as_graph_dict(self) -> dict:
        return {
            "@odata.type": "#microsoft.graph.fileAttachment",
            "name": self.name,
            "contentType": self.content_type or "application/octet-stream",
            "contentBytes": base64.b64encode(self.content_bytes).decode("ascii"),
        }


def send_reply(
    message_ref: str,
    comment: str,
    *,
    attachment: Attachment | None = None,
    timeout: int = 30,
) -> None:
    """Send `comment` as a reply to `message_ref` (a Graph message id) now.

    Plain text goes as Graph's `comment`. If it carries **bold** / *italic*
    markers (lists included) it goes as an HTML `message.body` instead (Graph rejects sending
    both) -- with the trade-off that a supplied body replaces the whole
    reply body, so the quoted earlier thread is not appended in that case.

    `attachment`, if given, rides in `message.attachments` regardless of
    which body shape above was chosen -- Graph's reply action accepts
    `message` and `comment` together, so attaching a file never costs the
    quoted thread the way bold/italic does.

    Raises GraphNotConfigured if the app registration is not set up yet, and
    GraphSendError on anything Graph itself rejects -- this never returns a
    false "sent" for a call that didn't go through, matching the fail-visible
    stance the rest of this codebase takes on the classifier and grounding
    gates.
    """
    if not configured():
        raise GraphNotConfigured(
            "GRAPH_TENANT_ID / GRAPH_CLIENT_ID / GRAPH_CLIENT_SECRET / "
            "GRAPH_MAILBOX are not all set -- see .env.example"
        )
    if not message_ref:
        raise GraphSendError("no Graph message id (ref) stored for this exchange")

    body_html = reply_html(comment)
    message: dict = {}
    payload: dict = {}
    if body_html is not None:
        message["body"] = {"contentType": "HTML", "content": body_html}
    else:
        payload["comment"] = comment
    if attachment is not None:
        message["attachments"] = [attachment.as_graph_dict()]
    if message:
        payload["message"] = message

    try:
        token = _access_token(timeout=timeout)
        resp = requests.post(
            f"{_GRAPH_BASE}/users/{MAILBOX}/messages/{message_ref}/reply",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        detail = ""
        response = getattr(exc, "response", None)
        if response is not None:
            detail = f" -- {response.status_code} {response.text[:300]}"
        raise GraphSendError(f"Graph send failed: {exc}{detail}") from exc
