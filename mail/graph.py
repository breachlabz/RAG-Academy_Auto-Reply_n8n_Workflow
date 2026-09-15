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

import os
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


def send_reply(message_ref: str, comment: str, *, timeout: int = 30) -> None:
    """Send `comment` as a reply to `message_ref` (a Graph message id) now.

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

    try:
        token = _access_token(timeout=timeout)
        resp = requests.post(
            f"{_GRAPH_BASE}/users/{MAILBOX}/messages/{message_ref}/reply",
            json={"comment": comment},
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
