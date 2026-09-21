// Talks straight to the classifier API's existing /review endpoints
// (api.py) -- unchanged by this frontend rewrite. Plain absolute fetches:
// they are NOT part of Next's own routing, so `basePath` in next.config.js
// (which only rewrites Next's own page/asset URLs) does not touch them --
// the literal "/review" prefix here has to stay hardcoded to match api.py.
const BASE = "/review";

// Both endpoints now return one entry per conversation --
// { conversation_id, conversation_subject, exchanges: [...] } -- instead of
// a flat list of turns, so a thread with several pending or sent replies
// renders as one card instead of several disconnected rows. See
// api.py's _group_by_conversation.

export async function fetchQueue() {
  const res = await fetch(`${BASE}/queue`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  return { groups: data.pending || [], dryRun: !!data.dry_run };
}

export async function fetchHistory() {
  const res = await fetch(`${BASE}/history`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  return data.history || [];
}

export async function sendReply(id, reply) {
  const res = await fetch(`${BASE}/${id}/send`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reply }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}
