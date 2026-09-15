// Talks straight to the classifier API's existing /review endpoints
// (api.py) -- unchanged by this frontend rewrite. Plain absolute fetches:
// they are NOT part of Next's own routing, so `basePath` in next.config.js
// (which only rewrites Next's own page/asset URLs) does not touch them --
// the literal "/review" prefix here has to stay hardcoded to match api.py.
const BASE = "/review";

export async function fetchQueue() {
  const res = await fetch(`${BASE}/queue`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  return { pending: data.pending || [], dryRun: !!data.dry_run };
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
