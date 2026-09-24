// The /knowledge endpoints in api.py -- the editable knowledge_chunks table
// (rag/knowledge.py). Every write re-embeds the chunk and updates Chroma
// server-side, so a save here changes what replies are drafted from.
//
// Chunk ids contain "#" ("EVH_Level_3_2.4.docx#3"), which a browser would
// otherwise treat as the start of a fragment -- always encode them.
const BASE = "/knowledge";

async function call(url, options) {
  const res = await fetch(url, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

function jsonBody(method, body) {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

export function fetchChunks() {
  return call(BASE);
}

export function updateChunk(id, { content, metadata }) {
  return call(`${BASE}/${encodeURIComponent(id)}`, jsonBody("PUT", { content, metadata }));
}

export function createChunk({ content, metadata }) {
  return call(BASE, jsonBody("POST", { content, metadata }));
}

export function deleteChunk(id) {
  return call(`${BASE}/${encodeURIComponent(id)}`, { method: "DELETE" });
}
