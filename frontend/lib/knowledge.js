// The /knowledge endpoints in api.py -- a read-only view of the
// knowledge_chunks table (rag/knowledge.py). The one write is adding new
// content, which the server appends to an Additions document and ingests.
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

// The whole knowledge base as a JSON file. The server sets
// Content-Disposition, so navigating here downloads instead of rendering.
export const EXPORT_URL = `${BASE}/export`;

// Appends to the current data/docs/Additions_N.docx and ingests it.
// Returns {file, entry, max, chunks}.
export function addContent({ title, content }) {
  return call(BASE, jsonBody("POST", { title, content }));
}
