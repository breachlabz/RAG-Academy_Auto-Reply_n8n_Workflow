"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import ClampedText from "./ClampedText";
import { createChunk, deleteChunk, fetchChunks, updateChunk } from "../lib/knowledge";

const NEW_METADATA = { source: "manual", heading: "" };

function pretty(metadata) {
  return JSON.stringify(metadata, null, 2);
}

// Parsed metadata, or an error string. Only shape is checked here; the
// server's validate_metadata() is the real rule (flat values, source/heading).
function parseMetadata(text) {
  try {
    const value = JSON.parse(text);
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      return { error: "Metadata must be a JSON object." };
    }
    return { value };
  } catch (e) {
    return { error: `Metadata is not valid JSON: ${e.message}` };
  }
}

function downloadJson(chunks) {
  const blob = new Blob([JSON.stringify(chunks, null, 2) + "\n"], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "knowledge_chunks.json";
  a.click();
  URL.revokeObjectURL(url);
}

// One chunk: read view by default, content + metadata editors on Edit.
// `chunk` is null for the "new chunk" form.
function ChunkCard({ chunk, onSaved, onDeleted, onCancelNew }) {
  const isNew = chunk === null;
  const [editing, setEditing] = useState(isNew);
  const [content, setContent] = useState(isNew ? "" : chunk.content);
  const [metaText, setMetaText] = useState(pretty(isNew ? NEW_METADATA : chunk.metadata));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  function startEdit() {
    setContent(chunk.content);
    setMetaText(pretty(chunk.metadata));
    setError(null);
    setEditing(true);
  }

  async function save() {
    const meta = parseMetadata(metaText);
    if (meta.error) return setError(meta.error);
    if (!content.trim()) return setError("Content cannot be empty.");
    setBusy(true);
    setError(null);
    try {
      const saved = isNew
        ? await createChunk({ content, metadata: meta.value })
        : await updateChunk(chunk.id, { content, metadata: meta.value });
      if (!isNew) setEditing(false);
      onSaved(saved, isNew);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function remove() {
    const note =
      chunk.origin === "ingest"
        ? "\n\nIt comes back the next time its document is re-ingested."
        : "";
    if (!window.confirm(`Delete ${chunk.id}? Replies stop using it immediately.${note}`)) return;
    setBusy(true);
    setError(null);
    try {
      await deleteChunk(chunk.id);
      onDeleted(chunk.id);
    } catch (e) {
      setError(e.message);
      setBusy(false);
    }
  }

  const meta = isNew ? null : chunk.metadata;
  return (
    <div className="kn-card">
      <div className="kn-card-head">
        <div className="kn-title">
          <span className="kn-heading">{isNew ? "New chunk" : meta.heading || "(no heading)"}</span>
          {!isNew && <span className="kn-id">{chunk.id}</span>}
        </div>
        <div className="kn-badges">
          {!isNew && chunk.origin === "manual" && <span className="kn-badge">Manual</span>}
          {!isNew && chunk.edited && <span className="kn-badge edited">Edited</span>}
        </div>
      </div>

      {!editing && (
        <>
          <ClampedText text={chunk.content} />
          <pre className="kn-meta">{pretty(meta)}</pre>
          <div className="kn-actions">
            <button type="button" onClick={startEdit}>Edit</button>
            <button type="button" className={"link danger" + (busy ? " busy" : "")} onClick={remove}>
              Delete
            </button>
          </div>
        </>
      )}

      {editing && (
        <div className="kn-editor">
          <label className="kn-label">Content <span>embedded and shown to the model as-is</span></label>
          <textarea
            className="kn-content"
            value={content}
            onChange={(e) => setContent(e.target.value)}
            spellCheck
          />
          <label className="kn-label">Metadata <span>JSON object; flat string / number / boolean values</span></label>
          <textarea
            className="kn-meta-edit"
            value={metaText}
            onChange={(e) => setMetaText(e.target.value)}
            spellCheck={false}
          />
          <div className="kn-actions">
            <button type="button" className={"primary" + (busy ? " busy" : "")} onClick={save}>
              {busy ? "Saving…" : isNew ? "Add chunk" : "Save"}
            </button>
            <button
              type="button"
              onClick={() => (isNew ? onCancelNew() : setEditing(false))}
              className={busy ? "busy" : ""}
            >
              Cancel
            </button>
          </div>
        </div>
      )}
      {error && <div className="status error">{error}</div>}
    </div>
  );
}

export default function KnowledgeView() {
  const [chunks, setChunks] = useState([]);
  const [sources, setSources] = useState([]);
  const [collection, setCollection] = useState("");
  const [error, setError] = useState(null);
  const [loaded, setLoaded] = useState(false);
  const [query, setQuery] = useState("");
  const [source, setSource] = useState("");
  const [adding, setAdding] = useState(false);
  const [notice, setNotice] = useState(null);

  const load = useCallback(async () => {
    try {
      const data = await fetchChunks();
      setChunks(data.chunks || []);
      setSources(data.sources || []);
      setCollection(data.collection || "");
      setError(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoaded(true);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return chunks.filter((c) => {
      if (source && c.metadata.source !== source) return false;
      if (!needle) return true;
      return (
        c.content.toLowerCase().includes(needle) ||
        c.id.toLowerCase().includes(needle) ||
        JSON.stringify(c.metadata).toLowerCase().includes(needle)
      );
    });
  }, [chunks, query, source]);

  function handleSaved(saved, isNew) {
    setNotice(`${isNew ? "Added" : "Saved"} ${saved.id}. Replies use it from now on.`);
    if (isNew) setAdding(false);
    setChunks((cs) => (isNew ? [...cs, saved] : cs.map((c) => (c.id === saved.id ? saved : c))));
    if (isNew) load(); // refresh the source list
  }

  function handleDeleted(id) {
    setNotice(`Deleted ${id}.`);
    setChunks((cs) => cs.filter((c) => c.id !== id));
  }

  const editedCount = chunks.filter((c) => c.edited).length;

  return (
    <section>
      <div className="kn-toolbar">
        <input
          type="search"
          className="kn-search"
          placeholder="Search content, id or metadata…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <select className="kn-select" value={source} onChange={(e) => setSource(e.target.value)}>
          <option value="">All sources</option>
          {sources.map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
        <span className="count">
          {visible.length} of {chunks.length} chunks
          {editedCount ? ` · ${editedCount} edited` : ""}
          {collection ? ` · collection "${collection}"` : ""}
        </span>
        <div className="kn-toolbar-end">
          <button type="button" onClick={() => downloadJson(chunks)} disabled={!chunks.length}>
            Export JSON
          </button>
          <button type="button" className="primary" onClick={() => setAdding(true)} disabled={adding}>
            Add chunk
          </button>
        </div>
      </div>

      <p className="kn-hint">
        Saving re-embeds the chunk and updates the vector store right away. Re-ingesting a
        document overwrites edits to its chunks; chunks you add here are kept.
      </p>

      {notice && <div className="status ok kn-notice">{notice}</div>}
      {error && <div id="loadError">Could not load the knowledge base: {error}</div>}
      {!error && loaded && chunks.length === 0 && (
        <div id="empty">
          The table is empty. Fill it from the live collection with{" "}
          <code>python -m rag knowledge pull</code>.
        </div>
      )}

      <div className="kn-list">
        {adding && (
          <ChunkCard
            chunk={null}
            onSaved={handleSaved}
            onDeleted={() => {}}
            onCancelNew={() => setAdding(false)}
          />
        )}
        {visible.map((chunk) => (
          <ChunkCard
            key={chunk.id + chunk.updated_at}
            chunk={chunk}
            onSaved={handleSaved}
            onDeleted={handleDeleted}
          />
        ))}
      </div>
    </section>
  );
}
