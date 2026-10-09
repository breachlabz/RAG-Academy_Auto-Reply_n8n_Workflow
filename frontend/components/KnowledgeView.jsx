"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import ClampedText from "./ClampedText";
import { addContent, EXPORT_URL, fetchChunks } from "../lib/knowledge";

const POLL_MS = 20000;

// The Knowledge tab: an "Add context" box, then a compact read-only list of
// what is already in the knowledge base. Each new entry is appended to the
// current data/docs/Additions_N.docx (20 per file) and ingested like any
// other document. Existing content cannot be edited or deleted here; "Export
// JSON" downloads all of it (every entry, whatever the search box says).
export default function KnowledgeView() {
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);
  const [chunks, setChunks] = useState([]);
  const [listError, setListError] = useState(null);
  const [query, setQuery] = useState("");

  const load = useCallback(async () => {
    try {
      setChunks((await fetchChunks()).chunks || []);
      setListError(null);
    } catch (e) {
      setListError(e.message);
    }
  }, []);

  // Poll like the review queue does, so entries added from another browser
  // show up without a reload.
  useEffect(() => {
    load();
    const id = setInterval(load, POLL_MS);
    return () => clearInterval(id);
  }, [load]);

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return chunks;
    return chunks.filter(
      (c) => c.content.toLowerCase().includes(needle) || c.id.toLowerCase().includes(needle)
    );
  }, [chunks, query]);

  async function save() {
    if (!content.trim()) return setError("Content cannot be empty.");
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const result = await addContent({ title, content });
      setNotice(`Added to ${result.file} (entry ${result.entry} of ${result.max}).`);
      setTitle("");
      setContent("");
      load();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
    <section className="kn-add">
      <h2 className="section-title">Add context</h2>
      <p className="kn-hint">
        New content is saved to an Additions document (20 entries each) and used in replies right away.
      </p>
      <input
        type="text"
        className="kn-input"
        placeholder="Title (optional)"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
      />
      <textarea
        className="kn-content"
        rows={3}
        placeholder="Content"
        value={content}
        onChange={(e) => setContent(e.target.value)}
        spellCheck
      />
      <div className="kn-actions">
        <button type="button" className={"primary" + (busy ? " busy" : "")} onClick={save}>
          {busy ? "Adding…" : "Add"}
        </button>
      </div>
      {notice && <div className="status ok">{notice}</div>}
      {error && <div className="status error">{error}</div>}
    </section>

    <section>
      <div className="kn-list-head">
        <h2 className="section-title">Already added</h2>
        <span className="count">{visible.length} of {chunks.length}</span>
        <a className="kn-export" href={EXPORT_URL} download title="Download all knowledge base content as a JSON file">
          Export JSON
        </a>
        <input
          type="search"
          className="kn-input kn-filter"
          placeholder="Search…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>
      {listError && <div id="loadError">Could not load the knowledge base: {listError}</div>}
      <ul className="kn-list">
        {visible.map((c) => (
          <li key={c.id}>
            <div className="kn-row-head">
              <span className="kn-heading">{c.metadata.heading || "(no heading)"}</span>
              <span className="kn-id">{c.id}</span>
            </div>
            <ClampedText text={c.content} />
            <details className="kn-meta-toggle">
              <summary>Metadata</summary>
              <pre className="kn-meta">{JSON.stringify(c.metadata, null, 2)}</pre>
            </details>
          </li>
        ))}
      </ul>
    </section>
    </>
  );
}
