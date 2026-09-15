"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ReviewRow from "../components/ReviewRow";
import HistoryRow from "../components/HistoryRow";
import RowHeader from "../components/RowHeader";
import { fetchHistory, fetchQueue } from "../lib/api";

const REVIEW_LABELS = ["Email", "Enquiry", "AI reply", "Edit & send"];
const HISTORY_LABELS = ["Email", "Enquiry", "AI reply", "Sent"];

const POLL_MS = 20000;

export default function Page() {
  const [rows, setRows] = useState([]);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);
  const [dryRun, setDryRun] = useState(false);
  // Rows mid Send -> fade-out. A poll landing in that ~800ms window would
  // otherwise see the row already gone from the backend (the send already
  // succeeded) and yank it out of `rows` immediately, cutting the fade short.
  const fadingRef = useRef(new Map());

  const load = useCallback(async () => {
    try {
      const [{ pending, dryRun }, historyRows] = await Promise.all([
        fetchQueue(),
        fetchHistory(),
      ]);
      setError(null);
      setDryRun(dryRun);
      setHistory(historyRows);
      setRows(() => {
        const merged = [...pending];
        for (const [id, snapshot] of fadingRef.current) {
          if (!merged.some((r) => r.id === id)) merged.push(snapshot);
        }
        return merged;
      });
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, POLL_MS);
    return () => clearInterval(id);
  }, [load]);

  function handleSendSuccess(row) {
    fadingRef.current.set(row.id, row);
  }

  function handleRemove(id) {
    fadingRef.current.delete(id);
    setRows((rs) => rs.filter((r) => r.id !== id));
    load(); // pulls the just-sent row into History right away, not on the next 20s tick
  }

  return (
    <div className="wrap">
      {dryRun && (
        <div className="dry-run-banner">Dry run</div>
      )}
      <header>
        <h1>Reply review</h1>
        <div className="toolbar">
          <span className="count">{rows.length ? `${rows.length} pending` : ""}</span>
          <button type="button" onClick={load}>Refresh</button>
        </div>
      </header>

      <section>
        <h2 className="section-title">Approval required to send</h2>
        {error && (
          <div id="loadError">Could not load the review queue: {error}</div>
        )}
        {!error && rows.length === 0 && (
          <div id="empty">Nothing waiting for review.</div>
        )}
        <div className="board">
          {rows.length > 0 && <RowHeader labels={REVIEW_LABELS} />}
          {rows.map((row) => (
            <ReviewRow
              key={row.id}
              row={row}
              onSendSuccess={handleSendSuccess}
              onRemove={handleRemove}
            />
          ))}
        </div>
      </section>

      <section>
        <h2 className="section-title">History</h2>
        {!error && history.length === 0 && (
          <div id="empty">Nothing sent yet.</div>
        )}
        <div className="board">
          {history.length > 0 && <RowHeader labels={HISTORY_LABELS} />}
          {history.map((row) => (
            <HistoryRow key={row.id} row={row} />
          ))}
        </div>
      </section>
    </div>
  );
}
