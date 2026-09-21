"use client";

import { useRef, useState } from "react";
import ClampedText from "./ClampedText";
import { sendReply } from "../lib/api";
import { applyCase, wrapSelection } from "../lib/textFormat";

function fmtDate(iso) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

// `row` is only ever used to *initialize* `text` -- React keeps this
// component (and its state) mounted across polls as long as `row.id` stays
// in the parent's list (see app/page.jsx's key={row.id}), so an in-progress
// edit or an expanded preview is never reset out from under the reviewer by
// a background refresh.
export default function ReviewRow({ row, onSendSuccess, onRemove }) {
  const [text, setText] = useState(row.reply || "");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState(null); // { kind: "ok" | "error", message }
  const [sent, setSent] = useState(false);
  const textareaRef = useRef(null);

  // Runs a transform (applyCase/wrapSelection from lib/textFormat) against
  // the textarea's current native selection, applies the result, then
  // restores focus and selection on the next tick -- setText re-renders
  // the textarea first, so the DOM selection has to be reapplied after,
  // not in the same handler tick.
  function runFormat(transform) {
    const el = textareaRef.current;
    if (!el) return;
    const { selectionStart, selectionEnd } = el;
    const result = transform(text, selectionStart, selectionEnd);
    setText(result.text);
    requestAnimationFrame(() => {
      el.focus();
      el.setSelectionRange(result.selectionStart, result.selectionEnd);
    });
  }

  async function handleSend() {
    if (busy) return;
    setBusy(true);
    setStatus(null);
    try {
      const result = await sendReply(row.id, text);
      setStatus({
        kind: "ok",
        message: result.dry_run ? "Recorded (dry run — nothing sent)." : "Sent.",
      });
      setSent(true);
      onSendSuccess(row); // tells the poll not to drop this row mid fade-out
      setTimeout(() => onRemove(row.id), 800);
    } catch (e) {
      setStatus({ kind: "error", message: e.message });
      setBusy(false);
    }
  }

  return (
    <div className={"row" + (sent ? " sent" : "")}>
      <div className="col meta">
        <div className="meta-row">
          <span className="meta-k">ID</span>
          <span className="meta-v">{row.id}</span>
        </div>
        <div className="meta-row">
          <span className="meta-k">{row.is_followup ? "Scheduled" : "Received"}</span>
          <span className="meta-v">{fmtDate(row.created_at)}</span>
        </div>
        <div className="meta-row">
          <span className="meta-k">Drafted</span>
          <span className="meta-v">{fmtDate(row.replied_at)}</span>
        </div>
      </div>

      <div className="col">
        {row.is_followup && <div className="followup-badge">Follow-up · no reply expected</div>}
        <ClampedText text={row.query || ""} />
        {row.query_gist && (
          <div className="gist"><span className="gist-label">Summary:</span> {row.query_gist}</div>
        )}
      </div>

      <div className="col">
        <ClampedText text={row.reply || ""} />
      </div>

      <div className="col edit-cell">
        <div className="format-toolbar">
          <button type="button" title="Capitalize each word" onClick={() => runFormat((t, s, e) => applyCase(t, s, e, "title"))}>
            Aa
          </button>
          <button type="button" title="UPPERCASE" onClick={() => runFormat((t, s, e) => applyCase(t, s, e, "upper"))}>
            AA
          </button>
          <button type="button" title="lowercase" onClick={() => runFormat((t, s, e) => applyCase(t, s, e, "lower"))}>
            aa
          </button>
          <span className="format-toolbar-sep" />
          <button type="button" title="Bold (**text**)" className="format-bold" onClick={() => runFormat((t, s, e) => wrapSelection(t, s, e, "**"))}>
            B
          </button>
          <button type="button" title="Italic (*text*)" className="format-italic" onClick={() => runFormat((t, s, e) => wrapSelection(t, s, e, "*"))}>
            I
          </button>
        </div>
        <textarea
          ref={textareaRef}
          className="edit-box"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        {status && <div className={"status " + status.kind}>{status.message}</div>}
        <div className="edit-actions">
          <button type="button" onClick={() => setText(row.reply || "")} title="Reset to AI reply">
            Reset
          </button>
          <button type="button" className={"primary" + (busy ? " busy" : "")} onClick={handleSend}>
            {sent ? "Sent" : busy ? "Sending…" : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}
