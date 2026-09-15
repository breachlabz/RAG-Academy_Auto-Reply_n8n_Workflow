"use client";

import { useState } from "react";
import ClampedText from "./ClampedText";
import { sendReply } from "../lib/api";

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

  const subject = row.conversation_subject || row.query_subject || "(no subject)";

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
          <span className="meta-k">Subject</span>
          <span className="meta-v" title={subject}>{subject}</span>
        </div>
        <div className="meta-row">
          <span className="meta-k">Conversation</span>
          <span className="meta-v" title={row.conversation_id || ""}>{row.conversation_id || ""}</span>
        </div>
        <div className="meta-row">
          <span className="meta-k">Received</span>
          <span className="meta-v">{fmtDate(row.created_at)}</span>
        </div>
        <div className="meta-row">
          <span className="meta-k">Drafted</span>
          <span className="meta-v">{fmtDate(row.replied_at)}</span>
        </div>
      </div>

      <div className="col">
        <ClampedText text={row.query || ""} />
        {row.query_gist && (
          <div className="gist"><span className="gist-label">Summary:</span> {row.query_gist}</div>
        )}
      </div>

      <div className="col">
        <ClampedText text={row.reply || ""} />
      </div>

      <div className="col edit-cell">
        <textarea
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
