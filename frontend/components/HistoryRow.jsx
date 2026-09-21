"use client";

import ClampedText from "./ClampedText";

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

// Read-only counterpart to ReviewRow: what actually went out, once it's
// already sent. `edited_reply` is the text that was really sent -- it only
// differs from `reply` (the original AI draft) when a human changed it
// before clicking Send.
export default function HistoryRow({ row }) {
  const wasEdited = (row.edited_reply || "") !== (row.reply || "");

  return (
    <div className="row history-row">
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
          <span className="meta-k">Sent</span>
          <span className="meta-v">{fmtDate(row.sent_at)}</span>
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

      <div className="col">
        {wasEdited && <div className="edited-note">Edited before sending</div>}
        <ClampedText text={row.edited_reply || row.reply || ""} />
      </div>
    </div>
  );
}
