"use client";

// One card per conversation, wrapping every pending or sent turn that
// belongs to it -- the subject/conversation id are shown once here instead
// of repeating per turn (see ReviewRow/HistoryRow, which used to carry a
// "Conversation" line of their own before this existed).
//
// `senderEmail` is shown as plain text so two cards from the same person
// can be told apart just by looking, regardless of open/sent status --
// `otherOpenThreads`/`otherSentThreads` below only surface a link when
// there's something to link to, so this is the one signal that always
// works, including two fully-resolved History cards that neither chip
// list would ever connect to each other.
//
// `otherOpenThreads`/`otherSentThreads` are the Case-1b signals: other
// threads from the same sender address that still have something open, and
// other threads already sent to them. Never acted on automatically -- a
// shared inbox alias can have different real people behind one address, so
// whether either one means anything is for the reviewer to judge, not this
// page. Each entry is a clickable chip: since every group on the page
// (queue and history alike) renders its own card with a stable id, clicking
// one can jump straight to the other thread instead of leaving the
// reviewer to scroll and hunt for it.

function jumpTo(conversationId) {
  const el = document.getElementById(`thread-${conversationId}`);
  if (!el) return; // not rendered on this page right now -- nothing to jump to
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.add("flash");
  setTimeout(() => el.classList.remove("flash"), 1200);
}

function ChipRow({ label, items }) {
  if (!items.length) return null;
  return (
    <div className="thread-other-open">
      <span className="thread-other-open-label">{label}</span>
      <div className="thread-other-open-list">
        {items.map((t) => (
          <button
            type="button"
            key={t.conversation_id}
            className="thread-other-open-chip"
            title={`Jump to: ${t.subject || "(no subject)"}`}
            onClick={() => jumpTo(t.conversation_id)}
          >
            {t.subject || "(no subject)"}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function ThreadCard({
  subject,
  conversationId,
  senderEmail,
  otherOpenThreads,
  otherSentThreads,
  children,
}) {
  const label = subject || "(no subject)";
  const open = otherOpenThreads || [];
  const sent = otherSentThreads || [];
  return (
    <div className="thread-card" id={`thread-${conversationId}`}>
      <div className="thread-card-header">
        <div className="thread-card-header-main">
          <span className="thread-subject" title={label}>{label}</span>
          <span className="thread-conv" title={conversationId || ""}>
            {senderEmail ? `${senderEmail} · ${conversationId || ""}` : (conversationId || "")}
          </span>
        </div>
        <ChipRow
          label={open.length === 1 ? "Also open:" : `${open.length} other open threads:`}
          items={open}
        />
        <ChipRow
          label={sent.length === 1 ? "Already sent:" : `${sent.length} already sent:`}
          items={sent}
        />
      </div>
      <div className="board">{children}</div>
    </div>
  );
}
