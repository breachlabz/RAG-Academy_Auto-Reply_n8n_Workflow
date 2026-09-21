"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ReviewRow from "../components/ReviewRow";
import HistoryRow from "../components/HistoryRow";
import RowHeader from "../components/RowHeader";
import ThreadCard from "../components/ThreadCard";
import { fetchHistory, fetchQueue } from "../lib/api";

const REVIEW_LABELS = ["Received / drafted", "Enquiry", "AI reply", "Edit & send"];
const HISTORY_LABELS = ["Received / sent", "Enquiry", "AI reply", "Sent"];

const POLL_MS = 20000;

function totalExchanges(groups) {
  return groups.reduce((n, g) => n + g.exchanges.length, 0);
}

// History only (see Case 1b discussion: pending stays in drafted order --
// oldest-first review order matters more there than clustering). Reorders,
// nothing else -- same cards, same ThreadCard, just placed so the same
// sender's threads sit next to each other instead of scattered by whenever
// they happened to send. Groups with no captured sender_email keep their
// own original position rather than clustering with each other, same
// reasoning as sender_threads()'s blank-address guard: two blanks are not
// evidence of the same person.
function clusterBySender(groups) {
  const withIndex = groups.map((group, i) => ({ group, i }));
  const firstIndexBySender = new Map();
  for (const { group, i } of withIndex) {
    if (group.sender_email && !firstIndexBySender.has(group.sender_email)) {
      firstIndexBySender.set(group.sender_email, i);
    }
  }
  const rankOf = ({ group, i }) =>
    group.sender_email ? firstIndexBySender.get(group.sender_email) : i;
  return withIndex
    .slice()
    .sort((a, b) => rankOf(a) - rankOf(b) || a.i - b.i)
    .map(({ group }) => group);
}

export default function Page() {
  // Each entry: { conversation_id, conversation_subject, exchanges: [...] }
  // -- one card per thread, not one row per reply. See lib/api.js.
  const [groups, setGroups] = useState([]);
  const [historyGroups, setHistoryGroups] = useState([]);
  const [error, setError] = useState(null);
  const [dryRun, setDryRun] = useState(false);
  // Exchanges mid Send -> fade-out, keyed by exchange id. A poll landing in
  // that ~800ms window would otherwise see the exchange already gone from
  // the backend (the send already succeeded) and yank it out immediately,
  // cutting the fade short -- so a fading exchange is stitched back into its
  // group (creating a one-off group if the thread had no other pending
  // exchange left) until its own timeout clears it via handleRemove.
  const fadingRef = useRef(new Map());

  const load = useCallback(async () => {
    try {
      const [{ groups: freshGroups, dryRun }, historyRows] = await Promise.all([
        fetchQueue(),
        fetchHistory(),
      ]);
      setError(null);
      setDryRun(dryRun);
      setHistoryGroups(historyRows);
      setGroups(() => {
        const merged = freshGroups.map((g) => ({ ...g, exchanges: [...g.exchanges] }));
        const byConversation = new Map(merged.map((g) => [g.conversation_id, g]));
        for (const [exchangeId, snapshot] of fadingRef.current) {
          let group = byConversation.get(snapshot.conversationId);
          if (!group) {
            group = {
              conversation_id: snapshot.conversationId,
              conversation_subject: snapshot.conversationSubject,
              exchanges: [],
            };
            byConversation.set(snapshot.conversationId, group);
            merged.push(group);
          }
          if (!group.exchanges.some((e) => e.id === exchangeId)) {
            group.exchanges.push(snapshot.exchange);
          }
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

  function handleSendSuccess(group, exchange) {
    fadingRef.current.set(exchange.id, {
      conversationId: group.conversation_id,
      conversationSubject: group.conversation_subject,
      exchange,
    });
  }

  function handleRemove(conversationId, exchangeId) {
    fadingRef.current.delete(exchangeId);
    setGroups((gs) =>
      gs
        .map((g) =>
          g.conversation_id === conversationId
            ? { ...g, exchanges: g.exchanges.filter((e) => e.id !== exchangeId) }
            : g
        )
        .filter((g) => g.exchanges.length > 0)
    );
    load(); // pulls the just-sent exchange into History right away, not on the next 20s tick
  }

  const pendingCount = totalExchanges(groups);

  return (
    <div className="wrap">
      {dryRun && (
        <div className="dry-run-banner">Dry run</div>
      )}
      <header>
        <h1>Reply review</h1>
        <div className="toolbar">
          <span className="count">{pendingCount ? `${pendingCount} pending` : ""}</span>
          <button type="button" onClick={load}>Refresh</button>
        </div>
      </header>

      <section>
        <h2 className="section-title">Approval required to send</h2>
        {error && (
          <div id="loadError">Could not load the review queue: {error}</div>
        )}
        {!error && groups.length === 0 && (
          <div id="empty">Nothing waiting for review.</div>
        )}
        {groups.map((group) => (
          <ThreadCard
            key={group.conversation_id}
            subject={group.conversation_subject}
            conversationId={group.conversation_id}
            senderEmail={group.sender_email}
            otherOpenThreads={group.other_open_threads}
            otherSentThreads={group.other_sent_threads}
          >
            <RowHeader labels={REVIEW_LABELS} />
            {group.exchanges.map((exchange) => (
              <ReviewRow
                key={exchange.id}
                row={exchange}
                onSendSuccess={(sentExchange) => handleSendSuccess(group, sentExchange)}
                onRemove={(exchangeId) => handleRemove(group.conversation_id, exchangeId)}
              />
            ))}
          </ThreadCard>
        ))}
      </section>

      <section>
        <h2 className="section-title">History</h2>
        {!error && historyGroups.length === 0 && (
          <div id="empty">Nothing sent yet.</div>
        )}
        {clusterBySender(historyGroups).map((group) => (
          <ThreadCard
            key={group.conversation_id}
            subject={group.conversation_subject}
            conversationId={group.conversation_id}
            senderEmail={group.sender_email}
            otherOpenThreads={group.other_open_threads}
            otherSentThreads={group.other_sent_threads}
          >
            <RowHeader labels={HISTORY_LABELS} />
            {group.exchanges.map((exchange) => (
              <HistoryRow key={exchange.id} row={exchange} />
            ))}
          </ThreadCard>
        ))}
      </section>
    </div>
  );
}
