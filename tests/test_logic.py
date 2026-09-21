"""Regression tests for this session's work: the ground()/refuses_in_prose()
salvage logic, exact-id targeting for record_reply, the follow-up
mechanism, sender identity + near-duplicate detection, and the review
queue's conversation grouping.

Run inside the classifier container, where the environment this code
actually expects (LiteLLM/Chroma reachability, env vars) is already correct:

    docker cp tests/test_logic.py email-classifier-api:/app/tests/test_logic.py
    docker exec email-classifier-api python3 -m unittest tests.test_logic -v

Two tiers, run together by default:

- Unit tests (the majority): no network. A fresh temp sqlite file per test
  case isolates threads.store state; ground()/render()/
  _group_by_conversation() are pure functions exercised directly with
  synthetic input -- no DB, no model calls.
- Integration tests (class-level `_require_network()` check, each skipped
  with a clear reason if the embedder isn't reachable): is_near_duplicate()
  needs a real bge-m3 call.

Everything here is a *regression* test for a bug that was actually found
and fixed this session, not a speculative "what if" -- see each test's
docstring for the failure it guards against.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rag.core import (  # noqa: E402
    DUPLICATE_MAX_DISTANCE,
    NO_ANSWER,
    ground,
    is_near_duplicate,
    refuses_in_prose,
)
from threads import context as thread_context  # noqa: E402
from threads import store as thread_store  # noqa: E402
from threads.store import Exchange  # noqa: E402


def _network_reachable() -> bool:
    try:
        is_near_duplicate("ping", "ping")
        return True
    except Exception:
        return False


_NETWORK_OK = _network_reachable()


# =============================================================================
# ground() / refuses_in_prose() -- the salvage logic behind the related-topic
# fallback. Every documented shape in ground()'s own docstring, plus the two
# new ones added this session.
# =============================================================================


class TestGround(unittest.TestCase):
    def test_bare_refusal_token(self):
        self.assertEqual(ground(NO_ANSWER), "")

    def test_empty_string(self):
        self.assertEqual(ground(""), "")
        self.assertEqual(ground("   "), "")

    def test_trailing_token_after_partial_answer(self):
        """A model asked for the whole reply sometimes answers what it can
        and appends the bare token for the part it can't, instead of
        replacing the whole reply as instructed -- that's a partial answer,
        not nothing, so only the token is stripped."""
        text = f"Level 1 covers basics of automotive networking. {NO_ANSWER}"
        self.assertEqual(
            ground(text), "Level 1 covers basics of automotive networking."
        )

    def test_trailing_honest_caveat_survives_unchanged(self):
        """'X, but the documentation does not specify Y' is a real answer
        with a caveat welded on, not a refusal -- must NOT be touched."""
        text = (
            "Level 1 does not include hands-on exercises, but the "
            "documentation does not specify whether handouts are allowed."
        )
        self.assertEqual(ground(text), text)

    def test_pure_refusal_prose_nothing_to_salvage(self):
        text = "The documentation does not specify pricing for this course."
        self.assertEqual(ground(text), "")

    def test_leading_refusal_with_real_content_is_salvaged(self):
        """Regression test: this exact case reached production. The model
        found genuinely relevant related-topic content but opened with a
        refusal sentence anyway despite explicit prompt instructions not
        to -- discarding the whole reply over the opening sentence would
        throw away a correct, useful answer."""
        text = (
            "The documentation does not specify a particular hardware kit "
            "or Bill of Materials (BOM) for the EVH Level 3 course. While "
            "Level 2 training includes a dedicated embedded hardware kit "
            "provided to each participant, Level 3 is designed as on-site "
            "training where practical exercises use real ECUs that can be "
            "provided by the client or selected by BreachLabz from their "
            "R&D department."
        )
        expected = (
            "While Level 2 training includes a dedicated embedded hardware "
            "kit provided to each participant, Level 3 is designed as "
            "on-site training where practical exercises use real ECUs that "
            "can be provided by the client or selected by BreachLabz from "
            "their R&D department."
        )
        self.assertEqual(ground(text), expected)

    def test_leading_refusal_with_trivial_remainder_still_discarded(self):
        text = "The documentation does not specify this. OK."
        self.assertEqual(ground(text), "")

    def test_double_leading_refusal_then_real_content_iterates(self):
        text = (
            "The documentation does not specify this. The extracts do not "
            "mention it either. Level 2 includes a hardware kit provided "
            "to each participant."
        )
        self.assertEqual(
            ground(text),
            "Level 2 includes a hardware kit provided to each participant.",
        )

    def test_normal_grounded_answer_unaffected(self):
        text = "Level 2 training includes a dedicated hardware kit provided to each participant."
        self.assertEqual(ground(text), text)

    def test_refuses_in_prose_ignores_negation_without_source_word(self):
        """'Level 1 does not include hands-on exercises' is a real answer
        about content -- a negation alone must not trip the refusal check,
        only a source word (documentation/extracts/...) negated together."""
        self.assertFalse(
            refuses_in_prose("Level 1 does not include hands-on exercises.")
        )


# =============================================================================
# threads.store -- exact-id targeting, follow-ups, sender identity.
# Each test gets its own temp sqlite file so nothing leaks between cases.
# =============================================================================


class StoreTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.db_path = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self.db_path.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            pathlib.Path(str(self.db_path) + suffix).unlink(missing_ok=True)


class TestRecordInboundAndReply(StoreTestCase):
    def test_record_inbound_returns_new_row_id(self):
        exchange_id = thread_store.record_inbound(
            "conv-1", "hello", path=self.db_path
        )
        self.assertIsInstance(exchange_id, int)
        self.assertGreater(exchange_id, 0)

    def test_record_inbound_duplicate_message_id_returns_none(self):
        first = thread_store.record_inbound(
            "conv-1", "hello", message_id="msg-1", path=self.db_path
        )
        second = thread_store.record_inbound(
            "conv-1", "hello again", message_id="msg-1", path=self.db_path
        )
        self.assertIsNotNone(first)
        self.assertIsNone(second)

    def test_record_reply_fallback_heuristic_targets_most_recent_unanswered(self):
        thread_store.record_inbound("conv-1", "question one", path=self.db_path)
        thread_store.record_reply("conv-1", "answer one", path=self.db_path)
        rows = {
            e.id: e for e in thread_store.history("conv-1", path=self.db_path)
        }
        self.assertEqual(list(rows.values())[0].reply, "answer one")

    def test_record_reply_fallback_excludes_followup_rows(self):
        """A pending follow-up placeholder must never steal a reply meant
        for a real, still-unanswered inbound turn via the fallback
        heuristic -- see record_reply's own docstring for why."""
        real_id = thread_store.record_inbound(
            "conv-1", "real question", path=self.db_path
        )
        thread_store.create_followup(
            "conv-1", "some future follow-up topic", "2099-01-01T00:00:00+00:00",
            path=self.db_path,
        )
        # The follow-up has a HIGHER id than the real turn -- exactly the
        # shape that broke the naive "highest id, reply IS NULL" heuristic.
        thread_store.record_reply("conv-1", "the real answer", path=self.db_path)
        real_row = thread_store.get_exchange(real_id, path=self.db_path)
        self.assertEqual(real_row["reply"], "the real answer")

    def test_record_reply_exact_id_ignores_a_stuck_unanswered_row(self):
        """Regression test for the exact misattachment bug found in this
        session: a near-duplicate suppressed by is_near_duplicate stays
        `reply IS NULL` forever (nothing ever drafts an answer for it), and
        if a LATER, unrelated reply is recorded via the "most recent
        unanswered" fallback it can land on that stuck row instead of the
        turn it was actually drafted for. Passing exchange_id must always
        target the exact row regardless of what else is unanswered.
        """
        first_id = thread_store.record_inbound(
            "conv-1", "When does Level 2 start?", path=self.db_path
        )
        stuck_id = thread_store.record_inbound(
            "conv-1", "What's the Level 2 start date?", path=self.db_path
        )  # simulates a near-duplicate that is never drafted -- stays NULL
        third_id = thread_store.record_inbound(
            "conv-1", "Do you offer a discount?", path=self.db_path
        )
        self.assertLess(first_id, stuck_id)
        self.assertLess(stuck_id, third_id)

        thread_store.record_reply(
            "conv-1", "Level 2 starts on the 1st.", exchange_id=first_id, path=self.db_path
        )
        thread_store.record_reply(
            "conv-1", "Yes, 10% early-bird.", exchange_id=third_id, path=self.db_path
        )

        first_row = thread_store.get_exchange(first_id, path=self.db_path)
        stuck_row = thread_store.get_exchange(stuck_id, path=self.db_path)
        third_row = thread_store.get_exchange(third_id, path=self.db_path)
        self.assertEqual(first_row["reply"], "Level 2 starts on the 1st.")
        self.assertEqual(third_row["reply"], "Yes, 10% early-bird.")
        self.assertIsNone(stuck_row["reply"])  # never touched


class TestFollowups(StoreTestCase):
    def test_create_followup_is_a_placeholder_not_a_real_query(self):
        exchange_id = thread_store.create_followup(
            "conv-1", "Payment link topic", "2020-01-01T00:00:00+00:00",
            path=self.db_path,
        )
        row = thread_store.get_exchange(exchange_id, path=self.db_path)
        self.assertEqual(row["is_followup"], 1)
        self.assertEqual(row["query"], "Payment link topic")
        self.assertIsNone(row["reply"])

    def test_create_followup_auto_derives_ref_from_latest_real_exchange(self):
        thread_store.record_inbound(
            "conv-1", "question", ref="graph-msg-123", path=self.db_path
        )
        exchange_id = thread_store.create_followup(
            "conv-1", "topic", "2020-01-01T00:00:00+00:00", path=self.db_path
        )
        row = thread_store.get_exchange(exchange_id, path=self.db_path)
        self.assertEqual(row["ref"], "graph-msg-123")

    def test_due_followups_excludes_future_and_drafted_rows(self):
        due_id = thread_store.create_followup(
            "conv-1", "due topic", "2020-01-01T00:00:00+00:00", path=self.db_path
        )
        future_id = thread_store.create_followup(
            "conv-1", "future topic", "2099-01-01T00:00:00+00:00", path=self.db_path
        )
        already_drafted_id = thread_store.create_followup(
            "conv-1", "drafted topic", "2020-01-01T00:00:00+00:00", path=self.db_path
        )
        thread_store.draft_followup(
            already_drafted_id, "already drafted reply", path=self.db_path
        )

        due = {row["id"] for row in thread_store.due_followups(path=self.db_path)}
        self.assertIn(due_id, due)
        self.assertNotIn(future_id, due)
        self.assertNotIn(already_drafted_id, due)

    def test_draft_followup_targets_exact_row_only(self):
        stage_a = thread_store.create_followup(
            "conv-1", "stage A", "2020-01-01T00:00:00+00:00", path=self.db_path
        )
        stage_b = thread_store.create_followup(
            "conv-1", "stage B", "2020-01-01T00:00:00+00:00", path=self.db_path
        )
        thread_store.draft_followup(stage_a, "reply for A", path=self.db_path)

        row_a = thread_store.get_exchange(stage_a, path=self.db_path)
        row_b = thread_store.get_exchange(stage_b, path=self.db_path)
        self.assertEqual(row_a["reply"], "reply for A")
        self.assertIsNone(row_b["reply"])  # independent -- no chaining coupling


class TestSenderIdentity(StoreTestCase):
    def test_sender_email_normalised_case_and_whitespace(self):
        thread_store.ensure_conversation(
            "conv-1", sender_email=" John.Doe@Example.com ", path=self.db_path
        )
        with thread_store.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT sender_email FROM conversations WHERE conversation_id = ?",
                ("conv-1",),
            ).fetchone()
        self.assertEqual(row["sender_email"], "john.doe@example.com")

    def test_ensure_conversation_keeps_first_sender_email(self):
        thread_store.ensure_conversation(
            "conv-1", sender_email="first@example.com", path=self.db_path
        )
        thread_store.ensure_conversation(
            "conv-1", sender_email="second@example.com", path=self.db_path
        )
        with thread_store.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT sender_email FROM conversations WHERE conversation_id = ?",
                ("conv-1",),
            ).fetchone()
        self.assertEqual(row["sender_email"], "first@example.com")

    def test_sender_threads_finds_other_open_threads(self):
        thread_store.record_inbound(
            "conv-a", "q1", sender_email="alex@example.com", path=self.db_path
        )
        thread_store.record_inbound(
            "conv-b", "q2", sender_email="Alex@Example.com", path=self.db_path
        )
        others = thread_store.sender_threads(
            "alex@example.com", exclude_conversation_id="conv-a", path=self.db_path
        )
        self.assertEqual([t["conversation_id"] for t in others], ["conv-b"])

    def test_sender_threads_excludes_already_sent_threads(self):
        thread_store.record_inbound(
            "conv-a", "q1", sender_email="alex@example.com", path=self.db_path
        )
        exchange_id = thread_store.record_inbound(
            "conv-b", "q2", sender_email="alex@example.com", path=self.db_path
        )
        thread_store.record_reply(
            "conv-b", "an answer", grounded=True, exchange_id=exchange_id,
            path=self.db_path,
        )
        thread_store.mark_sent(exchange_id, "an answer", path=self.db_path)

        others = thread_store.sender_threads(
            "alex@example.com", exclude_conversation_id="conv-a", path=self.db_path
        )
        self.assertEqual(others, [])  # conv-b is fully resolved, not "open"

    def test_sender_threads_blank_address_never_matches(self):
        thread_store.record_inbound("conv-a", "q1", path=self.db_path)  # no sender
        thread_store.record_inbound("conv-b", "q2", path=self.db_path)  # no sender
        others = thread_store.sender_threads(
            "", exclude_conversation_id="conv-a", path=self.db_path
        )
        self.assertEqual(others, [])


# =============================================================================
# threads.context.render() -- a follow-up must never be misattributed to
# the enquirer. Pure function, synthetic Exchange objects, no DB needed.
# =============================================================================


class TestRenderFollowupAttribution(unittest.TestCase):
    def test_normal_exchange_renders_as_enquirer_and_reply(self):
        out = thread_context.render(
            [Exchange(id=1, query="hi", reply="hello back")]
        )
        self.assertIn("Enquirer:\nhi", out)
        self.assertIn("Our reply:\nhello back", out)

    def test_unsent_followup_is_invisible_even_if_drafted(self):
        """Regression test: this exact bug shipped in the initial follow-up
        implementation -- render() attributed the follow-up's topic label
        to the enquirer as if they had asked it."""
        out = thread_context.render(
            [
                Exchange(
                    id=1,
                    query="payment link topic",
                    reply="here is the link",
                    is_followup=True,
                    sent=False,
                )
            ]
        )
        self.assertEqual(out, "")

    def test_sent_followup_renders_as_our_own_message_never_enquirer(self):
        out = thread_context.render(
            [
                Exchange(
                    id=1,
                    query="payment link topic",
                    reply="here is the link",
                    is_followup=True,
                    sent=True,
                )
            ]
        )
        self.assertNotIn("Enquirer:", out)
        self.assertIn("here is the link", out)

    def test_mixed_history_treats_each_row_correctly(self):
        out = thread_context.render(
            [
                Exchange(id=1, query="real question", reply="real answer"),
                Exchange(
                    id=2, query="followup topic", reply="followup text",
                    is_followup=True, sent=True,
                ),
                Exchange(id=3, query="second real question", reply=None),
            ]
        )
        self.assertIn("Enquirer:\nreal question", out)
        self.assertIn("Our reply:\nreal answer", out)
        self.assertNotIn("followup topic", out)  # the topic label itself never leaks
        self.assertIn("followup text", out)
        self.assertIn("Enquirer:\nsecond real question", out)
        # id 3 has no reply yet -- no "Our reply:" line for it.
        self.assertEqual(out.count("Our reply:"), 1)


# =============================================================================
# api._group_by_conversation() -- pure function, list of dicts in, grouped
# list out. Imported lazily so a missing FastAPI/uvicorn install doesn't
# block the rest of this file from running.
# =============================================================================


class TestGroupByConversation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from api import _group_by_conversation
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.py not importable here: {exc}")
        cls._group_by_conversation = staticmethod(_group_by_conversation)

    def test_folds_same_conversation_into_one_group(self):
        rows = [
            {"id": 1, "conversation_id": "c1", "conversation_subject": "Subj"},
            {"id": 2, "conversation_id": "c1", "conversation_subject": "Subj"},
        ]
        groups = self._group_by_conversation(rows)
        self.assertEqual(len(groups), 1)
        self.assertEqual([e["id"] for e in groups[0]["exchanges"]], [1, 2])

    def test_conversation_subject_hoisted_not_repeated_per_exchange(self):
        rows = [{"id": 1, "conversation_id": "c1", "conversation_subject": "Subj"}]
        groups = self._group_by_conversation(rows)
        self.assertEqual(groups[0]["conversation_subject"], "Subj")
        self.assertNotIn("conversation_subject", groups[0]["exchanges"][0])

    def test_different_conversations_stay_separate_in_first_seen_order(self):
        rows = [
            {"id": 1, "conversation_id": "c2", "conversation_subject": "B"},
            {"id": 2, "conversation_id": "c1", "conversation_subject": "A"},
            {"id": 3, "conversation_id": "c2", "conversation_subject": "B"},
        ]
        groups = self._group_by_conversation(rows)
        self.assertEqual([g["conversation_id"] for g in groups], ["c2", "c1"])
        self.assertEqual(len(groups[0]["exchanges"]), 2)
        self.assertEqual(len(groups[1]["exchanges"]), 1)


# =============================================================================
# is_near_duplicate() -- needs a real embedder call. Skipped as a whole class
# (not per-test) if the classifier's embedding backend isn't reachable.
# =============================================================================


@unittest.skipUnless(_NETWORK_OK, "embedder not reachable from this environment")
class TestNearDuplicate(unittest.TestCase):
    def test_reworded_question_is_a_near_duplicate(self):
        self.assertTrue(
            is_near_duplicate(
                "When does the next EVH Level 2 cohort start?",
                "Hi, just checking -- what's the start date for the upcoming Level 2 cohort?",
            )
        )

    def test_synonym_swap_is_a_near_duplicate(self):
        self.assertTrue(
            is_near_duplicate(
                "What is the price for EVH Level 2?",
                "How much does EVH Level 2 cost?",
            )
        )

    def test_genuinely_different_question_same_thread_is_not(self):
        self.assertFalse(
            is_near_duplicate(
                "When does the next EVH Level 2 cohort start?",
                "Also, do you offer any discount for early enrollment?",
            )
        )

    def test_different_topic_entirely_is_not(self):
        self.assertFalse(
            is_near_duplicate(
                "What is the price for EVH Level 2?",
                "What hardware kit do I need for EVH Level 3?",
            )
        )

    def test_blank_input_never_flags_or_crashes(self):
        self.assertFalse(is_near_duplicate("", "anything"))
        self.assertFalse(is_near_duplicate("anything", "   "))

    def test_default_threshold_is_positive_and_below_max_distance(self):
        # Sanity check on the constant itself, not just behaviour -- catches
        # a config typo (e.g. "1.5") that no functional test above would.
        self.assertGreater(DUPLICATE_MAX_DISTANCE, 0)
        self.assertLess(DUPLICATE_MAX_DISTANCE, 1.0)


class TestReplyHtml(unittest.TestCase):
    """The review edit box's **bold** / *italic* markers must reach the
    recipient as real formatting -- and text without them must keep going out
    as the plain `comment` (reply_html -> None), stray asterisks included."""

    def test_bold_and_italic(self):
        from mail.graph import reply_html

        self.assertEqual(
            reply_html("Hello **there** and *you*"),
            "<p>Hello <strong>there</strong> and <em>you</em></p>",
        )

    def test_plain_text_returns_none(self):
        from mail.graph import reply_html

        self.assertIsNone(reply_html("Just a plain reply.\n\n- one\n- two"))

    def test_stray_asterisks_stay_literal(self):
        from mail.graph import reply_html

        for text in ("5 * 3 * 2", "a*b*c", "a lone * here", "** **", "**"):
            self.assertIsNone(reply_html(text), text)

    def test_html_in_reply_is_escaped(self):
        from mail.graph import reply_html

        self.assertEqual(
            reply_html("**x** <script>alert(1)</script> & co"),
            "<p><strong>x</strong> &lt;script&gt;alert(1)&lt;/script&gt; &amp; co</p>",
        )

    def test_paragraphs_and_line_breaks(self):
        from mail.graph import reply_html

        self.assertEqual(
            reply_html("**Hi**\nline two\n\nSecond para"),
            "<p><strong>Hi</strong><br>line two</p><p>Second para</p>",
        )

    def test_send_reply_payload_shape(self):
        from unittest import mock

        import mail.graph as g

        with mock.patch.object(g, "configured", return_value=True), \
             mock.patch.object(g, "_access_token", return_value="t"), \
             mock.patch.object(g.requests, "post") as post:
            g.send_reply("msg-1", "plain")
            self.assertEqual(post.call_args.kwargs["json"], {"comment": "plain"})
            g.send_reply("msg-1", "**bold**")
            self.assertEqual(
                post.call_args.kwargs["json"],
                {"message": {"body": {"contentType": "HTML", "content": "<p><strong>bold</strong></p>"}}},
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
