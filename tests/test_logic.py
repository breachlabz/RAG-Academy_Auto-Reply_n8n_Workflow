"""Regression tests for this session's work: the ground()/refuses_in_prose()
salvage logic, exact-id targeting for record_reply, the follow-up
mechanism, sender identity + near-duplicate detection, the review queue's
conversation grouping, the end-to-end answering shapes (grounded/ungrounded/
related-topic through answer() itself, not just ground()), reply-format
detection (list vs prose), the standalone-question rewrite's fail-safe
degrade, and the /generate-reply endpoint's dedupe/near-duplicate/routing
branches.

Run inside the classifier container, where the environment this code
actually expects (LiteLLM/Chroma reachability, env vars) is already correct:

    docker cp tests/test_logic.py email-classifier-api:/app/tests/test_logic.py
    docker exec email-classifier-api python3 -m unittest tests.test_logic -v

Two tiers, run together by default:

- Unit tests (the majority): no network. A fresh temp sqlite file per test
  case isolates threads.store state; ground()/render()/
  _group_by_conversation() are pure functions exercised directly with
  synthetic input; answer()/standalone_question()/generate_reply() are
  exercised with retrieve()/requests.post()/classify()/rag_answer() mocked
  out, so the branching logic is tested without a model or Chroma call --
  no DB, no model calls.
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
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import rag.core as rag_core  # noqa: E402
import threads.context as thread_context_module  # noqa: E402
from rag.core import (  # noqa: E402
    DUPLICATE_MAX_DISTANCE,
    NO_ANSWER,
    Answer,
    Chunk,
    answer,
    ground,
    is_near_duplicate,
    refuses_in_prose,
)
from classifier.core import Result  # noqa: E402
from rag.core import NO_INFO_REPLY  # noqa: E402
from rag.format_hint import as_bullets, resolve, wants_list  # noqa: E402
from threads import context as thread_context  # noqa: E402
from threads import store as thread_store  # noqa: E402
from threads.store import Exchange  # noqa: E402


def _mock_response(payload: dict) -> mock.Mock:
    """A requests.Response stand-in for a chat-completion call: .json()
    returns `payload`, .raise_for_status() is a no-op (the happy path)."""
    resp = mock.Mock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    return resp


def _chat_payload(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


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
# rag.core.answer() -- the end-to-end answering shapes a single enquiry can
# take, with retrieve() and the chat-completion call mocked out so the gate
# and grounding logic is exercised without a model or Chroma. Complements
# TestGround above, which tests ground() directly on hand-written model
# output; these test answer() as a whole, including the retrieval-distance
# gate that ground() never sees.
# =============================================================================


class TestAnswerGrounding(unittest.TestCase):
    def _chunk(self, distance: float, heading: str = "EVH > Level 2") -> Chunk:
        return Chunk(text="Level 2 body text.", source="evh.docx", heading=heading, distance=distance)

    def test_content_in_docs_is_grounded_with_source(self):
        """A question the documents answer directly: single query, single
        prose reply, source attached."""
        chunk = self._chunk(0.15)
        with mock.patch.object(rag_core, "retrieve", return_value=[chunk]), \
             mock.patch.object(
                 rag_core.requests, "post",
                 return_value=_mock_response(_chat_payload("Level 2 covers CAN and UDS.")),
             ):
            result = answer("What does level 2 cover?")
        self.assertTrue(result.ok)
        self.assertTrue(result.grounded)
        self.assertEqual(result.text, "Level 2 covers CAN and UDS.")
        self.assertEqual(result.sources, ["EVH > Level 2"])

    def test_content_not_in_docs_is_ungrounded(self):
        """The model emits the bare refusal token -- content genuinely absent
        from the corpus, not a retrieval miss."""
        chunk = self._chunk(0.4)
        with mock.patch.object(rag_core, "retrieve", return_value=[chunk]), \
             mock.patch.object(
                 rag_core.requests, "post",
                 return_value=_mock_response(_chat_payload(NO_ANSWER)),
             ):
            result = answer("What is the CVSS score for a CAN injection attack?")
        self.assertTrue(result.ok)
        self.assertFalse(result.grounded)
        self.assertEqual(result.text, "")
        self.assertIn("member of our team", result.reply)

    def test_related_topic_answer_salvaged_end_to_end(self):
        """The model opens with a refusal sentence but real, related-topic
        content follows -- answer() must salvage it the same way ground()
        does on its own (see TestGround), not just when called directly."""
        leading_refusal = (
            "The documentation does not specify a hardware kit for Level 3. "
            "Level 2 training includes a dedicated embedded hardware kit "
            "provided to each participant."
        )
        chunk = self._chunk(0.3)
        with mock.patch.object(rag_core, "retrieve", return_value=[chunk]), \
             mock.patch.object(
                 rag_core.requests, "post",
                 return_value=_mock_response(_chat_payload(leading_refusal)),
             ):
            result = answer("What hardware kit does level 3 use?")
        self.assertTrue(result.grounded)
        self.assertEqual(
            result.text,
            "Level 2 training includes a dedicated embedded hardware kit "
            "provided to each participant.",
        )

    def test_no_chunks_retrieved_is_ungrounded_not_an_error(self):
        with mock.patch.object(rag_core, "retrieve", return_value=[]):
            result = answer("Do you teach basket weaving?")
        self.assertTrue(result.ok)
        self.assertFalse(result.grounded)
        self.assertEqual(result.chunks, [])

    def test_chunks_past_max_distance_are_ungrounded(self):
        """Retrieval found something, but nothing close enough -- distinct
        from the no-chunks-at-all case: `chunks` is populated (so a caller
        can still log the closest distance) even though nothing was kept."""
        far_chunks = [self._chunk(1.4), self._chunk(1.8)]
        with mock.patch.object(rag_core, "retrieve", return_value=far_chunks):
            result = answer("Unrelated question", max_distance=1.0)
        self.assertTrue(result.ok)
        self.assertFalse(result.grounded)
        self.assertEqual(result.chunks, far_chunks)

    def test_retrieval_failure_is_an_error_not_ungrounded(self):
        """A failed request says nothing about whether the documents cover
        the question -- callers must be able to tell this apart from a
        genuine, successful ungrounded answer (see api.py's /answer, which
        only records a reply on the ok-but-ungrounded path)."""
        with mock.patch.object(rag_core, "retrieve", side_effect=RuntimeError("chroma down")):
            result = answer("What does level 1 cover?")
        self.assertFalse(result.ok)
        self.assertFalse(result.grounded)
        self.assertIn("chroma down", result.error)

    def test_model_request_failure_is_an_error(self):
        chunk = self._chunk(0.2)
        with mock.patch.object(rag_core, "retrieve", return_value=[chunk]), \
             mock.patch.object(
                 rag_core.requests, "post",
                 side_effect=rag_core.requests.RequestException("connection refused"),
             ):
            result = answer("What does level 1 cover?")
        self.assertFalse(result.ok)
        self.assertFalse(result.grounded)

    def test_multi_question_forced_list_coerces_paragraph_into_bullets(self):
        """Enquirer asked several things (as_list=True, the caller's job --
        see TestFormatHint) but the small model still answered in one
        paragraph -- the as_bullets() net inside answer() must split it."""
        paragraph = (
            "Level 1 covers the fundamentals of automotive networking. "
            "Level 2 adds hands-on hardware exercises with a provided kit. "
            "Level 3 is on-site training using real client ECUs."
        )
        chunk = self._chunk(0.2)
        with mock.patch.object(rag_core, "retrieve", return_value=[chunk]), \
             mock.patch.object(
                 rag_core.requests, "post",
                 return_value=_mock_response(_chat_payload(paragraph)),
             ):
            result = answer(
                "What does each level cover, and how is level 3 different, and "
                "does level 2 include hardware?",
                as_list=True,
            )
        self.assertTrue(result.grounded)
        self.assertTrue(result.text.startswith("- "))
        self.assertEqual(result.text.count("\n- "), 2)  # 3 bullets total


# =============================================================================
# rag.format_hint -- pure functions, no mocking. Decides list vs prose from
# the enquirer's own wording, and the as_bullets() net that coerces a
# paragraph the model produced anyway.
# =============================================================================


class TestFormatHint(unittest.TestCase):
    def test_own_bulleted_list_wants_list(self):
        self.assertTrue(wants_list("Questions:\n- price?\n- start date?"))

    def test_own_numbered_list_wants_list(self):
        self.assertTrue(wants_list("1. What is the price?\n2. When does it start?"))

    def test_two_questions_wants_list(self):
        self.assertTrue(wants_list("What is the price? When does it start?"))

    def test_explicit_ask_for_points_wants_list(self):
        self.assertTrue(wants_list("Can you break this down for me in points?"))

    def test_set_shaped_question_wants_list(self):
        self.assertTrue(wants_list("What are the prerequisites for level 2?"))

    def test_plain_single_question_is_prose(self):
        self.assertFalse(wants_list("What does level 2 cost?"))

    def test_empty_text_is_prose(self):
        self.assertFalse(wants_list(""))

    def test_resolve_list_mode_forces_list_regardless_of_text(self):
        self.assertIs(resolve("list", "What does level 2 cost?"), True)

    def test_resolve_prose_mode_forces_prose_regardless_of_text(self):
        self.assertIs(resolve("prose", "1. a?\n2. b?"), False)

    def test_resolve_auto_defers_to_wants_list(self):
        self.assertIs(resolve("auto", "What is the price? When does it start?"), True)
        self.assertIsNone(resolve("auto", "What does level 2 cost?"))

    def test_as_bullets_noop_when_already_a_list(self):
        text = "- one\n- two"
        self.assertEqual(as_bullets(text), text)

    def test_as_bullets_noop_when_single_sentence(self):
        text = "Level 2 costs one thousand dollars."
        self.assertEqual(as_bullets(text), text)

    def test_as_bullets_splits_multi_sentence_paragraph(self):
        text = (
            "Level 1 covers networking fundamentals. Level 2 adds hardware "
            "exercises. Level 3 is on-site with real ECUs."
        )
        out = as_bullets(text)
        self.assertEqual(out.count("\n- ") + 1, 3)
        self.assertTrue(out.startswith("- "))

    def test_as_bullets_keeps_short_lead_in_line(self):
        text = "You will need: A laptop capable of running a VM. A CAN interface for the exercises."
        out = as_bullets(text)
        self.assertTrue(out.startswith("You will need:\n- "))


# =============================================================================
# threads.context.standalone_question -- the follow-up rewrite step, mocked
# at the chat-completion call. Its fail-safe contract matters as much as the
# happy path: a follow-up must never be dropped or mangled just because the
# rewrite call failed.
# =============================================================================


class TestStandaloneQuestionRewrite(unittest.TestCase):
    def test_no_prior_history_skips_rewrite_entirely(self):
        with mock.patch.object(thread_context_module.requests, "post") as post:
            out = thread_context.standalone_question("What are the levels?", [])
        self.assertEqual(out, "What are the levels?")
        post.assert_not_called()

    def test_follow_up_rewritten_using_prior_context(self):
        prior = [Exchange(id=1, query="What are the levels?", reply="Three levels.")]
        with mock.patch.object(
            thread_context_module.requests, "post",
            return_value=_mock_response(
                _chat_payload("What does the second level cover?")
            ),
        ):
            out = thread_context.standalone_question("And the second one?", prior)
        self.assertEqual(out, "What does the second level cover?")

    def test_rewrite_call_failure_degrades_to_original_text(self):
        """The rewrite is an enhancement, not a requirement -- a thread that
        cannot reach the model must still produce a draft from the raw
        (if under-specified) question rather than fail the email."""
        prior = [Exchange(id=1, query="What are the levels?", reply="Three levels.")]
        with mock.patch.object(
            thread_context_module.requests, "post",
            side_effect=thread_context_module.requests.RequestException("timeout"),
        ):
            out = thread_context.standalone_question("And the second one?", prior)
        self.assertEqual(out, "And the second one?")

    def test_rewrite_that_answers_instead_of_rewriting_is_discarded(self):
        """A rewrite far longer than the original is the model having padded
        or answered the question rather than condensed it -- the original is
        safer to embed than an invented one."""
        prior = [Exchange(id=1, query="What are the levels?", reply="Three levels.")]
        bloated = "Well, " * 200 + "what does the second level cover?"
        with mock.patch.object(
            thread_context_module.requests, "post",
            return_value=_mock_response(_chat_payload(bloated)),
        ):
            out = thread_context.standalone_question("And the second one?", prior)
        self.assertEqual(out, "And the second one?")


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

    def test_mark_sent_stores_attachment_name(self):
        """Only the filename crosses into storage -- api.py never passes the
        attachment's bytes here, and mark_sent has nowhere to put them even
        if it did (no such column). See attachment_name's comment on
        _ADDED_COLUMNS."""
        exchange_id = thread_store.record_inbound("conv-1", "q", path=self.db_path)
        thread_store.record_reply("conv-1", "a", path=self.db_path)
        thread_store.mark_sent(
            exchange_id, "a", attachment_name="notes.pdf", path=self.db_path
        )
        row = thread_store.get_exchange(exchange_id, path=self.db_path)
        self.assertEqual(row["attachment_name"], "notes.pdf")

    def test_mark_sent_without_attachment_leaves_it_null(self):
        exchange_id = thread_store.record_inbound("conv-1", "q", path=self.db_path)
        thread_store.mark_sent(exchange_id, "a", path=self.db_path)
        row = thread_store.get_exchange(exchange_id, path=self.db_path)
        self.assertIsNone(row["attachment_name"])


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


class TestReplyAttachment(unittest.TestCase):
    """The review edit box's one-file attachment (see lib/attachment.js):
    only the filename is ever meant to survive on this side -- these guard
    the size cap, the Graph payload shape, and that a small file round-trips
    unmodified through base64 into Graph's fileAttachment dict."""

    def test_over_limit_rejected(self):
        from mail.graph import Attachment, AttachmentTooLarge, MAX_ATTACHMENT_BYTES

        with self.assertRaises(AttachmentTooLarge):
            Attachment(
                name="big.bin",
                content_type="application/octet-stream",
                content_bytes=b"x" * (MAX_ATTACHMENT_BYTES + 1),
            )

    def test_at_limit_accepted(self):
        from mail.graph import Attachment, MAX_ATTACHMENT_BYTES

        Attachment(
            name="exact.bin",
            content_type="application/octet-stream",
            content_bytes=b"x" * MAX_ATTACHMENT_BYTES,
        )  # must not raise

    def test_as_graph_dict_round_trips_bytes(self):
        import base64

        from mail.graph import Attachment

        att = Attachment(name="a.txt", content_type="text/plain", content_bytes=b"hello")
        d = att.as_graph_dict()
        self.assertEqual(d["@odata.type"], "#microsoft.graph.fileAttachment")
        self.assertEqual(d["name"], "a.txt")
        self.assertEqual(d["contentType"], "text/plain")
        self.assertEqual(base64.b64decode(d["contentBytes"]), b"hello")

    def test_missing_content_type_defaults(self):
        from mail.graph import Attachment

        d = Attachment(name="a", content_type="", content_bytes=b"x").as_graph_dict()
        self.assertEqual(d["contentType"], "application/octet-stream")

    def test_send_reply_plain_text_keeps_comment_and_adds_attachment(self):
        """Plain text (no bold/italic) + an attachment: comment is still sent
        as-is (quoted thread kept) and the attachment rides in message."""
        from unittest import mock

        import mail.graph as g

        att = g.Attachment(name="a.txt", content_type="text/plain", content_bytes=b"hi")
        with mock.patch.object(g, "configured", return_value=True), \
             mock.patch.object(g, "_access_token", return_value="t"), \
             mock.patch.object(g.requests, "post") as post:
            g.send_reply("msg-1", "plain reply", attachment=att)
            payload = post.call_args.kwargs["json"]
            self.assertEqual(payload["comment"], "plain reply")
            self.assertEqual(payload["message"]["attachments"], [att.as_graph_dict()])

    def test_send_reply_formatted_text_keeps_attachment_alongside_body(self):
        """**bold** + an attachment: message carries both body and attachments
        in the same dict, no comment (matches the plain bold/italic case)."""
        from unittest import mock

        import mail.graph as g

        att = g.Attachment(name="a.txt", content_type="text/plain", content_bytes=b"hi")
        with mock.patch.object(g, "configured", return_value=True), \
             mock.patch.object(g, "_access_token", return_value="t"), \
             mock.patch.object(g.requests, "post") as post:
            g.send_reply("msg-1", "**bold**", attachment=att)
            payload = post.call_args.kwargs["json"]
            self.assertNotIn("comment", payload)
            self.assertEqual(payload["message"]["attachments"], [att.as_graph_dict()])
            self.assertEqual(
                payload["message"]["body"],
                {"contentType": "HTML", "content": "<p><strong>bold</strong></p>"},
            )

    def test_send_reply_no_attachment_has_no_message_key_for_plain_text(self):
        """Regression guard: adding attachment support must not start sending
        an empty `message` key on the ordinary plain-text, no-attachment path."""
        from unittest import mock

        import mail.graph as g

        with mock.patch.object(g, "configured", return_value=True), \
             mock.patch.object(g, "_access_token", return_value="t"), \
             mock.patch.object(g.requests, "post") as post:
            g.send_reply("msg-1", "plain")
            self.assertEqual(post.call_args.kwargs["json"], {"comment": "plain"})


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

    def test_lists_render_as_ul_and_ol_alongside_formatting(self):
        from mail.graph import reply_html

        self.assertEqual(
            reply_html("Intro **now**\n- one\n- *two*\n\n1. first\n2. second\nafter"),
            "<p>Intro <strong>now</strong></p><ul><li>one</li><li><em>two</em></li></ul>"
            "<ol><li>first</li><li>second</li></ol><p>after</p>",
        )

    def test_list_alone_stays_plain_text(self):
        from mail.graph import reply_html

        self.assertIsNone(reply_html("- one\n- two\n\n1. a\n2. b"))

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


# =============================================================================
# api.generate_reply() -- the Outlook path's gate/dedupe/routing branches,
# with classify() and rag_answer() mocked out (no model, no Chroma) so this
# tests the endpoint's own branching, not the model's judgement. Real
# threads.store, pointed at a temp sqlite file per test.
# =============================================================================

_RAG_ROUTE_RESULT = Result(
    flags={"academic": True, "administrative": False, "spam": False},
    probs={"academic": 0.95, "administrative": 0.0, "spam": 0.0},
)
_HUMAN_ROUTE_RESULT = Result(
    flags={"academic": True, "administrative": True, "spam": False},
    probs={"academic": 0.95, "administrative": 0.95, "spam": 0.0},
)


class GenerateReplyFlowTestCase(unittest.TestCase):
    def setUp(self):
        try:
            import api as api_module
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.py not importable here: {exc}")
        self.api = api_module

        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.db_path = pathlib.Path(self._tmp.name)
        self._orig_db_path = thread_store.DB_PATH
        thread_store.DB_PATH = self.db_path

    def tearDown(self):
        thread_store.DB_PATH = self._orig_db_path
        self.db_path.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            pathlib.Path(str(self.db_path) + suffix).unlink(missing_ok=True)

    def _request(self, **overrides):
        fields = {
            "email_text": "What are the levels?",
            "is_html": False,
            "conversation_id": "conv-1",
            "subject": "Enquiry",
        }
        fields.update(overrides)
        return self.api.ReplyRequest(**fields)

    def test_duplicate_message_id_produces_no_second_draft(self):
        grounded = Answer(
            text="Three levels.", grounded=True,
            chunks=[Chunk(text="x", source="s", heading="h", distance=0.1)],
        )
        with mock.patch.object(self.api, "classify", return_value=_RAG_ROUTE_RESULT), \
             mock.patch.object(self.api, "rag_answer", return_value=grounded) as rag_mock:
            first = self.api.generate_reply(self._request(message_id="m1"))
            second = self.api.generate_reply(self._request(message_id="m1"))

        self.assertFalse(first["duplicate"])
        self.assertTrue(first["answered"])

        self.assertTrue(second["duplicate"])
        self.assertFalse(second["answered"])
        self.assertEqual(second["reason"], "already handled (duplicate message_id)")
        rag_mock.assert_called_once()  # never reached on the duplicate

    def test_near_duplicate_blocks_second_draft_without_reaching_rag(self):
        """First email is routed to a human and so stays unanswered
        (`reply IS NULL`) -- the exact condition that makes it `pending` for
        the near-duplicate check on the second email."""
        with mock.patch.object(self.api, "classify", return_value=_HUMAN_ROUTE_RESULT):
            first = self.api.generate_reply(
                self._request(email_text="Why hasn't my invoice been refunded?", message_id="m1")
            )
        self.assertEqual(first["reason"], "not routed to rag")

        with mock.patch.object(self.api, "classify", return_value=_RAG_ROUTE_RESULT), \
             mock.patch.object(self.api, "is_near_duplicate", return_value=True), \
             mock.patch.object(self.api, "rag_answer") as rag_mock:
            second = self.api.generate_reply(
                self._request(email_text="Any update on my refund?", message_id="m2")
            )

        self.assertTrue(second["near_duplicate"])
        self.assertFalse(second["answered"])
        self.assertEqual(
            second["reason"],
            "near-duplicate of a still-unanswered question in this thread",
        )
        rag_mock.assert_not_called()

    def test_not_routed_to_rag_produces_no_draft_and_skips_rag_answer(self):
        with mock.patch.object(self.api, "classify", return_value=_HUMAN_ROUTE_RESULT), \
             mock.patch.object(self.api, "rag_answer") as rag_mock:
            payload = self.api.generate_reply(
                self._request(email_text="Please refund my last payment.", message_id="m1")
            )
        self.assertFalse(payload["answered"])
        self.assertEqual(payload["reason"], "not routed to rag")
        rag_mock.assert_not_called()

    def test_grounded_reply_is_recorded_to_the_thread(self):
        grounded = Answer(
            text="Three levels: 1, 2 and 3.",
            grounded=True,
            chunks=[Chunk(text="x", source="s", heading="EVH > Overview", distance=0.1)],
        )
        with mock.patch.object(self.api, "classify", return_value=_RAG_ROUTE_RESULT), \
             mock.patch.object(self.api, "rag_answer", return_value=grounded):
            payload = self.api.generate_reply(self._request(message_id="m1"))

        self.assertTrue(payload["answered"])
        self.assertEqual(payload["answer"], "Three levels: 1, 2 and 3.")
        exchanges = thread_store.history(payload["conversation_id"])
        self.assertEqual(len(exchanges), 1)
        self.assertTrue(exchanges[0].grounded)
        self.assertIn("Three levels", exchanges[0].reply)

    def test_ungrounded_reply_is_still_recorded_so_the_thread_shows_it(self):
        """A no-information draft is a real reply the enquirer will have
        seen -- it must land in the thread even though `answered` is False,
        so a later follow-up doesn't read as though the question vanished."""
        ungrounded = Answer(
            text="",
            grounded=False,
            chunks=[Chunk(text="x", source="s", heading="h", distance=0.5)],
        )
        with mock.patch.object(self.api, "classify", return_value=_RAG_ROUTE_RESULT), \
             mock.patch.object(self.api, "rag_answer", return_value=ungrounded):
            payload = self.api.generate_reply(
                self._request(email_text="What is the CVSS score for a CAN attack?", message_id="m1")
            )

        self.assertFalse(payload["answered"])
        self.assertEqual(payload["reason"], "no grounded answer in the documents")
        exchanges = thread_store.history(payload["conversation_id"])
        self.assertEqual(len(exchanges), 1)
        self.assertFalse(exchanges[0].grounded)
        self.assertIn(NO_INFO_REPLY, exchanges[0].reply)


if __name__ == "__main__":
    unittest.main(verbosity=2)
