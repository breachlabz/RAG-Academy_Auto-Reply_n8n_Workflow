"""Dense regression suite: one test per behaviour that would break production
if it regressed, with similar cases table-driven via subTest.

Covers classifier routing, ground() salvage, answer() grounding, list/prose
formatting, threads.store dedupe + exact-id replies, the /generate-reply
branches, and the editable knowledge table. Model, Chroma and Graph calls are
mocked; only TestNearDuplicate needs the real embedder and skips without it.

    .venv/bin/python -m unittest tests.test_logic -v
    # or inside the container:
    docker cp tests/test_logic.py email-classifier-api:/app/tests/test_logic.py
    docker exec email-classifier-api python3 -m unittest tests.test_logic -v
"""

from __future__ import annotations

import contextlib
import math
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import rag.core as rag_core  # noqa: E402
from classifier.core import Result, classify, route  # noqa: E402
from rag.core import (  # noqa: E402
    NO_ANSWER,
    NO_INFO_REPLY,
    Answer,
    Chunk,
    answer,
    ground,
    is_near_duplicate,
)
from rag.format_hint import as_bullets, wants_list  # noqa: E402
from threads import store as thread_store  # noqa: E402


def _mock_response(payload: dict) -> mock.Mock:
    resp = mock.Mock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    return resp


def _chat(content: str) -> mock.Mock:
    return _mock_response({"choices": [{"message": {"content": content}}]})


def _result(academic: float, non_academic: float) -> Result:
    return Result(
        flags={"academic": academic >= 0.5, "non_academic": non_academic >= 0.5},
        probs={"academic": academic, "non_academic": non_academic},
    )


def _temp_db(case: unittest.TestCase) -> pathlib.Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    path = pathlib.Path(tmp.name)
    for suffix in ("", "-wal", "-shm"):
        case.addCleanup(pathlib.Path(str(path) + suffix).unlink, missing_ok=True)
    return path


class TestClassifier(unittest.TestCase):
    def test_routing(self):
        """Only a confident, academic-only email is auto-answered."""
        cases = [
            (_result(0.97, 0.02), "rag"),
            (_result(0.99, 0.55), "rag"),    # weak non_academic flag doesn't block
            (_result(0.01, 0.99), "human"),
            (_result(0.99, 0.95), "human"),  # mixed: never auto-answer
            (_result(0.70, 0.05), "human"),  # low confidence
            (Result(error="boom"), "human"),
        ]
        for result, expected in cases:
            with self.subTest(probs=result.probs, error=result.error):
                self.assertEqual(route(result), expected)

    def test_classify_maps_logprobs_to_labels(self):
        tok = lambda p: {"token": "true", "logprob": 0.0,  # noqa: E731
                         "top_logprobs": [{"token": "true", "logprob": math.log(p)}]}
        payload = {"choices": [{
            "message": {"content": '{"academic": true, "non_academic": false}'},
            "logprobs": {"content": [tok(0.96), tok(0.03)]},
        }]}
        with mock.patch("classifier.core.requests.post", return_value=_mock_response(payload)):
            result = classify("When is the Level 2 exam?")
        self.assertAlmostEqual(result.probs["academic"], 0.96)
        self.assertAlmostEqual(result.probs["non_academic"], 0.03)
        self.assertEqual(route(result), "rag")


class TestGround(unittest.TestCase):
    def test_ground_shapes(self):
        answer_text = "Level 2 includes a hardware kit provided to each participant."
        caveat = ("Level 1 does not include hands-on exercises, but the "
                  "documentation does not specify whether handouts are allowed.")
        cases = [
            (NO_ANSWER, ""),
            ("   ", ""),
            ("The documentation does not specify pricing for this course.", ""),
            ("The documentation does not specify this. OK.", ""),
            (f"Level 1 covers networking. {NO_ANSWER}", "Level 1 covers networking."),
            (caveat, caveat),  # honest caveat is not a refusal
            (answer_text, answer_text),
            # leading refusal(s) followed by real content: salvage the content
            ("The documentation does not specify a kit for Level 3. " + answer_text, answer_text),
            ("The documentation does not specify this. The extracts do not "
             "mention it either. " + answer_text, answer_text),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw[:50]):
                self.assertEqual(ground(raw), expected)


class TestAnswer(unittest.TestCase):
    CHUNK = Chunk(text="Level 2 body.", source="evh.docx", heading="EVH > Level 2", distance=0.2)

    def _answer(self, model_output=None, chunks=None, **kw):
        post = (mock.patch.object(rag_core.requests, "post", return_value=_chat(model_output))
                if model_output is not None else mock.patch.object(rag_core.requests, "post"))
        with mock.patch.object(rag_core, "retrieve", return_value=[self.CHUNK] if chunks is None else chunks), post:
            return answer("What does level 2 cover?", **kw)

    def test_grounded_with_source(self):
        result = self._answer("Level 2 covers CAN and UDS.")
        self.assertTrue(result.grounded)
        self.assertEqual(result.text, "Level 2 covers CAN and UDS.")
        self.assertEqual(result.sources, ["EVH > Level 2"])

    def test_ungrounded_is_ok_not_error(self):
        for label, kw in [("refusal token", {"model_output": NO_ANSWER}),
                          ("no chunks", {"chunks": []})]:
            with self.subTest(label):
                result = self._answer(**kw)
                self.assertTrue(result.ok)
                self.assertFalse(result.grounded)

    def test_retrieval_failure_is_error(self):
        with mock.patch.object(rag_core, "retrieve", side_effect=RuntimeError("chroma down")):
            result = answer("What does level 1 cover?")
        self.assertFalse(result.ok)
        self.assertIn("chroma down", result.error)

    def test_forced_list_coerces_paragraph(self):
        result = self._answer(
            "Level 1 covers networking. Level 2 adds hardware. Level 3 is on-site.",
            as_list=True,
        )
        self.assertTrue(result.text.startswith("- "))
        self.assertEqual(result.text.count("\n- "), 2)


class TestFormatHint(unittest.TestCase):
    def test_wants_list(self):
        cases = [
            ("Questions:\n- price?\n- start date?", True),
            ("What is the price? When does it start?", True),
            ("Can you break this down for me in points?", True),
            ("What does level 2 cost?", False),
            ("", False),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(wants_list(text), expected)

    def test_as_bullets(self):
        self.assertEqual(as_bullets("- one\n- two"), "- one\n- two")
        self.assertEqual(as_bullets("One sentence."), "One sentence.")
        out = as_bullets("Level 1 covers basics. Level 2 adds hardware. Level 3 is on-site.")
        self.assertTrue(out.startswith("- "))
        self.assertEqual(out.count("\n- "), 2)


class TestThreadStore(unittest.TestCase):
    def setUp(self):
        self.db = _temp_db(self)

    def test_duplicate_message_id_rejected(self):
        self.assertIsNotNone(thread_store.record_inbound("c", "hi", message_id="m1", path=self.db))
        self.assertIsNone(thread_store.record_inbound("c", "hi again", message_id="m1", path=self.db))

    def test_reply_targets_exact_exchange(self):
        """A stuck unanswered row must never receive another turn's reply."""
        first, stuck, third = (thread_store.record_inbound("c", q, path=self.db)
                               for q in ("start date?", "start date again?", "discount?"))
        thread_store.record_reply("c", "the 1st", exchange_id=first, path=self.db)
        thread_store.record_reply("c", "10% off", exchange_id=third, path=self.db)
        get = lambda i: thread_store.get_exchange(i, path=self.db)["reply"]  # noqa: E731
        self.assertEqual((get(first), get(stuck), get(third)), ("the 1st", None, "10% off"))

    def test_mark_sent(self):
        eid = thread_store.record_inbound("c", "q", path=self.db)
        thread_store.mark_sent(eid, "a", attachment_name="notes.pdf", path=self.db)
        row = thread_store.get_exchange(eid, path=self.db)
        self.assertTrue(row["sent"])
        self.assertEqual(row["attachment_name"], "notes.pdf")


class TestGenerateReply(unittest.TestCase):
    RAG = _result(0.95, 0.0)
    HUMAN = _result(0.95, 0.95)
    GROUNDED = Answer(text="Three levels.", grounded=True,
                      chunks=[Chunk(text="x", source="s", heading="h", distance=0.1)])

    def setUp(self):
        try:
            import api
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.py not importable here: {exc}")
        self.api = api
        patcher = mock.patch.object(thread_store, "DB_PATH", _temp_db(self))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _send(self, route_result, rag_result=None, message_id="m1", **patches):
        req = self.api.ReplyRequest(email_text="What are the levels?", is_html=False,
                                    conversation_id="c", subject="Enquiry", message_id=message_id)
        with mock.patch.object(self.api, "classify", return_value=route_result), \
             mock.patch.object(self.api, "rag_answer", return_value=rag_result) as rag, \
             mock.patch.multiple(self.api, **patches) if patches else contextlib.nullcontext():
            return self.api.generate_reply(req), rag

    def test_grounded_reply_recorded(self):
        payload, _ = self._send(self.RAG, self.GROUNDED)
        self.assertTrue(payload["answered"])
        [exchange] = thread_store.history(payload["conversation_id"])
        self.assertTrue(exchange.grounded)
        self.assertIn("Three levels", exchange.reply)

    def test_ungrounded_reply_still_recorded(self):
        payload, _ = self._send(self.RAG, Answer(text="", grounded=False, chunks=[]))
        self.assertFalse(payload["answered"])
        [exchange] = thread_store.history(payload["conversation_id"])
        self.assertIn(NO_INFO_REPLY, exchange.reply)

    def test_blocked_paths_never_reach_rag(self):
        self._send(self.RAG, self.GROUNDED, message_id="dup")
        cases = [
            ("duplicate", dict(route_result=self.RAG, message_id="dup"), "duplicate"),
            ("not routed", dict(route_result=self.HUMAN, message_id="h1"), None),
            ("near duplicate", dict(route_result=self.RAG, message_id="n1",
                                    is_near_duplicate=mock.Mock(return_value=True)), "near_duplicate"),
        ]
        for label, kw, flag in cases:
            with self.subTest(label):
                payload, rag = self._send(**kw)
                self.assertFalse(payload["answered"])
                if flag:
                    self.assertTrue(payload[flag])
                rag.assert_not_called()


class TestKnowledgeStore(unittest.TestCase):
    def setUp(self):
        from rag import knowledge

        self.kn = knowledge
        self.db = _temp_db(self)
        self.push = mock.patch.object(knowledge, "_push").start()
        mock.patch.object(knowledge, "_drop").start()
        self.addCleanup(mock.patch.stopall)
        knowledge.record_ingest(
            [("a.docx#0", "A\n\nfirst", {"source": "a.docx", "heading": "A", "chunk_index": 0})],
            path=self.db,
        )

    def test_update_pushes_then_saves(self):
        out = self.kn.update_chunk("a.docx#0", content="new", path=self.db)
        self.push.assert_called_once()
        self.assertTrue(out["edited"])

    def test_failed_push_leaves_row_unchanged(self):
        self.push.side_effect = RuntimeError("embedder down")
        with self.assertRaises(RuntimeError):
            self.kn.update_chunk("a.docx#0", content="changed", path=self.db)
        self.assertEqual(self.kn.get_chunk("a.docx#0", path=self.db)["content"], "A\n\nfirst")

    def test_reset_ingest_keeps_manual_chunks(self):
        manual = self.kn.create_chunk("keep me", {"heading": "Manual"}, path=self.db)
        self.kn.record_ingest([], reset=True, path=self.db)
        self.assertEqual([r["id"] for r in self.kn.list_chunks(path=self.db)], [manual["id"]])


class TestNearDuplicate(unittest.TestCase):
    """Needs the real bge-m3 embedder; skipped when it isn't reachable."""

    @classmethod
    def setUpClass(cls):
        try:
            is_near_duplicate("ping", "ping")
        except Exception as exc:
            raise unittest.SkipTest(f"embedder not reachable: {exc}")

    def test_near_duplicate(self):
        cases = [
            ("What is the price for EVH Level 2?", "How much does EVH Level 2 cost?", True),
            ("When does the next Level 2 cohort start?", "Do you offer an early-bird discount?", False),
            ("", "anything", False),
        ]
        for a, b, expected in cases:
            with self.subTest(a=a, b=b):
                self.assertEqual(is_near_duplicate(a, b), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
