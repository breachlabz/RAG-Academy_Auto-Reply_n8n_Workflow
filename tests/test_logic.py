"""Dense regression suite: one test per behaviour that would break production
if it regressed, with similar cases table-driven via subTest.

Covers classifier routing, ground() salvage, answer() grounding, list/prose
formatting, threads.store dedupe + exact-id replies, the /generate-reply
branches, and the editable knowledge table. Model and Chroma calls are
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
import shutil
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
            # a real fact with a source caveat welded on: keep the fact only
            (caveat, "Level 1 does not include hands-on exercises."),
            (answer_text, answer_text),
            # leading refusal(s) followed by real content: salvage the content
            ("The documentation does not specify a kit for Level 3. " + answer_text, answer_text),
            ("The documentation does not specify this. The extracts do not "
             "mention it either. " + answer_text, answer_text),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw[:50]):
                self.assertEqual(ground(raw), expected)

    def test_echoed_format_instruction_is_dropped(self):
        """The prompt's "Format:" line must never reach the reply."""
        body = "The certificate is valid for three years."
        for echo in (
            "- **Format:** reply in plain prose, at most four sentences, with no list.",
            "Format: reply as a bulleted list -- one \"- \" item per point.",
        ):
            self.assertEqual(rag_core.ground(f"{body}\n\n{echo}"), body)
        kept = "Format: the exam is multiple choice."
        self.assertEqual(rag_core.ground(kept), kept)

    def test_gap_about_the_question_is_ungrounded(self):
        captions_q = ("Does the ACP Level 2 course offer live captions or a "
                      "sign language interpreter during the live sessions?")
        ects_q = ("Does the ACP Level 1 training give university ECTS credits? "
                  "I would like to count it towards my master's degree.")
        cases = [
            # real agent drafts: a related fact with the admission welded on
            ("The ACP Level 2 “Advanced Engineering” live sessions are "
             "interactive and include plenty of time for questions and "
             "discussions, but the documentation does not mention any provision "
             "for live captions or a sign language interpreter.", captions_q, False),
            ('The ACP Level 1 "Foundation" training provides a CYEQT Certificate '
             "of Attendance, but the documentation does not mention university "
             "ECTS credits. You would need to consult your university.", ects_q, False),
            # leading refusal about what was asked, related content after it
            ("The documentation does not specify whether ECTS credits are "
             "awarded. The training ends with a TÜV Rheinland exam.", ects_q, False),
            # admission about a side detail nobody asked for: fact survives
            ("The ACP Level 2 live sessions offer no captions. The documentation "
             "does not mention discounts.", captions_q, True),
            # a plain answer is untouched
            ("ACP Level 2 consists of six live sessions.", captions_q, True),
        ]
        for raw, question, kept in cases:
            with self.subTest(raw=raw[:50]):
                self.assertEqual(bool(ground(raw, question=question)), kept)

    def test_source_attribution_stripped(self):
        cases = [
            ('The documentation states that for ACP Level 1 "Foundation" training, '
             "no prior knowledge is required.",
             'For ACP Level 1 "Foundation" training, no prior knowledge is required.'),
            ("Level 1 is for beginners. According to the documentation, Level 2 "
             "builds on it.",
             "Level 1 is for beginners. Level 2 builds on it."),
            ("- The documentation says that CAN bus is covered\n"
             "- Based on the course materials, Lab setup is included",
             "- CAN bus is covered\n- Lab setup is included"),
            ("Per the documentation: the course lasts three days.",
             "The course lasts three days."),
            # not an attribution lead-in: left alone
            ("The course materials include a hardware kit.",
             "The course materials include a hardware kit."),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw[:50]):
                self.assertEqual(ground(raw), expected)

    def test_source_caveats_dropped(self):
        cases = [
            ("- Level 1 is priced at €890.\n"
             "- The documentation does not mention a discount for booking both.\n"
             "- A tailored quote can be provided on request.",
             "- Level 1 is priced at €890.\n"
             "- A tailored quote can be provided on request."),
            ("Level 1 costs €890. The documentation does not mention discounts.",
             "Level 1 costs €890."),
            ("Level 1 has no exams, but the documentation does not specify "
             "whether handouts are allowed.",
             "Level 1 has no exams."),
            # nothing but caveats: ungrounded, goes to a human
            ("The documentation does not mention discounts. The extracts do "
             "not specify dates.", ""),
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

    def test_manual_queue(self):
        """Unanswerable enquiry: queued with no reply, then answered by hand."""
        eid = thread_store.record_inbound("c", "what is the dress code?", path=self.db)
        self.assertEqual(thread_store.pending_manual(path=self.db), [])
        thread_store.queue_manual(eid, "http://n8n/resume", path=self.db)
        self.assertEqual([r["id"] for r in thread_store.pending_manual(path=self.db)], [eid])
        self.assertEqual(thread_store.pending_review(path=self.db), [])
        self.assertFalse(thread_store.history("c", path=self.db)[0].answered)
        self.assertTrue(thread_store.record_manual_reply(eid, "Smart casual.", path=self.db))
        self.assertFalse(thread_store.record_manual_reply(eid, "again", path=self.db))
        row = thread_store.get_exchange(eid, path=self.db)
        self.assertEqual((row["reply"], row["edited_reply"], row["sent"], row["grounded"]),
                         ("Smart casual.", "Smart casual.", 1, 0))
        self.assertEqual(thread_store.pending_manual(path=self.db), [])

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


class TestFinalizeSearched(unittest.TestCase):
    def test_reply_without_a_search_is_ungrounded(self):
        import api
        draft = "The ACP Level 1 TÜV exam is 90 minutes long and consists of 40 multiple-choice questions."
        cases = [(False, False), (True, True), (None, True)]
        for searched, grounded in cases:
            with self.subTest(searched=searched):
                out = api.finalize_email(api.FinalizeRequest(output=draft, searched=searched))
                self.assertEqual(out["grounded"], grounded)
                self.assertEqual(bool(out["reply"]), grounded)
        out = api.finalize_email(api.FinalizeRequest(output=draft, searched=False))
        self.assertIn("without searching", out["reason"])


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


class TestAdditions(unittest.TestCase):
    """Real .docx files and the real chunker; only the embed/Chroma push is mocked."""

    def setUp(self):
        from rag import additions, knowledge

        self.add = additions
        self.docs = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.docs, ignore_errors=True)
        db = _temp_db(self)
        mock.patch.object(knowledge, "DB_PATH", db).start()
        self.push = mock.patch.object(rag_core, "_collection").start()
        mock.patch.object(rag_core, "embed", side_effect=lambda docs: [[0.0]] * len(docs)).start()
        self.addCleanup(mock.patch.stopall)
        self.kn = knowledge

    def test_each_entry_is_its_own_chunk_with_doc_metadata(self):
        self.add.add("Fee is 500 EUR.\nPayable in advance.", "Level 2 fee", docs_dir=self.docs)
        out = self.add.add("Starts in March.", docs_dir=self.docs)
        self.assertEqual((out["file"], out["entry"]), ("Additions_1.docx", 2))
        [chunk] = out["chunks"]
        self.assertEqual(chunk["id"], "Additions_1.docx#1")
        self.assertEqual(
            chunk["metadata"],
            {"source": "Additions_1.docx", "heading": "Starts in March.", "chunk_index": 1},
        )
        first = self.kn.get_chunk("Additions_1.docx#0")
        self.assertIn("Level 2 fee", first["content"])
        self.assertIn("Payable in advance.", first["content"])

    def test_rolls_over_after_max_entries(self):
        for i in range(self.add.MAX_PER_DOC):
            self.add.add(f"entry {i}", docs_dir=self.docs)
        out = self.add.add("one more", docs_dir=self.docs)
        self.assertEqual((out["file"], out["entry"]), ("Additions_2.docx", 1))
        self.assertEqual(self.add.entry_count(self.docs / "Additions_1.docx"), self.add.MAX_PER_DOC)


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
