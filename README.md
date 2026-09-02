# email-classifier

Phase 1 gate: decide whether an incoming email may be auto-answered by RAG, or
must go to a human.

## Design

**Multi-label, not single-label.** An email asking about the Level 2 syllabus
*and* about an unpaid invoice is genuinely both academic and administrative.
Forcing a single label makes the hardest cases arbitrary and turns the ground
truth in your holdout into coin flips. Three independent booleans instead:
`academic`, `administrative`, `spam`.

**Measured confidence, not self-reported.** Asking a model for
`"confidence": "high|medium|low"` gets you the model role-playing calibration.
Because every field is a boolean under a constrained grammar, the logprob of
the `true`/`false` token *is* P(label) — a real number you can sweep a
threshold over and plot a precision curve against.

**Fail safe.** `route()` returns `"human"` for anything administrative, spam,
non-academic, below threshold, malformed, or errored. `"rag"` requires a
positive academic flag, both other flags clear, and confidence over threshold.

## Setup

```sh
pip install -r requirements.txt
```

`.env` holds the endpoint and key and is loaded automatically. All calls go
through the **LiteLLM proxy on `:4001`** (the compose file maps `4001:4000`,
so the `4000` in your notes is wrong).

Routing through LiteLLM is not optional for the 27B. LiteLLM applies
`enable_thinking: false` from `litellm-config.yaml`; calling `:8081` directly
without it, the model spends its entire token budget on reasoning and emits no
JSON at all — 800 tokens of `reasoning_content` and an empty `content`.

## Use

```sh
python -m classifier "when does level 2 start? also I havent paid"
# route=human  academic=0.94  administrative=1.00  spam=0.02

python -m classifier --large --threshold 0.95 "..."   # 27B instead of 9B
cat email.txt | python -m classifier
```

```python
from classifier import classify, route

result = classify(body)
if route(result) == "rag":
    ...
```

## Evaluating

```sh
python scripts/generate.py --out data              # ~380 emails, 27B
python scripts/generate.py --out data --resume     # continue if interrupted
python scripts/evaluate.py data/holdout.jsonl --save data/results.jsonl
```

The generator writes each batch to `data/raw.jsonl` as it completes, so an
interrupted run keeps its work. Generation runs on the 27B and classification
on the 9B on purpose — one model for both gives correlated blind spots and
flatters the score.

`scripts/rag_eval.py` is the Phase 2 equivalent — it measures whether the
answering path refuses when it should, which is the gate that actually holds.
It fails the run only on a *leak* (answered without support); a *miss* costs
human queue volume and harms nobody.

`evaluate.py` reports precision on the RAG route (a false positive means the
pipeline auto-replied to something it shouldn't), recall at that precision
(what it costs you in human queue volume), and which spec the leaks came from.

Two things to do before trusting any of it:

- **Hand-label the holdout.** Don't have the 80B label the set you use to judge
  the 9B — you inherit its errors as ground truth on exactly the edge cases
  that matter.
- **Get real email in.** Synthetic numbers are a smoke test. Set the production
  threshold against real mail.

## Phase 2: RAG

Once `route()` says `rag`, `rag.answer()` answers out of the document set —
and only out of it.

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements-rag.txt
.venv/bin/python -m rag ingest                 # everything in data/docs
.venv/bin/python -m rag ask "does level 2 cover CAN?"
```

`data/docs` holds `.md`, `.txt` and `.docx`. Ingest runs from the CLI only —
it is not exposed over HTTP — and writes to the same Chroma the API container
reads, so adding a document needs no rebuild and no restart.

```python
from classifier import classify, route
from rag import answer

result = classify(body)
if route(result) == "rag":
    result = answer(body)
    send(result.reply)            # the answer, or the no-information line
    if not result.grounded:
        human_queue(body)         # nothing in the documents supported it
```

**Two independent gates, both failing to `human`.** Phase 1 asks *may* this be
auto-answered; Phase 2 asks *can* it be, from what we actually hold. An email
can clear the classifier and still have no answer in the documents — that is
`grounded=False`, not a guess. `answer()` never raises; failures arrive as
`Answer.error`, same as `Result.error`.

**When the documents do not cover it**, `Answer.reply` is the
no-information line (`RAG_NO_INFO_REPLY`) instead of an empty string, so there
is one field to show a sender and a separate one to route on. `reply` is
deliberately not the same field as `text`: `text` stays empty unless an answer
was genuinely grounded, so code that branches on it cannot mistake the fallback
for a real answer. On a request *error* `reply` is empty too — a failed call
says nothing about whether the documents cover the question, and claiming
otherwise would be a statement about the corpus that nothing supports.

The line is a fixed string, not generated. The one thing known for certain in
that branch is that there was nothing to ground an answer in, so asking a 9B to
phrase the apology just reopens the door to it filling the gap from memory.

Administrative and spam mail does **not** get that line. "We don't have
relevant information related to this query" is untrue and unhelpful for an
unpaid-invoice email, which is not unanswerable — just not ours to answer
automatically.

The refusal is enforced twice. The retrieved chunks are dropped if their cosine
distance exceeds `RAG_MAX_DISTANCE`, and the prompt tells the model to emit
`NOT_IN_DOCUMENTS` when the extracts do not cover the question. In practice
only the second one discriminates — see "The distance gate cannot do this job"
below, where ten measured questions show the two distance ranges overlapping.

**Storage.** Chroma on `:8010` (the compose file in `ai-server`, already up),
collection `docs`. Embedding is **bge-m3** (1024-dim) on its own llama-server,
`llama-bge-m3` on `:8183`, reached through LiteLLM as model `bge-m3` like every
other model. See "Embedding server" below.

Chroma is never given an embedding function. Every vector is computed in
`rag.core.embed()` and passed in explicitly, and collections are built with
`embedding_function=None`. That keeps the embedder a property of this code
rather than of the collection — and stops Chroma reconstructing its default
MiniLM on every `get_collection`, which would download an 80MB ONNX blob into a
container that never needs it and would silently embed with the wrong model if
anyone passed `query_texts` instead of `query_embeddings`.

Vectors from two embedders are not comparable, so changing `RAG_EMBED_MODEL`
means `ingest --reset`. The dimensions differ (384 against 1024), so a mixed
collection fails outright rather than degrading quietly — which is the good
outcome.

Chunking is heading-aware: sections split at `##`/`###`, each chunk carrying
its heading trail into the embedded text. That trail is what makes "does level
2 cover CAN" match a table row reading *"Module: CAN Communication — …"*, which
never says which level it belongs to. It is also what `Answer.sources` reports.

Ingest is idempotent on ids (`file.docx#3`), so re-ingesting an edited file
overwrites in place. Sections *deleted* from a file leave orphan chunks behind;
`--reset` after deletions.

### .docx ingest

`rag/docx_text.py` converts Word files to the markdown the chunker expects.
It was written against `EVH Training - Overview v2(5).docx`, which has the
shape a designed document usually has and a structured one never does:

- **No heading styles.** Every paragraph is `Normal`; headings are marked only
  by being *entirely* bold. A partly bold paragraph is body text with a bold
  lead-in ("**Level 1** gives the participant…") and must not become a heading,
  which is why the test is `all(run.bold)` and not `any`.
- **Hard-wrapped prose.** Single sentences arrive split across three or four
  paragraphs, so consecutive body paragraphs are re-joined before chunking.
  Without that, chunks split mid-sentence and embed badly.
- **Content in tables.** Both curriculum tables are rendered one row per line
  with the header repeated (`Module: X — Know-how: Y`) rather than as a pipe
  table, so a row stays meaningful when it lands in a chunk without its header.

`docx_text.verify()` compares characters reachable through paragraphs and
tables against every `w:t` run in the file. Run it on any new document before
trusting retrieval over it — text inside shapes and text boxes is *not* picked
up, and the only symptom is the model refusing questions it should answer.

It ignores images entirely. The EVH file is 2.8MB holding 8KB of text: whatever
those three images say is invisible to retrieval, and if a diagram carries
content you expect answers about, it has to be typed out somewhere.

### Embedding server

`llama-bge-m3` in `ai-server/docker-compose.yml`, serving `:8183`, plus one
`model_list` entry in `litellm-config.yaml`. Nothing else in that config was
touched — the embedder is deliberately **not** in `fallbacks`, which are
chat-completion routes and cannot stand in for an embedder.

Three settings there are load-bearing:

- `--embedding` puts llama-server in a mode where it serves `/v1/embeddings`
  and refuses chat, which is why this is a separate container rather than a
  flag on an existing one.
- `--pooling cls` is bge-m3's actual pooling. The default is mean, which
  produces vectors that look perfectly healthy and rank worse — a failure that
  never announces itself.
- `--ubatch-size` must be at least `--ctx-size` for a pooled embedder: a
  sequence has to fit in one micro-batch to be pooled, and llama.cpp rejects
  longer input rather than truncating it. Both are 8192 here.

FP16, not a quant: bge-m3 is 560M parameters, so the whole file is 1.1GB and
quantising an embedder costs retrieval quality for memory that is not scarce.
bge-m3 also takes raw text on both sides — it has no `query:` / `passage:`
convention, and adding one (as e5 and bge-v1.5 require) would shift every
vector off where it should sit.

### What bge-m3 fixed, measured

The MiniLM failure was real: asked *"what are the levels and who is each
for"*, it ranked the Level 2 paragraph **sixth** (0.809), and the single best
chunk — *"we have split the training into three levels"* — did not make its
top eight at all. At `k=4` the model never saw Level 2 and said so.

bge-m3 puts that best chunk **first at 0.494** and Level 2 **third at 0.532**.
`RAG_TOP_K` went back to **4**; the 6 was only ever propping up a weak
embedder.

### The distance gate cannot do this job

Nineteen hand-labelled questions, ten answerable from the document and nine
not (`scripts/rag_eval.py`):

```
answerable distances   : 0.402 - 0.501
unanswerable distances : 0.424 - 0.564
separable by one cutoff: no
19/19 correct, 0 leaks
```

The ranges **overlap**, so no value of `RAG_MAX_DISTANCE` both admits "what are
the three levels" (0.501) and rejects "what is the price" (0.424). That is not
a tuning problem, it is what distance measures: a price question *is* about the
training, so it retrieves the right sections and scores close — the document
simply has no price in it. Cosine distance sees topical similarity, never
whether a fact is present.

Treat `RAG_MAX_DISTANCE` as a crash-guard for input with no topical relation to
the corpus at all. It is not, and cannot be tuned into, a correctness gate.

### The refusal gate leaks, so there are two of them

With distance unable to discriminate, everything rests on the model declining
to answer — and the `NOT_IN_DOCUMENTS` token alone was **not** reliable.
Measured on seven questions the document cannot answer, the 9B emitted the
token for four and answered the other three in prose:

> "The provided documentation does not specify the number of days the Level 3
> training runs for."

That arrives as a well-formed answer that happens to say nothing, and it set
`answered=true`. It also breaks the prompt rule against mentioning the
documents, so the model was disobeying two instructions at once.

`refuses_in_prose()` is the second net. It is deliberately narrow: a sentence
must contain **both** a source word (document, text, extract, provided, …)
**and** a negated reporting verb (does not specify / mention / contain / …).
A sentence with only a negation is left alone, because "Level 1 does not
include hands-on exercises" is a real answer about content and must survive,
while "the provided text does not detail Level 2" is a refusal wearing an
answer's clothes. Eleven cases pin that boundary in `scripts/rag_eval.py`.

It is a regex over model prose, so it will rot as prompts and models change.
Run the eval when either moves.

### Chunk size is a retrieval parameter, not a formatting one

`MAX_CHUNK_CHARS` was 1200, which split "Who can take the EVH training?" into
three chunks, one per level. Asked "what are the three levels?", retrieval
returned Level 1 and Level 3 and ranked Level 2 **seventh** — so the answer
named two levels and reported the third as missing.

At 3000 the section survives whole (11 chunks became 6) and the question is
answered correctly at `k=4`. A question about a section wants the section, not
a third of it. This is affordable because the corpus is small; it will need
revisiting when it is not.

### Still weak

- **The 3 images are invisible.** 2.8MB of file, 8KB of text. Anything a
  diagram says is unreachable; it has to be typed out somewhere.
- **One document, 11 chunks.** At this size the whole corpus (~2.5k tokens)
  fits in a prompt and retrieval is not yet earning its keep. It will as the
  corpus grows — the infrastructure is now in place for that, which is the
  point.

Over HTTP the pipeline is `POST /answer` (see `n8n/README.md`), which the n8n
form now calls. Ingest is not exposed over HTTP — documents go in from the CLI
against the same Chroma instance the container reads, so there is nothing to
re-ingest after a rebuild.

## Phase 3: threading

Phases 1 and 2 are stateless: an email arrives, it is classified, it is answered
from the documents, nothing is remembered. `threads/` is the memory —
SQLite, keyed by Outlook's `conversationId`, deduplicated on
`internetMessageId`. `threads/context.py` turns a stored thread into the two
strings answering needs, and conflating them is the mistake it exists to avoid:
a **history block** for the generation prompt, and a **rewritten standalone
question** for retrieval, because embeddings have no memory and *"and the second
one?"* retrieves nothing.

### A follow-up does not classify as an email

The gate was measuring the wrong thing, and nothing about it looked broken:

| Email | academic | spam | route |
|---|---|---|---|
| `and what does the second one cover?` — alone | 0.2195 | **0.7835** | human |
| the same email, with the thread as context | **0.9903** | 0.0010 | rag |

A follow-up carries almost none of its own subject matter — on its own wording
it is a bag of stopwords with no institute in sight. So every conversation died
at the gate on its *second* message, which is the one case threading exists to
serve, and it died as **spam**, which is why nothing downstream looked wrong.

`classify()` now takes an optional `context`. It changes what the model is
*shown*, never what it is asked: the label still describes the latest email
alone, and `CONTEXT_SYSTEM` says so twice, because classifying the *thread*
rather than the *email* is how this fix would become a hole. It does not:
an invoice question inside an academic thread still comes back administrative at
0.9998 and still gets no draft.

Pass `context` for a follow-up and leave it empty for a first email — with no
context the prompt is byte-identical to what it was, so the Phase 1 threshold
sweep still describes the gate.

## Layout

```
classifier/core.py       classify() + route(), prompt and schema
classifier/__main__.py   CLI
rag/core.py              ingest() + retrieve() + answer(), embed(), chunking
rag/docx_text.py         .docx -> markdown, plus verify() for extraction loss
rag/__main__.py          CLI: ingest / ask
threads/store.py         SQLite: conversations + messages, dedupe on message id
threads/context.py       history block + standalone-question rewrite + summary
mail/text.py             html -> prose, and cutting off the quoted reply
api.py                   the HTTP surface n8n calls (/classify /answer /generate-reply)
n8n/                     the pipeline workflow + a test form, and how to import them
data/docs/               source documents for the vector store
scripts/generate.py      synthetic set via 27B, batched + resumable
scripts/evaluate.py      threshold sweep over a holdout (Phase 1)
scripts/rag_eval.py      refusal/leak measurement over the docs (Phase 2)
scripts/thread_eval.py   follow-up resolution over multi-turn threads (Phase 3)
```

`POST /generate-reply` is the whole email pipeline in one call — strip the HTML
and quoted history, classify, gate, retrieve, answer, remember — and is what
`n8n/outlook-academy-workflow.json` uses. The safety logic (the classifier
gate, the refusal detection, the email shell) lives here and only here, so a
workflow is a few nodes of plumbing with nothing to keep in sync.

`POST /answer` is the same minus the memory: stateless, no thread, no HTML
stripping. It is what the test form calls and what `scripts/rag_eval.py`
measures against.

## Gotcha

`SYSTEM` and `SCHEMA` in `core.py` must stay in sync — same field names, same
order. When the prompt doesn't describe the schema, grammar-constrained
decoding forces tokens the model finds unlikely (observed: the `label` key at
logprob **-7.46**, the model wanting `subject` instead). The logprobs then
measure the grammar overriding the model, not the classification, and your
confidence signal is silently garbage.
