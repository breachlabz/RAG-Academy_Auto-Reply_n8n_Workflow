# Academy Auto-Reply

An assistant that watches an Outlook mailbox and, for each incoming email:

1. **Classifies** it as *academic*, *administrative*, or *spam*.
2. For academic-only email, **retrieves** the relevant passages from a set of
   Word documents about the training programmes and **drafts a reply grounded
   in them** — nothing else.
3. Queues that reply in the **review page** (`/review`) for a human to read,
   optionally edit, and send. Nothing goes out until a person clicks Send.

Anything about payments, records, refunds or account issues — and anything the
classifier is not confident about, or that the documents do not answer — gets
**no draft** and is left for a human. **Nothing sends automatically.** Every
reply this system produces waits in the review queue for an explicit human
Send.

This is the single reference for deploying and running it. Follow it top to
bottom.

---

## 1. How it works

```
                    ┌───────────────────── this docker compose ─────────────────────┐
                    │                                                               │
  Outlook mailbox ──┼──▶ n8n ──▶ classifier API ──┬──▶ chromadb   (vector store)    │
                    │            (FastAPI :8100)  └──▶ embedder   (bge-m3, CPU)      │
                    │              │        ▲                                        │
                    │              │        │  the n8n AI Agent node also retrieves  │
                    │              ▼        │  + drafts, via its own Chroma tool     │
                    │      review queue                                              │
                    │      (/review, a human reads/edits/sends)                      │
                    └──────────────┼────────┼───────────────────────────────────────┘
                                   │        │
                                   ▼        ▼
                    Microsoft Graph      your chat model  (LiteLLM or Ollama)
                    (app-only, Send)          already running, elsewhere
                                   │
                                   ▼
                            Outlook mailbox (sent)
```

Four containers come up: `n8n`, `classifier`, `chromadb`, `embedder`. All
publish on `127.0.0.1` only. The **chat model is yours** — an existing LiteLLM
or Ollama endpoint named once in `.env`; this project adds no model of its own
and needs no GPU. It runs one small CPU embedding model (`bge-m3`) locally.
Sending is a separate, app-only Microsoft Graph credential the review page
calls directly (§6e) — independent of the n8n Outlook connection, which only
needs to *read* the mailbox now.

### The pipeline, node by node

The n8n workflow `Academy Agent (Outlook)` is thin plumbing over the classifier
API. The safety logic (the gate, the refusal detection, the email shell) lives
in the API and only there.

```
New Outlook email ─▶ Prepare ─▶ Is label "Academy"? ─true─▶ AI Agent ─▶ Finalize ─▶ Grounded answer? ─true─▶ Record reply
 (polls inbox/min)   POST         ($json.proceed)          (drafts from    POST        ($json.grounded)      POST
                     /emails/                               academy_docs)  /emails/                          /threads/reply
                     prepare                                               finalize                          (queues it for
                                   └─false─▶ Do nothing                    └──────────────false─▶ Human queue  /review)
```

The workflow's last node is **Record reply** — n8n's job ends once the reply is
recorded. A human takes it from there at **`/review`** (§6e): read it, edit it
if needed, click **Send**. That call goes straight from the classifier API to
Microsoft Graph, not back through n8n.

| Node | Does |
|---|---|
| **Prepare** → `POST /emails/prepare` | strips HTML + quoted history, records the inbound message (deduplicated on `internetMessageId`), loads the thread, runs the **classifier gate**, and rewrites a follow-up into a standalone retrieval question. Returns `proceed` (the gate), `history` / `email_text` / `query` / `format` (the agent's prompt), and `ref` (the Graph message id, carried onto the stored row for `/review` to reply to later). |
| **Is label "Academy"?** | branches on `proceed`. False → *Do nothing* (payments, spam, low confidence, or a duplicate). |
| **AI Agent — generate reply** | an n8n LangChain agent with the `academy_docs` Chroma tool. Must search the docs before answering; replies with the bare token `NOT_IN_DOCUMENTS` when they do not cover the question. Uses `format` to pick a bulleted or prose reply. Writes the body only — no greeting or sign-off. |
| **Finalize** → `POST /emails/finalize` | runs the agent's output through the grounding net (rejects a bare/embedded `NOT_IN_DOCUMENTS` and prose that merely *reports the documents as silent*), then wraps a grounded body in the greeting/sign-off shell. Returns `grounded` and the ready-to-send `reply`. |
| **Grounded answer?** | branches on `grounded`. False → *Human queue*. |
| **Record reply** → `POST /threads/reply` | records the drafted reply against the conversation (so the next email in the thread has its history) and puts the row in the **review queue** — it now sits at `/review` until a human sends it. |

---

## 2. Prerequisites

| Requirement | Detail |
|---|---|
| Linux host with **Docker** + **Docker Compose v2** | `docker compose version` prints v2.x |
| ~8 GB free disk | model cache + images |
| Your **chat model** on an OpenAI-compatible endpoint (LiteLLM or Ollama), **reachable from a container** | see §3 and §5 |
| A **Microsoft Entra (Azure AD) app registration** you can create | for the Outlook connection — §6 |
| The mailbox account | whose inbox this watches |

A capable instruction model is expected for `CHAT_MODEL` — it is used for
**both** classification and drafting. A ~27B-class model is the reference; a 9B
works with more misses. See §5 for the hard requirements.

---

## 3. Bring up the stack

```sh
cd email-classifier
cp .env.example .env
```

Edit `.env` and set three values:

```ini
# Your chat model's OpenAI-compatible endpoint, AS SEEN FROM INSIDE A CONTAINER.
#   LiteLLM, another box:  http://<its-ip>:4000/v1
#   Ollama,  another box:  http://<its-ip>:11434/v1
#   Same host:             http://host.docker.internal:<port>/v1   ← see note below
LLM_URL=http://host.docker.internal:4000/v1

# API key. Blank for Ollama, or a LiteLLM proxy with no auth.
LLM_KEY=

# The model name the endpoint exposes for chat.
#   LiteLLM: the model_list name.   Ollama: the pulled tag (`ollama list`).
CHAT_MODEL=qwen3-27b
```

> **Reaching a model server on the same host.** `host.docker.internal` resolves
> to the host's docker-bridge gateway, **not** `127.0.0.1`. A model server bound
> to `127.0.0.1` only (many LiteLLM/Ollama setups are) is **unreachable** from a
> container and every email silently routes to a human. Fix one of:
> - bind the model server to `0.0.0.0` (Ollama: `OLLAMA_HOST=0.0.0.0`; LiteLLM:
>   publish `0.0.0.0:<port>` or a `172.17.0.1:<port>` mapping), **or**
> - put it on a docker network this stack also joins — copy
>   `docker-compose.override.yml.example` to `docker-compose.override.yml`, set
>   the network name, and use `LLM_URL=http://<service-name>:<container-port>/v1`.

> **If your model is served by Ollama, read §5 now** — you very likely need
> `EC_ALLOW_UNCALIBRATED=1`, without which the assistant drafts nothing.

Then:

```sh
docker compose up -d --build
```

First boot downloads the embedding model (~2 GB) into a docker volume:

```sh
docker compose logs -f embedder      # wait for "Ready" / "Starting HTTP server", then Ctrl-C
```

Check the stack:

```sh
docker compose ps
curl -s http://127.0.0.1:8100/health ; echo      # {"ok":true}
```

If `/health` is not `ok`, see §9.

---

## 4. Load the training documents

The Word documents live in `data/docs/`. Load them into the vector store:

```sh
docker compose exec classifier python -m rag ingest --reset
#   "NNN chunks -> collection 'docs'"
```

Verify retrieval end to end (this calls your chat model):

```sh
docker compose exec classifier python -m rag ask "what are the training levels?"
#   A real answer drawn from the documents, plus the source headings.
#   If it prints the "we don't have relevant information" line or an error → §9.
```

To change the documents later: edit `data/docs/`, re-run `ingest --reset`.
Supported: `.docx .pdf .md .txt .html`. Check extraction first with
`docker compose exec classifier python -m rag check` — a multi-MB PDF reporting
a few hundred characters is a scan with no text layer and no setting will
rescue it.

---

## 5. Chat model requirements

`CHAT_MODEL` runs both classification and drafting.

### 5a. The hard dependency — token logprobs

The classifier reads the probability of the `true`/`false` token in the model's
JSON to decide confidence. **No logprobs → classification fails for every email
→ the whole mailbox goes to the human queue**, silently.

- **LiteLLM** over llama.cpp or vLLM: returns logprobs. Nothing to do.
- **Ollama**: its OpenAI endpoint did not return logprobs until ~**v0.12**.
  Check (§5c). If empty and you cannot upgrade, set in `.env`:
  ```ini
  EC_ALLOW_UNCALIBRATED=1
  ```
  Routing then uses the model's plain yes/no. Still fails safe — any hint of
  spam or a billing/account matter still goes to a human — but the confidence
  threshold no longer applies. Responses carry `"calibrated": false`.

### 5b. Other requirements

- **Strict JSON-schema decoding** (`response_format: {type: json_schema}`).
  Any current LiteLLM/llama.cpp/vLLM, and Ollama ≥ 0.5.
- **No hidden "thinking".** For a reasoning model (e.g. Qwen3):
  - **LiteLLM**: add `chat_template_kwargs: {"enable_thinking": false}` to its
    `model_list` entry.
  - **Ollama**: pull a non-thinking tag, or `PARAMETER think false` in the
    Modelfile.
  If wrong, classification comes back empty → everything routes to a human.
- **Context window ≥ 8192 tokens.** The drafting prompt (doc extracts + thread
  history) is long. **Ollama defaults to 4096 and silently truncates.** Set
  `OLLAMA_CONTEXT_LENGTH=8192` (or `PARAMETER num_ctx 8192`); fallback
  `EC_NUM_CTX=8192` in `.env`.
- **Ollama — keep the model resident.** `OLLAMA_KEEP_ALIVE=-1`, or the model
  unloads after 5 min idle and the next email stalls ~30 s on reload.

### 5c. One-shot check

```sh
curl -s http://<endpoint>/v1/chat/completions \
  -H "Authorization: Bearer <key-or-anything>" -H "Content-Type: application/json" \
  -d '{"model":"<model>","messages":[{"role":"user","content":"Reply with the word ok"}],
       "logprobs":true,"top_logprobs":3,"max_tokens":10}' | jq '.choices[0]'
```

- `message.content` should be `"ok"` (non-empty → thinking is not eating output).
- `logprobs.content` should be a populated array. `null` → you need
  `EC_ALLOW_UNCALIBRATED=1`.

---

## 6. Connect the Outlook mailbox

### 6a. Register an app in Microsoft Entra

1. [Azure Portal](https://portal.azure.com) → **Microsoft Entra ID** → **App
   registrations** → **New registration**.
2. Name e.g. `academy-email-autoreply`. **Single tenant** is fine.
3. **Redirect URI**: platform **Web**, value exactly:
   ```
   http://localhost:5678/rest/oauth2-credential/callback
   ```
   (Microsoft rejects plain-http redirects *except* `localhost` — which is why
   this stack keeps n8n on localhost.)
4. **Register**. Copy the **Application (client) ID** from **Overview**.
5. **Certificates & secrets** → **New client secret** → copy the **Value** now
   (hidden after you leave the page).
6. **API permissions** → **Add a permission** → **Microsoft Graph** →
   **Delegated** → add **`Mail.Read`** and **`offline_access`**. Click
   **Grant admin consent** if your tenant shows it.
   (Only *read* — this app registration polls the inbox. It no longer needs
   write access: nothing in n8n creates a draft or sends any more. Sending is
   a separate, app-only registration — §6e.)

### 6b. Open the n8n UI

n8n is pinned to `127.0.0.1:5678` and never exposed on the network (it runs
arbitrary code). How you reach the UI depends on where you are:

- **At the machine itself** (it has a desktop): just open
  <http://localhost:5678> in a local browser.
- **Headless box, administering from your laptop**: forward the port, then use
  your laptop browser:
  ```sh
  ssh -L 5678:127.0.0.1:5678 <user>@<this-host>
  # then open http://localhost:5678 on your laptop
  ```

Either way the address must be exactly `http://localhost:5678` — Microsoft Entra
only accepts `http://localhost` as a plain-HTTP OAuth redirect (§6c), not an IP
or hostname. Create the owner account when prompted (local, stays on the box).

This is only for setup and later maintenance. Once the workflow is Active the
pipeline runs headless — no tunnel, no browser.

### 6c. Add credentials

1. **Microsoft Outlook OAuth2 API** — paste the Client ID and Secret from 6a,
   confirm the redirect URL matches, **Connect my account** → sign in as the
   mailbox account. Should show **Connected**.
2. **LiteLLM / chat model** — a credential of type **OpenAI API** with
   **Base URL** = your `LLM_URL` (reachable *from the n8n container* — same
   rules as §3) and the API key. Used by the AI Agent's *Local Model* and
   *Embeddings bge-m3* nodes.
3. **Chroma** — type **Chroma API** (self-hosted), **Base URL**
   `http://email-classifier-chroma:8000`, no auth. Used by the `academy_docs`
   node. This is the same store `python -m rag ingest` writes to.

### 6d. Import and wire the workflows

1. **Workflows → ⋯ → Import from File** → `n8n/academy-agent-workflow.json`.
2. Attach credentials where nodes show a warning:
   - **New Outlook email** → the Outlook credential.
   - **Local Model** and **Embeddings bge-m3** → the chat-model credential.
   - **academy_docs** → the Chroma credential.
3. Optionally open **New Outlook email** to set folder / poll interval
   (default: Inbox, every minute).
4. Toggle **Active**.

`n8n/email-classifier-form-workflow.json` is an optional browser form for
trying classification + retrieval by hand (`POST /answer`), no mailbox. Import
it the same way; it needs no credentials.

> **Re-importing overwrites credential bindings and the active flag.** Re-attach
> and re-activate after any import.

### 6e. Set up sending (the review queue)

The workflow above only gets a reply as far as the **review queue**
(`http://127.0.0.1:8100/review`) — nothing sends on its own. Opening that page
without going further is fine: you can read and edit drafts, the **Send**
button will just fail clearly ("not configured") until this step is done.

Sending needs its own Microsoft Graph credential — deliberately **not** the
Outlook OAuth2 credential from §6a. That one is *delegated*: it only works
because a person signed in interactively, which is exactly what a background
Send button cannot wait on. This is an **app-only** registration instead — it
authenticates as itself, with a client id/secret, no sign-in ever:

1. [Azure Portal](https://portal.azure.com) → **Microsoft Entra ID** → **App
   registrations** → **New registration**. A second, separate app from §6a
   (e.g. `academy-email-send`) — do not reuse that one or add these
   permissions to it.
2. **API permissions** → **Add a permission** → **Microsoft Graph** →
   **Application permissions** → add **`Mail.Send`** → **Grant admin
   consent** (a tenant admin must click this; application permissions have no
   per-user consent).
3. **Certificates & secrets** → **New client secret** → copy the **Value**
   now.
4. Copy the **Application (client) ID** and **Directory (tenant) ID** from
   **Overview**.
5. In `.env`:
   ```ini
   GRAPH_TENANT_ID=<Directory (tenant) ID>
   GRAPH_CLIENT_ID=<Application (client) ID>
   GRAPH_CLIENT_SECRET=<the secret VALUE>
   GRAPH_MAILBOX=<the mailbox's sign-in email, e.g. training@yourorg.com>
   ```
6. `docker compose up -d` to pick up the new values.

Reopen `/review` and Send should work. `mail/graph.py` is the whole
implementation — about eighty lines, nothing more happens on the way to
Graph.

---

## 7. Verify end to end

Send a test email to the mailbox:

> Subject: *Course question* — Body: *What are the training levels and who is
> Level 2 aimed at?*

Within ~1–2 minutes a **row appears at `/review`** listing the levels, drawn
from the documents, with your edit box pre-filled; the n8n execution ends at
**Record reply**. Edit it or not, then click **Send** (§6e must be done first)
— the reply goes out via Graph and the row leaves the queue.

Then send: *My invoice still shows unpaid, can you check?* → **nothing in the
queue**, the execution ends at **Do nothing** (`reason: not routed to rag`).
Correct — billing is never auto-answered.

If the first test produced no queue row, open the failing execution and read
the node output:

| Where it stops / `reason` | Meaning | Fix |
|---|---|---|
| **Prepare**, `error: expected 3 bools, got 0` | endpoint returned no logprobs | §5a — `EC_ALLOW_UNCALIBRATED=1` (Ollama) |
| **Prepare**, empty/garbled `flags` | model returned no usable classification | §5b — thinking not disabled |
| **Prepare**, `error: request failed …` | classifier can't reach your model | §3 note + §9 |
| **Do nothing**, `reason: not routed to rag` on an academic email | gate said human | check `flags`/`probs` in the Prepare output; raise nothing, the gate is conservative by design |
| **Do nothing**, `reason: already handled (duplicate message_id)` | same email seen before | expected on a re-poll; send a fresh email |
| **Human queue** | classified academic, but the agent found no grounded answer | re-run `ingest --reset` (§4); confirm the topic is in `data/docs/`; on Ollama check context length (§5b) |
| row appears in `/review`, `"calibrated": false` | flag-only mode (§5a) | expected on Ollama without logprobs |

---

## 8. Day-2 operations

**Change the training documents** — edit `data/docs/`, then
`docker compose exec classifier python -m rag ingest --reset`. The n8n
`academy_docs` node reads the same store, so nothing else to do.

**Update the code** — `docker compose up -d --build classifier`.

**Reply wording** (greeting, sign-off) — in `.env`:
```ini
RAG_EMAIL_GREETING=Hello,
RAG_EMAIL_SIGNOFF=Best regards,\nThe Academy Team
```
then `docker compose up -d`. A literal `\n` becomes a line break.

**Gate strictness** — `EC_THRESHOLD` in `.env` (default `0.9`; higher → more
email to a human).

**Review and send replies** — `http://127.0.0.1:8100/review`. Every grounded
reply waits there until a human sends it (§6e); nothing sends on its own.

**Reply format** — automatic: a numbered/bulleted enquiry, an "in points" ask,
or several questions produces a `- ` list; otherwise prose. `Prepare` decides
it (`format` field) and `Finalize` enforces it. Test with
`python -m rag ask --list "…"` / `--prose`.

**Back up** — the stateful things are docker volumes plus one file:
- `email-classifier_chroma-data` — ingested vectors (rebuildable from
  `data/docs/`).
- `email-classifier_n8n_data` — workflows, credentials, execution history.
- `./data/threads.db` — conversation memory (a bind mount).
```sh
docker run --rm -v email-classifier_n8n_data:/v -v "$PWD":/out alpine \
  tar czf /out/n8n-backup.tgz -C /v .
```

**Stop / start**
```sh
docker compose stop        # keeps everything
docker compose down        # removes containers, keeps volumes
docker compose down -v     # ALSO deletes volumes — loses n8n creds + vectors
```

---

## 9. Troubleshooting

Run everything from `email-classifier/`.

**`docker compose up` fails** — check Compose is v2 (`docker compose version`).

**`embedder` keeps restarting** — still downloading, or out of disk
(`docker compose logs embedder`, needs ~3 GB).

**`/health` not ok / API logs show `chromadb` connection errors** —
`docker compose restart classifier`; if it persists,
`docker compose down && docker compose up -d`.

**`rag ask` error mentions the embeddings endpoint** — `embedder` not ready
yet (`docker compose logs embedder`, wait for `Ready`).

**`rag ask` error mentions chat/completions, or connection refused/timeout** —
`classifier` cannot reach your model at `LLM_URL`:
- from the host: `curl $LLM_URL/models` — does it answer?
- same-host model server: see the §3 note (must not be `127.0.0.1`-only).
- another machine: firewall open to this host, Ollama bound to `0.0.0.0`.
- after editing `.env`: `docker compose up -d` to recreate.

**Everything routes to the human queue** — the **Prepare** node output:
- `error: expected 3 bools, got 0` → no logprobs → §5a.
- empty/malformed `flags` → model is "thinking" → §5b, restart the model server.

**Outlook "Connect my account" fails / redirect error** — the browser must
reach n8n as exactly `http://localhost:5678` (via the SSH tunnel), and the
Entra redirect URI must match `http://localhost:5678/rest/oauth2-credential/callback`
character for character.

**Workflow runs, `grounded: true`, but nothing shows at `/review`** — check
`docker compose logs classifier` around the `Record reply` call; a reply is
only queued once `/threads/reply` has actually recorded it. If it's there but
**Send** fails with *"not configured"*, §6e hasn't been done yet — that's
expected, not a bug, until `GRAPH_*` is set in `.env`.

**Send fails with a 502 from Graph** — the row stays in the queue (nothing is
lost) and the error in the page names what Graph rejected. Common causes: the
app registration's `Mail.Send` permission was added but admin consent was
never granted, or `GRAPH_MAILBOX` isn't a real mailbox in the tenant.

**Container logs** — `docker compose logs -f classifier` (or `n8n`, `embedder`,
`chromadb`).

---

## 10. HTTP API

`http://classifier:8100` inside the compose network, `http://127.0.0.1:8100`
from the host. The safety logic lives here; the workflows are plumbing.

| Endpoint | Purpose |
|---|---|
| `GET /health` | `{"ok": true}` |
| `POST /classify` | `{"text": "..."}` → `type`, `route`, `flags`, `probs`, `calibrated` |
| `POST /answer` | `{"text": "..."}` → classify + retrieve + grounded reply, **stateless**. What the test form and `scripts/rag_eval.py` use. |
| `POST /generate-reply` | `{"email_text", "conversation_id", "subject", "message_id"}` → the whole pipeline in one call, thread-aware. An alternative to the split `prepare`/`finalize` for a one-HTTP-node workflow. |
| `POST /emails/prepare` | `{"body", "is_html", "subject", "conversation_id", "message_id", "ref"}` → `proceed`, `history`, `email_text`, `query`, `format`, `ref`, `duplicate`, `reason`. Front half of the Outlook path. |
| `POST /emails/finalize` | `{"output", "conversation_id", "subject", "format"}` → `grounded`, `reply`, `subject`, `agent_output`, `reason`. Back half. |
| `POST /threads/reply` | `{"conversation_id", "reply", "subject", "grounded"}` → records the drafted reply and, when grounded, queues it for review. |
| `GET /threads` / `GET /threads/{id}` | stored conversations / one conversation's turns |
| `GET /review` | the review queue page — open this in a browser |
| `GET /review/queue` | `{"pending": [...]}` → grounded replies awaiting a human, with a cached one-line `query_gist` |
| `POST /review/{id}/send` | `{"reply": "..."}` → sends it via Graph (§6e) and records the sent text. 503 if Graph isn't configured yet, 502 if Graph rejected it — either way the row stays queued so nothing is silently lost. |

Notes:
- **`type`** collapses the three flags by priority **spam > administrative >
  academic** — a mixed course-and-payment email is `administrative`, never
  auto-answered.
- **`reply`** is the grounded answer, or the no-information line when the
  documents do not cover it, or empty for administrative/spam/error.
- **`format`** (`/answer`, `/generate-reply`, and echoed through
  `prepare`→`finalize`): `"auto"` (default) decides list vs prose from the
  enquirer's wording; `"list"` / `"prose"` force it.
- **`sources`** / **`closest`** are diagnostic only — see §11.

```sh
curl -s -X POST http://127.0.0.1:8100/answer -H 'Content-Type: application/json' \
  -d '{"text":"does level 2 include hands-on hardware work?"}'
```

---

## 11. Design notes

**Multi-label, not single-label.** An email about the Level 2 syllabus *and* an
unpaid invoice is genuinely both. Three independent booleans — `academic`,
`administrative`, `spam` — set under a constrained grammar, so the logprob of
each `true`/`false` token *is* P(label): a real number to sweep a threshold
over, not a model role-playing "confidence: high".

**Two independent gates, both failing to `human`.** Phase 1 (`classifier`) asks
*may* this be auto-answered — `route()` returns `human` for anything
administrative, spam, non-academic, below threshold, malformed or errored.
Phase 2 (`rag` / the agent) asks *can* it be, from what we actually hold — an
email can clear the classifier and still have no answer in the documents
(`grounded=false`). Neither step ever raises; failures arrive as an `error`
field.

**The distance gate cannot judge correctness.** Over nineteen hand-labelled
questions the cosine distances for answerable and unanswerable ones *overlap* —
a price question is topically about the training, it just has no answer in the
docs. So `RAG_MAX_DISTANCE` is only a crash-guard for wholly-unrelated input.
Everything rests on the model declining to answer, and the bare
`NOT_IN_DOCUMENTS` token alone leaked (the 9B answered ~3/7 unanswerables in
prose that *reports the docs as silent*). `rag.core.refuses_in_prose()` is the
second net: a sentence must contain **both** a source word (document, text,
provided, …) **and** a negated reporting verb (does not specify / mention /
contain / …). `scripts/rag_eval.py` pins that boundary — run it whenever the
prompt or model changes, it is a regex over model prose and will rot.

**Chunk size is a retrieval parameter.** At 1200 chars "Who can take the EVH
training?" split one-chunk-per-level and "what are the three levels?" returned
two and reported the third missing. At 3000 the section survives whole. A
question about a section wants the section.

**Threading — a follow-up does not classify as an email.** *"and what does the
second one cover?"* alone scores academic 0.22 / spam 0.78 and routes to a
human — it is a bag of stopwords with no institute in sight, so every
conversation used to die at the gate on its *second* message. `threads/` is the
memory (SQLite, keyed by Outlook `conversationId`, deduped on
`internetMessageId`). `classify()` takes optional `context`: it changes what
the model is *shown*, never what it is asked — the label still describes the
latest email alone, so an invoice question inside an academic thread still
routes to a human. `threads/context.py` also rewrites the follow-up into a
standalone question before it is embedded, because embeddings have no memory.

**Embedding — bge-m3, 1024-dim, CPU.** Chroma is never given an embedding
function; every vector is computed in `rag.core.embed()` and passed in, and
collections are built with `embedding_function=None` (otherwise Chroma
reconstructs its default MiniLM and would silently embed with the wrong model
if anyone passed `query_texts`). Vectors from two embedders are not comparable,
so changing `RAG_EMBED_MODEL` means `ingest --reset`. bge-m3 takes raw text on
both sides — no `query:` / `passage:` prefix.

**.docx extraction is lossy.** `rag/docx_text.py` converts Word to the markdown
the chunker expects, handling designed documents (headings marked only by being
*entirely* bold, hard-wrapped sentences, curriculum content in tables). It
**ignores images entirely** — a 2.8 MB file holding 8 KB of text means whatever
three diagrams say is invisible to retrieval. Run
`python -m rag check` on any new document; text in shapes and text boxes is not
picked up and the only symptom is the model refusing questions it should
answer.

**Gotcha — `SYSTEM` and `SCHEMA` in `classifier/core.py` must stay in sync**
(same field names, same order). When the prompt does not describe the schema,
grammar-constrained decoding forces tokens the model finds unlikely and the
logprobs then measure the grammar overriding the model, not the
classification — your confidence signal goes silently garbage.

---

## 12. Evaluating

```sh
python scripts/generate.py --out data           # ~380 synthetic emails, on the large model
python scripts/evaluate.py data/holdout.jsonl   # threshold sweep, Phase 1
python scripts/rag_eval.py                       # refusal / leak measurement, Phase 2
python scripts/thread_eval.py                    # follow-up resolution, Phase 3
```

Generation runs on a *different* (larger) model than classification on purpose —
one model for both gives correlated blind spots and flatters the score.
`rag_eval.py` fails the run only on a **leak** (answered without support); a
**miss** costs human-queue volume and harms nobody. Before trusting any number:
hand-label the holdout, and set the production threshold against **real** mail.

---

## 13. Repo layout

```
api.py                    the HTTP surface (classify / answer / generate-reply /
                          emails.prepare / emails.finalize / threads.reply / review)
classifier/core.py        classify() + route(), prompt and JSON schema
classifier/__main__.py    CLI:  python -m classifier "…"
rag/core.py               ingest() + retrieve() + answer(), embed(), chunking, grounding
rag/docx_text.py          .docx -> markdown, plus verify() for extraction loss
rag/format_hint.py        list-vs-prose decision + as_bullets() coercion
rag/__main__.py           CLI:  python -m rag  {ingest | ask | check | manifest}
threads/store.py          SQLite: conversations + turns, dedupe on message id, review queue
threads/context.py        history block + standalone-question rewrite + rolling summary
threads/gist.py           one-line query summary for the review queue, cached per row
mail/text.py              html -> prose, and cutting the quoted reply
mail/graph.py             app-only Microsoft Graph client -- the review queue's Send
frontend/                 the review queue: a Next.js app, statically exported
                          (`npm run build` -> frontend/out/) and served by
                          api.py at /review -- no Node process at runtime
n8n/academy-agent-workflow.json          the mailbox pipeline
n8n/email-classifier-form-workflow.json  the optional manual test form
data/docs/                source documents for the vector store
scripts/                  generate / evaluate / rag_eval / thread_eval
docker-compose.override.yml.example      per-host model-network wiring
```

---

## 14. Guarantees

- **Nothing sends automatically.** Every grounded reply stops in the review
  queue (`/review`); the only thing that sends it is a human clicking Send
  there, optionally after editing it. Nothing in the classify/retrieve/draft
  path (Prepare, the AI Agent, Finalize, Record reply) has network access to
  Graph at all — only `POST /review/{id}/send`, and only when it is called,
  ever reaches `mail/graph.py`.
- A failed or not-yet-configured Send never gets recorded as sent. The row
  stays in the queue and the human sees why, instead of the system silently
  believing something went out that didn't (§9).
- No container is reachable off `127.0.0.1`. The review page has no login of
  its own — it inherits that same "trusted local network" boundary. Put it
  behind real auth before exposing it any wider.
- Administrative and spam email never receive an automated reply, and never
  reach the review queue at all.
- Every factual sentence in a reply comes from `data/docs/`. If the documents
  do not cover a question, the email goes to the human queue rather than
  getting a guessed answer.
