# Handoff — running the Academy email auto-reply on your hardware

You received a folder (`email-classifier/`) and this document. Follow it top to
bottom. It assumes no prior contact with the project.

---

## 1. What this is

An assistant that watches an Outlook mailbox and, for each incoming email:

1. **Classifies** it as *academic*, *administrative*, or *spam*.
2. For academic-only email, **retrieves** the relevant passages from a set of
   Word documents about the training programmes and **drafts a reply** grounded
   in them.
3. Saves that reply as an **Outlook draft** in the same conversation. A human
   reviews and sends it.

Anything about payments, records, refunds, or account issues — and anything the
classifier is not confident about — gets **no draft** and is left for a human.
The system **never sends mail**. It only creates drafts.

---

## 2. What you need before starting

| Requirement | Detail |
|---|---|
| A Linux host with **Docker** + **Docker Compose v2** | `docker compose version` should print v2.x |
| ~8 GB free disk | model cache + images |
| Your **27B chat model** already running behind an OpenAI-compatible endpoint (LiteLLM or Ollama), reachable from this host | see the requirements in §6 |
| A **Microsoft Entra (Azure AD) app registration** you can create | for the Outlook connection — §7 walks through it |
| The mailbox account | whose inbox this should watch |

You do **not** need a GPU for anything in this folder. The chat model (which
does need a GPU) is *your existing* model — this project reuses it and adds no
model of its own. It only runs a small CPU embedding model locally.

---

## 3. Architecture

```
                    ┌─────────────────── this docker compose ───────────────────┐
                    │                                                           │
  Outlook mailbox ──┼──▶ n8n ──▶ classifier API ──┬──▶ chromadb  (vector store) │
       ▲            │            (FastAPI :8100)  └──▶ embedder  (bge-m3, CPU)   │
       │            │                 │                                         │
   draft created ◀──┼─────────────────┘                                         │
                    └───────────────────────────────┼───────────────────────────┘
                                                    │
                                                    ▼
                              your chat model  (LiteLLM or Ollama)
                                   already running, elsewhere
```

Four containers come up: `n8n`, `classifier`, `chromadb`, `embedder`. All are
bound to `127.0.0.1` only — nothing is exposed on the network.

---

## 4. Bring up the stack

```sh
cd email-classifier

cp .env.example .env
```

Edit `.env` and set three values:

```ini
# Your chat model's OpenAI-compatible endpoint, as seen from inside a container.
#   LiteLLM, same host:   http://host.docker.internal:4000/v1
#   LiteLLM, another box: http://<its-ip>:4000/v1
#   Ollama,  same host:   http://host.docker.internal:11434/v1
#   Ollama,  another box: http://<its-ip>:11434/v1
LLM_URL=http://host.docker.internal:11434/v1

# API key. Blank for Ollama, or for a LiteLLM proxy with no auth.
LLM_KEY=

# The model name the endpoint exposes.
#   LiteLLM: the model_list name.   Ollama: the pulled tag (run `ollama list`).
CHAT_MODEL=qwen3:32b
```

> **If your model is served by Ollama, also read §6 now** — there is one
> setting (`EC_ALLOW_UNCALIBRATED`) you very likely need, and without it the
> assistant drafts nothing and routes every email to a human.

Then:

```sh
docker compose up -d --build
```

First boot downloads the embedding model (~2 GB) into a docker volume. Watch it
finish:

```sh
docker compose logs -f embedder
#   ... "Ready" / "Starting HTTP server" — then Ctrl-C to stop following
```

Check all four are up and the API is healthy:

```sh
docker compose ps
curl -s http://127.0.0.1:8100/health ; echo
#   {"ok":true}
```

If `/health` is not `ok`, jump to §9.

---

## 5. Load the training documents

The Word documents live in `data/docs/`. Load them into the vector store:

```sh
docker compose exec classifier python -m rag ingest --reset
#   ... "NNN chunks -> collection 'docs'"
```

Verify retrieval actually works end to end (this calls your chat model):

```sh
docker compose exec classifier python -m rag ask "what are the training levels?"
#   Should print a real answer drawn from the documents, plus the source
#   headings. If it prints the "we don't have relevant information" line or an
#   error, jump to §9.
```

To change the documents later: edit files in `data/docs/`, then re-run the
`ingest --reset` command above.

---

## 6. Requirements on your chat model

`CHAT_MODEL` is used for **both** classification and reply drafting.

### 6a. The one hard dependency: token logprobs

The classifier decides confidence by reading the probability of the
`true`/`false` token in the model's JSON output. If the endpoint does not
return logprobs, classification fails for every email and **the assistant
sends the entire mailbox to the human queue** — silently, nothing errors
visibly.

- **LiteLLM** over llama.cpp or vLLM: returns logprobs. Nothing to do.
- **Ollama**: its OpenAI-compatible `/v1/chat/completions` did **not** return
  logprobs until roughly **v0.12**. Check yours (§6c). If it comes back empty
  and you cannot upgrade, set in `.env`:
  ```ini
  EC_ALLOW_UNCALIBRATED=1
  ```
  This routes on the model's plain yes/no answer instead. The gate still fails
  safe — any hint of spam or a billing/account matter still goes to a human —
  but the confidence threshold no longer applies, so a genuinely borderline
  email ("course question, and also I haven't paid") is decided by a single
  yes/no rather than held for review. The classifier response carries
  `"calibrated": false` whenever this mode is active.

### 6b. Other requirements

- **Strict JSON-schema decoding** (`response_format: {type: json_schema}`).
  Any current LiteLLM/llama.cpp/vLLM, and Ollama ≥ 0.5, support this.
- **No hidden "thinking".** If the model is a reasoning model (e.g. Qwen3):
  - **LiteLLM**: add to its `litellm-config.yaml` entry —
    `chat_template_kwargs: {"enable_thinking": false}`
  - **Ollama**: pull a non-thinking tag, or add `PARAMETER think false` to its
    Modelfile. (Structured output usually suppresses the think block anyway.)
  If this is wrong, classification comes back empty → everything routes to a
  human.
- **Context window ≥ 8192 tokens.** The drafting prompt (document extracts +
  thread history) is long. **Ollama defaults to 4096 and silently truncates**,
  so answers come out thin or refuse. Fix on the Ollama service:
  ```
  OLLAMA_CONTEXT_LENGTH=8192
  ```
  (or `PARAMETER num_ctx 8192` in the Modelfile). If you cannot set either, put
  `EC_NUM_CTX=8192` in `.env` as a fallback.
- **Ollama only — keep the model resident.** Set `OLLAMA_KEEP_ALIVE=-1` on the
  Ollama service, or the 27B unloads after 5 min idle and the next email stalls
  ~30 s while it reloads.

### 6c. One-shot check

From this host, filling in your endpoint / key / model:

```sh
curl -s http://<endpoint>/v1/chat/completions \
  -H "Authorization: Bearer <key-or-anything>" -H "Content-Type: application/json" \
  -d '{"model":"<model>","messages":[{"role":"user","content":"Reply with the word ok"}],
       "logprobs":true,"top_logprobs":3,"max_tokens":10}' | jq '.choices[0]'
```

- `message.content` should be `"ok"` (non-empty → thinking is not eating the
  output).
- `logprobs.content` should be a populated array. If it is `null` → you need
  `EC_ALLOW_UNCALIBRATED=1` (§6a).

---

## 7. Connect the Outlook mailbox

### 7a. Register an app in Microsoft Entra

1. [Azure Portal](https://portal.azure.com) → **Microsoft Entra ID** → **App
   registrations** → **New registration**.
2. Name: e.g. `academy-email-autoreply`. Supported account types: **single
   tenant** (this directory only) is fine.
3. **Redirect URI**: platform **Web**, value exactly:
   ```
   http://localhost:5678/rest/oauth2-credential/callback
   ```
   (Microsoft rejects plain-http redirects *except* `localhost` — that is why
   this stack keeps n8n on localhost.)
4. **Register**.
5. On the **Overview** page copy the **Application (client) ID**.
6. **Certificates & secrets** → **New client secret** → copy the **Value**
   immediately (it is hidden after you leave the page).
7. **API permissions** → **Add a permission** → **Microsoft Graph** →
   **Delegated permissions** → add **`Mail.ReadWrite`** and
   **`offline_access`**. If your tenant shows a **Grant admin consent** button,
   click it.

### 7b. Open the n8n UI

n8n is loopback-only. From your workstation:

```sh
ssh -L 5678:127.0.0.1:5678 <user>@<this-host>
```

Then open <http://localhost:5678> in a browser. Create the owner account when
prompted (local, stays on the box).

### 7c. Add the credential

1. **Credentials** → **New** → search **Microsoft Outlook OAuth2 API**.
2. Paste the **Client ID** and **Client Secret** from step 7a.
3. Confirm the **OAuth Redirect URL** shown matches what you registered
   (`http://localhost:5678/rest/oauth2-credential/callback`).
4. Click **Connect my account** → sign in as the mailbox account → accept.
   The credential should show **Connected**.

### 7d. Import and wire the workflow

1. **Workflows** → **⋯** (top right) → **Import from File** → choose
   `n8n/outlook-academy-workflow.json`. (The other file in `n8n/`,
   `email-classifier-workflow.json`, is the optional test form from §8 — not
   the mailbox pipeline.)

2. Open the workflow. Two nodes need the credential attached (they will show a
   warning until you do):
   - **New Outlook email** (the trigger) → select your Outlook credential.
   - **Create Outlook draft reply** → select the same credential.
3. Optionally open **New Outlook email** and set the folder / poll interval
   (default: Inbox, every minute).
4. Toggle the workflow **Active** (top right).

---

## 8. Verify end to end

Send a test email to the mailbox:

> Subject: *Course question*
> Body: *What are the training levels and who is Level 2 aimed at?*

Within ~1–2 minutes:

- **A draft reply appears** in that conversation in the mailbox, listing the
  levels, drawn from the documents.
- In n8n, **Executions** shows the run ending at **Create Outlook draft
  reply**.

Now send a second one:

> Body: *My invoice still shows unpaid, can you check?*

- **No draft.** The n8n execution ends at **No draft (human queue)** with
  reason *not routed to rag*. This is correct — billing questions are not
  auto-answered.

If the first test produced no draft, check the n8n execution's
**Classify + RAG reply** node output:

| `reason` in the output | Meaning | Fix |
|---|---|---|
| `not routed to rag` + `error: expected 3 bools, got 0` | endpoint returned no logprobs | §6a — set `EC_ALLOW_UNCALIBRATED=1` (Ollama) |
| `not routed to rag` + empty/garbled `flags` | model returned no usable classification | §6b — thinking not disabled |
| `no grounded answer in the documents` | classified academic, but retrieval found nothing usable | re-run `ingest --reset` (§5); confirm the topic is in `data/docs/`; on Ollama check context length (§6b) |
| `retrieval or model error` + `error` field | API could not reach your model or the embedder | §9 |
| `already handled (duplicate message_id)` | same email seen before | expected on a re-poll; send a fresh email |
| draft appears but `"calibrated": false` in the output | running in flag-only mode (§6a) | expected on Ollama without logprobs; upgrade Ollama to restore calibrated routing |

---

## 9. Troubleshooting

Run everything from the `email-classifier/` folder.

**`docker compose up` fails**
Check Docker Compose is v2 (`docker compose version`, not `docker-compose`).

**`embedder` keeps restarting**
It is still downloading the model, or ran out of disk. `docker compose logs
embedder`. Needs ~3 GB free.

**`/health` not ok, or API logs show connection errors to `chromadb`**
`docker compose restart classifier`. If it persists, `docker compose down &&
docker compose up -d`.

**`python -m rag ask` returns `error=...` mentioning the embeddings endpoint**
The `embedder` container is not ready. `docker compose logs embedder` — wait
for `Ready`, retry.

**`python -m rag ask` returns `error=...` mentioning chat/completions or a
connection refused / timeout**
`classifier` cannot reach your model at `LLM_URL`.
- From the host: `curl $LLM_URL/models` — does it answer? (For Ollama the
  path is the same: `http://<host>:11434/v1/models`.)
- If the model server is on this same host, `LLM_URL` must use
  `http://host.docker.internal:<port>/v1`, not `http://localhost:...`
  (localhost inside the container is the container itself).
- If on another machine, make sure its firewall allows this host, and that
  Ollama is bound to `0.0.0.0` (`OLLAMA_HOST=0.0.0.0`), not just localhost.
After changing `.env`: `docker compose up -d` (recreates with new values).

**Classifier routes everything to the human queue**
Check the **Classify + RAG reply** node output in n8n:
- `error: expected 3 bools, got 0` → the endpoint returns no logprobs. On
  Ollama, set `EC_ALLOW_UNCALIBRATED=1` in `.env` and `docker compose up -d`
  (§6a). On LiteLLM, this should not happen — check the proxy.
- empty or malformed `flags` → the model is "thinking" instead of answering.
  Disable it (§6b) and restart the model server.

**Outlook "Connect my account" fails / redirect error**
The browser must reach n8n as exactly `http://localhost:5678` (via the SSH
tunnel), and the Entra redirect URI must be
`http://localhost:5678/rest/oauth2-credential/callback` character for
character.

**n8n workflow runs but no draft, node output looks fine, `answered: true`**
Check the **Create Outlook draft reply** node — usually the Outlook credential
is missing there (it is a separate attachment from the trigger).

**Logs for a specific container**
```sh
docker compose logs -f classifier      # or n8n, embedder, chromadb
```

---

## 10. Day-2 operations

**Change the training documents**
Edit / add / remove files in `data/docs/`, then:
```sh
docker compose exec classifier python -m rag ingest --reset
```
Supported: `.docx .pdf .md .txt .html`. Scanned PDFs (no text layer) yield
nothing — check with `docker compose exec classifier python -m rag check`.

**Update the code** (if you receive a new version of this folder)
```sh
docker compose up -d --build classifier
```

**Adjust the reply wording** (greeting, sign-off)
These are environment variables read by the API. Add to `.env`, e.g.:
```ini
RAG_EMAIL_GREETING=Hello,
RAG_EMAIL_SIGNOFF=Best regards,\nThe Academy Team
```
then `docker compose up -d`.

**Tune how strict the "is this academic" gate is**
`EC_THRESHOLD` in `.env` (default `0.9`; higher = more emails go to a human).

**Reply format (prose vs. bulleted list)**
Automatic: when the enquirer writes a numbered/bulleted list, asks "in points",
or asks several questions, the draft comes back as a `- ` list; otherwise it is
prose. To force it, send `"format": "list"` or `"format": "prose"` in the
`/generate-reply` body (add the field to the *Classify + RAG reply* node in
n8n). Test from the shell with `python -m rag ask --list "..."`.

**Back up**
The only stateful things are docker volumes:
- `email-classifier_chroma-data` — the ingested vectors (rebuildable from
  `data/docs/` via `ingest --reset`).
- `email-classifier_n8n-data` — the workflow, credentials, execution history.
- `./data/threads.db` — conversation memory (a bind mount, so it is just a file
  in `data/`).
```sh
docker run --rm -v email-classifier_n8n-data:/v -v "$PWD":/out alpine \
  tar czf /out/n8n-backup.tgz -C /v .
```

**Stop / start**
```sh
docker compose stop        # keeps data
docker compose down        # removes containers, keeps volumes
docker compose down -v     # ALSO deletes volumes — you lose n8n creds + vectors
```

---

## 11. Guarantees

- Nothing in this stack **sends** email. The Outlook node calls `createReply`,
  which creates a draft. Sending is a human action.
- No container is reachable off `127.0.0.1`.
- Administrative and spam email never receives an automated reply.
- Every factual sentence in a draft comes from `data/docs/`. If the documents
  do not cover a question, the email goes to the human queue rather than
  getting a guessed answer.
