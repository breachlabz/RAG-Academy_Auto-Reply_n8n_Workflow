# Academy Auto-Reply

Watches an Outlook mailbox and, for each incoming email:

1. **Classifies** it as `academic` (course content, levels, exams, schedules,
   course prices) or `non_academic` (invoices, refunds, enrolment records,
   account issues, spam — anything else).
2. For confident academic-only email, **retrieves** passages from the training
   documents in `data/docs/` and **drafts a reply grounded only in them**.
3. Queues the draft on the **review page** (`/review`). A person reads, edits
   and sends it. **Nothing is sent automatically.**

Non-academic email, anything the classifier is unsure about, and anything the
documents don't answer get no draft and are left for a person.

This file is the single reference for deploying, operating and changing the
system.

---

## Contents

1. [Architecture](#1-architecture)
2. [Requirements](#2-requirements)
3. [Host preparation](#3-host-preparation)
4. [Chat model](#4-chat-model)
5. [Install and configure](#5-install-and-configure)
6. [Microsoft 365 app registrations](#6-microsoft-365-app-registrations)
7. [n8n workflow](#7-n8n-workflow)
8. [Go-live verification](#8-go-live-verification)
9. [Operations](#9-operations)
10. [Troubleshooting](#10-troubleshooting)
11. [HTTP API](#11-http-api)
12. [Security model and guarantees](#12-security-model-and-guarantees)
13. [Design notes](#13-design-notes)
14. [Evaluation](#14-evaluation)
15. [Repository layout](#15-repository-layout)

---

## 1. Architecture

```
                    ┌──────────────────────── docker compose (this repo) ────────────────────────┐
                    │                                                                            │
  Outlook mailbox ──┼──▶ n8n ──▶ classifier API ──┬──▶ local_chromadb  (vector store, :8000)      │
   (Graph, read)    │            (FastAPI :8100)  └──▶ embedder        (bge-m3, CPU)             │
                    │              │     ▲   the n8n AI Agent retrieves from the same Chroma     │
                    │              ▼     │                                                       │
                    │      /review + Knowledge tab  (human approves every send)                  │
                    └──────────────┼─────┼────────────────────────────────────────────────────────┘
                                   │     │
                                   ▼     ▼
                   Microsoft Graph        chat model server (OpenAI-compatible,
                   (app-only Mail.Send)   llama.cpp / LiteLLM / Ollama — §4)
```

| Container | Image / build | Published | State |
|---|---|---|---|
| `email-classifier-api` | `Dockerfile` (FastAPI + static Next.js review UI) | `127.0.0.1:8100` | `./data` bind mount (`threads.db`) |
| `email-classifier-chroma` | `chromadb/chroma:1.5.9` | `127.0.0.1:8011` | volume `chroma-data` |
| `email-classifier-embedder` | `text-embeddings-inference:cpu-1.5` (bge-m3) | — | volume `embed-cache` |
| `n8n` | `n8nio/n8n` | `127.0.0.1:5678` | volume `n8n_data` |

Nothing is published beyond loopback. The chat model runs outside the stack.

### Pipeline

```
New Outlook email ─▶ Prepare ─▶ Is label "Academy"? ─true─▶ AI Agent ─▶ Finalize ─▶ Grounded answer? ─true─▶ Record reply ─▶ Wait for review
 (polls every min)   POST          ($json.proceed)          (academy_docs  POST        ($json.grounded)      POST
                     /emails/prepare                         Chroma tool)  /emails/finalize                  /threads/reply
                                   └─false─▶ Do nothing                                └─false─▶ Human queue
```

| Node | Responsibility |
|---|---|
| **Prepare** → `/emails/prepare` | Strips HTML and quoted history, records the inbound message (deduplicated on `internetMessageId`), loads the thread, runs the **classifier gate**, rewrites a follow-up into a standalone query. Returns `proceed`, the agent prompt fields and `ref` (Graph message id). |
| **Is label "Academy"?** | Branches on `proceed`. False → *Do nothing* (non-academic, low confidence, duplicate). |
| **AI Agent** | Searches `academy_docs` and drafts the body, or returns `NOT_IN_DOCUMENTS`. |
| **Finalize** → `/emails/finalize` | Grounding net (rejects `NOT_IN_DOCUMENTS` and prose that only reports the documents as silent), wraps the body in greeting/sign-off. |
| **Record reply** → `/threads/reply` | Stores the draft on the thread and queues it at `/review`. |
| **Wait for review** | Paused until `/review/{id}/send` resumes it. |

The safety logic (gate, grounding, email shell) lives in the API only; the
workflow is plumbing. Sending happens from the API straight to Graph when a
reviewer clicks Send.

---

## 2. Requirements

| Item | Requirement |
|---|---|
| OS | Linux x86_64 (Ubuntu 22.04/24.04 LTS assumed below) |
| Docker | Engine ≥ 24 with Compose v2 |
| CPU / RAM | 8 vCPU, 32 GB RAM recommended (embedder + API + n8n ≈ 6 GB; rest for the model host if co-located) |
| Disk | ≥ 60 GB free if the model is hosted here (weights ≈ 20 GB), otherwise ≥ 15 GB |
| GPU (if hosting the model) | NVIDIA, ≥ 24 GB VRAM for a 32B Q4 model; 16 GB for 14B |
| Chat model | OpenAI-compatible endpoint with **token logprobs**, **JSON-schema decoding**, **no reasoning output**, context ≥ 8192 — §4 |
| Network egress | `ghcr.io`, `docker.io`, `huggingface.co`, `login.microsoftonline.com`, `graph.microsoft.com` |
| Network ingress | SSH only. All service ports are loopback-bound. |
| Microsoft 365 | Rights to create two app registrations, and a tenant admin for consent — §6 |
| Repository access | Read access to this repo |

---

## 3. Host preparation

Skip the GPU parts if the chat model is hosted elsewhere.

```sh
sudo apt update && sudo apt -y upgrade
sudo apt -y install ca-certificates curl git jq gnupg

# Docker Engine + Compose v2
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"            # re-login to apply
docker compose version                     # v2.x

# NVIDIA driver (GPU hosts)
sudo apt -y install ubuntu-drivers-common && sudo ubuntu-drivers autoinstall && sudo reboot
nvidia-smi

# NVIDIA Container Toolkit (GPU hosts)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt -y install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If `nvidia-smi` fails after the driver install, Secure Boot is blocking the
module: disable it or enrol the MOK key.

**Firewall.** Allow inbound SSH only. Docker-published ports bypass `ufw`
(Docker writes its own iptables rules), which is why every port in this stack,
and the model server below, is bound to a loopback or bridge address rather
than relying on the host firewall.

---

## 4. Chat model

One instruction model does both classification and drafting. Hard
requirements:

| Requirement | Why | Failure mode if missing |
|---|---|---|
| Token **logprobs** on `/v1/chat/completions` | Confidence = P(`true`) of each classifier flag | Every email routes to a human, silently |
| **JSON-schema** `response_format` | Classifier output is grammar-constrained | Classification errors → human |
| **No reasoning/thinking output** | Output must be the JSON/body only | Empty classification → human |
| Context ≥ 8192 tokens | Drafting prompt carries doc extracts + thread | Truncated extracts, refusals |

Reference: **Qwen2.5-32B-Instruct Q4_K_M** on llama.cpp (non-reasoning, so
nothing to disable). Qwen2.5-14B on a 16 GB GPU works with more borderline mail
sent to people.

### 4a. llama.cpp on this host

```sh
sudo mkdir -p /opt/models && sudo chown "$USER" /opt/models
curl -L -o /opt/models/qwen2.5-32b-instruct-q4_k_m.gguf \
  "https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q4_K_M.gguf?download=true"

docker run -d --name llama --restart unless-stopped --gpus all \
  -p 172.17.0.1:8080:8080 \
  -v /opt/models:/models:ro \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /models/qwen2.5-32b-instruct-q4_k_m.gguf --alias qwen2.5-32b \
  --host 0.0.0.0 --port 8080 -c 16384 -ngl 999 --jinja

docker logs -f llama     # until "server is listening"
```

- `-p 172.17.0.1:8080:8080` publishes on the Docker bridge only: reachable
  from the stack's containers as `http://host.docker.internal:8080/v1`, not
  from the network. Check the bridge IP with
  `docker network inspect bridge -f '{{(index .IPAM.Config 0).Gateway}}'`.
- `--alias` is the value for `CHAT_MODEL`.
- Out of VRAM: lower `-ngl` (e.g. `40`) to offload layers to CPU.
- Qwen3 or another reasoning model: add `--reasoning-budget 0`.

### 4b. Other backends

- **LiteLLM** over llama.cpp/vLLM: returns logprobs. For reasoning models set
  `chat_template_kwargs: {"enable_thinking": false}` in the `model_list` entry.
- **Ollama**: ≥ 0.12 for logprobs; `OLLAMA_HOST=0.0.0.0`,
  `OLLAMA_CONTEXT_LENGTH=8192`, `OLLAMA_KEEP_ALIVE=-1`, a non-thinking tag.
  Without logprobs set `EC_ALLOW_UNCALIBRATED=1` (routing then uses the bare
  flags; still fails safe, but the confidence threshold no longer applies and
  responses carry `"calibrated": false`). If the context can't be raised
  server-side, `EC_NUM_CTX=8192`.

### 4c. Acceptance check

```sh
EP=http://172.17.0.1:8080/v1; M=qwen2.5-32b
curl -s $EP/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the word ok\"}],\"logprobs\":true,\"top_logprobs\":3,\"max_tokens\":10}" \
  | jq '{content: .choices[0].message.content, logprobs: (.choices[0].logprobs.content | length)}'
# {"content": "ok", "logprobs": >0}

curl -s $EP/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"JSON object with key ok set to true\"}],\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"t\",\"schema\":{\"type\":\"object\",\"properties\":{\"ok\":{\"type\":\"boolean\"}},\"required\":[\"ok\"]}}},\"max_tokens\":50}" \
  | jq -r '.choices[0].message.content'
# {"ok": true}
```

Both must pass before continuing.

---

## 5. Install and configure

Clone into a directory named `email-classifier`: Compose derives volume names
from it (`email-classifier_n8n_data`, …), and §9's backup commands assume it.

```sh
sudo mkdir -p /opt/email-classifier && sudo chown "$USER" /opt/email-classifier
git clone https://github.com/breachlabz/RAG-Academy_Auto-Reply_n8n_Workflow.git /opt/email-classifier
cd /opt/email-classifier
git checkout <release tag or commit>        # deploy a pinned revision, not a moving branch

cp .env.example .env && chmod 600 .env
```

### 5a. Configuration (`.env`)

Required:

| Variable | Value |
|---|---|
| `LLM_URL` | Model endpoint **as seen from a container**, e.g. `http://host.docker.internal:8080/v1` (§4a) |
| `LLM_KEY` | API key, blank for llama.cpp/Ollama |
| `CHAT_MODEL` | Model name the endpoint exposes (llama.cpp `--alias`) |

Set after §6 (sending):

| Variable | Value |
|---|---|
| `GRAPH_TENANT_ID`, `GRAPH_CLIENT_ID`, `GRAPH_CLIENT_SECRET` | App-only send registration (§6b) |
| `GRAPH_MAILBOX` | UPN of the mailbox replies are sent from |

Optional:

| Variable | Default | Effect |
|---|---|---|
| `EC_THRESHOLD` | `0.9` | Minimum P(academic) to draft; higher → more mail to people |
| `EC_NON_ACADEMIC_THRESHOLD` | `EC_THRESHOLD` | P(non_academic) at which an email goes to a person even if academic |
| `EC_ALLOW_UNCALIBRATED` | unset | Flag-only routing for backends without logprobs (§4b) |
| `EC_NUM_CTX` | unset | Ollama `num_ctx` per request |
| `RAG_EMAIL_GREETING` / `RAG_EMAIL_SIGNOFF` | `Hello,` / `Best regards,\nThe Training Team` | Reply shell; `\n` = line break |
| `CHAT_MODEL_LARGE` | `CHAT_MODEL` | Larger model for evaluation runs only |
| `REVIEW_DRY_RUN` | unset | **Testing only.** Send marks rows sent without calling Graph. Must be unset in production. |

### 5b. Start and load documents

```sh
docker compose up -d --build
docker compose logs -f embedder          # first boot pulls bge-m3 (~2 GB); wait for "Ready"
docker compose ps                        # 4 containers Up, API (healthy)
curl -fsS http://127.0.0.1:8100/health   # {"ok":true}

docker compose exec classifier python -m rag check            # extraction stats per document
docker compose exec classifier python -m rag ingest --reset   # "NN chunks -> collection 'docs'"
docker compose exec classifier python -m rag ask "what are the training levels?"
```

`ingest` also populates the `knowledge_chunks` table (§9c).

Classifier smoke test:

```sh
c() { curl -s -X POST http://127.0.0.1:8100/classify -H 'Content-Type: application/json' -d "{\"text\":\"$1\"}" | jq -c '{type,route,probs}'; }
c "What does EVH Level 3 module 2.4 cover?"                 # academic / rag
c "I was charged twice for my course, please refund me."    # non_academic / human
```

---

## 6. Microsoft 365 app registrations

Two separate registrations, least privilege each. Record client IDs, secret
values and **secret expiry dates**.

### 6a. Inbox read (delegated, used by n8n)

Entra ID → App registrations → New registration:

- Name `academy-email-read`, single tenant.
- Redirect URI (Web): `http://localhost:5678/rest/oauth2-credential/callback`
  — Entra accepts plain HTTP only for `localhost`, which is why n8n is reached
  through an SSH tunnel at exactly `http://localhost:5678`.
- Certificates & secrets → new client secret.
- API permissions → Microsoft Graph → **Delegated**: `Mail.Read`,
  `offline_access` → Grant admin consent.

### 6b. Send (application, used by `/review` Send)

- New registration `academy-email-send`, single tenant, no redirect URI.
- API permissions → Microsoft Graph → **Application**: `Mail.Send` → Grant
  admin consent.
- Certificates & secrets → new client secret.
- Put tenant ID, client ID, secret value and mailbox UPN in `.env` (§5a),
  then `docker compose up -d`.

**Scope it to the one mailbox.** An application `Mail.Send` grant can send as
any mailbox in the tenant until restricted. Restrict it in Exchange Online
(application access policy, or RBAC for Applications where your tenant uses
it), e.g.:

```powershell
Connect-ExchangeOnline
New-ApplicationAccessPolicy -AppId <send-app-client-id> `
  -PolicyScopeGroupId <mail-enabled-security-group-containing-the-mailbox> `
  -AccessRight RestrictAccess -Description "Academy auto-reply send scope"
Test-ApplicationAccessPolicy -Identity <mailbox-upn> -AppId <send-app-client-id>   # Granted
Test-ApplicationAccessPolicy -Identity <any-other-upn> -AppId <send-app-client-id> # Denied
```

---

## 7. n8n workflow

### 7a. Access

n8n and the review UI are loopback-only. From an admin workstation:

```sh
ssh -N -L 5678:127.0.0.1:5678 -L 8100:127.0.0.1:8100 <user>@<host>
```

n8n: `http://localhost:5678` (create the owner account on first load; store
it). Review UI: `http://localhost:8100/review`.

### 7b. Credentials

| Name | Type | Settings |
|---|---|---|
| Outlook (read) | Microsoft Outlook OAuth2 API | §6a client ID/secret → **Connect my account** as the mailbox → *Account connected* |
| Chat model | OpenAI API | Base URL = `LLM_URL`, API key = `LLM_KEY` or any non-empty string |
| Embedder | OpenAI API | Base URL `http://embedder:80/v1`, API key any non-empty string |
| Chroma | Chroma API (self-hosted) | Base URL `http://email-classifier-chroma:8000`, no auth |

The Chroma credential must point at the same store the API writes to
(`local_chromadb`), otherwise knowledge edits and ingests don't reach the
agent.

### 7c. Import

1. Workflows → Import from File → `n8n/academy-agent-workflow.json`.
2. Bind credentials: **New Outlook email** → Outlook; **Local Model** → Chat
   model; **Embeddings bge-m3** → Embedder; **academy_docs** → Chroma.
3. **Local Model**: model = `CHAT_MODEL`. **academy_docs**: collection `docs`.
   **New Outlook email**: folder and poll interval (default Inbox, 1 min).
4. Save. Leave inactive until §8.

Re-importing a workflow drops credential bindings and deactivates it; redo
steps 2–4 after any import. `n8n/email-classifier-form-workflow.json` is an
optional manual test form (no credentials).

---

## 8. Go-live verification

1. n8n → run the workflow from **Test: run manually**. Expect `Finalize`
   `grounded: true` and a new row at `/review`.
2. Activate the workflow.
3. From an external account, email the mailbox: *"What are the training
   levels and who is Level 2 aimed at?"* → row at `/review` within ~2 min →
   **Send** → reply arrives with the original quoted.
4. Email *"My invoice still shows unpaid, can you check?"* → no row; execution
   ends at *Do nothing*, `reason: not routed to rag`.
5. Confirm `REVIEW_DRY_RUN` is unset:
   `docker exec email-classifier-api printenv REVIEW_DRY_RUN` prints nothing
   (the review page also shows a *Dry run* badge when it's set).
6. Take the first backup (§9e) and record secret expiry dates.

---

## 9. Operations

All commands from `/opt/email-classifier`.

### 9a. Daily use

- Reviewers open `/review` through the SSH tunnel (§7a), approve or edit
  drafts, click **Send**. A failed Send leaves the row queued with the Graph
  error shown; nothing is recorded as sent unless Graph accepted it.
- Only non-reviewable mail (non-academic, low confidence, undocumented) stays
  in the Outlook inbox for manual handling.

### 9b. Training documents

```sh
# replace/add files in data/docs/ (.docx .pdf .md .txt .html), then:
docker compose exec classifier python -m rag check
docker compose exec classifier python -m rag ingest --reset
```

`check` flags documents that extract little text (scanned PDFs, text in
images/shapes are invisible to retrieval).

### 9c. Knowledge base tuning (`knowledge_chunks`)

Every chunk in the collection has a row in `knowledge_chunks` in
`data/threads.db`: `content` (the exact text embedded and shown to the model)
and `metadata` (JSON: `source`, `heading`, `chunk_index`, plus any flat keys
you add). Any write re-embeds the chunk and upserts it into Chroma before the
row is committed, so the change applies to the next email on both reply paths.
If the embedder or Chroma rejects the write, the row is left unchanged.

- **UI:** `/review` → **Knowledge** tab: search, edit, add, delete, export JSON.
- **API:** `GET/POST /knowledge`, `GET/PUT/DELETE /knowledge/{id}` (§11).
- **Bulk:**
  ```sh
  docker compose exec -T classifier python -m rag knowledge export > chunks.json
  docker compose exec -T classifier python -m rag knowledge import - < chunks.json
  ```
  Import validates the whole file first, updates changed chunks, creates
  entries with no/unknown id as manual chunks, never deletes.

Rules:

- **Documents win.** `ingest` overwrites the rows of every file it loads
  (edits included) and removes rows for sections no longer present. Durable
  changes belong in the source document.
- Manual chunks (`origin: manual`, id `manual:<hex>`) are never touched by
  ingest and are re-embedded after `--reset`. Put the subject in `content`;
  metadata is not embedded.
- A manual chunk that contradicts a document chunk is not resolved for you —
  both can be retrieved. Fix or delete the stale one.
- Metadata values must be flat (string, number, boolean); `source` and
  `heading` are required.
- Table empty on an existing deployment (collection ingested before the table
  existed): `docker compose exec classifier python -m rag knowledge pull`.

### 9d. Configuration changes

Edit `.env`, then `docker compose up -d` (recreates only changed services).
Reply wording, thresholds and Graph credentials are all `.env`.

### 9e. Backups

| Asset | Location | Method |
|---|---|---|
| Conversations, review queue, knowledge edits, manual chunks | `data/threads.db` (+ `-wal`) | SQLite online backup (below) — don't `cp` a live WAL database |
| n8n workflows, credentials, execution history | volume `email-classifier_n8n_data` | tar of the volume |
| Configuration and secrets | `.env` | copy, encrypted at rest |
| Vector store | volume `email-classifier_chroma-data` | not required: rebuilt by `ingest --reset` (manual chunks restored from `threads.db`) |

```sh
B=/var/backups/email-classifier/$(date +%F); sudo mkdir -p "$B" && sudo chown "$USER" "$B"
docker compose exec -T classifier python -c \
  "import sqlite3; sqlite3.connect('/app/data/threads.db').backup(sqlite3.connect('/app/data/threads.backup.db'))" \
  && mv data/threads.backup.db "$B/threads.db"
docker run --rm -v email-classifier_n8n_data:/v:ro -v "$B":/out alpine tar czf /out/n8n_data.tgz -C /v .
install -m 600 .env "$B/env"
```

Schedule it (e.g. `/etc/cron.d/email-classifier`, daily) and ship `$B` off
host. Retention per your policy.

Restore:

```sh
docker compose stop classifier n8n
cp <backup>/threads.db data/threads.db && rm -f data/threads.db-wal data/threads.db-shm
docker run --rm -v email-classifier_n8n_data:/v -v <backup>:/in alpine sh -c "find /v -mindepth 1 -delete && tar xzf /in/n8n_data.tgz -C /v"
docker compose up -d
docker compose exec classifier python -m rag ingest --reset   # if chroma-data was lost
```

### 9f. Upgrades and rollback

```sh
# 1. back up (9e)
git fetch --tags && git checkout <new tag or commit>
docker compose up -d --build            # rebuilds the API image; recreates changed services
docker compose ps && curl -fsS http://127.0.0.1:8100/health
# 2. run the classifier smoke test (5b) and apply any release-specific steps
```

Rollback: `git checkout <previous tag>` and `docker compose up -d --build`.
Schema changes are additive (new tables/columns only), so older code runs
against a newer `threads.db`. If a release re-ingested the collection, re-run
`ingest --reset` after rolling back.

Release-specific steps for the version introducing `academic`/`non_academic`
classification and the Knowledge tab, when upgrading an existing deployment:

- `docker compose exec classifier python -m rag knowledge pull`
- `docker compose up -d` also recreates `n8n` (new `host.docker.internal`
  mapping); state is in its volume.
- Re-import the optional form workflow if used (score fields renamed).
- `EC_ADMIN_THRESHOLD` was renamed `EC_NON_ACADEMIC_THRESHOLD`.

### 9g. Monitoring

| Signal | Check |
|---|---|
| API liveness | `curl -fsS http://127.0.0.1:8100/health`; container health `docker inspect -f '{{.State.Health.Status}}' email-classifier-api` |
| All services up | `docker compose ps` (+ `docker ps -f name=llama` for the model) |
| Model reachable from the stack | `docker exec email-classifier-api python -c "from classifier.core import BASE_URL,auth_headers;import urllib.request as u;print(u.urlopen(u.Request(BASE_URL+'/models',headers=auth_headers()),timeout=5).status)"` → `200` |
| Mail flowing | n8n → Executions: successful runs every poll; failures on *Prepare* indicate model/API problems |
| Review backlog | `curl -s http://127.0.0.1:8100/review/queue \| jq '[.pending[].exchanges[]] \| length'` |
| Logs | `docker compose logs --since 1h classifier` (also `n8n`, `embedder`, `local_chromadb`) |
| Secret expiry | both Entra client secrets (§6) — rotate before expiry: update the n8n Outlook credential and `GRAPH_CLIENT_SECRET` + `docker compose up -d` |

### 9h. Lifecycle

```sh
docker compose stop | start | restart classifier
docker compose down          # removes containers, keeps volumes and data/
docker compose down -v       # DESTROYS n8n credentials/workflows and vectors — never in production
```

All services use `restart: unless-stopped` and come back after a host reboot.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| *Prepare*: `error: expected 2 bools, got 0` | Backend returns no logprobs | §4c; Ollama → `EC_ALLOW_UNCALIBRATED=1` |
| *Prepare*: `error: request failed …` / `rag ask` connection refused | API can't reach `LLM_URL` | Test from the container (§9g); same-host server must be bound to the bridge/0.0.0.0, not 127.0.0.1 |
| Empty or malformed `flags` | Model emits reasoning | §4a `--reasoning-budget 0` / §4b `enable_thinking: false` |
| Everything goes to *Human queue* on academic mail | Documents don't answer it, or truncated context | `ingest --reset`; confirm topic in `data/docs/`; context ≥ 8192 |
| *Do nothing*, `already handled (duplicate message_id)` | Same message polled again | Expected |
| `/health` fails or Chroma errors in API logs | Chroma/embedder not ready | `docker compose logs local_chromadb embedder`; `docker compose restart classifier` |
| `embedder` restarting | Model download in progress or disk full | `docker compose logs embedder`; ≥ 3 GB free |
| Knowledge save returns 502 | Embedder or Chroma unreachable | Row unchanged; fix the service and retry |
| n8n Outlook *Connect* fails | Redirect mismatch | Browser must be at exactly `http://localhost:5678`; redirect URI must match §6a |
| n8n agent answers from stale content | n8n Chroma credential points elsewhere | Must be `http://email-classifier-chroma:8000` (§7b) |
| Send → 503 *not configured* | `GRAPH_*` missing | §6b, then `docker compose up -d` |
| Send → 502 | Graph rejected | Row stays queued; check admin consent, `GRAPH_MAILBOX`, access policy scope |
| Rows marked sent but no mail delivered | `REVIEW_DRY_RUN` set | Unset it, `docker compose up -d` |

---

## 11. HTTP API

`http://classifier:8100` inside the compose network, `http://127.0.0.1:8100`
on the host. No authentication — loopback/tunnel access only (§12).

| Endpoint | Purpose |
|---|---|
| `GET /health` | `{"ok": true}` |
| `POST /classify` | `{"text"}` → `type`, `route`, `flags`, `probs`, `calibrated` |
| `POST /answer` | `{"text"}` → classify + retrieve + grounded reply, stateless (test form, `scripts/rag_eval.py`) |
| `POST /generate-reply` | `{"email_text", "conversation_id", "subject", "message_id"}` → full pipeline in one call, thread-aware |
| `POST /emails/prepare` | `{"body", "is_html", "subject", "conversation_id", "message_id", "ref"}` → `proceed`, `history`, `email_text`, `query`, `format`, `ref`, `duplicate`, `reason` |
| `POST /emails/finalize` | `{"output", "conversation_id", "subject", "format"}` → `grounded`, `reply`, `subject`, `agent_output`, `reason` |
| `POST /threads/reply` | `{"conversation_id", "reply", "subject", "grounded"}` → records the draft; queues it when grounded |
| `GET /threads`, `GET /threads/{id}` | conversations / one conversation's turns |
| `GET /review` | review UI (Reply review + Knowledge tabs) |
| `GET /review/queue`, `GET /review/history` | pending / sent replies grouped by conversation |
| `POST /review/{id}/send` | `{"reply", "attachment_*"?}` → sends via Graph, records it. 503 not configured, 502 Graph rejected (row stays queued), 413 attachment > 3 MB |
| `GET /knowledge` | all chunks `{id, content, metadata, origin, edited, created_at, updated_at}` + `sources` |
| `POST /knowledge` | `{"content", "metadata"}` → manual chunk, embedded and upserted |
| `GET`/`PUT`/`DELETE /knowledge/{id}` | one chunk; `PUT {"content"?, "metadata"?}` re-embeds (metadata replaces the whole object). URL-encode ids (`#` → `%23`). 404 / 422 invalid / 502 embedder or Chroma failure |

- `type` is `academic` only when academic and nothing else; otherwise
  `non_academic`. `unknown` = classification failed.
- `route` is the field to branch on (`rag` / `human`).
- `format`: `auto` (list vs prose from the enquirer's wording), `list`, `prose`.

---

## 12. Security model and guarantees

- **No automatic sending.** Only `POST /review/{id}/send` reaches
  `mail/graph.py`; nothing in the classify/retrieve/draft path has Graph
  access. A failed Send is never recorded as sent.
- **No network exposure.** Every service binds to `127.0.0.1` (model server to
  the Docker bridge). The review UI, Knowledge editor and API have **no
  authentication** — access is via SSH tunnel only. Exposing them requires an
  authenticating reverse proxy (SSO) in front; knowledge edits change live
  replies.
- **n8n executes arbitrary code** — never publish port 5678.
- **Least privilege in Graph:** read is delegated `Mail.Read`; send is
  application `Mail.Send` scoped to one mailbox (§6b).
- **Secrets** live in `.env` (mode 600) and the n8n volume. Neither is in git.
  Rotate both Entra secrets before expiry.
- **Non-academic mail never gets a draft**, and every factual sentence in a
  draft comes from `data/docs/` or the knowledge table; unsupported questions
  go to a person.
- **Attachments** added at Send are passed straight to Graph (≤ 3 MB) and
  never stored; only the file name is recorded.

---

## 13. Design notes

**Two labels, multi-label.** `academic` and `non_academic` are independent
booleans under a constrained grammar, so the logprob of each `true`/`false`
token is P(label) — a real number to threshold, not a model-stated
confidence. A course question with a payment issue sets both and goes to a
person.

**Two independent gates, both failing to `human`.** The classifier asks
whether an email *may* be auto-answered: `route()` returns `human` for
non-academic (≥ `EC_NON_ACADEMIC_THRESHOLD`), not academic, below
`EC_THRESHOLD`, malformed or errored. Retrieval/drafting asks whether it *can*
be answered from what we hold (`grounded`). Neither raises; failures arrive as
`error`.

**The distance gate cannot judge correctness.** Distances for answerable and
unanswerable questions overlap (a price question is on-topic but may have no
answer). `RAG_MAX_DISTANCE` is only a guard against unrelated input. The model
declining is the real gate, backed by `rag.core.refuses_in_prose()`, which
catches prose that reports the documents as silent (a source word **and** a
negated reporting verb). `scripts/rag_eval.py` pins that boundary — rerun it
whenever the prompt or model changes.

**Chunk size is a retrieval parameter.** At 1200 chars a per-level section
split and "what are the three levels?" came back with two. At 3000 sections
survive whole.

**Threading.** A follow-up ("and the second one?") carries no subject matter
of its own. `threads/` stores conversations (SQLite, keyed by Outlook
`conversationId`, deduplicated on `internetMessageId`); `classify()` is shown
the thread as context but labels the latest email only, so an invoice question
inside an academic thread still goes to a person. Follow-ups are rewritten into
standalone questions before embedding.

**Embedding.** bge-m3, 1024-dim, CPU. Vectors are always computed in
`rag.core.embed()` and passed to Chroma (collections use
`embedding_function=None`). Changing `RAG_EMBED_MODEL` requires
`ingest --reset`.

**.docx extraction is lossy.** `rag/docx_text.py` handles bold-only headings,
hard-wrapped text and tables, but ignores images, shapes and text boxes. Run
`python -m rag check` on every new document.

**`SYSTEM` and `SCHEMA` in `classifier/core.py` must stay in sync** (same
keys, same order). Otherwise constrained decoding forces tokens the model
finds unlikely and the logprobs stop measuring the classification.

---

## 14. Evaluation

```sh
python scripts/generate.py --out data           # synthetic set on the large model, 70/30 split
python scripts/evaluate.py data/holdout.jsonl   # classifier threshold sweep
python scripts/rag_eval.py                       # grounding: leaks vs misses
python scripts/thread_eval.py                    # follow-up resolution
docker cp tests email-classifier-api:/app/ && docker compose exec classifier python -m unittest tests.test_logic   # tests are not baked into the image
```

Generation uses a different model than classification to avoid correlated
blind spots. `rag_eval.py` fails only on a **leak** (answered without
support). Set the production threshold against hand-labelled real mail, not
the synthetic set.

---

## 15. Repository layout

```
api.py                      HTTP API (classify, answer, prepare/finalize, threads, review, knowledge)
classifier/core.py          classify() + route(): prompt, JSON schema, thresholds
rag/core.py                 ingest(), retrieve(), answer(), embed(), chunking, grounding
rag/knowledge.py            knowledge_chunks table, synced to Chroma on every write
rag/docx_text.py            .docx → markdown, extraction checks
rag/format_hint.py          list-vs-prose decision
rag/__main__.py             CLI: python -m rag {ingest|ask|check|manifest|knowledge}
threads/store.py            conversations, turns, dedupe, review queue (SQLite)
threads/context.py          history block, standalone-question rewrite, rolling summary
threads/gist.py             one-line enquiry summary for the review UI
mail/text.py                HTML → text, quoted-reply stripping
mail/graph.py               app-only Graph send (reply, HTML rendering, attachment)
frontend/                   review UI (Next.js static export → served at /review)
n8n/academy-agent-workflow.json          mailbox pipeline
n8n/email-classifier-form-workflow.json  optional manual test form
data/docs/                  source documents
tests/test_logic.py         unit and regression tests
scripts/                    generate / evaluate / rag_eval / thread_eval
docker-compose.yml          the stack
```
