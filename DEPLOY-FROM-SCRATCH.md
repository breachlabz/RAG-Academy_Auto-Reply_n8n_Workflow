# Academy Auto-Reply — full setup on new hardware

Handover guide for **Davide**. Start from a bare Linux box with a GPU and end
with the Outlook auto-reply pipeline running headless.

This covers the parts `README.md` assumes you already have — the OS, the GPU
stack and the chat model server — then hands back to `README.md` for the
application itself. Where a step is just "do what the README says", it says so;
keep `README.md` open alongside this.

**What you are building**

```
  Outlook mailbox ──▶ n8n ──▶ classifier API ──┬──▶ chromadb   (vector store)   ┐
                              (FastAPI :8100)  └──▶ embedder   (bge-m3, CPU)    │ docker compose
                                │        review queue (/review, human sends)   │ (this repo)
                                │              │                               ┘
                                ▼              ▼
                    chat model server     Microsoft Graph (app-only, Send)
                    (llama.cpp, ~27–32B,       │
                     on the GPU)               ▼
                    ← you set this up    Outlook mailbox (sent)
```

The chat model does the classification **and** the reply drafting. Everything
else is CPU. Nothing sends automatically — every reply waits in the review
queue until a human sends it (§10a, README §14).

---

## 0. Before you start — what you need

| Thing | Detail |
|---|---|
| Linux host, root/sudo | Ubuntu 22.04 or 24.04 LTS assumed below. Adjust package names for other distros. |
| NVIDIA GPU, **≥ 24 GB VRAM** | for a 32B model at Q4. 16 GB works with a 14B (more misses — see README §5). No GPU → see §3 note. |
| ~60 GB free disk | model weights (~20 GB) + embedding model (~2 GB) + docker images + headroom |
| Outbound internet | to pull images, model weights, and reach Microsoft Graph |
| A Microsoft 365 tenant where you can **create an app registration** | §7 |
| The mailbox account credentials | the inbox this will watch |
| This repo | `git clone https://github.com/breachlabz/RAG-Academy_Auto-Reply_n8n_Workflow.git` (or the folder handed to you) |

Time: about 1–2 hours, most of it model download and the Azure screens.

---

## 1. Base OS packages

```sh
sudo apt update && sudo apt -y upgrade
sudo apt -y install curl git jq build-essential ca-certificates gnupg
sudo reboot        # if the upgrade pulled a new kernel
```

Optional but recommended — a non-root user in the `docker` group (created in §4).

---

## 2. NVIDIA driver

```sh
sudo apt -y install ubuntu-drivers-common
sudo ubuntu-drivers autoinstall           # or: sudo apt -y install nvidia-driver-550
sudo reboot
```

After the reboot:

```sh
nvidia-smi        # must print the GPU, driver version, and CUDA version
```

If `nvidia-smi` fails: secure boot can block the unsigned module — either
disable secure boot in the BIOS or enrol the MOK key the installer offered.

---

## 3. Docker + NVIDIA Container Toolkit

**Docker Engine + Compose v2:**

```sh
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"           # log out/in afterwards for this to take effect
docker compose version                    # must print v2.x
```

**NVIDIA Container Toolkit** (lets containers see the GPU):

```sh
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt -y install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify the GPU is visible from a container:

```sh
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

> **No GPU at all?** You can run the chat model on CPU (llama.cpp works, just
> slow — expect 30–90 s per email) or point `LLM_URL` at a chat model on
> another machine. Skip §2 and the `--gpus all` flag in §5; everything else is
> unchanged.

---

## 4. The chat model server (llama.cpp)

We run **one** instruction model behind an OpenAI-compatible endpoint. The
classifier needs three things from it (all satisfied by current llama.cpp):

- **token logprobs** — the classifier reads P(`true`/`false`) to score
  confidence. No logprobs ⇒ every email routes to a human, silently.
- **JSON-schema constrained decoding** (`response_format: json_schema`).
- **no hidden "thinking" tokens** in the output.

Using a **non-reasoning** model (Qwen2.5-32B-Instruct) means there is no
thinking to disable — the simplest correct choice. Qwen3 also works but needs
its thinking mode turned off (see the note at the end of this section).

### 4a. Download the weights

```sh
sudo mkdir -p /opt/models && sudo chown "$USER" /opt/models
cd /opt/models
# Qwen2.5-32B-Instruct, Q4_K_M (~19.9 GB). Fits a 24 GB card with an 8–16k context.
curl -L -o qwen2.5-32b-instruct-q4_k_m.gguf \
  "https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q4_K_M.gguf?download=true"
```

Smaller card: use `bartowski/Qwen2.5-14B-Instruct-GGUF` → `Q4_K_M` (~9 GB) and
set `CHAT_MODEL=qwen2.5-14b` throughout. The classifier still fails safe; it
just sends more borderline mail to the human queue (README §5, §11).

### 4b. Run it

```sh
docker run -d --name llama --restart unless-stopped --gpus all \
  -p 8080:8080 \
  -v /opt/models:/models \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /models/qwen2.5-32b-instruct-q4_k_m.gguf \
  --alias qwen2.5-32b \
  --host 0.0.0.0 --port 8080 \
  -c 16384 -ngl 999 --jinja
```

- `--host 0.0.0.0` — **required** so the compose stack can reach it via
  `host.docker.internal` (README §3). Since port 8080 is now open on the box,
  keep the machine behind a firewall / not on the public internet, or bind to
  the docker bridge IP `172.17.0.1:8080` instead of `0.0.0.0`.
- `-c 16384` — context window. Must be ≥ 8192 (drafting prompt is long).
- `-ngl 999` — all layers on the GPU. Drop to a number (e.g. `-ngl 40`) if you
  run out of VRAM; the rest runs on CPU.
- `--alias qwen2.5-32b` — the model name the endpoint reports. This is what
  goes in `CHAT_MODEL`.

Watch it load:

```sh
docker logs -f llama        # wait for "server is listening on http://0.0.0.0:8080", then Ctrl-C
```

### 4c. Verify logprobs, JSON and no-thinking — do not skip

```sh
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-32b","messages":[{"role":"user","content":"Reply with the word ok"}],
       "logprobs":true,"top_logprobs":3,"max_tokens":10}' | jq '.choices[0]'
```

Pass criteria:

- `.message.content` is exactly `"ok"` — non-empty, no `<think>` block. Good.
- `.logprobs.content` is a **populated array**. If it is `null`, logprobs are
  off and the classifier cannot calibrate — fix the server before continuing
  (with current llama.cpp it should just work).

JSON schema check:

```sh
curl -s http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-32b","messages":[{"role":"user","content":"Give me a JSON object with one key ok set to true"}],
       "response_format":{"type":"json_schema","json_schema":{"name":"t","schema":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}}},
       "max_tokens":50}' | jq -r '.choices[0].message.content'
#   -> {"ok": true}
```

> **If you use Qwen3 instead:** add `--reasoning-budget 0` to the llama.cpp
> command (disables thinking). Re-run 4c and confirm `.message.content` has no
> `<think>…</think>` wrapper. If it does, the classifier will get empty
> classifications and route everything to a human.

> **Alternatives to llama.cpp:**
> - **Ollama** — easy, but its OpenAI endpoint returned no logprobs before
>   ~v0.12. If you go this route: install Ollama ≥ 0.12, `ollama pull
>   qwen2.5:32b`, set `OLLAMA_HOST=0.0.0.0`, `OLLAMA_CONTEXT_LENGTH=8192`,
>   `OLLAMA_KEEP_ALIVE=-1`, and run the 4c check. If logprobs come back `null`,
>   set `EC_ALLOW_UNCALIBRATED=1` in `.env` (README §5a) — still safe, just no
>   confidence threshold.
> - **LiteLLM** in front of llama.cpp — only needed if you want one endpoint
>   for several models or central keys. Not required here.

---

## 5. Get the project onto the box

```sh
cd ~
git clone https://github.com/breachlabz/RAG-Academy_Auto-Reply_n8n_Workflow.git email-classifier
cd email-classifier
```

(Or drop the handed-over folder here and `cd` into it.)

The training documents are already in `data/docs/` (EVH + ACP overviews, FAQs,
level syllabi). To change them later: edit that folder and re-run the ingest in
§7 of the README.

---

## 6. Configure and start the stack

```sh
cp .env.example .env
```

Edit `.env` — set exactly these three:

```ini
# llama.cpp on this same host, bound to 0.0.0.0 in §4b:
LLM_URL=http://host.docker.internal:8080/v1
LLM_KEY=
CHAT_MODEL=qwen2.5-32b        # must match --alias from §4b
```

If the model runs on a **different** machine, use
`LLM_URL=http://<that-ip>:8080/v1` instead and make sure the firewall allows it.

Bring it up (from `email-classifier/`):

```sh
docker compose up -d --build
docker compose logs -f embedder      # wait for "Ready" / "Starting HTTP server", then Ctrl-C (first boot pulls ~2 GB)
docker compose ps                    # n8n, classifier, local_chromadb, embedder all "Up"
curl -s http://127.0.0.1:8100/health ; echo      # {"ok":true}
```

Load the documents into the vector store and prove retrieval works end to end
(this call hits your llama.cpp):

```sh
docker compose exec classifier python -m rag ingest --reset
#   "NNN chunks -> collection 'docs'"
docker compose exec classifier python -m rag ask "what are the training levels?"
#   a real answer drawn from the docs + source headings
```

If either fails, README §9 has the symptom→fix table. The usual culprit is
`LLM_URL` not being reachable from the container — recheck §4b (`--host
0.0.0.0`) and that `curl http://127.0.0.1:8080/v1/models` answers on the host.

---

## 7. Register the app in Microsoft Entra (Azure AD)

This is what lets n8n read the inbox and create drafts. You need the **Client
ID** and a **Client secret** at the end.

1. <https://portal.azure.com> → **Microsoft Entra ID** → **App registrations**
   → **New registration**.
2. **Name**: `academy-email-autoreply`. **Supported account types**: *Accounts
   in this organizational directory only* (single tenant) is fine.
3. **Redirect URI**: platform **Web**, value **exactly**:
   ```
   http://localhost:5678/rest/oauth2-credential/callback
   ```
   Microsoft rejects plain-`http` redirects *except* on `localhost` — this is
   why n8n stays on localhost and you reach it over an SSH tunnel (§8).
4. **Register**.
5. **Overview** → copy the **Application (client) ID**.
6. **Certificates & secrets** → **Client secrets** → **New client secret** →
   pick 12–24 months → **Add** → copy the **Value** immediately (it is hidden
   once you leave the page). This is the **Client secret**.
7. **API permissions** → **Add a permission** → **Microsoft Graph** →
   **Delegated permissions** → add:
   - `Mail.Read`
   - `offline_access`

   Then **Grant admin consent for <tenant>** (needs an admin; if you are not
   one, ask your Microsoft 365 admin to click it). Both should show a
   green tick.

   Read-only on purpose: this app registration is only for **polling the
   inbox**. Nothing in n8n creates a draft or sends any more — that moved to
   the review queue (§10a/§12), which uses a *second*, app-only registration
   with its own `Mail.Send` permission, not this one.

Keep the Client ID and secret to hand for §9.

---

## 8. Reach the n8n UI (SSH tunnel)

n8n is bound to `127.0.0.1:5678` on the box and is never exposed — it runs
arbitrary workflow code. From your laptop:

```sh
ssh -L 5678:127.0.0.1:5678 <user>@<the-box>
# leave that session open, then on your laptop browser open:
#   http://localhost:5678
```

The address must be exactly `http://localhost:5678` — Entra only accepts
`http://localhost` (not `127.0.0.1`, not a hostname) as the OAuth redirect.

On first load n8n asks you to create an **owner account** — this is local to the
box, use any email/password and keep it in the password manager.

The tunnel and browser are only for setup and later maintenance. Once the
workflow is **Active**, the pipeline runs headless.

---

## 9. Create the n8n credentials

**Credentials** (left sidebar) → **Add credential**, four of them:

### 9a. Microsoft Outlook OAuth2 API
- **Client ID** / **Client Secret**: from §7.
- Confirm the **OAuth Redirect URL** shown matches the Entra one character for
  character.
- **Connect my account** → sign in as the **mailbox account** → consent.
- Must end on **Connected / Account connected**.

### 9b. Chat model — type **OpenAI API**
- Name it `Chat model (llama.cpp)`.
- **Base URL**: `http://host.docker.internal:8080/v1`
  *(the n8n container reaches the host the same way the classifier does — if
  your model is on another box, use that address here too).*
- **API Key**: any non-empty string (llama.cpp ignores it) — e.g. `x`.

### 9c. Embedder — type **OpenAI API**  ← separate from 9b, easy to miss
- Name it `Embedder (bge-m3)`.
- **Base URL**: `http://embedder:80/v1`  *(the embedder container, inside the
  compose network — this is **not** your chat model)*.
- **API Key**: any non-empty string — e.g. `x`.

### 9d. Chroma — type **Chroma API** (self-hosted)
- Name it `Chroma (academy_docs)`.
- **Base URL**: `http://email-classifier-chroma:8000`
- No authentication.
- This is the same store `python -m rag ingest` writes to.

---

## 10. Import and wire the workflow

1. **Workflows → ⋯ (top-right) → Import from File** →
   `n8n/academy-agent-workflow.json`.
2. Open the nodes flagged with a red credential warning and attach:

   | Node | Credential |
   |---|---|
   | **New Outlook email** (trigger) | Outlook OAuth2 (9a) |
   | **Local Model** | Chat model (9b) |
   | **Embeddings bge-m3** | **Embedder (9c)** — not the chat model |
   | **academy_docs** | Chroma (9d) |

3. Open **Local Model** and set the model to your alias — `qwen2.5-32b`
   (the imported file has an older name cached).
4. Open **academy_docs** and confirm the collection is **`docs`**.
5. Open **New Outlook email** — set the folder (default **Inbox**) and poll
   interval (default **every minute**).
6. **Save**.

> Re-importing the workflow later **wipes credential bindings and the Active
> toggle**. Re-attach and re-activate after any import.

The imported topology is `Prepare → Is label "Academy"? → AI Agent → Finalize →
Grounded answer? → Record reply`, with the false branches going to *Do
nothing* / *Human queue*. That is the intended wiring — nothing to re-wire.
`Record reply` is the workflow's last node now: it hands the reply to the
**review queue** (§9e) rather than creating an Outlook draft. (README §1 has
the node-by-node table.)

`n8n/email-classifier-form-workflow.json` is an optional manual test form — import
it the same way, it needs no credentials.

---

## 10a. Set up sending (review queue, app-only Graph)

The review queue lives at `http://<box>:8100/review` — same host as the
classifier API, no tunnel needed once you're on that network. It reads and
lets you edit every grounded reply for free; the **Send** button additionally
needs its own Graph credential, separate from §7/9a because sending has to
work with nobody signed in:

1. Entra ID → App registrations → **New registration** — a second app, e.g.
   `academy-email-send`. Do not reuse §7's app or add this permission to it.
2. API permissions → Microsoft Graph → **Application permissions** (not
   delegated) → add `Mail.Send` → **Grant admin consent**.
3. Certificates & secrets → new client secret → copy the **Value**.
4. Copy the **Application (client) ID** and **Directory (tenant) ID**.
5. In `.env`:
   ```ini
   GRAPH_TENANT_ID=<tenant id>
   GRAPH_CLIENT_ID=<client id>
   GRAPH_CLIENT_SECRET=<secret value>
   GRAPH_MAILBOX=<mailbox sign-in email>
   ```
6. `docker compose up -d`.

Until this is done, `/review` still works for reading/editing; **Send**
returns a clear "not configured" error instead of pretending to send. Full
detail: README §6e.

---

## 11. End-to-end test (workflow still inactive)

Use the built-in manual trigger first — in the workflow, click **Test: run
manually**. It feeds a canned academic email through `Prepare → … → Finalize`
without touching Outlook. Expect the execution to reach **Finalize** with
`grounded: true` and a drafted reply body.

Then flip it live:

1. Toggle the workflow **Active**.
2. From another account, email the mailbox:
   > **Subject:** Course question
   > **Body:** What are the training levels and who is Level 2 aimed at?
3. Within ~1–2 minutes a **row appears at `http://<box>:8100/review`**
   listing the levels from the documents, with an edit box pre-filled. The
   n8n execution ends at **Record reply**. Click **Send** there (needs §10a
   done) to actually deliver it.
4. Send a second email: *"My invoice still shows unpaid, can you check?"* →
   **nothing queued**, execution ends at **Do nothing** (`reason: not routed
   to rag`). This is correct — billing is never auto-answered.

If the first test produced no queue row, open the failed execution and read
the node output against README §7's table. Most common:

| Symptom (in the **Prepare** node output) | Fix |
|---|---|
| `error: expected 3 bools, got 0` | model returned no logprobs — recheck §4c; on Ollama set `EC_ALLOW_UNCALIBRATED=1` |
| empty / garbled `flags` | model is emitting "thinking" — use the non-thinking model or `--reasoning-budget 0` (§4) |
| `error: request failed …` | classifier can't reach llama.cpp — §6, README §9 |
| ends at **Human queue** on an academic email | agent found no grounded answer — re-run `ingest --reset`, confirm the topic is in `data/docs/` |
| row is in `/review`, but **Send** returns "not configured" | §10a not done yet — expected until `GRAPH_*` is set in `.env` |
| **Send** returns a 502 from Graph | row stays queued, nothing lost — the error names what Graph rejected; check `Mail.Send` admin consent and that `GRAPH_MAILBOX` is a real mailbox |

---

## 12. Going live / handover state

Once the two test emails behave correctly:

- Leave the workflow **Active**.
- Close the SSH tunnel — the pipeline now runs headless on its polling trigger.
- **Nothing sends automatically.** Every grounded reply waits at `/review`
  until a human reads it, optionally edits it, and clicks Send.

### Day-2 operations (full detail in README §8)

| Task | Command (from `email-classifier/`) |
|---|---|
| Change the training docs | edit `data/docs/`, then `docker compose exec classifier python -m rag ingest --reset` |
| Update the app code | `git pull && docker compose up -d --build classifier` |
| Change greeting / sign-off | `RAG_EMAIL_GREETING` / `RAG_EMAIL_SIGNOFF` in `.env`, then `docker compose up -d` |
| Gate strictness | `EC_THRESHOLD` in `.env` (default `0.9`; higher → more mail to humans) |
| Restart the model | `docker restart llama` |
| Stop everything (keep data) | `docker compose stop` |

### Back up (these hold all the state)

```sh
# n8n workflows + credentials + history
docker run --rm -v email-classifier_n8n_data:/v -v "$PWD":/out alpine \
  tar czf /out/n8n-backup.tgz -C /v .
# conversation memory
cp data/threads.db data/threads.db.bak
# the vector store is rebuildable from data/docs/ with `ingest --reset` — no backup needed
```

Never run `docker compose down -v` — the `-v` deletes the volumes, losing the
n8n credentials (including the Outlook OAuth connection) and the vectors. The
`GRAPH_*` send credential lives in `.env`, not a volume — back that up
however you already handle secrets on this box.

---

## 13. Quick checklist

- [ ] `nvidia-smi` works on the host and inside a container
- [ ] `docker compose version` is v2.x
- [ ] llama.cpp up, `--host 0.0.0.0`, §4c logprobs check passes
- [ ] `.env` has `LLM_URL`, `LLM_KEY`, `CHAT_MODEL` (matches `--alias`)
- [ ] `docker compose ps` — 4 containers Up; `/health` ok
- [ ] `python -m rag ingest --reset` ran; `python -m rag ask` returns a real answer
- [ ] Entra trigger app: `Mail.Read` + `offline_access`, admin consent granted
- [ ] n8n reached at `http://localhost:5678` via SSH tunnel; owner account created
- [ ] 4 credentials created; Outlook shows **Connected**
- [ ] workflow imported, 4 credential attachments done, Local Model set to your alias
- [ ] manual test reaches Finalize `grounded: true`
- [ ] second Entra app-only registration: `Mail.Send`, admin consent granted; `GRAPH_*` set in `.env` (§10a)
- [ ] live: academic email → row at `/review`, Send delivers it; billing email → nothing queued
- [ ] workflow left **Active**; backups taken

---

*Questions on the internals — the gate, the grounding net, the threading model,
the eval scripts — are all in `README.md` §10–§12.*
