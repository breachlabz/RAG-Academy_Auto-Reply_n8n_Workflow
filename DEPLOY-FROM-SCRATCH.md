# Academy Auto-Reply — production setup, step by step

Handover guide for whoever sets up the production server. It assumes **no
prior experience** with Linux servers, Docker, n8n or Azure: every step says
what to type, what you should see, and what to do if you don't.

Start from a bare Linux server with a GPU. Finish with the Outlook auto-reply
pipeline running on its own, and a review page where a person approves every
reply before it is sent.

`README.md` has the deeper explanations (how the classifier decides, how
replies are grounded, troubleshooting). You do not need it to finish this
guide; it is referenced where it helps.

---

## Read this first (5 minutes)

**What you are building**

```
  Outlook mailbox ──▶ n8n ──▶ classifier API ──┬──▶ Chroma      (stores the training documents)
                              (port 8100)      └──▶ embedder    (turns text into vectors, CPU)
                                │        review page (/review): a person reads, edits, sends
                                ▼              │
                    chat model server          ▼
                    (llama.cpp on the GPU)   Microsoft Graph ──▶ reply leaves the mailbox
```

- Every incoming email is classified as **academic** (a question about courses,
  levels, exams, prices) or **non-academic** (invoices, refunds, account
  problems, spam — anything else).
- Only academic emails get a drafted reply, written **only** from the Word
  documents in `data/docs/`. Non-academic emails are left for a person.
- **Nothing is ever sent automatically.** Drafts wait on the review page until
  a person clicks Send.

**How to use this guide**

- Do the sections **in order**. Each one ends with a **✅ Check** — do not move
  on until it passes.
- Lines in grey boxes are commands. Copy them **one block at a time** into the
  terminal and press Enter. Lines starting with `#` are comments — they are
  there to explain, pasting them does nothing harmful.
- Anything in `<angle brackets>` is a placeholder: replace the whole thing,
  brackets included, with your real value. Example: `ssh <user>@<server-ip>`
  becomes `ssh maria@10.0.0.25`.
- `sudo` runs a command as administrator. The first time it asks for **your**
  password; nothing appears on screen while you type it — that is normal.
- If a command prints an error you don't understand, **stop**, copy the full
  error text, and ask. Do not improvise fixes on a production server.

**Editing a file on the server (you will need this in §6)**

We use `nano`, a simple text editor that runs in the terminal:

```sh
nano .env            # opens the file
```

- Move with the arrow keys and type normally.
- **Save:** press `Ctrl+O`, then `Enter`.
- **Quit:** press `Ctrl+X`.
- Pasting: right-click in most terminals, or `Ctrl+Shift+V`.

**Never do these on the production server**

- `docker compose down -v` — the `-v` **deletes** the n8n login, the Outlook
  connection and the document store.
- Delete or overwrite the `data/` folder — it holds every conversation, the
  review queue and your knowledge-base edits.
- Share or commit the `.env` file — it contains secrets.

---

## 0. Collect these before you start

Ask for anything you don't have **before** starting — you will be blocked
halfway otherwise.

| You need | From whom / where | Used in |
|---|---|---|
| SSH access to the server: its **IP address**, a **username**, and your password or key | whoever provides the server | everywhere |
| `sudo` (administrator) rights on it | same | §1–§4 |
| The server has an **NVIDIA GPU with ≥ 24 GB memory** (16 GB works with a smaller model) | same | §2, §4 |
| ~60 GB free disk space | same | §4 |
| The server can reach the internet | same | downloads, Microsoft Graph |
| Access to the Git repository (GitHub account with read access), **or** the project folder handed to you | project owner | §5 |
| A Microsoft 365 **admin**, or someone who can create an "App registration" and click "Grant admin consent" | your Microsoft 365 / IT admin | §7, §10a |
| The **mailbox** this will watch: its email address and a way to sign in to it | IT / mailbox owner | §9a, §10a |
| A password manager to store the secrets you create | — | throughout |

Time: about 2–3 hours for a first-timer, most of it waiting for downloads and
the Azure screens.

**Connect to the server** from your own computer (Windows: open *PowerShell*;
Mac/Linux: open *Terminal*):

```sh
ssh <user>@<server-ip>
```

The first time it asks *"Are you sure you want to continue connecting?"* —
type `yes`. Every command in §1–§6 is typed **in this SSH session**.

✅ **Check:** your prompt now shows the server's name, e.g. `maria@academy-srv:~$`.

---

## 1. Base system packages

```sh
sudo apt update && sudo apt -y upgrade
sudo apt -y install curl git jq nano build-essential ca-certificates gnupg
```

If the upgrade mentions a new kernel or asks to restart:

```sh
sudo reboot
```

This disconnects you. Wait one minute, then `ssh <user>@<server-ip>` again.

✅ **Check:** `git --version` prints a version number.

---

## 2. NVIDIA driver (lets the server use the GPU)

```sh
sudo apt -y install ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot
```

Reconnect with `ssh` after a minute, then:

```sh
nvidia-smi
```

✅ **Check:** a table showing the GPU name, its memory (e.g. `24576MiB`), a
driver version and a CUDA version.

If it says `command not found` or `couldn't communicate with the NVIDIA
driver`: the server's *Secure Boot* is probably blocking the driver. This
needs someone with access to the server's BIOS/console — ask the server
provider to disable Secure Boot or enrol the driver key, then re-run
`nvidia-smi`.

> **No GPU at all?** Skip §2 and remove `--gpus all` from §4b. The model then
> runs on the CPU — it works, but each email takes 30–90 seconds.

---

## 3. Docker (runs every part of the system in containers)

**Install Docker:**

```sh
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
```

The second line lets you use Docker without `sudo`, but only after you log
out and back in:

```sh
exit
```

then `ssh <user>@<server-ip>` again.

✅ **Check:** `docker compose version` prints `Docker Compose version v2.…`
(no `sudo` needed). If it says *permission denied*, you did not log out and
back in.

**Let containers use the GPU** (skip if you have no GPU):

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

```sh
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

✅ **Check:** the same GPU table as in §2, this time printed from inside a
container.

---

## 4. The chat model (the AI that classifies and writes replies)

One instruction model runs on the GPU and answers on port 8080. The
classifier needs three things from it — the checks in §4c confirm all three:

- **token probabilities ("logprobs")** — used to measure how sure the model
  is. Without them every email silently goes to a person.
- **JSON output on request.**
- **no hidden "thinking" text** in its answers.

We use **Qwen2.5-32B-Instruct**, which meets all three out of the box.

### 4a. Download the model (~20 GB — this takes a while)

```sh
sudo mkdir -p /opt/models && sudo chown "$USER" /opt/models
cd /opt/models
curl -L -o qwen2.5-32b-instruct-q4_k_m.gguf \
  "https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q4_K_M.gguf?download=true"
```

A progress bar runs until the download is done.

**GPU with only 16 GB?** Download the smaller model instead:
`bartowski/Qwen2.5-14B-Instruct-GGUF` → file `Qwen2.5-14B-Instruct-Q4_K_M.gguf`
(~9 GB), and everywhere below use `qwen2.5-14b` instead of `qwen2.5-32b`.

✅ **Check:** `ls -lh /opt/models` shows the `.gguf` file at roughly 19–20 GB.

### 4b. Start the model server

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

What the important options mean (don't change them unless told to):

- `--alias qwen2.5-32b` — the model's name. You will type this again in §6
  and §10.
- `--host 0.0.0.0` — lets the other containers reach it. **Port 8080 is then
  open on the server**: the server must be behind a firewall and not directly
  on the public internet. If unsure, ask whoever provides the server.
- `-c 16384` — how much text the model can read at once. Keep it ≥ 8192.
- `-ngl 999` — put the whole model on the GPU. If the log in the next step
  says *out of memory*, run `docker rm -f llama` and repeat §4b with
  `-ngl 40`.

Watch it load (1–3 minutes):

```sh
docker logs -f llama
```

Wait for a line containing `server is listening on http://0.0.0.0:8080`, then
press `Ctrl+C` (this only stops *watching* the log; the server keeps running).

### 4c. Verify the model — do not skip

**Check 1 — answers and probabilities:**

```sh
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-32b","messages":[{"role":"user","content":"Reply with the word ok"}],
       "logprobs":true,"top_logprobs":3,"max_tokens":10}' | jq '.choices[0]'
```

✅ Pass: `"content": "ok"` (no `<think>` text), **and** a `"logprobs"` section
containing a list of entries. If `logprobs` is `null`, stop and ask — the
classifier cannot work without it.

**Check 2 — JSON output:**

```sh
curl -s http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-32b","messages":[{"role":"user","content":"Give me a JSON object with one key ok set to true"}],
       "response_format":{"type":"json_schema","json_schema":{"name":"t","schema":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}}},
       "max_tokens":50}' | jq -r '.choices[0].message.content'
```

✅ Pass: prints `{"ok": true}` (spacing may differ).

> **Using a Qwen3 model instead?** Add `--reasoning-budget 0` to the §4b
> command (turns off "thinking") and re-run both checks.
>
> **Using Ollama instead of llama.cpp?** Needs Ollama ≥ 0.12 for logprobs. If
> Check 1 shows `logprobs: null`, set `EC_ALLOW_UNCALIBRATED=1` in `.env`
> (§6) — README §5a explains the trade-off.

---

## 5. Get the project onto the server

```sh
cd ~
git clone https://github.com/breachlabz/RAG-Academy_Auto-Reply_n8n_Workflow.git email-classifier
cd email-classifier
```

If Git asks for a username and password: the password must be a GitHub
**personal access token**, not your GitHub password (GitHub → Settings →
Developer settings → Personal access tokens). If you were handed the folder
instead, copy it to `~/email-classifier` and `cd` into it.

✅ **Check:**

```sh
ls
```

shows (among others) `docker-compose.yml`, `.env.example` is present
(`ls -a` shows hidden files), and `ls data/docs` lists the training `.docx`
files.

From here on, **every command runs inside `~/email-classifier`**. If you
reconnect later, first run `cd ~/email-classifier`.

> You do **not** need `docker-compose.override.yml.example` on a normal
> server — ignore it. It is only for a machine where the chat model lives in
> another Docker stack's private network.

---

## 6. Configure and start the stack

### 6a. Create the settings file

```sh
cp .env.example .env
nano .env
```

Change exactly these three lines (the file explains each one):

```ini
LLM_URL=http://host.docker.internal:8080/v1
LLM_KEY=
CHAT_MODEL=qwen2.5-32b
```

- `LLM_URL` — where the model from §4 answers, as seen from a container. If
  the model runs on a **different** machine, use
  `http://<that-machine-ip>:8080/v1` instead.
- `LLM_KEY` — leave empty for llama.cpp.
- `CHAT_MODEL` — must be **exactly** the `--alias` from §4b.

Leave everything else as it is. In particular, **do not** add
`REVIEW_DRY_RUN=1` — that is a testing switch that makes the Send button
pretend to send.

Save (`Ctrl+O`, `Enter`) and quit (`Ctrl+X`).

✅ **Check:** `grep -E '^(LLM_URL|CHAT_MODEL)=' .env` prints your two values.

### 6b. Build and start everything

```sh
docker compose up -d --build
```

The first run downloads and builds several parts — expect 5–15 minutes. It
ends with lines like `Container email-classifier-api  Started`.

The embedder downloads its own model (~2 GB) on first start. Watch it:

```sh
docker compose logs -f embedder
```

Wait for `Ready` or `Starting HTTP server`, then `Ctrl+C`.

✅ **Check:**

```sh
docker compose ps
```

Four rows — `email-classifier-api`, `email-classifier-chroma`,
`email-classifier-embedder`, `n8n` — all with status `Up` (the API also shows
`(healthy)` after about 30 seconds). Then:

```sh
curl -s http://127.0.0.1:8100/health ; echo
```

prints `{"ok":true}`.

### 6c. Load the training documents

```sh
docker compose exec classifier python -m rag ingest --reset
```

✅ **Check:** prints `NNN chunks -> collection 'docs'` (a number around 50–60)
followed by the list of documents.

This also fills the **knowledge table** — the editable copy of the documents
you will see in the review page's **Knowledge** tab (§12).

Prove the whole chain works (this asks the model a real question):

```sh
docker compose exec classifier python -m rag ask "what are the training levels?"
```

✅ **Check:** a short answer about the EVH/ACP levels, followed by source
headings. If instead you get *"we don't have relevant information"* or an
error, see §11's table — nearly always `LLM_URL` is wrong or the model from §4
isn't running (`docker ps` should list `llama`).

**Check the classifier:**

```sh
curl -s -X POST http://127.0.0.1:8100/classify -H 'Content-Type: application/json' \
  -d '{"text":"What does EVH Level 3 module 2.4 cover?"}' ; echo
curl -s -X POST http://127.0.0.1:8100/classify -H 'Content-Type: application/json' \
  -d '{"text":"I was charged twice for my course, please refund me."}' ; echo
```

✅ **Check:** the first prints `"type":"academic","route":"rag"`, the second
`"type":"non_academic","route":"human"`.

---

## 7. Register the inbox-reading app in Microsoft Entra (Azure)

This lets n8n read the mailbox. At the end you will have a **Client ID** and a
**Client secret** — store both in your password manager as you go.

You need a Microsoft 365 account that can create app registrations. Step 7
needs an **admin** to click one button; if that isn't you, do steps 1–6 and
send the admin the app's name.

1. Open <https://portal.azure.com> → search for **Microsoft Entra ID** → left
   menu **App registrations** → **New registration**.
2. **Name**: `academy-email-autoreply`. **Supported account types**: *Accounts
   in this organizational directory only*.
3. **Redirect URI**: choose platform **Web**, and enter **exactly**:
   ```
   http://localhost:5678/rest/oauth2-credential/callback
   ```
4. Click **Register**.
5. On the **Overview** page, copy the **Application (client) ID** → this is the
   **Client ID**.
6. Left menu **Certificates & secrets** → **Client secrets** → **New client
   secret** → expiry 12–24 months → **Add**. Copy the **Value** column
   **immediately** — it is hidden forever once you leave the page. This is the
   **Client secret**. (Put a reminder in your calendar for its expiry date —
   the pipeline stops reading mail when it expires.)
7. Left menu **API permissions** → **Add a permission** → **Microsoft Graph**
   → **Delegated permissions** → tick `Mail.Read` and `offline_access` → **Add
   permissions**. Then **Grant admin consent for <your organisation>** → Yes.

✅ **Check:** both permissions show a green tick under *Status*.

This app can only **read** mail. Sending uses a second, separate app (§10a).

---

## 8. Open the web pages from your computer (SSH tunnel)

For safety, n8n (port 5678) and the review page (port 8100) only listen **on
the server itself** — they are not reachable from the network. You reach them
through an "SSH tunnel": your computer forwards those ports over your SSH
connection.

On **your own computer**, open a **new** terminal window (keep it open while
you use the pages):

```sh
ssh -L 5678:127.0.0.1:5678 -L 8100:127.0.0.1:8100 <user>@<server-ip>
```

Now in your browser:

- n8n: **<http://localhost:5678>** — must be exactly `localhost`, not
  `127.0.0.1` (Microsoft only accepts `localhost` for the sign-in in §9a).
- Review page: **<http://localhost:8100/review>**

First time in n8n it asks you to create an **owner account** — it exists only
on this server. Use any email and a strong password, and store it in the
password manager.

✅ **Check:** both pages load. The review page says *"Nothing waiting for
review."*

Every time you want to use either page later, open this tunnel first.

---

## 9. Create the four n8n credentials

In n8n: left sidebar **Credentials** (or **Overview → Credentials**) → **Add
credential** (or **Create → Credential**). Create these four.

### 9a. Microsoft Outlook OAuth2 API
- Search for and pick **Microsoft Outlook OAuth2 API**.
- **Client ID** / **Client Secret**: from §7.
- Check the **OAuth Redirect URL** shown on the form is character-for-character
  the one you entered in §7 step 3.
- Click **Connect my account** → sign in **as the mailbox** → accept.

✅ **Check:** the credential shows **Account connected**.

### 9b. Chat model — type **OpenAI API**
- Name it `Chat model (llama.cpp)`.
- **Base URL**: `http://host.docker.internal:8080/v1` (same address as
  `LLM_URL` in §6a).
- **API Key**: type `x` (llama.cpp ignores it, but the field can't be empty).

### 9c. Embedder — type **OpenAI API** (a second one — easy to miss)
- Name it `Embedder (bge-m3)`.
- **Base URL**: `http://embedder:80/v1` — this is **not** the chat model.
- **API Key**: `x`.

### 9d. Chroma — type **Chroma API** (self-hosted)
- Name it `Chroma (academy_docs)`.
- **Base URL**: `http://email-classifier-chroma:8000`
- No authentication.

This is the same document store §6c loaded, so n8n and the review page's
Knowledge tab always see the same content.

---

## 10. Import and connect the workflow

1. n8n → **Workflows** → **⋯** menu (top right) → **Import from File** →
   choose `n8n/academy-agent-workflow.json`. (You need the file on your
   computer: download it from the repository on GitHub, or copy it off the
   server with `scp <user>@<server-ip>:email-classifier/n8n/academy-agent-workflow.json .`
   run on your computer.)
2. Nodes with a missing credential show a red warning. Open each and pick the
   credential:

   | Node | Credential |
   |---|---|
   | **New Outlook email** (the trigger) | Outlook OAuth2 (9a) |
   | **Local Model** | Chat model (9b) |
   | **Embeddings bge-m3** | **Embedder (9c)** — not the chat model |
   | **academy_docs** | Chroma (9d) |

3. Open **Local Model** → set the model to `qwen2.5-32b` (your `--alias`).
4. Open **academy_docs** → the collection must be **`docs`**.
5. Open **New Outlook email** → folder **Inbox**, poll **every minute**.
6. Click **Save**.

Do **not** turn the workflow on (Active) yet — that is §11.

> Re-importing this file later **removes** the credential links and switches
> the workflow off. Redo steps 2–6 after any re-import.

Optional: import `n8n/email-classifier-form-workflow.json` the same way. It's
a manual test form and needs no credentials.

✅ **Check:** no red warnings remain on any node.

---

## 10a. Set up sending (review page → Microsoft Graph)

The **Send** button needs its own Microsoft app, because it has to work with
nobody signed in. Until this is done, the review page still works for
reading and editing, and Send shows a clear *"not configured"* message instead
of sending.

1. <https://portal.azure.com> → **Microsoft Entra ID** → **App registrations**
   → **New registration** → name `academy-email-send` → **Register**. (A
   *new* app — do not reuse the one from §7.)
2. **API permissions** → **Add a permission** → **Microsoft Graph** →
   **Application permissions** (not *Delegated*) → tick `Mail.Send` → **Add**
   → **Grant admin consent**.
3. **Certificates & secrets** → new client secret → copy the **Value** now.
4. **Overview** → copy **Application (client) ID** and **Directory (tenant)
   ID**.
5. On the server:
   ```sh
   cd ~/email-classifier
   nano .env
   ```
   Add these four lines at the end (no spaces around `=`):
   ```ini
   GRAPH_TENANT_ID=<directory (tenant) id>
   GRAPH_CLIENT_ID=<application (client) id>
   GRAPH_CLIENT_SECRET=<secret value>
   GRAPH_MAILBOX=<the mailbox's email address>
   ```
   Save and quit.
6. Apply the change:
   ```sh
   docker compose up -d
   ```

✅ **Check:** `docker compose ps` shows the API `Up (healthy)` again after
~30 seconds.

> `Mail.Send` as an application permission can send as **any** mailbox in the
> tenant. Ask your Microsoft 365 admin whether to restrict this app to the one
> mailbox (Exchange Online supports this for app registrations).

---

## 11. End-to-end test

**Dry run inside n8n (no real email):** open the workflow → click **Test
workflow** / **Execute workflow** on the *Test: run manually* trigger.

✅ **Check:** the run reaches **Finalize** (nodes turn green) with
`grounded: true`, and a new row appears on the review page.

**Live test:**

1. In n8n, switch the workflow to **Active** (toggle top right) → confirm.
2. From a **different** email account, send the mailbox:
   > **Subject:** Course question
   > **Body:** What are the training levels and who is Level 2 aimed at?
3. Within 1–2 minutes a row appears on **<http://localhost:8100/review>**
   (tunnel open) with a drafted reply built from the documents.
4. Read it, edit if you like, click **Send**.

✅ **Check:** the reply arrives in the sending account's inbox, with the
original message quoted underneath.

5. Send a second email: *"My invoice still shows unpaid, can you check?"*

✅ **Check:** **no** row appears on the review page. In n8n the run ends at
**Do nothing** with `reason: not routed to rag`. This is correct — billing
is non-academic and is always left for a person.

**If something doesn't match**, open the run in n8n (**Executions** in the
left menu), click the red or last node, and compare with this table:

| What you see | What to do |
|---|---|
| **Prepare** output has `error: expected 2 bools, got 0` | the model returned no probabilities — redo §4c; on Ollama set `EC_ALLOW_UNCALIBRATED=1` in `.env`, then `docker compose up -d` |
| **Prepare** output has `error: request failed …` | the API can't reach the model — check `LLM_URL` in `.env` (§6a) and that `docker ps` lists `llama` |
| empty or garbled `flags` | the model is "thinking" — use Qwen2.5, or add `--reasoning-budget 0` (§4) |
| an academic email ends at **Human queue** | the documents don't answer it — re-run §6c, and check the topic really is in `data/docs/` |
| the trigger never fires | the Outlook credential (9a) shows *not connected* — reconnect it; the workflow must be **Active** |
| **Send** says "not configured" | §10a not done, or `.env` has a typo — check the four `GRAPH_` lines, then `docker compose up -d` |
| **Send** returns a 502 error | nothing is lost, the row stays. The message names what Microsoft rejected: usually admin consent for `Mail.Send` is missing, or `GRAPH_MAILBOX` is not a real mailbox |
| review page won't load | the SSH tunnel from §8 isn't open |

Still stuck: `docker compose logs --tail 100 classifier` shows the API's
recent log — copy it into your question. README §9 has more symptoms.

---

## 12. Going live — and daily use

Once both test emails behave:

- Leave the workflow **Active**.
- Close the SSH tunnel. The pipeline keeps running on its own on the server.
- A person checks **/review** regularly (tunnel first): read each draft, edit
  if needed, **Send**. Nothing leaves without that click.

### The Knowledge tab (tuning replies)

The review page has a second tab, **Knowledge**. It lists every piece
("chunk") of the training documents the replies are written from — its text
and its details (*metadata*, shown as JSON).

- **Edit** a chunk and **Save**: replies use the new text from the next email
  on. Both the review page and n8n see it immediately.
- **Add chunk**: add a fact that isn't in the documents yet. Write the topic
  into the text itself ("EVH Level 2 kit calibration: …"), not only into the
  heading — the text is what gets searched.
- **Export JSON**: a copy of everything, for backup or review.
- ⚠ **Re-loading the documents (§6c command) overwrites edits** to chunks that
  came from a document. Chunks you *added* are kept. A permanent change
  belongs in the Word document itself.

The same thing from the command line: README §4a.

### Routine tasks

Run these from `~/email-classifier` on the server.

| Task | Command |
|---|---|
| Updated Word documents | replace the files in `data/docs/` (e.g. with `scp`), then `docker compose exec classifier python -m rag ingest --reset` |
| Install a new version of the app | see §13 |
| Change the greeting / sign-off | set `RAG_EMAIL_GREETING` / `RAG_EMAIL_SIGNOFF` in `.env`, then `docker compose up -d` |
| Make the classifier stricter / looser | `EC_THRESHOLD` in `.env` (default `0.9`; higher → more mail to people), then `docker compose up -d` |
| Is everything running? | `docker compose ps` and `docker ps` (the model is `llama`) |
| Restart the model | `docker restart llama` |
| Restart the app | `docker compose restart classifier` |
| Stop everything (data kept) | `docker compose stop`; start again with `docker compose start` |
| After a server reboot | nothing — everything restarts on its own (check with `docker compose ps`) |

### Backups — set these up on day one

```sh
cd ~/email-classifier
# 1. conversations, review queue, knowledge-base edits and added chunks
docker compose exec classifier python -c "import sqlite3; s=sqlite3.connect('/app/data/threads.db'); d=sqlite3.connect('/app/data/threads.db.bak'); s.backup(d)"
cp data/threads.db.bak ~/threads-$(date +%F).db
# 2. n8n: workflows, credentials, run history
docker run --rm -v email-classifier_n8n_data:/v -v "$HOME":/out alpine \
  tar czf /out/n8n-backup-$(date +%F).tgz -C /v .
# 3. settings and secrets
cp .env ~/env-backup-$(date +%F)
```

Copy those three files off the server to wherever your organisation keeps
backups (they contain secrets — treat them like passwords). The document
store needs no backup: `ingest --reset` rebuilds it from `data/docs/`, and
added chunks come back from `threads.db`.

Secrets expire: both Azure client secrets (§7, §10a) stop working on the
date you chose. Renew them in the Azure portal before then, and update the
Outlook credential in n8n and `GRAPH_CLIENT_SECRET` in `.env`.

---

## 13. Installing a newer version later

```sh
cd ~/email-classifier
# back up first — §12 "Backups"
git pull
docker compose up -d --build classifier
docker compose ps          # API "Up (healthy)" after ~30 s
```

Then read the release notes / commit message for extra steps. For the
version that introduced the academic / non-academic classifier and the
Knowledge tab, on a server that was **already running** an older version:

1. Fill the knowledge table once from the existing document store:
   ```sh
   docker compose exec classifier python -m rag knowledge pull
   ```
   ✅ prints `NN chunks copied from collection 'docs' into knowledge_chunks`.
2. If you use the optional test form, re-import
   `n8n/email-classifier-form-workflow.json` (it shows the new scores).
3. Re-run the two classifier checks at the end of §6c.

---

## 14. Final checklist

- [ ] `nvidia-smi` works on the server and inside a container (§2, §3)
- [ ] `docker compose version` is v2.x, works without `sudo`
- [ ] model server `llama` running; both §4c checks pass
- [ ] `.env` has `LLM_URL`, `CHAT_MODEL` (= the `--alias`); **no** `REVIEW_DRY_RUN`
- [ ] `docker compose ps` — 4 containers Up; `/health` returns `{"ok":true}`
- [ ] `ingest --reset` done; `rag ask` gives a real answer; `/classify` checks pass
- [ ] Entra app 1: `Mail.Read` + `offline_access`, admin consent ✔
- [ ] n8n reached via SSH tunnel at `http://localhost:5678`; owner account saved
- [ ] 4 n8n credentials; Outlook shows **Account connected**
- [ ] workflow imported, 4 credentials attached, Local Model = your alias, saved
- [ ] Entra app 2: `Mail.Send` (application), admin consent ✔; 4 `GRAPH_` lines in `.env`
- [ ] test: academic email → row at `/review` → Send delivers it
- [ ] test: invoice email → nothing queued
- [ ] workflow **Active**; Knowledge tab shows the chunks
- [ ] backups taken and copied off the server; secret expiry dates in the calendar

---

*How it works inside — the classifier gate, the grounding checks, threading,
the knowledge table, the evaluation scripts — is in `README.md` §4a and
§10–§12.*
