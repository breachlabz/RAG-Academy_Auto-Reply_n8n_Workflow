# n8n integration

Two workflows, one API.

| File | n8n name | What it is |
|---|---|---|
| `outlook-academy-workflow.json` | Academy Auto-Reply (Outlook) | **the pipeline** — mailbox in, Outlook draft out |
| `email-classifier-workflow.json` | Email Classifier | a web form for trying the classifier + retrieval by hand, no mailbox |

First-run setup — the Entra app, the Outlook credential, ingesting the docs —
is in [`../HANDOFF.md`](../HANDOFF.md). This file is the reference for the two
workflows themselves: what each node does, how to re-import, and the gotchas.

---

## Academy Auto-Reply — the pipeline

```
New Outlook email ─▶ Classify + RAG reply ─▶ Grounded answer? ──true──▶ Create Outlook draft reply
 (polls Inbox/min)    POST /generate-reply        │
                                                  └──false──▶ No draft (human queue)
```

Five nodes, no JavaScript, one HTTP call carrying the work. **`/generate-reply`
does all of it:** pull the email out of the HTML body, cut quoted history, load
the conversation, classify the latest message against it, apply the gate,
retrieve, draft a grounded reply, and record what was drafted. The workflow's
only jobs are polling the mailbox, branching on the result, and turning a
grounded reply into a Graph draft.

**No label filter.** Every inbox message goes to `/generate-reply`; the gate
inside the API decides. A payment question, spam, or a low-confidence
classification comes back `answered: false` and lands at *No draft (human
queue)*. Nothing is auto-answered that the classifier did not clear.

**Branch on `answered`, not `route`.** `answered` is true only when the email
was routed to `rag` *and* the documents actually supported a grounded answer;
the response's `reason` names which of the two failed. The *Grounded answer?* IF
node tests exactly `{{ $json.answered }}`.

**`Create Outlook draft reply` calls Graph `createReply`** — it creates a
**draft** in the original conversation and **never sends**. Sending is a
separate Graph call this workflow does not contain. The reply goes in as the
`comment` field: plain text, so a bulleted answer shows as `- ` lines.

**Re-delivery is handled.** A trigger that re-fires on an already-processed
email comes back `duplicate: true` / `reason: already handled` and stops — no
second draft.

### Credentials

One: a **Microsoft Outlook OAuth2** credential (`microsoftOutlookOAuth2Api`,
scope `Mail.ReadWrite`), attached to **both** *New Outlook email* and *Create
Outlook draft reply*. Walkthrough in HANDOFF.md §7.

### Why the thread matters — follow-ups

A follow-up carries almost none of its own subject matter, and classified on
its own words it misroutes badly:

| Email | academic | spam | route |
|---|---|---|---|
| `and what does the second one cover?` — alone | 0.22 | **0.78** | human |
| the same email, with the thread as context | **0.99** | 0.001 | rag |

Without the thread, every conversation died at the gate on its *second* message
— and died as *spam*, so nothing downstream looked wrong. `/generate-reply`
loads the thread from `data/threads.db` and classifies the latest email
*against* it. The label still describes the latest email only, so an invoice
question inside an academic thread still comes back administrative and still
gets no draft. It also rewrites the follow-up into a standalone question before
embedding — *"and the second one?"* is clear to a reader and a bag of stopwords
to bge-m3; the rewrite shows up as `retrieval_query` in the response.

### Activating

1. Import `outlook-academy-workflow.json` (**Workflows → ⋯ → Import from File**).
2. Attach the Outlook credential to the two nodes above.
3. Optionally open *New Outlook email* to change the folder or poll interval
   (default: Inbox, every minute).
4. Toggle **Active**.

### Re-importing after editing the JSON

UI import is simplest. From the shell the CLI needs a top-level `id`, which the
checked-in file omits (it would collide on UI import), so inject one:

```sh
python3 -c "
import json; wf = json.load(open('n8n/outlook-academy-workflow.json'))
wf['id'] = 'academyautoreply01'
json.dump(wf, open('/tmp/wf.json','w'))"
docker compose cp /tmp/wf.json n8n:/tmp/wf.json
docker compose exec n8n n8n import:workflow --input=/tmp/wf.json
docker compose restart n8n
```

Import overwrites the credential bindings and the active flag — re-attach the
Outlook credential and re-activate afterwards.

---

## Email Classifier — the test form

A form (button + text box) that classifies one pasted email as **academic**,
**administrative**, or **spam**, and shows the answer it would draft.

```
n8n form ──POST /answer──▶ classifier API ──▶ your chat model     (classify + draft)
                                          └──▶ embedder + chromadb (retrieve)
```

No mailbox, no credentials. Use it to confirm classification and retrieval work
before wiring up Outlook, or to demo the gate. `/answer` runs the classifier
*and* retrieval together, so nothing can reach the RAG path around the gate.

Live at `http://localhost:5678/form/email-classifier-form` once imported and
active.

### Re-importing

```sh
python3 -c "
import json; wf = json.load(open('n8n/email-classifier-workflow.json'))
wf['id'], wf['active'] = 'emailclassifier01', True
json.dump(wf, open('/tmp/wf.json','w'))"
docker compose cp /tmp/wf.json n8n:/tmp/wf.json
docker compose exec n8n n8n import:workflow --input=/tmp/wf.json
docker compose exec n8n n8n publish:workflow --id=emailclassifier01
docker compose restart n8n
```

### Form gotchas

- **`curl`-test with `-F 'field-0=...'`**, not the field label — n8n indexes
  submitted fields by position. `-F 'email=...'` arrives as `null` and the API
  422s.
- Answering is slow enough that the form returns a `formWaitingUrl` instead of
  the result page; a `curl` test must follow that URL:
  ```sh
  curl -s -X POST http://localhost:5678/form/email-classifier-form -F 'field-0=what does level 2 cover?'
  # {"formWaitingUrl":"http://localhost:5678/form-waiting/NN?signature=..."}
  curl -s "<that url>"
  ```
- If you rebuild the form: the trigger needs an explicit `parameters.path`, and
  any field holding a `{{ }}` expression must start with `=` or n8n renders it
  literally.

---

## The API these workflows call

Both workflows are thin wrappers over the classifier API — `http://classifier:8100`
inside the compose network, `http://127.0.0.1:8100` from the host. Endpoints
worth hitting directly when debugging:

```sh
# classify only
curl -s -X POST http://127.0.0.1:8100/classify -H 'Content-Type: application/json' \
  -d '{"text":"when does level 2 start? also I havent paid"}'
# {"type":"administrative","route":"human","flags":{...},"probs":{...},"calibrated":true}

# classify + retrieve + answer  (what the form uses)
curl -s -X POST http://127.0.0.1:8100/answer -H 'Content-Type: application/json' \
  -d '{"text":"does level 2 include hands-on hardware work?"}'
# {..., "answered":true, "answer":"...", "sources":[...], "closest":0.43}

# the full mailbox call  (what the Outlook workflow uses), with thread memory
curl -s -X POST http://127.0.0.1:8100/generate-reply -H 'Content-Type: application/json' \
  -d '{"email_text":"and what does the second one cover?","conversation_id":"AAQk...","subject":"RE: Levels","message_id":"<abc@x>"}'
# {..., "answered":true, "reply":"Hello,\n\n...", "retrieval_query":"and what does ACP Level 2 cover?"}

curl -s http://127.0.0.1:8100/threads/AAQk...     # the whole conversation
```

- **`type`** collapses the three flags by priority **spam > administrative >
  academic**, so a mixed course-and-payment email is `administrative` and never
  auto-answered.
- **`reply`** is the grounded answer when there is one, the no-information line
  when the documents don't cover it, and empty for administrative/spam mail or
  on error.
- **`calibrated: false`** means the chat endpoint returned no logprobs and
  routing fell back to bare flags — see HANDOFF.md §6a (Ollama).
- **`format`** (optional, `/answer` and `/generate-reply`): `"auto"` (default)
  makes the reply a `- ` bulleted list when the enquirer wrote a list, asked
  "in points", or asked several questions, and prose otherwise; `"list"` /
  `"prose"` force it. `retrieval_query` is echoed back only when a follow-up
  was rewritten.
- **`sources`** / **`closest`** are diagnostic only. Measured over nineteen
  questions the cosine distances for answerable and unanswerable ones overlap,
  so `closest` cannot tell you whether an answer was possible and
  `RAG_MAX_DISTANCE` cannot be tuned to it. The refusal detection in `rag.core`
  does that job; see the main README.
