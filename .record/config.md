# Edge Proxy — GPT-5.5 / Claude / Gemini OpenAI-Compatible Gateway

A single HTTPS endpoint that exposes **GPT-5.x (incl. 5.5)**, **Claude (incl.
Opus 4.8)**, **Gemini 3.x**, and a handful of legacy Azure OpenAI models behind
one OpenAI-compatible API. Works as drop-in base URL for any client that speaks
OpenAI / Anthropic / Codex.

- **Base endpoint:** `https://edge-fsd.japaneast.cloudapp.azure.com`
- **Region:** Japan East
- **Auth:** Bearer token (one shared key, request a personal one if you need
  isolated quota / revocation)
- **Owner:** taoli1ms

---

## Quickstart — verify from your laptop

```bash
curl https://edge-fsd.japaneast.cloudapp.azure.com/v1/models \
  -H "Authorization: Bearer <YOUR_TOKEN>"
```

Expected: `200 OK` with a JSON list of available models.

---

## Client setup

The proxy speaks three API dialects on the same base URL:

| SDK / client                      | Base URL to configure                                          | Notes                          |
| --------------------------------- | -------------------------------------------------------------- | ------------------------------ |
| **OpenAI SDK / Codex CLI**        | `https://edge-fsd.japaneast.cloudapp.azure.com/v1`             | Trailing `/v1` is required     |
| **Anthropic SDK / Claude Code**   | `https://edge-fsd.japaneast.cloudapp.azure.com`                | SDK adds `/v1` itself          |
| **OpenAI Responses API (Codex)**  | `https://edge-fsd.japaneast.cloudapp.azure.com/v1`             | See gpt-5.5 caveat below       |

Every request must include:

```
Authorization: Bearer <YOUR_TOKEN>
```

### Environment variable conventions

```bash
# OpenAI-compatible clients (most common)
export OPENAI_API_KEY="<YOUR_TOKEN>"
export OPENAI_BASE_URL="https://edge-fsd.japaneast.cloudapp.azure.com/v1"

# Anthropic SDK / Claude Code
export ANTHROPIC_API_KEY="<YOUR_TOKEN>"
export ANTHROPIC_BASE_URL="https://edge-fsd.japaneast.cloudapp.azure.com"
```

---

## Full endpoint reference

| Purpose                | Full URL                                                                        |
| ---------------------- | ------------------------------------------------------------------------------- |
| Model list             | `https://edge-fsd.japaneast.cloudapp.azure.com/v1/models`                       |
| Chat completion        | `https://edge-fsd.japaneast.cloudapp.azure.com/v1/chat/completions`             |
| Responses API (Codex)  | `https://edge-fsd.japaneast.cloudapp.azure.com/v1/responses`                    |
| Messages API (Claude)  | `https://edge-fsd.japaneast.cloudapp.azure.com/v1/messages`                     |
| Embeddings             | `https://edge-fsd.japaneast.cloudapp.azure.com/v1/embeddings`                   |
| Usage / quota          | `https://edge-fsd.japaneast.cloudapp.azure.com/usage`                           |

---

## Available models

Pulled live from `/v1/models`. Pick by `id`.

**OpenAI (latest)**
- `gpt-5.5` — flagship, **Responses API only** (see caveat)
- `gpt-5.4`, `gpt-5.4-mini`
- `gpt-5.3-codex`
- `gpt-5-mini`

**Anthropic**
- `claude-opus-4.8`, `claude-opus-4.7`, `claude-opus-4.6`, `claude-opus-4.5`
- `claude-sonnet-4.6`, `claude-sonnet-4.5`
- `claude-haiku-4.5`

**Google**
- `gemini-3.1-pro-preview`
- `gemini-3.5-flash`, `gemini-3-flash-preview`
- `gemini-2.5-pro`

**Microsoft / experimental**
- `mai-code-1-flash-internal`
- `trajectory-compaction`

**Azure OpenAI legacy** — `gpt-4.1`, `gpt-4o`, `gpt-4o-mini`, `gpt-4`,
`gpt-3.5-turbo`, `text-embedding-3-small`, `text-embedding-ada-002`, …

---

## Caveat — gpt-5.5 routing

`gpt-5.5` (and other reasoning-first variants) are **served only via the
Responses API**. Calling `/v1/chat/completions` with `model: "gpt-5.5"` returns:

```json
{"error":{"message":"model \"gpt-5.5\" is not accessible via the /chat/completions endpoint","code":"unsupported_api_for_model"}}
```

**Fix:** point your client at `/v1/responses` instead, or configure your wrapper
to route gpt-5.x models through the Responses API. (For projects using the
OpenAI Python SDK: use `client.responses.create(...)` instead of
`client.chat.completions.create(...)`.)

The other models (gpt-4o/4.1, Claude, Gemini) work fine on `/v1/chat/completions`.

---

## Examples

### `curl` — Chat (gpt-4o)

```bash
curl https://edge-fsd.japaneast.cloudapp.azure.com/v1/chat/completions \
  -H "Authorization: Bearer <YOUR_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### `curl` — Responses (gpt-5.5)

```bash
curl https://edge-fsd.japaneast.cloudapp.azure.com/v1/responses \
  -H "Authorization: Bearer <YOUR_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.5",
    "input": "Say hi"
  }'
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://edge-fsd.japaneast.cloudapp.azure.com/v1",
    api_key="<YOUR_TOKEN>",
)

# Chat (works for gpt-4o / 4.1 / Claude / Gemini)
r = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)

# Responses (required for gpt-5.5)
r = client.responses.create(model="gpt-5.5", input="Say hi")
print(r.output_text)
```

### Python (Anthropic SDK)

```python
import anthropic

client = anthropic.Anthropic(
    base_url="https://edge-fsd.japaneast.cloudapp.azure.com",
    api_key="<YOUR_TOKEN>",
)
r = client.messages.create(
    model="claude-opus-4.8",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello"}],
)
```

### Claude Code CLI

```bash
ANTHROPIC_BASE_URL=https://edge-fsd.japaneast.cloudapp.azure.com \
ANTHROPIC_API_KEY=<YOUR_TOKEN> \
ANTHROPIC_MODEL=claude-opus-4.8 \
claude
```

### Codex CLI

```bash
OPENAI_BASE_URL=https://edge-fsd.japaneast.cloudapp.azure.com/v1 \
OPENAI_API_KEY=<YOUR_TOKEN> \
codex --model gpt-5.5
```

---

## Operations

### Check your usage / quota

```bash
curl https://edge-fsd.japaneast.cloudapp.azure.com/usage \
  -H "Authorization: Bearer <YOUR_TOKEN>"
```

Returns the backing GitHub Copilot seat's plan, organization list, and quota
snapshots (chat, completions, premium_interactions). For the enterprise tier
these are typically `unlimited: true`; the reset date is shown as
`quota_reset_date`.

---

## Troubleshooting

| Symptom                                                                                     | Cause                                                       | Fix                                                                                |
| ------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `400 unsupported_api_for_model` on gpt-5.5                                                  | Wrong endpoint                                              | Use `/v1/responses` instead of `/v1/chat/completions`                              |
| `401 Unauthorized`                                                                          | Missing or wrong token                                      | Confirm `Authorization: Bearer <token>` header is present and not URL-encoded      |
| Tool-call response crashes client with `'dict' object has no attribute 'model_dump'`         | OpenAI SDK strict mode + Responses API tool calls           | Wrap `tool_calls` as `ChatCompletionMessageToolCall` Pydantic objects in the SDK adapter |
| Slow first response                                                                         | Cold container in Japan East                                | Subsequent requests are ~1–2 s                                                     |

---

## Notes

- **Token is sensitive.** Treat it like a credential. Request a personal key if
  you need revocation / per-user quota isolation.
- The proxy is single-region (Japan East). Expect 80–120 ms RTT from East Asia,
  200–250 ms from North America / Europe.
- Streaming (`stream: true`) is supported on `/v1/chat/completions` and
  `/v1/responses`.
- For bug reports or feature requests, ping the proxy owner (taoli1ms).
