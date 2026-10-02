# Fieldnote

Fieldnote explains unfamiliar paper forms from a camera scan or pasted text. It describes visible fields in the reader’s chosen language; it does not fill or submit forms.

## Public demo architecture

- Render serves this page and the small FastAPI backend from one web service.
- The backend sends the selected image or pasted excerpt to Tinker’s hosted vision model (`thinkingmachines/Inkling-Small`) through Tinker’s Anthropic-compatible API. Visitors do not install or download a model.
- `TINKER_API_KEY` stays in the server environment. It is never included in frontend JavaScript.
- The page asks for consent before each inference request. Fieldnote does not write submitted form contents to disk or log them. Tinker receives the content to run inference; read Tinker’s current data-handling terms before using real personal documents.
- In-memory per-IP and global daily request limits provide basic demo protection. They reset when the service restarts and are not a substitute for provider-side spending limits or stronger abuse protection.

Form images and text may contain personal data. The consent notice tells visitors where their content goes. Avoid using highly sensitive documents, and verify translations and form requirements with the issuing office. AI may misread low-quality or handwritten forms. Fieldnote does not provide legal, medical, or financial advice.

## Deploy on Render

The repository-level [`render.yaml`](../../render.yaml) describes one Python web service. In Render, create a Blueprint from the Git repository containing this project. The Blueprint installs `requirements.txt` and starts FastAPI on Render’s assigned port.

Before the first useful request, set `TINKER_API_KEY` in the Render service’s Environment settings. The model name defaults to `thinkingmachines/Inkling-Small`; `TINKER_MODEL_NAME` can override it with another Tinker vision model available to the account. Keep keys in Render environment variables, never in the repository or browser.

Tinker currently describes its Anthropic-compatible inference endpoint as a low-traffic beta for testing and internal workflows. This is a small public demo with request caps, not a high-traffic service. Expect model availability and latency to vary, and set a spending limit with the provider before sharing it broadly. Tinker may need to enable beta API access for your account.

## Run locally

1. Use Python 3.10 or newer.
2. Install dependencies with `pip install -r requirements.txt`.
3. Set `TINKER_API_KEY` in your local shell and run:

   ```powershell
   uvicorn server:app --host 127.0.0.1 --port 8000
   ```

4. Open <http://localhost:8000>.

The camera works on localhost or HTTPS. The hosted AI requires a Tinker API key and may incur provider usage charges.

## Challenge fit

Fieldnote uses an open-weight vision model through hosted inference so the visitor can start immediately without downloading gigabytes. The current Hacktoberfest Weekend Challenge prompt asks for an open-source AI project built for a real person; its deadline is October 5, 2026 at 6:59 AM UTC. For a Tinker category entry, describe Tinker’s meaningful role and be transparent that model inference is hosted.
