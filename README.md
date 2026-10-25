# SkulTok

A Flask-based bilingual tutor for Papua New Guinea, served with a single-origin HTML frontend and Gemini API backend.

## Run locally on Windows

Use Python 3.13 or another supported Python version. From this directory:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Set provider keys in a local `.env` file for development: `GROQ_API_KEY` and/or `OPENROUTER_API_KEY` (also accepts `OPENROUTER_KEY`), with `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) as the final fallback. The app tries configured providers in Groq, OpenRouter, Gemini order using the `requests` HTTP client; the Groq and OpenAI SDK packages are not required. Groq and OpenRouter models can be changed using `GROQ_MODEL` and `OPENROUTER_MODEL`; the defaults are `openai/gpt-oss-20b` and `google/gemma-4-31b-it:free`. These replace retired/unavailable model IDs. Local greetings and basic arithmetic are answered without an AI provider. Provider availability, model names, and free quotas may change; check each provider's current terms. The `.env` file is ignored by Docker and gcloud deployment contexts; do not copy it into the image or commit it. The app listens on `http://localhost:8080` by default and honors `PORT` when set.

## Deploy to Cloud Run

1. Enable the Cloud Run, Cloud Build, and Secret Manager APIs in your Google Cloud project.
2. Create a Secret Manager secret named `gemini-api-key` containing your Gemini API key. Do not put the key in a deploy command, Dockerfile, or source file.
3. Ensure the Cloud Run runtime service account has the **Secret Manager Secret Accessor** role on that secret.
4. From this directory, deploy the source (Cloud Build uses the included Dockerfile):

```powershell
gcloud run deploy skultok `
  --source . `
  --region us-central1 `
  --allow-unauthenticated `
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest
```

Choose the region appropriate for your users and data requirements. Remove `--allow-unauthenticated` if the service should require authentication. The app defaults to the stable Gemini model `gemini-3.8-flash`; override it with the `GEMINI_MODEL` environment variable if needed for your account or model availability.

For the Groq/OpenRouter fallback chain, configure `GROQ_API_KEY` and/or `OPENROUTER_API_KEY` (or `OPENROUTER_KEY`) as secrets/environment variables in the deployment platform. Keep `GEMINI_API_KEY` configured if Gemini should remain the final fallback. Confirm the exact variable names in the deployment platform's environment settings; local `.env` settings are not automatically deployed.

After deployment, test the returned service URL:

```powershell
Invoke-RestMethod "$SERVICE_URL/health"
Invoke-RestMethod -Method Post "$SERVICE_URL/api/chat" `
  -ContentType 'application/json' `
  -Body (@{message='Explain fractions using kaukau'; grade='Grade 8'; lang='English + Tok Pisin'} | ConvertTo-Json)
```

The frontend is served by Flask at `/`, and calls `/api/chat` on the same origin. `/health` is a lightweight liveness check. Chat requests accept Grades 5-12 and the three language options shown in the UI. Combined-language answers use a natural Lae/Morobe Tok Pisin style first, followed by English in parentheses; English-origin action verbs used in the Tok Pisin portion follow the requested `-im` suffix rule. Answers are limited to 250 output tokens. The voice button reads the displayed text without rewriting words, preferring an Australian or New Zealand voice for Tok Pisin and an English voice for the parenthetical translation when the browser provides them. Pronunciation accuracy depends on the voices installed in the user's browser or operating system. The app retries transient connection failures and Gemini HTTP 429, 500, 502, 503, and 504 responses up to three attempts with backoff, using a 3-second connect and 12-second read timeout per attempt. Errors for temporary unavailability, timeouts, and oversized requests are returned as JSON. The Gemini API key must be supplied through `GEMINI_API_KEY` (or Secret Manager on Cloud Run), never hard-coded. The service uses Gunicorn and binds to Cloud Run's injected `PORT`.

All `/api/` errors, including unexpected server failures and unsupported routes or methods, return JSON with an appropriate HTTP status. Unexpected errors are logged server-side; internal exception details are not sent to clients.

Changing the language selector immediately localizes the page controls and guidance. The selected language also determines the language used for new tutor replies.
