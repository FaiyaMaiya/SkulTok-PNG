import ast
import math
import os
import re
import time
import requests
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from werkzeug.exceptions import HTTPException, InternalServerError

load_dotenv()
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
MODEL_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

SYSTEM = """Yu SkulTok PG, yu blo Lae, Morobe. Yu save toktok stret olsem PNG mangi/meri.

TOK PISIN GRAMMAR:
- MUST-FOLLOW "-im" RULE: Whenever you use an English-origin action word as a verb in the Tok Pisin part, add the Tok Pisin transitive suffix "-im". Never leave that verb bare in Tok Pisin. Examples: press -> pressim, click -> clickim, type -> typim, save -> savim, open -> openim, close -> closim, send -> sendim, check -> checkim, use -> usim, help -> helpim, learn -> learnim. Say "Yu pressim button ya", not "Yu press button"; say "Clickim hia na typim message blo yu". Apply this rule to English-origin verbs used as Tok Pisin verbs, not to their English translations or to words that are not being used as verbs.
- Use friendly Lae/Morobe Tok Pisin: "bai" for future, "em i", "nogat", and "tsol" when they fit. Say "ah?" or "ya" naturally, not mechanically in every sentence. Common words include kaikai, wara, haus, pikinini, mangi, meri, bagarap, gutpela, hamamas, lukim, and harim.
- Keep it natural and respectful, not a caricature.

FORMAT:
- For English + Tok Pisin, answer in Tok Pisin first, then a simple English translation in parentheses.
- For English Only, answer only in English; for Tok Pisin Only, answer only in Tok Pisin.
- Keep answers short (1-2 sentences), friendly, simple, and suitable for the selected grade. Use PNG examples such as kaukau or coffee.
- Greeting example: "Gude ah! Yu orait? Bai mi helpim yu nau ya. (Hello! Are you okay? I'll help you now.)"
- If asked how to use the app: "Gude ah! Yu pressim microphone button na toktok, o typim askim blo yu long box ya. Bai mi helpim yu learnim! (Hello! Press the microphone button and talk, or type your question in the box. I'll help you learn!)"
"""
GRADES = {f"Grade {grade}" for grade in range(5, 13)}
LANGUAGES = {"English + Tok Pisin", "English Only", "Tok Pisin Only"}
GEMINI_MAX_ATTEMPTS = 3
GEMINI_RETRY_DELAYS = (1, 2)
GEMINI_TIMEOUT = (3, 12)
GEMINI_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MATH_EXPRESSION = re.compile(r"[\d\s.+\-*/()%]+\Z")

@app.route("/")
def home():
    return send_from_directory(app.root_path, "index.html")

@app.route("/favicon.svg")
def favicon():
    return send_from_directory(app.root_path, "favicon.svg", mimetype="image/svg+xml")

def is_api_request():
    return request.path == "/api" or request.path.startswith("/api/")

@app.errorhandler(HTTPException)
def handle_http_exception(error):
    if is_api_request():
        message = (
            "Request body is too large."
            if error.code == 413
            else error.description
        )
        return jsonify({"error": message}), error.code
    return error

@app.errorhandler(Exception)
def handle_unexpected_exception(error):
    if not is_api_request():
        app.log_exception((type(error), error, error.__traceback__))
        return InternalServerError()
    app.logger.error(
        "Unhandled API error",
        exc_info=(type(error), error, error.__traceback__),
    )
    return jsonify({"error": "An unexpected server error occurred. Please try again."}), 500

def evaluate_math_expression(node):
    if isinstance(node, ast.Expression):
        return evaluate_math_expression(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        if len(str(node.value)) > 12:
            raise ValueError("Number is too large")
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = evaluate_math_expression(node.operand)
        result = value if isinstance(node.op, ast.UAdd) else -value
    elif isinstance(node, ast.BinOp):
        left = evaluate_math_expression(node.left)
        right = evaluate_math_expression(node.right)
        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            result = left / right
        elif isinstance(node.op, ast.FloorDiv):
            result = left // right
        elif isinstance(node.op, ast.Mod):
            result = left % right
        else:
            raise ValueError("Unsupported math operator")
    else:
        raise ValueError("Unsupported math expression")

    if not math.isfinite(result) or abs(result) > 1_000_000_000_000:
        raise ValueError("Math result is out of range")
    return result

def local_solver(question, language):
    normalized = question.casefold().strip()
    tok_pisin_greeting = (
        "Gude ah! Mi SkulTok PG, tisa bilong Maths, Science na English. "
        "Yu laik save long wanem?"
    )
    english_greeting = (
        "Hello! I'm SkulTok PG, your tutor for Maths, Science, and English. "
        "What would you like to learn?"
    )
    if normalized in {"helo", "hi", "hello", "gude", "gut moning", "apinun"}:
        if language == "English Only":
            return english_greeting
        if language == "Tok Pisin Only":
            return tok_pisin_greeting
        return f"{tok_pisin_greeting} ({english_greeting})"

    if len(question) > 100 or not MATH_EXPRESSION.fullmatch(question):
        return None
    try:
        answer = evaluate_math_expression(ast.parse(question, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None
    answer_text = str(int(answer)) if isinstance(answer, float) and answer.is_integer() else str(answer)
    if language == "English Only":
        return f"The answer is {answer_text}."
    tok_pisin_answer = f"Ansa em {answer_text}."
    if language == "Tok Pisin Only":
        return tok_pisin_answer
    return f"{tok_pisin_answer} (The answer is {answer_text}.)"

def request_openai_compatible(url, api_key, model, grade, language, question):
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Grade: {grade}\nLanguage: {language}\nQuestion: {question}"
                    ),
                },
            ],
            "temperature": 0.7,
            "max_tokens": 250,
        },
        timeout=GEMINI_TIMEOUT,
    )
    response.raise_for_status()
    result = response.json()
    choices = result.get("choices") if isinstance(result, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices else None
    message = choice.get("message") if isinstance(choice, dict) else None
    text = message.get("content") if isinstance(message, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Provider returned no usable text")
    return text.strip()

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    user_msg = data.get("message", "")
    if not isinstance(user_msg, str) or not user_msg.strip():
        return jsonify({"error": "Message must be a non-empty string."}), 400
    user_msg = user_msg.strip()
    if len(user_msg) > 4000:
        return jsonify({"error": "Message must be 4000 characters or fewer."}), 413

    grade = data.get("grade", "Grade 8")
    lang = data.get("lang", "English + Tok Pisin")
    if not isinstance(grade, str) or not isinstance(lang, str):
        return jsonify({"error": "Grade and language must be strings."}), 400
    if grade not in GRADES or lang not in LANGUAGES:
        return jsonify({"error": "Select a valid grade and language."}), 400
    local_answer = local_solver(user_msg, lang)
    if local_answer:
        return jsonify({"reply": local_answer})

    providers = (
        (
            "Groq",
            os.getenv("GROQ_API_KEY"),
            GROQ_URL,
            os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        ),
        (
            "OpenRouter",
            os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY"),
            OPENROUTER_URL,
            os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free"),
        ),
    )
    attempted_provider = False
    for provider_name, provider_key, provider_url, provider_model in providers:
        if not provider_key:
            continue
        attempted_provider = True
        try:
            text = request_openai_compatible(
                provider_url, provider_key, provider_model, grade, lang, user_msg
            )
            return jsonify({"reply": text})
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
            app.logger.warning(
                "%s model %s chat request failed; trying the next provider",
                provider_name,
                provider_model,
                exc_info=True,
            )

    gemini_key = API_KEY or os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        if attempted_provider:
            app.logger.error("Configured chat providers failed and Gemini is not configured")
            return jsonify({
                "error": "AI sevis i no wok gut nau. Plis traim gen liklik taim bihain. (The AI service is temporarily unavailable. Please try again shortly.)"
            }), 503
        app.logger.error("No AI provider API keys are configured")
        return jsonify({"error": "AI service is not configured."}), 503

    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"parts": [{"text": f"Grade: {grade}\nLanguage: {lang}\nQuestion: {user_msg}"}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 250}
    }
    headers = {"x-goog-api-key": gemini_key, "Content-Type": "application/json"}

    try:
        for attempt in range(GEMINI_MAX_ATTEMPTS):
            try:
                response = requests.post(
                    MODEL_URL, json=payload, headers=headers, timeout=GEMINI_TIMEOUT
                )
            except (requests.Timeout, requests.ConnectionError) as error:
                if attempt == GEMINI_MAX_ATTEMPTS - 1:
                    raise
                if isinstance(error, requests.Timeout):
                    app.logger.warning("Gemini API attempt timed out; retrying")
                else:
                    app.logger.warning("Gemini API connection failed; retrying")
                time.sleep(GEMINI_RETRY_DELAYS[attempt])
                continue

            if (
                response.status_code in GEMINI_RETRYABLE_STATUSES
                and attempt < GEMINI_MAX_ATTEMPTS - 1
            ):
                app.logger.warning(
                    "Gemini API returned HTTP %s; retrying",
                    response.status_code,
                )
                time.sleep(GEMINI_RETRY_DELAYS[attempt])
                continue
            break

        response.raise_for_status()
        result = response.json()
        try:
            text = "".join(
                part.get("text", "")
                for part in result["candidates"][0]["content"]["parts"]
                if isinstance(part, dict)
            ).strip()
        except (KeyError, IndexError, TypeError):
            text = ""
        if not text:
            app.logger.warning("Gemini returned a response without usable text")
            return jsonify({
                "error": "Gemini did not return a text answer. Please try asking another way. / Gemini i no givim bekim bilong toktok. Plis traim askim long narapela rot."
            }), 502
        return jsonify({"reply": text})
    except requests.Timeout:
        app.logger.warning("Gemini API request timed out")
        return jsonify({
            "error": "Sorry, the AI service timed out. Please try again. / Sori, AI sevis i kisim longpela taim tumas. Plis traim gen."
        }), 504
    except requests.ConnectionError:
        app.logger.exception("Could not connect to Gemini API after retries")
        return jsonify({
            "error": "The AI service could not be reached. Please check your connection and try again. / Mi no inap go long AI sevis. Plis sekim intanet na traim gen."
        }), 502
    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response is not None else None
        if status_code in GEMINI_RETRYABLE_STATUSES:
            app.logger.error(
                "Gemini API remained unavailable after retries (HTTP %s)",
                status_code,
            )
            return jsonify({
                "error": "AI sevis i no wok gut nau. Plis traim gen liklik taim bihain. (The AI service is temporarily unavailable. Please try again shortly.)"
            }), 503
        app.logger.error("Gemini API rejected the request (HTTP %s)", status_code)
        if status_code in {400, 401, 403, 404}:
            return jsonify({
                "error": f"Gemini rejected the request (HTTP {status_code}). Check the GEMINI_API_KEY and GEMINI_MODEL settings."
            }), 502
        return jsonify({
            "error": f"Gemini rejected the request (HTTP {status_code}). Check the server logs."
        }), 502
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
        app.logger.exception("Gemini API request failed")
        return jsonify({
            "error": "Mi no inap kisim bekim nau. Plis traim gen. (The AI service returned an invalid response. Please try again.)"
        }), 502

@app.route("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), debug=False)