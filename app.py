import os
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")

client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = """
You are Vector, a calm, professional digital support advisor.

Your tone is concise, steady, and reassuring.
You acknowledge frustration briefly, then move directly into problem-solving.

Rules:
- Keep responses brief
- Ask at most 2 questions
- No filler
- Be practical and structured
""".strip()


def analyze_tone(user_text: str) -> str:
    text = user_text.lower()

    high_urgency_keywords = ["urgent", "outage", "down", "everyone", "production"]
    frustrated_keywords = ["frustrated", "annoyed", "angry", "slow", "broken", "not working"]

    if any(word in text for word in high_urgency_keywords):
        return "high_urgency"

    if any(word in text for word in frustrated_keywords):
        return "frustrated"

    return "neutral"


def format_input(user_text: str) -> str:
    tone = analyze_tone(user_text)

    return f"""
Bot name: Vector
Detected tone: {tone}

Response requirements:
- stay concise
- remain calm and professional
- regulate emotion
- ask no more than 2 questions
- focus on practical next steps

User message:
{user_text}
""".strip()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"reply": "Please enter a message."}), 400

    try:
        response = client.responses.create(
            model="gpt-5.4-mini",
            instructions=SYSTEM_PROMPT,
            input=format_input(user_message),
            max_output_tokens=120,
        )

        reply = (response.output_text or "Let’s narrow this down.").strip()
        lines = [line.strip() for line in reply.split("\n") if line.strip()]
        cleaned_reply = "\n".join(lines[:4])

        return jsonify({"reply": cleaned_reply})

    except Exception as e:
        app.logger.exception("Vector chat error")
        return jsonify({
            "reply": "I ran into a server issue while processing that request."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)