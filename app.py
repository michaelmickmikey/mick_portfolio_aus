import os
from flask import Flask, render_template, request, jsonify
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are Mick's portfolio assistant.

You help visitors learn about Mick McCallion's:
- software engineering experience
- AI and machine learning projects
- MSc AI dissertation
- portfolio projects

Keep answers concise and helpful.
"""

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/projects/qlearning")
def qlearning_demo():
    return render_template("projects/qlearning.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_message = data.get("message", "")

        response = client.chat.completions.create(
            model="gpt-4.1-mini",   # 👈 CHEAP MODEL
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=300
        )

        reply = response.choices[0].message.content

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({
            "reply": "Sorry — something went wrong calling the AI.",
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001)