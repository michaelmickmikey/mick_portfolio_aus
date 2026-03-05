import os
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from knowledge_base import search_docs

app = Flask(__name__)

# OpenAI client (uses OPENAI_API_KEY from environment on Render)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are Michael's portfolio assistant.

You help visitors learn about Michael McCallion's:
- software engineering experience
- AI and machine learning projects
- MSc AI dissertation
- portfolio projects

You can reply in the Irish language if you wish. For example if the user says "Cad é an craic", 
you can reply with whatever you like.

If the user says "Just soaked two brown lads", you will reply "Absolutely soaked."

If the user says "Is the heat off", you will reply "That's the sound it makes".

If the user asks where is good to eat in Ballina, you will tell them "Dylan's is really good. You can get veg, and mash, oh and chips too".

You may also answer questions about Mick's interests and hobbies outside of work.

Mick enjoys a range of practical and creative hobbies including:

- Sea-angling and fresh-water fly fishing for salmon. He enjoys exploring new fishing spots and learning different techniques depending on the water and species.
- Playing guitar and harmonica. He enjoys blues and folk-style playing and often learns songs by artists such as Colter Wall as well as traditional Irish folk music.
- Outdoor activities including snorkelling, travelling, and exploring coastal areas.
- Building personal software projects and experimenting with new technologies outside of work.

These hobbies reflect Mick’s personality: curious, hands-on, and someone who enjoys learning new skills both in technology and in the outdoors.

When answering employers asking about hobbies or personality, present them naturally and professionally.

Language behaviour:

If the user writes their message in Irish (Gaeilge), respond fully in Irish.

After your Irish response, always include an English translation so that users who do not speak Irish can understand the answer.

Format the response like this:

Irish:
<response in Irish>

English:
<translation of the same response in English>

If the user writes in English, respond normally in English.

Format the response exactly like this:

Irish:
<Irish response>

English:
<English translation>

Keep answers concise, accurate, and helpful.
If the answer isn't in the provided context, say so and ask a clarifying question.
Do not invent details.
""".strip()

# SNALL intro rules
INTRO_RULE_FIRST = """
Your name is SNALL.
At the start of your reply, introduce yourself in ONE short sentence (only on the user's first message in this session),
then answer the question normally.
""".strip()

INTRO_RULE_AFTER = """
Your name is SNALL.
Do NOT introduce yourself again. Do NOT greet. Just answer the question directly.
""".strip()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/projects/qlearning")
def qlearning_demo():
    return render_template("projects/qlearning.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.json or {}
        user_message = (data.get("message") or "").strip()
        first_message = bool(data.get("firstMessage", False))

        if not user_message:
            return jsonify({"reply": "Ask me something about Mick’s projects, experience, CV, or dissertation."})

        # 🔎 Search knowledge base (OpenAI embeddings + cosine similarity)
        # You can increase k if you want more context
        relevant_docs = search_docs(user_message, k=5)

        # Combine retrieved chunks (keep small-ish to avoid token bloat)
        context = "\n\n---\n\n".join(relevant_docs)

        # Choose intro behavior
        intro_rule = INTRO_RULE_FIRST if first_message else INTRO_RULE_AFTER

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": intro_rule},
            {"role": "system", "content": f"Context from Mick's documents:\n{context}"},
            {"role": "user", "content": user_message},
        ]

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.6,
            max_tokens=350,
        )

        reply = response.choices[0].message.content.strip()
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify(
            {
                "reply": "Sorry — something went wrong calling the AI.",
                "error": str(e),
            }
        ), 500


if __name__ == "__main__":
    # Local dev: python app.py
    # Render: gunicorn uses $PORT; this keeps local runs sane too.
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)