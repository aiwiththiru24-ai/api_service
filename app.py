import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)

CORS(app, resources={
    r"/api/*": {
        "origins": [
            "https://id-preview--73d01875-3401-47b1-97fb-e7e891aa818f.lovable.app",
            # If you also have a production domain, add it here too
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/ask")
def ask():
    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()

    if not query:
        return jsonify({"error": "query is required"}), 400

    model = str(data.get("model") or DEFAULT_MODEL)
    instructions = str(data.get("instructions") or "You are a helpful assistant.")

    try:
        resp = client.responses.create(
            model=model,
            instructions=instructions,
            input=query,
        )
        return jsonify({
            "answer": resp.output_text,
            "response_id": resp.id,
            "model": model
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
