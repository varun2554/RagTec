from flask import Flask, render_template, request, jsonify
import os
from backend.rag import RAGModel
from backend.extract_text import extract_text_from_pdf

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load Retrieval-Augmented Generation Model
rag_model = RAGModel()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    filename = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filename)

    # Extract text and store in RAG system
    extracted_text = extract_text_from_pdf(filename)
    rag_model.add_document(filename, extracted_text)

    return jsonify({"message": "File uploaded successfully"}), 200

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    answer = rag_model.answer_question(question)
    return jsonify({"answer": answer}), 200

if __name__ == "__main__":
    app.run(debug=True)
