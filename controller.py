import os
import uuid
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import chromadb
from helpers import document_date_extractor, document_text_extractor, document_classifier
from helpers import RAG

app = Flask(__name__)

classifier = document_classifier.DocumentClassifier()
text_extractor = document_text_extractor.DocumentTextExtractor()
date_extractor = document_date_extractor.DocumentDateExtractor()

chroma_client = chromadb.PersistentClient(path="vector_db")
collection = chroma_client.get_or_create_collection(name="document_collection")

rag = RAG.LocalRAG()

UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/api/document', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if file:
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)

        text = text_extractor.extract_text(temp_path)
        date = date_extractor.extract_primary_date(text)
        classification = classifier.classify_text(text)

        target_directory = os.path.join("result", str(classification), str(date))
        os.makedirs(target_directory, exist_ok=True)
        target_path_string = os.path.join(target_directory, filename)

        os.rename(temp_path, target_path_string)

        text_chunks = rag.chunk_text(text)
        chunk_ids = []
        chunk_metadatas = []

        for i, chunk in enumerate(text_chunks):
            chunk_ids.append(f"{filename}_chunk_{i}_{str(uuid.uuid4())[:8]}")
            chunk_metadatas.append({
                "source_file": filename,
                "original_path": temp_path,
                "target_path": target_path_string,
                "classification": classification,
                "date": date
            })

        if text_chunks:
            collection.add(
                documents=text_chunks,
                metadatas=chunk_metadatas,
                ids=chunk_ids
            )

        return jsonify({
            "message": "Document processed and stored successfully",
            "file": filename,
            "classification": classification,
            "date": date,
            "chunks_added": len(text_chunks)
        }), 200

    @app.route('/api/chat', methods=['POST'])
    def chat_endpoint():
        data = request.get_json()

        if not data or 'question' not in data:
            return jsonify({"error": "Invalid payload, 'question' field is required"}), 400

        question = data['question']

        response = rag.ask_question(question)

        return jsonify({
            "question": question,
            "response": response
        }), 200

    if __name__ == "__main__":
        app.run(host="0.0.0.0", port=8000)