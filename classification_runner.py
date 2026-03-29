import os
import uuid
import chromadb
from helpers import document_date_extractor, document_text_extractor, document_classifier, RAG

document_classifier = document_classifier.DocumentClassifier()
text_extractor = document_text_extractor.DocumentTextExtractor()
date_extractor = document_date_extractor.DocumentDateExtractor()
rag = RAG.LocalRAG()

chroma_client = chromadb.PersistentClient(path="vector_db")
collection = chroma_client.get_or_create_collection(name="document_collection")

def document_classification(full_path: str) -> tuple:
    text = text_extractor.extract_text(full_path)
    date = date_extractor.extract_primary_date(text)
    classification = document_classifier.classify_text(text)
    return text, date, classification

for root, directories, files in os.walk('dataset'):
    for file_name in files:
        full_path_string = os.path.join(root, file_name)

        text, date, classification = document_classification(full_path_string)

        target_directory = os.path.join("result", classification, date)
        os.makedirs(target_directory, exist_ok=True)
        target_path_string = os.path.join(target_directory, file_name)

        with open(full_path_string, 'rb') as source_file:
            with open(target_path_string, 'wb+') as dest_file:
                dest_file.write(source_file.read())

        text_chunks = rag.chunk_text(text)

        chunk_ids = []
        chunk_metadatas = []

        for i, chunk in enumerate(text_chunks):
            chunk_ids.append(f"{file_name}_chunk_{i}_{str(uuid.uuid4())[:8]}")

            chunk_metadatas.append({
                "source_file": file_name,
                "original_path": full_path_string,
                "target_path": target_path_string,
                "classification": classification,
                "date": date
            })

        collection.add(
            documents=text_chunks,
            metadatas=chunk_metadatas,
            ids=chunk_ids
        )

