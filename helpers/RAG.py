import requests
import chromadb
import json
from helpers import document_date_extractor


class LocalRAG:
    def __init__(self, db_path: str = "vector_db", collection_name: str = "document_collection",
                 api_endpoint: str = "http://localhost:1234/api/v1/chat"):

        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        self.api_endpoint = api_endpoint
        self.date_extractor = document_date_extractor.DocumentDateExtractor()

    def _extract_metadata_from_query(self, query: str) -> tuple:
        query_lower = query.lower()
        target_class = None

        target_date = self.date_extractor.extract_primary_date(query_lower)

        class_mapping = {
            "receipt": "Receipt",
            "contract": "Contract",
            "resume": "Resume",
            "scientific paper": "Scientific Paper"
        }

        for keyword, class_name in class_mapping.items():
            if keyword in query_lower:
                target_class = class_name
                break

        return target_date, target_class

    def _retrieve_context(self, question: str, n_results: int = 3) -> str:
        target_date, target_class = self._extract_metadata_from_query(question)

        valid_date = target_date and target_date != 'Unknown'
        valid_class = target_class and target_class != 'Unknown'

        where_filters = None
        if valid_date and valid_class:
            where_filters = {"$and": [{"date": target_date}, {"classification": target_class}]}
        elif valid_date:
            where_filters = {"date": target_date}
        elif valid_class:
            where_filters = {"classification": target_class}

        if where_filters:
            results = self.collection.query(
                query_texts=[question],
                n_results=n_results,
                where=where_filters
            )
        else:
            results = self.collection.query(
                query_texts=[question],
                n_results=n_results
            )

        if not results['documents'] or not results['documents'][0]:
            return "No relevant context found in the database."

        retrieved_chunks = results['documents'][0]

        return "\n...\n".join(retrieved_chunks)

    def chunk_text(self, text: str, chunk_size: int = 5000, overlap: int = 50) -> list:
        if not text:
            return []
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def ask_question(self, question: str) -> str:
        context = self._retrieve_context(question)

        system_prompt = (
            "You are an analytical assistant. Answer the user's question using ONLY the provided context. If the context does not contain the answer, state: 'Insufficient data to answer this question.' Do not use outside knowledge."
        )

        augmented_input = f"Context Data:\n{context}\n\nUser Question:\n{question}"

        payload = {
            "model": "qwen3.5-2b",
            "system_prompt": system_prompt,
            "input": augmented_input
        }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(self.api_endpoint, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()

            return response_data.get("output", response_data.get("response", json.dumps(response_data)))

        except requests.exceptions.RequestException as e:
            return f"API Connection failed: {str(e)}"
