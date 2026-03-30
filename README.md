# Document Classification pipeline

## Features
### 1.Text and metadata extraction from files
The app checks the file's mime type. It either extracts text from reading the file, either sends it to OCR (pdf files with little to no text or images). 

The date is extracted from the file text and used as a a parameter, either for file routing (Classification/Date), either for filtering the vector database.  

### 2.File classification
The system checks for keywords inside the text. If the keyword search isn't conclusive (doesn't have enough keywords and a solid difference from the second best category), the app falls back to classifying the app using ML. The file is copied to the corresponding file path (result/[Classification]/[Date]) and the text is chunked and saved into the vector database (using chromadb)

### 3.Question answering (RAG)
Using an LLM (Qwen3.5-2B), the user can ask questions about specific documents. The saved chunks from the vector database are used to give context to the LLM.

## Approach taken
Documents are processed via a `classification_runner.py` or by a call to the Flask API endpoint `http://127.0.0.1:8000/api/document`. Text is extracted from the file, or OCR-ized from the images in the file utilizing pytesseract. Primary dates are identified using datefinder. Classification is determined using the following approach: keyword heuristics evaluate the text, and an automated ML fallback (using the model **facebook/bart-large-mnli**, the task **zero-shot-classification**) if the top score or the margin between the top two heuristic scores is inconclusive. Files are physically copied to a structured file system hierarchy (`result/[classification]/[date]/`), with the original file metadata.

Extracted text is chunked into fixed-size strings (5000 characters with a 50-character overlap) to fit within context window constraints. The chunks are embedded and stored in a local ChromaDB persistent instance. Extracted dates, classifications, and file paths are added to each chunk as metadata.

Document questions are processed using datefinder and a keyword mapping dictionary to extract target dates and document classifications. These extracted entities construct a metadata filter (where_filters) to restrict the vector database search space. The context will contain the mathematically nearest text chunks. A local LLM (Qwen3.5-2B) is hosted via software like Ollama or LM Studio (or any external tool, `start_LLM.py` starts an api for Qwen3.5-2B, although it's probably more unoptimized than a specialized tool). Retrieved context is injected into the system prompt instructing the model to synthesize an answer based only on the provided text, returning an "Insufficient data" text if the context lacks the necessary information.

### Limitations

The whole OCR process is not perfect. A proper OCR solution should have a more thorough and dynamic preprocessing and postprocessing. Current implementation may produce deformed strings.

The keyword-based classification needs more (quality) keywords to properly differentiate between other.

The ML-based classification would benefit from finetuning before using. Current classification process relies only on the assumption the downloaded model has decent weights for our use case.

The chunks have a fixed 5000 character size. This implementation risks splitting sentences in 2 separate chunks. For this specific task, the approach taken was to make the size big enough each document will have at most 2 chunks. This can be solved by using NLP tools like `spaCy` to find boundaries. 

The AI model used was a 2B parameter one because of the hardware limitations. While it does it's job nicely, a model with more parameters would work way better.

## How to run

Create a venv (the project was tested on python 3.12)
```commandline
python3 -m venv .venv
```
Enter the venv (depending on OS)
```commandline
source .venv/bin/activate #Linux/MacOS
```
```commandline
.venv\Scripts\activate.bat # Windows
```
```commandline
.venv\Scripts\Activate.ps1 # Windows Powershell
```
Install dependencies
```commandline
pip install -r requirements.txt
```
Run the classification script (this might take a while, depending on dataset size)
```commandline
python3 classification_runner.py
```

**The LLM used for testing was qwen3.5-2b. For testing I used an external tool (LM Studio), but the LLM server can be started with the following script. Feel free to change the model to your liking**
```commandline
python3 start_LLM.py
```
Run the chat script (CLI document question)
```commandline
python3 chat.py
```

Run the API
```commandline
python3 controller.py
```

### Using the API
/api/document (sends a file to the classification pipeline)
```commandline
curl -X POST http://localhost:8000/api/document \
  -F "file=@example.pdf"
```
response:
```json
{
  "message": "Document processed and stored successfully",
  "file": "example.pdf",
  "classification": "invoice",
  "date": "2024-01-15",
  "chunks_added": 12
}
```
...

/api/chat (provide a question regarding a document)
```commandline
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the uploaded documents"}'
```

response:
```json
{
  "question": "What invoices do I have from January?",
  "response": "You have 3 invoices dated January 2024..."
}
```