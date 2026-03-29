# Document Classification pipeline

## Features
### 1.Text and metadata extraction from files
The app checks the file's mime type. It either extracts text from reading the file, either sends it to OCR (pdf files with little to no text or images). 

The date is extracted from the file text and used as a a parameter, either for file routing (Classification/Date), either for filtering the vector database.  

### 2.File classification
The system checks for keywords inside the text. If the keyword search isn't conclusive (doesn't have enough keywords and a solid difference from the second best category), the app falls back to classifying the app using the model `facebook/bart-large-mnli`, using the task `zero-shot-classification`. The file is copied to the corresponding file path (result/[Classification]/[Date]) and the text is chunked and saved into the vector database (using chromadb)

### 3.Question answering (RAG)
Using an LLM (Qwen3.5-2B), the user can ask questions about specific documents. The saved chunks from the vector database are used to give context to the LLM.

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
Run the chat script
```commandline
python3 chat.py
```