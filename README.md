# 🧑‍💼 HR AI Agent

An offline Retrieval-Augmented Generation (RAG) application built with **Streamlit**, **LangChain**, **Ollama**, and **ChromaDB**. The application allows users to upload HR/legal documents (PDF and JSON), index them into a vector database, and ask natural language questions grounded in the uploaded documents.

---

## Features

* Upload **PDF** documents
* Upload **JSON** documents
* Automatic document indexing into **ChromaDB**
* Semantic search using **nomic-embed-text**
* Question answering using **Gemma 3**
* Displays retrieved context used for answering
* Fully local (no external API required)

---

## Tech Stack

* Streamlit
* LangChain
* ChromaDB
* Ollama
* Gemma 3
* Nomic Embed Text

---

## Project Structure

```text
project/
│
├── app.py
├── database/
├── uploads/
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd hr-ai-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama

Download and install Ollama from:

https://ollama.com

---

## Download Required Models

Embedding model

```bash
ollama pull nomic-embed-text
```

LLM

```bash
ollama pull gemma3:4b
```

Start Ollama

```bash
ollama serve
```

---

## Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## Supported File Types

### PDF

Examples

* Company HR Policy
* Employee Handbook
* Labour Law Notifications
* Offer Letters
* Employment Contracts

### JSON

Any structured JSON document such as

* Acts
* Rules
* Policies
* Knowledge Base
* FAQs

---

## Workflow

1. Upload a PDF or JSON file.
2. The document is converted into LangChain Documents.
3. The documents are embedded using **nomic-embed-text**.
4. Embeddings are stored in **ChromaDB**.
5. Enter an HR-related question.
6. Relevant documents are retrieved through semantic search.
7. **Gemma 3** generates an answer using only the retrieved context.
8. Retrieved source documents are displayed for transparency.

---

## Example Questions

* What is the time limit for filing a workplace sexual harassment complaint?
* What are the employer's responsibilities under the POSH Act?
* What information must an LLP include on its invoices?
* When did the Industrial Relations Code, 2020 come into force?
* Which provisions of the Code on Wages became effective on 21 November 2025?

---

## Requirements

```text
streamlit
langchain
langchain-core
langchain-community
langchain-ollama
langchain-chroma
chromadb
pypdf
```

---

## Notes

* Documents are persisted inside the `database/` directory.
* Ollama must be running before starting the application.
* The application answers questions using only the indexed documents. If the required information is not present in the uploaded data, the model should indicate that it cannot find the answer.

---

## Future Improvements

* Conversation memory
* Hybrid search (BM25 + Vector Search)
* Metadata filtering
* Source highlighting
* Multi-document collections
* Document management (delete/re-index)
* Citation-aware responses
* Support for DOCX and TXT files

---

## License

This project is intended for educational and research purposes. Modify and use it according to your project's licensing requirements.