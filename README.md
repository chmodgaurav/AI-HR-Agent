# 🧑‍💼 HR AI Agent

An offline Retrieval-Augmented Generation (RAG) application with **Conversational Memory** built with **Streamlit**, **LangChain**, **Ollama**, and **ChromaDB**. The application allows users to upload HR/legal documents (PDF and JSON), index them into a vector database, and conduct multi-turn conversations with the AI that remembers context and maintains session isolation.

---

## ✨ Features

### Core RAG Features
* Upload **PDF** documents
* Upload **JSON** documents
* Automatic document indexing into **ChromaDB**
* Semantic search using **nomic-embed-text**
* Question answering using **Gemma 3**
* Displays retrieved context used for answering
* Fully local (no external API required)

### Conversational Memory Features ⭐ NEW
* **Session Memory** - Conversation history maintained per employee
* **Multi-turn Conversations** - Ask questions across multiple turns without repeating information
* **Session Isolation** - Complete isolation between employees (no data leakage)
* **Context-Aware Responses** - AI understands pronouns and previous context
* **Multi-turn Workflows** - Complete leave applications across multiple exchanges
* **Automatic Information Extraction** - Extract dates, leave types, employee IDs from conversations
* **TTL-based Session Cleanup** - Sessions auto-expire after 2 hours of inactivity

---

## Tech Stack

* **Frontend:** Streamlit
* **LLM Framework:** LangChain
* **Vector Database:** ChromaDB
* **LLM Runtime:** Ollama
* **Language Model:** Gemma 3 (4B)
* **Embeddings:** Nomic Embed Text
* **Memory System:** In-memory Session Management with TTL-based cleanup

---

## Project Structure

```text
AI-HR-Agent/
│
├── app.py                          # Main Streamlit application
├── conversation_memory.py          # Session and memory management
├── memory_chat_engine.py           # LLM integration with memory
├── tools/
│   └── apply_leave.py              # Leave application tool
├── database/
│   └── chroma/                     # Vector store
├── dataset/
│   ├── pdf/                        # PDF documents
│   └── json/                       # JSON documents
├── requirements.txt
├── README.md
├── QUICKSTART.md                   # 5-minute quick start
├── README_MEMORY.md                # Memory feature overview
├── MEMORY_IMPLEMENTATION.md        # Technical documentation
└── IMPLEMENTATION_GUIDE.md         # User guide
```

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd AI-HR-Agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama

Download and install Ollama from: https://ollama.com

### 4. Download Required Models

**Embedding model:**
```bash
ollama pull nomic-embed-text
```

**LLM:**
```bash
ollama pull gemma3:4b
```

**Start Ollama:**
```bash
ollama serve
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

---

## Quick Start

For a 5-minute quick start guide, see [QUICKSTART.md](QUICKSTART.md)

---

## Requirements

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

### Document Upload & Indexing
1. Upload a PDF or JSON file via the "Upload" tab
2. The document is converted into LangChain Documents
3. The documents are embedded using **nomic-embed-text**
4. Embeddings are stored in **ChromaDB**

### Multi-turn Conversation with Memory
1. Enter your Employee ID in the sidebar (creates a unique session)
2. Go to "HR Tools" tab to apply for leave conversationally
3. Or go to "Ask HR" tab to ask questions with conversation memory
4. The system remembers your previous messages and context
5. You don't need to repeat information
6. Leave application works across multiple turns
7. Session persists for 2 hours of inactivity

### Question Answering
1. Enter an HR-related question
2. Relevant documents are retrieved through semantic search
3. **Gemma 3** generates an answer using only the retrieved context
4. Retrieved source documents are displayed for transparency

---

## Usage Examples

### Example 1: Multi-turn Leave Application
```
You: "I want to apply for leave"
AI: "What type of leave do you need?"

You: "Casual leave"
AI: "When do you need the leave?"

You: "2024-02-15 to 2024-02-17"
AI: "Please provide a reason"

You: "Medical appointment"
AI: "Your leave request has been submitted successfully!"

✅ No information repeated!
✅ Full context remembered!
✅ Leave applied automatically!
```

### Example 2: Context-Aware Q&A
```
You: "What is the policy on medical leave?"
AI: [Retrieves from knowledge base] "Medical leave policy is..."

You: "How many days do I get?"
AI: [Remembers you asked about medical leave] "You are entitled to..."

You: "Can I use them for family emergencies?"
AI: [Understands 'them' refers to medical leave] "Yes, medical leave can be used for..."

✅ AI understands context!
✅ No need to repeat questions!
```

---

## Application Tabs

### Upload Tab
Upload HR/legal documents (PDF or JSON) to build your knowledge base. Documents are automatically indexed and searchable.

### HR Tools Tab ⭐ NEW
**Multi-turn Leave Application** - Conversationally apply for leave without repeating information:
- The system asks questions step by step
- Remembers your responses
- Automatically validates dates and information
- Submits your leave request when all information is collected

### Ask HR Tab
**Context-Aware Q&A** - Ask HR questions with conversation memory:
- Questions are answered based on indexed documents
- Previous questions and answers are remembered
- Context is maintained across multiple turns
- Retrieved documents are displayed for transparency

---

## Example Questions

### Q&A Examples
* What is the time limit for filing a workplace sexual harassment complaint?
* What are the employer's responsibilities under the POSH Act?
* What information must an LLP include on its invoices?
* When did the Industrial Relations Code, 2020 come into force?
* Which provisions of the Code on Wages became effective on 21 November 2025?

### Memory Examples
* "I need to apply for leave" → System asks for type, dates, reason
* "What's the leave policy?" → System retrieves policy and remembers context
* "How many days do I get?" → System understands you're asking about your approved leave type
* "What about sick leave?" → System can compare different leave types

---

## Requirements

```text
streamlit>=1.20
langchain>=1.0
langchain-community
langchain-core>=0.1
langchain-text-splitters
langchain-chroma
langchain-ollama
chromadb>=0.3
pypdf
jq
python-dotenv
pandas
tqdm
```

---

## Session Management

⭐ **Conversational Memory Features (v1.0 - NEW)**
* Each employee gets a unique session identified by their Employee ID
* Sessions maintain conversation history for 2 hours of inactivity
* All conversations are completely isolated between employees
* No cross-user data leakage
* Information is automatically extracted and reused

### Session Lifecycle
1. Enter your Employee ID in the sidebar
2. Start a conversation in "HR Tools" or "Ask HR" tab
3. Session is created automatically
4. Your conversation history is maintained
5. After 2 hours of inactivity, session expires automatically
6. Previous sessions can be viewed by clearing history and restarting

---

## Documentation

📖 **Comprehensive Guides Available:**
* **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
* **[README_MEMORY.md](README_MEMORY.md)** - Memory features overview
* **[MEMORY_IMPLEMENTATION.md](MEMORY_IMPLEMENTATION.md)** - Technical architecture
* **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)** - Configuration and troubleshooting

---

## Notes

* Documents are persisted inside the `database/chroma/` directory
* Ollama must be running before starting the application (run `ollama serve` in a separate terminal)
* The application answers questions using only the indexed documents
* If the required information is not in the knowledge base, the model will indicate it cannot find the answer
* All sessions are stored in-memory and will be cleared when the application restarts
* For production use with persistent sessions, consider Phase 2 enhancements

---

## Future Enhancements

### Phase 2 (Planned)
* Persistent session storage (Redis/PostgreSQL)
* Conversation summarization for long chats
* Advanced information extraction (NER)
* Session analytics and monitoring
* Conversation export/import

### Phase 3 (Future)
* LangGraph checkpointing for durable state
* Human handoff with conversation context
* Multi-modal input support
* Workflow automation
* Cross-session insights

### Other Improvements
* Hybrid search (BM25 + Vector Search)
* Metadata filtering
* Multi-document collections
* Document management (delete/re-index)
* Citation-aware responses
* Support for DOCX and TXT files

---

## Testing & Verification

To verify the conversational memory system is working correctly:

```bash
# Quick verification (all core tests)
python test_quick.py
# Expected output: ✓ ALL TESTS PASSED

# Run full unit test suite
pytest test_conversation_memory.py -v
# Expected: 18 tests passing

# Run example scenarios
python examples.py
```

---

## License

This project is intended for educational and research purposes. Modify and use it according to your project's licensing requirements.