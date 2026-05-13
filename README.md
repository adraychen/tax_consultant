# 🇨🇦 Canada Tax Law AI Assistant (RAG Prototype)

An intelligent Retrieval-Augmented Generation (RAG) assistant designed to help users navigate the **Canadian Income Tax Act and Regulations**. This app uses official government tax documentation as its source of truth to provide grounded, accurate answers.

---

## 🚀 Features

- **Semantic Search:** Uses `nlpaueb/legal-bert-base-uncased`, a BERT model pre-trained on legal text, to find relevant tax law provisions even when exact keywords don't match.
- **Query Rewriting:** Automatically translates plain language questions into precise Income Tax Act terminology before retrieval.
- **Sub-Question Decomposition:** Complex multi-part questions are broken into focused sub-questions, each retrieving independently before answers are synthesized.
- **RAG Architecture:** Grounded in actual legislation to minimize hallucinations.
- **Fast UI:** Built with Streamlit for a smooth, chat-like experience.
- **Privacy First:** All sensitive API keys and database credentials are managed via environment variables and Streamlit Secrets.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Language Model** | Google Gemini 2.5 Flash (via Google AI Studio) |
| **Embedding Model** | `nlpaueb/legal-bert-base-uncased` (local, no API key required) |
| **Vector Database** | Supabase (PostgreSQL + pgvector extension) |
| **Orchestration** | LlamaIndex (ingestion, retrieval, sub-question engine) |
| **Frontend** | Streamlit |
| **Deployment** | Streamlit Community Cloud |

---

## 📂 Project Structure

```
canada-tax-rag-bot/
├── app.py              # Streamlit web application
├── ingest.py           # Data ingestion pipeline
├── requirements.txt    # Python dependencies
├── .env                # Local environment variables (never commit)
├── .gitignore          # Excludes .env and sensitive files
└── README.md           # This file
```

---

## 🗄️ Database Schema & Setup

This project uses **Supabase (PostgreSQL)** as the vector store. The `vecs` library (used internally by LlamaIndex's Supabase integration) manages its own schema.

### 1. Enable pgvector

In your Supabase SQL Editor, run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2. Pre-create the Knowledge Table

The `vecs` library creates the table automatically, but defaults to 1536 dimensions. Since `legal-bert-base-uncased` produces **768-dimensional** vectors, you must pre-create the table before ingesting:

```sql
CREATE TABLE IF NOT EXISTS vecs.tax_knowledge (
    id TEXT PRIMARY KEY,
    vec vector(768) NOT NULL,
    metadata JSONB
);
```

> ⚠️ **Important:** If you need to re-ingest, run `DELETE FROM vecs.tax_knowledge;` first. Do NOT drop the table — `vecs` will recreate it with the wrong dimensions (1536).

### 3. Connection Strings

Two connection strings are needed depending on context:

| Context | Connection Type | Port |
|---|---|---|
| **Streamlit Cloud** | Session Pooler | 5432 |
| **Local / PyCharm** | Session Pooler or Direct | 5432 |

Get these from: **Supabase Dashboard → Settings → Database → Connection string**

**Session Pooler format:**
```
postgresql://postgres.[PROJECT-REF]:[PASSWORD]@aws-X-region.pooler.supabase.com:5432/postgres
```

> ⚠️ **Note:** The Direct connection (`db.[PROJECT-REF].supabase.co:5432`) may not resolve from all networks. Use the Session Pooler URL for reliability.

---

## ⚙️ Installation & Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/canada-tax-rag-bot.git
cd canada-tax-rag-bot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the root directory:

```
GEMINI_API_KEY=your_google_ai_studio_key
SUPABASE_DB_URL=postgresql://postgres.[PROJECT-REF]:[PASSWORD]@aws-X-region.pooler.supabase.com:5432/postgres
```

> ⚠️ **Important:** Never commit your `.env` file. Ensure `.gitignore` includes `.env`.

### 4. Pre-create the Supabase Table

Run the SQL from the Database Setup section above before ingesting.

### 5. Run Ingestion

```bash
python ingest.py
```

Expected output:
```
Loaded 1 document(s). Starting ingestion...
Created 3895 chunks.
Ingestion complete.
```

> The embedding model (`legal-bert-base-uncased`) downloads automatically on first run (~400MB).

### 6. Launch the Web App

```bash
streamlit run app.py
```

---

## 📥 Ingestion Pipeline

`ingest.py` performs the following steps:

1. **Scrapes** the full text of the Income Tax Act from `laws-lois.justice.gc.ca`
2. **Chunks** the document using `SentenceSplitter` with `chunk_size=512` and `chunk_overlap=50`
3. **Embeds** each chunk using `nlpaueb/legal-bert-base-uncased` (768 dimensions, runs locally)
4. **Stores** chunks and embeddings into Supabase's `vecs.tax_knowledge` table

### Chunking Strategy

| Parameter | Value | Reason |
|---|---|---|
| `chunk_size` | 512 tokens | Small enough for precise retrieval of specific legal provisions |
| `chunk_overlap` | 50 tokens | Prevents cutting sentences mid-provision |
| Total chunks | ~3,895 | Full Income Tax Act coverage |

### Data Source

| Source | URL |
|---|---|
| Income Tax Act (Full Text) | https://laws-lois.justice.gc.ca/eng/acts/i-3.3/FullText.html |

---

## ☁️ Deployment (Streamlit Cloud)

1. Push your code to GitHub (ensure `.env` is in `.gitignore`)
2. Connect your repo to [Streamlit Community Cloud](https://streamlit.io/cloud)
3. In **Settings → Secrets**, add:

```toml
GEMINI_API_KEY = "your-gemini-api-key"
SUPABASE_DB_URL = "postgresql://postgres.[PROJECT-REF]:[PASSWORD]@aws-X-region.pooler.supabase.com:5432/postgres"
```

> Use the **Session Pooler** URL (port 5432) for Streamlit Cloud — the Direct connection uses IPv6 which Streamlit Cloud does not support.

---

## 🔍 Retrieval Architecture

The app uses a three-layer retrieval strategy:

```
User Question
     │
     ▼
Query Rewriting (Gemini)
Translates plain language → legal terminology
     │
     ▼
Sub-Question Decomposition (LlamaIndex)
Breaks complex questions into focused sub-questions
     │
     ▼
Vector Similarity Search (Supabase + legal-bert)
similarity_top_k=30 chunks retrieved per sub-question
     │
     ▼
Response Synthesis (Gemini 2.5 Flash)
Combines retrieved chunks into a coherent answer
```

---

## 📦 Requirements

```
streamlit
llama-index
llama-index-vector-stores-supabase
llama-index-llms-google-genai
llama-index-embeddings-huggingface
llama-index-readers-web
llama-index-retrievers-bm25
python-dotenv
google-genai
psycopg2-binary
sentence-transformers
```

---

## ⚠️ Known Limitations

- **Vocabulary gap:** The Income Tax Act uses formal legal language that may not match plain language queries. Query rewriting mitigates but does not fully solve this.
- **Single data source:** Only the Income Tax Act is ingested. CRA interpretation bulletins and guides would significantly improve answer quality for practical questions.
- **Embedding model:** `legal-bert-base-uncased` was trained primarily on EU and contract law, not Canadian tax law specifically. OpenAI `text-embedding-3-large` would improve retrieval quality.
- **No chat history:** Each question is answered independently with no memory of previous questions.

---

## ⚖️ Disclaimer

This application is a technical prototype. It does not provide professional tax, legal, or financial advice. Always consult with a qualified professional or refer to the official [CRA website](https://www.canada.ca/en/revenue-agency.html) for tax matters.
