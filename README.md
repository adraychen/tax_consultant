# 🇨🇦 Canada Tax Law AI Assistant (RAG Prototype)

An intelligent Retrieval-Augmented Generation (RAG) assistant designed to help users navigate the **Canadian Income Tax Act and Regulations**. This app uses official government tax documentation as its source of truth to provide grounded, accurate answers.

---

## 🚀 Features

- **Semantic Search:** Uses Google's `gemini-embedding-2` (3072 dimensions) to bridge plain language queries with formal legal terminology.
- **Section Metadata Indexing:** Each chunk is tagged with its section number, section title, Part, and Division during ingestion for more precise retrieval.
- **Multi-Source Knowledge Base:** Ingests both the full Income Tax Act and CRA plain language guides, bridging the vocabulary gap between user questions and legal text.
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
| **Embedding Model** | `gemini-embedding-2` (3072 dimensions, via Google AI Studio) |
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

The `vecs` library creates the table automatically but defaults to 1536 dimensions. Since `gemini-embedding-2` produces **3072-dimensional** vectors, you must pre-create the table before ingesting:

```sql
CREATE TABLE IF NOT EXISTS vecs.tax_knowledge (
    id TEXT PRIMARY KEY,
    vec vector(3072) NOT NULL,
    metadata JSONB
);
```

> ⚠️ **Important:** If you need to re-ingest, run `DELETE FROM vecs.tax_knowledge;` first. Do NOT drop the table — `vecs` will recreate it with the wrong dimensions (1536).

### 3. Enable Row Level Security (RLS)

Enable RLS on the table while allowing read access:

```sql
ALTER TABLE public.tax_knowledge ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow public read access"
ON public.tax_knowledge
FOR SELECT
USING (true);
```

### 4. Connection Strings

| Context | Connection Type | Port |
|---|---|---|
| **Streamlit Cloud** | Session Pooler | 5432 |
| **Local / PyCharm** | Session Pooler | 5432 |

Get these from: **Supabase Dashboard → Settings → Database → Connection string**

**Session Pooler format:**
```
postgresql://postgres.[PROJECT-REF]:[PASSWORD]@aws-X-region.pooler.supabase.com:5432/postgres
```

> ⚠️ **Note:** The Direct connection (`db.[PROJECT-REF].supabase.co:5432`) may not resolve from all networks. Streamlit Cloud does not support IPv6 used by the Direct connection. Use the Session Pooler URL for reliability.

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
Loading Income Tax Act...
Created 3895 chunks from Income Tax Act.
Loading CRA guides...
Created 19 chunks from CRA guides.
Total chunks to ingest: 3914
Ingested batch 1 of 79 (50/3914 chunks)
...
Ingestion complete.
```

> ⚠️ **Note:** Ingestion makes ~3,914 API calls to `gemini-embedding-2`. A `time.sleep(1)` delay is applied between calls to avoid rate limits. Expect 45-60 minutes total. If ingestion is interrupted, check the row count in Supabase with `SELECT COUNT(*) FROM vecs.tax_knowledge;` and update `START_FROM` in `ingest.py` to resume from the last saved batch.

### 6. Launch the Web App

```bash
streamlit run app.py
```

---

## 📥 Ingestion Pipeline

`ingest.py` performs the following steps:

1. **Scrapes** the full Income Tax Act and four CRA plain language guides
2. **Chunks** each document using `SentenceSplitter` with `chunk_size=512` and `chunk_overlap=50`
3. **Tags** each chunk with section metadata (section number, section title, Part, Division, source)
4. **Embeds** each chunk using `gemini-embedding-2` (3072 dimensions) via Google AI Studio
5. **Stores** chunks, embeddings, and metadata into Supabase's `vecs.tax_knowledge` table in batches of 50

### Chunking Strategy

| Parameter | Value | Reason |
|---|---|---|
| `chunk_size` | 512 tokens | Small enough for precise retrieval of specific legal provisions |
| `chunk_overlap` | 50 tokens | Prevents cutting sentences mid-provision |
| `batch_size` | 50 chunks | Prevents Supabase connection timeouts during long ingestion |
| Total chunks | ~3,914 | Income Tax Act + CRA guides |

### Data Sources

| Source | Type | URL |
|---|---|---|
| Income Tax Act (Full Text) | Legislation | https://laws-lois.justice.gc.ca/eng/acts/i-3.3/FullText.html |
| CRA Business Expenses Overview | CRA Guide | https://www.canada.ca/en/revenue-agency/services/tax/businesses/topics/sole-proprietorships-partnerships/business-expenses.html |
| CRA Motor Vehicle Expenses | CRA Guide | https://www.canada.ca/en/revenue-agency/services/tax/businesses/topics/sole-proprietorships-partnerships/business-expenses/motor-vehicle-expenses.html |
| CRA Home Office Expenses | CRA Guide | https://www.canada.ca/en/revenue-agency/services/tax/businesses/topics/sole-proprietorships-partnerships/business-expenses/home-office-expenses.html |
| CRA Hiring Family Members | CRA Guide | https://www.canada.ca/en/revenue-agency/services/tax/businesses/topics/sole-proprietorships-partnerships/business-expenses/hire-family-members.html |

### Section Metadata

Each Income Tax Act chunk is automatically tagged with:

| Metadata Field | Example Value |
|---|---|
| `source` | `"Income Tax Act"` |
| `part` | `"PART I"` |
| `division` | `"DIVISION I"` |
| `section_number` | `"162"` |
| `section_title` | `"Failure to file return of income"` |

---

## ☁️ Deployment (Streamlit Cloud)

1. Push your code to GitHub (ensure `.env` is in `.gitignore`)
2. Connect your repo to [Streamlit Community Cloud](https://streamlit.io/cloud)
3. In **Settings → Secrets**, add:

```toml
GEMINI_API_KEY = "your-gemini-api-key"
SUPABASE_DB_URL = "postgresql://postgres.[PROJECT-REF]:[PASSWORD]@aws-X-region.pooler.supabase.com:5432/postgres"
```

> Use the **Session Pooler** URL (port 5432) for Streamlit Cloud.

---

## 🔍 Retrieval Architecture

The app uses a four-layer retrieval strategy:

```
User Question
     │
     ▼
Query Rewriting (Gemini 2.5 Flash)
Translates plain language → precise legal terminology
     │
     ▼
Sub-Question Decomposition (LlamaIndex)
Breaks complex multi-part questions into focused sub-questions
     │
     ▼
Vector Similarity Search (Supabase + gemini-embedding-2)
similarity_top_k=15 chunks retrieved per sub-question
Metadata-tagged chunks improve ranking precision
     │
     ▼
Response Synthesis (Gemini 2.5 Flash)
Combines retrieved chunks into a coherent, grounded answer
```

---

## 📦 Requirements

```
streamlit
llama-index
llama-index-vector-stores-supabase
llama-index-llms-google-genai
llama-index-readers-web
python-dotenv
google-genai
psycopg2-binary
```

---

## ⚠️ Known Limitations

- **No chat history:** Each question is answered independently with no memory of previous questions in the session.
- **Static knowledge base:** The Income Tax Act and CRA guides are ingested at a point in time. Re-ingestion is required when legislation is updated.
- **Rate limits:** `gemini-embedding-2` has per-minute rate limits. A delay is applied during ingestion to manage this, but large re-ingestions take 45-60 minutes.

---

## ⚖️ Disclaimer

This application is a technical prototype. It does not provide professional tax, legal, or financial advice. Always consult with a qualified professional or refer to the official [CRA website](https://www.canada.ca/en/revenue-agency.html) for tax matters.
