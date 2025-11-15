# Movie Recommender Chatbot

## Overview
This project is a **Retrieval-Augmented Generation (RAG)** chatbot designed to recommend and discuss movies using real data from the **Wikipedia Movie Plots** dataset on Kaggle.  
The chatbot understands user queries about movies by title, genre, country, or release year and responds in a **natural, conversational way**.
These system combines **information retrieval**, **semantic embeddings**, and **LLM-based response generation**.  

The chatbot can:
- Search relevant movie plots from a dataset.
- Retrieve additional web context using **Serper API (Google Search)**.
- Generate friendly, concise, and accurate responses using **Groq’s Llama 3.1 model**.

---

## System Design & Architecture

### 1️. Data Source
- **Wikipedia Movie Plots dataset (Kaggle)**
  - Source: [`jrobischon/wikipedia-movie-plots`](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots)
  - Contains >34,000 movies with title, genre, director, cast, and full plot summaries.

### 2️. Embedding & Vector Search
- **Sentence Transformer**: `BAAI/bge-small-en-v1.5`
- **FAISS (Facebook AI Similarity Search)**

### 3️. Retrieval Logic
The chatbot retrieves information in two stages:
1. **Vector Search:** Finds top movie chunks relevant to user query.
2. **Web Context (Serper API):** Fetches extra context if retrieval confidence is low.

### 4️. Generation
- **Groq API** (Llama 3.1-8B Instant) is used to generate natural responses.
- The model receives:
  - Movie context (from FAISS)
  - Optional web snippets (from Serper)
  - The user’s question.

---

## Design Choices

| Component | Choice | Reason |
|------------|---------|--------|
| **Model** | `BAAI/bge-small-en-v1.5` | Small but high-quality English embedding model (fast + accurate). |
| **Vector Store** | FAISS | Lightweight and fast for large movie datasets. |
| **LLM Provider** | Groq (Llama 3.1-8B Instant) | Excellent latency + accuracy balance for natural responses. |
| **UI Framework** | Gradio | Simplifies web deployment and supports custom CSS for chat-style UI. |
| **Web Search** | Serper API | Adds dynamic context for more recent or missing info. |
| **Dataset Handling** | Kaggle API | Automatically fetches dataset without uploading large CSVs. |

---

## Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/maxijmoff/movie-chatbot.git
cd movie-chatbot
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Add API Keys
The chatbot requires three keys:
1. **Groq API Key** from [`https://console.groq.com/`](https://console.groq.com/)
2. **Serper API Key** from [`https://serper.dev/`](https://serper.dev/)
3. **Kaggle API Key** from [`https://www.kaggle.com/settings`](https://www.kaggle.com/settings)

**Add Kaggle credentials:**
1. Download 'kaggle.json' from Kaggle.
2. Place it in:
```bash
~/.kaggle/kaggle.json
```
3. Set permissions:
```bash
chmod 600 ~/.kaggle/kaggle.json
```

### 4. Run the Application
```bash
python app.py
```
