import re
import requests
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from kaggle_secrets import UserSecretsClient

# 1. API Key Setup
user_secrets = UserSecretsClient()
GROQ_API_KEY = user_secrets.get_secret("GROQ_API_KEY")
SERPER_API_KEY = user_secrets.get_secret("SERPER_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# 2. Load & Clean Dataset
df = pd.read_csv("/kaggle/input/wikipedia-movie-plots/wiki_movie_plots_deduped.csv")

df = df.rename(columns={
    "Title": "title",
    "Release Year": "year",
    "Origin/Ethnicity": "origin",
    "Director": "director",
    "Cast": "cast",
    "Genre": "genre",
    "Wiki Page": "link",
    "Plot": "plot"
})

df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
df = df.fillna("Unknown")
df_cleaned = df.dropna(subset=["title", "plot"]).drop_duplicates(subset=["title", "year"])

# 3. Chunking Plot
def split_plot_into_chunks(text, max_words=150, overlap=30):
    if not isinstance(text, str) or not text.strip():
        return []
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk_words = words[i:i + max_words]
        chunks.append(" ".join(chunk_words))
        i += max_words - overlap
    return chunks

docs, metas = [], []

for i, row in df_cleaned.iterrows():
    title = str(row["title"]).strip()
    year = int(row["year"])
    origin = str(row["origin"]).strip()
    genre = str(row["genre"]).strip()
    director = str(row["director"]).strip()
    cast = str(row["cast"]).strip()
    plot = str(row["plot"]).strip()

    meta_text = f"Title: {title}\nYear: {year}\nOrigin: {origin}\nGenre: {genre}\nDirector: {director}\nCast: {cast}"
    docs.append(meta_text)
    metas.append({
        "movie_id": i,
        "chunk_type": "meta",
        "title": title,
        "year": year,
        "origin": origin,
        "genre": genre,
        "director": director,
        "cast": cast
    })

    for ch in split_plot_into_chunks(plot):
        docs.append(ch)
        metas.append({
            "movie_id": i,
            "chunk_type": "plot",
            "title": title,
            "year": year,
            "origin": origin,
            "genre": genre,
            "director": director,
            "cast": cast
        })

# 4. Embedding Model & FAISS
EMB_MODEL = "BAAI/bge-small-en-v1.5"
emb_model = SentenceTransformer(EMB_MODEL)

embeddings = emb_model.encode(
    docs, batch_size=64, show_progress_bar=True,
    convert_to_numpy=True, normalize_embeddings=True
).astype("float32")

d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embeddings)

# 5. Helper Functions
def normalize_title(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_tokens(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9, /]+", "", s)
    tokens = re.split(r"[,/ ]+", s)
    return [t for t in tokens if t]

def match_filter(meta, genre=None, origin=None, min_year=None, max_year=None):
    y = meta["year"]
    genre_tokens = normalize_tokens(meta["genre"] or "")
    origin_tokens = normalize_tokens(meta["origin"] or "")
    if min_year and y < min_year: return False
    if max_year and y > max_year: return False
    if genre and genre.lower().strip() not in genre_tokens: return False
    if origin and origin.lower().strip() not in origin_tokens: return False
    return True

def title_boost(query, candidates):
    q = normalize_title(query)
    for c in candidates:
        title = normalize_title(c["meta"]["title"])
        if q == title:
            c["score"] += 5.0
        elif q in title and len(q.split()) > 1:
            c["score"] += 2.5
        elif len(set(q.split()) & set(title.split())) >= 2:
            c["score"] += 1.0
    return candidates

# 6. Search Functions
def search_chunks(query, top_k=200):
    q_emb = emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idx = index.search(q_emb, top_k)
    results = []
    for s, i_doc in zip(scores[0], idx[0]):
        results.append({
            "score": float(s),
            "chunk_id": int(i_doc),
            "text": docs[i_doc],
            "meta": metas[i_doc],
        })
    return results

def search_movies(query, top_k_movies=5, top_k_chunks=200, genre=None, origin=None, min_year=None, max_year=None):
    chunks = search_chunks(query, top_k=top_k_chunks)
    movie_map = {}
    for ch in chunks:
        meta = ch["meta"]
        movie_id = meta["movie_id"]
        if not match_filter(meta, genre, origin, min_year, max_year):
            continue
        if movie_id not in movie_map or ch["score"] > movie_map[movie_id]["score"]:
            movie_map[movie_id] = {"movie_id": movie_id, "score": ch["score"], "text": ch["text"], "meta": meta}
    if not movie_map:
        for ch in chunks:
            meta = ch["meta"]
            movie_id = meta["movie_id"]
            if movie_id not in movie_map or ch["score"] > movie_map[movie_id]["score"]:
                movie_map[movie_id] = {"movie_id": movie_id, "score": ch["score"], "text": ch["text"], "meta": meta}
    movie_list = list(movie_map.values())
    movie_list = title_boost(query, movie_list)
    return sorted(movie_list, key=lambda x: x["score"], reverse=True)[:top_k_movies]

# 7. Web Search (Serper)
def clean_web_text(text):
    return re.sub(r"\s+", " ", text).strip() if text else ""

def web_search_serper(query):
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": 5}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=7)
        data = resp.json()
    except Exception:
        return None
    collected = []
    for source in ["answerBox", "knowledgeGraph", "organic"]:
        items = data.get(source)
        if not items:
            continue
        if isinstance(items, dict):
            items = [items]
        for it in items:
            snippet = clean_web_text(it.get("snippet", ""))
            title = clean_web_text(it.get("title", ""))
            desc = clean_web_text(it.get("description", ""))
            combined = "\n".join(filter(None, [title, snippet, desc]))
            if combined:
                collected.append(combined)
    return "\n\n".join(collected[:5]) if collected else None

# 8. Prompt Builder & Groq
def build_prompt(query, movies, web_context=None):
    movie_blocks = []
    for m in movies:
        meta = m["meta"]
        block = f"""
Judul: {meta['title']}
Tahun: {meta['year']}
Asal: {meta['origin']}
Genre: {meta['genre']}
Director: {meta['director']}

Ringkasan:
{m['text']}
""".strip()
        movie_blocks.append(block)
    ctx_movies = "\n\n---\n\n".join(movie_blocks)
    ctx_web = web_context.strip() if web_context else "Tidak ada konteks tambahan."
    return f"""
Kamu adalah teman ngobrol santai yang paham banyak film.
Jawablah hanya berdasarkan informasi di bawah ini.

KONTEKS FILM:
{ctx_movies}

KONTEKS WEB:
{ctx_web}

Pertanyaan:
{query}

Jawaban:
"""

def groq_generate(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.35,
        max_tokens=400,
    )
    return response.choices[0].message.content

# 9. Chatbot Function
def movie_chatbot(query, top_k_movies=5, genre=None, origin=None, min_year=None, max_year=None, score_threshold=0.25):
    query = query.strip()
    movies = search_movies(query=query, top_k_movies=top_k_movies, genre=genre, origin=origin,
                           min_year=min_year, max_year=max_year)
    best_score = movies[0]["score"] if movies else 0.0
    web_ctx = None if best_score >= score_threshold else web_search_serper(query)
    prompt = build_prompt(query, movies, web_context=web_ctx)
    answer = groq_generate(prompt)
    return answer, movies
