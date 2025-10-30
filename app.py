from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import requests, os

app = FastAPI()


def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        r = requests.get(url)
        with open(filename, "wb") as f:
            f.write(r.content)

download_file(
    "https://huggingface.co/datasets/kunal957/search/resolve/main/wiki_index.faiss",
    "wiki_index.faiss"
)
download_file(
    "https://huggingface.co/datasets/kunal957/search/resolve/main/wikipedia_abstracts.pkl",
    "wikipedia_abstracts.pkl"
)


model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("wiki_index.faiss")
df = pd.read_pickle("wikipedia_abstracts.pkl")


@app.post("/search")
def search_post(query: str):
    q_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_vec, k=5)
    results = [
        {
            "rank": rank + 1,
            "title": df.iloc[idx]["title"],
            "abstract": df.iloc[idx]["abstract_clean"][:250] + "…"
        }
        for rank, idx in enumerate(I[0])
    ]
    return {"query": query, "results": results}

@app.get("/search")
def search_get(query: str):
    q_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_vec, k=5)
    results = [
        {
            "rank": rank + 1,
            "title": df.iloc[idx]["title"],
            "abstract": df.iloc[idx]["abstract_clean"][:250] + "…"
        }
        for rank, idx in enumerate(I[0])
    ]
    return {"query": query, "results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
