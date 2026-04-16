import os
import re
import time
import json
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np

import faiss  # faiss-cpu
from openai import OpenAI

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except Exception:
    YouTubeTranscriptApi = None

# PDF text extraction 
try:
    import pdfplumber
except Exception:
    pdfplumber = None

# Configuration - Optimized for better retrieval
DEFAULT_LINKS_FILE = "palav_url_links.txt"
CHUNK_CHARS = 3000        
CHUNK_OVERLAP = 500       
TOP_K = 10                
MIN_SIM_THRESHOLD = 0.22  

# Models
EMBED_MODEL = "text-embedding-3-small"  
ANSWER_MODEL_DEFAULT = "gpt-4o-mini"

# Persistence
INDEX_DIR = ".palav_index_cache"  

@dataclass
class DocChunk:
    id: str
    source_url: str
    title: str
    text: str

# --- Utilities ---
def normalize_whitespace(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def is_pdf_url(url: str) -> bool:
    return url.lower().split("?")[0].endswith(".pdf")

def is_youtube_url(url: str) -> bool:
    u = url.lower()
    return ("youtube.com/watch" in u) or ("youtu.be/" in u)

def extract_youtube_video_id(url: str) -> Optional[str]:
    if "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0].split("&")[0]
    m = re.search(r"[?&]v=([^&]+)", url)
    if m:
        return m.group(1)
    return None

# --- Fetching & Extraction ---
def fetch_html_text(url: str, timeout: int = 20) -> Tuple[str, str]:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else url
    main = soup.find("main") or soup.find("article")
    text = main.get_text("\n", strip=True) if main else soup.get_text("\n", strip=True)
    return title, normalize_whitespace(text)

def fetch_pdf_text(url: str, timeout: int = 30) -> Tuple[str, str]:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is not installed. Add it to requirements.txt")
    
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    
    from io import BytesIO
    title = url
    pages_text = []
    
    with pdfplumber.open(BytesIO(r.content)) as pdf:
        if pdf.metadata and 'Title' in pdf.metadata:
            title = pdf.metadata['Title']
            
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
                
    return title, normalize_whitespace("\n".join(pages_text))

def fetch_youtube_transcript_text(url: str) -> Tuple[str, str]:
    if YouTubeTranscriptApi is None:
        raise RuntimeError("youtube-transcript-api is not installed.")
    vid = extract_youtube_video_id(url)
    if not vid:
        raise RuntimeError("Could not parse YouTube video id")
    try:
        transcript = YouTubeTranscriptApi.get_transcript(vid, languages=["en"])
    except Exception:
        transcript = YouTubeTranscriptApi.get_transcript(vid)
    text = " ".join([x.get("text", "") for x in transcript])
    return f"YouTube transcript: {vid}", normalize_whitespace(text)

# --- Processing & Indexing ---
def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text: return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        piece = text[start:end].strip()
        if piece: chunks.append(piece)
        if end == n: break
        start = max(0, end - overlap)
    return chunks

def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    faiss.normalize_L2(vecs)
    return vecs

def build_faiss_index(vectors: np.ndarray):
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index

def load_allowed_urls(path: str) -> List[str]:
    if not os.path.exists(path): return []
    urls = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            found = re.findall(r"https?://\S+", line)
            urls.extend(found if found else [line])
    return list(dict.fromkeys(urls))

def ensure_index_dir():
    os.makedirs(INDEX_DIR, exist_ok=True)

def index_key(links_file: str) -> str:
    if not os.path.exists(links_file): return "missing_links_file"
    h = file_sha1(links_file)
    settings = f"{EMBED_MODEL}|{CHUNK_CHARS}|{CHUNK_OVERLAP}"
    return sha1(h + "|" + settings)

def index_paths(key: str) -> Dict[str, str]:
    return {
        "faiss": os.path.join(INDEX_DIR, f"{key}.faiss"),
        "vectors": os.path.join(INDEX_DIR, f"{key}.npy"),
        "chunks": os.path.join(INDEX_DIR, f"{key}.chunks.jsonl"),
        "report": os.path.join(INDEX_DIR, f"{key}.report.json"),
        "meta": os.path.join(INDEX_DIR, f"{key}.meta.json"),
    }

def index_exists(paths: Dict[str, str]) -> bool:
    return all(os.path.exists(paths[p]) for p in ["faiss", "vectors", "chunks", "meta"])

def save_index(paths: Dict[str, str], index, vectors: np.ndarray, chunks: List[DocChunk], report: Dict):
    faiss.write_index(index, paths["faiss"])
    np.save(paths["vectors"], vectors)
    with open(paths["chunks"], "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
    with open(paths["report"], "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump({"embed_model": EMBED_MODEL, "chunk_chars": CHUNK_CHARS, "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

def load_index(paths: Dict[str, str]) -> Tuple[object, np.ndarray, List[DocChunk], Dict]:
    index = faiss.read_index(paths["faiss"])
    vectors = np.load(paths["vectors"])
    chunks = []
    with open(paths["chunks"], "r", encoding="utf-8") as f:
        for line in f: chunks.append(DocChunk(**json.loads(line)))
    report = {}
    if os.path.exists(paths["report"]):
        with open(paths["report"], "r", encoding="utf-8") as f: report = json.load(f)
    return index, vectors, chunks, report

def ingest_sources(links_file: str) -> Tuple[List[DocChunk], Dict]:
    urls = load_allowed_urls(links_file)
    report = {"total_urls": len(urls), "ok": 0, "failed": []}
    chunks = []
    for url in urls:
        try:
            if is_youtube_url(url): title, text = fetch_youtube_transcript_text(url)
            elif is_pdf_url(url): title, text = fetch_pdf_text(url)
            else: title, text = fetch_html_text(url)
            if len(text) < 200: raise RuntimeError("Text too short.")
            for i, piece in enumerate(chunk_text(text)):
                chunks.append(DocChunk(id=sha1(url + f"::{i}"), source_url=url, title=title, text=piece))
            report["ok"] += 1
        except Exception as e:
            report["failed"].append({"url": url, "error": repr(e)})
        time.sleep(0.1)
    return chunks, report

def build_or_load(links_file: str, api_key: str, force_rebuild: bool = False):
    ensure_index_dir()
    key = index_key(links_file)
    paths = index_paths(key)
    if (not force_rebuild) and index_exists(paths):
        index, vectors, chunks, report = load_index(paths)
        return index, vectors, chunks, report, key, paths, True
    chunks, report = ingest_sources(links_file)
    client = OpenAI(api_key=api_key)
    vectors = embed_texts(client, [c.text for c in chunks])
    index = build_faiss_index(vectors)
    save_index(paths, index, vectors, chunks, report)
    return index, vectors, chunks, report, key, paths, False

# --- Chat & Retrieval Logic ---
SYSTEM_INSTRUCTIONS = """You are a breastfeeding education chatbot for an NGO.

RULE 1: If the question is about BREASTFEEDING or MATERNAL HEALTH, you MUST provide an answer. 
- You are encouraged to respond in the language used by the user (e.g., Burmese, Spanish, etc.) by translating the relevant information.
- First, check the provided SOURCES for the answer. 
- If the SOURCES do not have the specific answer, use your general training data. 
- If you use general training data, you MUST start your response with "EXTERNAL_KNOWLEDGE:".

RULE 2: If the question is totally UNRELATED to breastfeeding (e.g., broken bones, car repair, history), reply: 
"I do not have required information. Please try different question"

Keep the tone parent-friendly and practical."""

def retrieve(client: OpenAI, index, chunks: List[DocChunk], query: str, top_k: int = TOP_K):
    qvec = embed_texts(client, [query])
    sims, idxs = index.search(qvec, top_k)
    results = []
    for score, i in zip(sims[0], idxs[0]):
        if i != -1: results.append((float(score), chunks[i]))
    return results

def make_answer(client: OpenAI, model: str, question: str, retrieved: List[Tuple[float, DocChunk]]) -> str:
    fallback_text = "I do not have required information. Please try different question"
    
    # Prepare Context
    context_blocks = [f"URL: {ch.source_url}\nSNIPPET: {ch.text}" for _, ch in retrieved]

    # Enhanced instructions to filter for truly used sources
    FILTER_INSTRUCTIONS = SYSTEM_INSTRUCTIONS + (
        "\nAt the end of your response, if you used information from the provided SOURCES, "
        "provide a list of those specific URLs after the tag 'USED_URLS:'. Example: USED_URLS: [\"url1\", \"url2\"]"
    )

    # Request LLM Answer
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": FILTER_INSTRUCTIONS},
            {"role": "user", "content": f"QUESTION: {question}\n\nSOURCES:\n" + "\n".join(context_blocks)}
        ],
        temperature=0
    )
    full_content = resp.choices[0].message.content.strip()

    # 1. Handle strict rejection
    if fallback_text.lower() in full_content.lower():
        return fallback_text

    # 2. Extract answer and used URLs
    if "USED_URLS:" in full_content:
        answer, url_part = full_content.split("USED_URLS:", 1)
        try:
            # Simple list cleaning if not valid JSON
            used_urls = re.findall(r'https?://\S+', url_part.replace('"', '').replace('[', '').replace(']', ''))
        except:
            used_urls = []
    else:
        answer = full_content
        used_urls = []

    # 3. Handle External Knowledge (Internet Fallback)
    if "EXTERNAL_KNOWLEDGE:" in answer:
        clean_answer = answer.replace("EXTERNAL_KNOWLEDGE:", "").strip()
        footer = "\n\n---\n*Note: This topic is not included in the manual; information is being provided from the internet.*"
        return clean_answer + footer

    # 4. Handle Source-Based Answer with filtered links
    if used_urls:
        source_list = "\n".join([f"- {u.rstrip(',').rstrip('.')}" for u in list(dict.fromkeys(used_urls))])
        return f"{answer.strip()}\n\nAdditional Resources:\n{source_list}"
    
    return answer.strip()

# --- Streamlit UI ---
st.set_page_config(page_title="Palav Breastfeeding Chatbot", layout="centered")
st.title("Palav Breastfeeding Userguide")

api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
if not api_key:
    st.error("OPENAI_API_KEY is not set.")
    st.stop()

ADMIN_MODE = str(st.secrets.get("ADMIN_MODE", "false")).lower() in {"true", "1"}

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to Palav Breastifeeding Userguide. You can ask me any breastfeeding question. \n\n Disclaimer: The information that I provide is for education purpose and is not meant to replace medical advice. I am not HIPPA compliant, please do not enter PII or PHI information such as name, SSN, address, billing, medical record etc."}
    ]

try:
    index, vectors, chunks, report, key, paths, loaded = build_or_load(DEFAULT_LINKS_FILE, api_key)
except Exception as e:
    st.error(f"Failed to load index: {e}")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            client = OpenAI(api_key=api_key)
            matches = retrieve(client, index, chunks, prompt)
            ans = make_answer(client, ANSWER_MODEL_DEFAULT, prompt, matches)
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})