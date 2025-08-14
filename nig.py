import io
import os
import re
import math
from typing import List, Tuple
from collections import Counter

import streamlit as st

# Optional deps: rank_bm25
try:
    from rank_bm25 import BM25Okapi
    HAS_RANK_BM25 = True
except Exception:
    BM25Okapi = None
    HAS_RANK_BM25 = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except Exception:
    HAS_PYPDF2 = False

# Gemini API client (online LLM)
try:
    from openai import OpenAI  # Assuming Gemini uses OpenAI-compatible API
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False

# ---------------------------
# Page Config & Styles
# ---------------------------
st.set_page_config(page_title="📚 StudyMate BM25 + Gemini", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3, h4 { color: #ffffff; text-shadow: 2px 2px 4px rgba(0,0,0,0.18); font-weight: 800; }
.answer-box, .passage-box, .notice-box {
    background-color: rgba(255,255,255,0.95);
    padding: 16px;
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.10);
    font-size: 16px;
    line-height: 1.6;
    color: #222;
}
.badge {
    display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; background:#111; color:#fff; margin-left:8px;
}
.score-chip {
    display:inline-block; background:#eee; padding:2px 8px; border-radius:999px; font-size:12px; margin-left:6px;
}
</style>
""", unsafe_allow_html=True)

st.title("📚 StudyMate — Offline NCERT Q&A (BM25) + Gemini Online LLM")
st.write("Upload PDFs and ask questions. If the answer is not in the PDFs, Gemini LLM will try to answer online.")

# ---------------------------
# Sidebar Controls
# ---------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_files = st.file_uploader("Upload NCERT PDFs", type=["pdf"], accept_multiple_files=True)
    max_words = st.slider("Max words per chunk", 50, 400, 150, 10)
    top_k = st.slider("Top passages", 1, 10, 5)

    st.markdown("---")
    st.subheader("🤖 Gemini API")
    enable_gemini = st.checkbox("Enable Gemini LLM", value=False)
    gemini_api_key_input = st.text_input("Enter Gemini API Key", type="password", placeholder="Paste your key here")
    min_score = st.slider("BM25 'in-PDF' threshold", 0.0, 10.0, 0.6, 0.05)

# ---------------------------
# PDF Processing
# ---------------------------
def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    if not HAS_PYPDF2:
        return ""
    text_parts: List[str] = []
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)

def clean_text(s: str) -> str:
    s = s.replace("\r", " ").replace("\t", " ")
    s = re.sub(r"\u00A0", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def split_into_paragraphs(text: str) -> List[str]:
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if p.strip()]

def chunk_paragraphs(paragraphs: List[str], max_words: int = 150) -> List[str]:
    chunks: List[str] = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= max_words:
            chunks.append(para)
        else:
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i : i + max_words]))
    return chunks

def tokenize(text: str) -> List[str]:
    text = clean_text(text).lower()
    return re.findall(r"\b[a-z0-9]+\b", text)

# ---------------------------
# Simple BM25 fallback
# ---------------------------
class SimpleBM25:
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / max(1, len(self.doc_len))
        self.df = Counter()
        for doc in corpus:
            for word in set(doc):
                self.df[word] += 1
        N = len(corpus)
        self.idf = {w: math.log(1 + (N - f + 0.5) / (f + 0.5)) for w, f in self.df.items()}

    def score(self, query_tokens: List[str], index: int) -> float:
        score = 0.0
        if index >= len(self.corpus):
            return score
        doc = self.corpus[index]
        tf = Counter(doc)
        for term in query_tokens:
            if term not in self.idf:
                continue
            numer = tf[term] * (self.k1 + 1.0)
            denom = tf[term] + self.k1 * (1 - self.b + self.b * len(doc) / max(1.0, self.avgdl))
            score += self.idf[term] * numer / max(1e-9, denom)
        return score

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        return [self.score(query_tokens, i) for i in range(len(self.corpus))]

# ---------------------------
# Build Index
# ---------------------------
@st.cache_data
def build_index(files, max_words: int) -> Tuple[List[str], List[List[str]]]:
    all_chunks: List[str] = []
    tokenized_chunks: List[List[str]] = []
    if not files:
        return all_chunks, tokenized_chunks
    for f in files:
        try:
            raw = f.read()
            text = extract_text_from_pdf_bytes(raw)
        except Exception:
            text = ""
        paragraphs = split_into_paragraphs(text)
        chunks = chunk_paragraphs(paragraphs, max_words)
        cleaned = [clean_text(c) for c in chunks]
        all_chunks.extend(cleaned)
        tokenized_chunks.extend([tokenize(c) for c in cleaned])
    return all_chunks, tokenized_chunks

all_chunks, tokenized_chunks = build_index(uploaded_files, max_words)

# ---------------------------
# Build BM25
# ---------------------------
@st.cache_data
def get_bm25(tokenized_chunks: List[List[str]]):
    if not tokenized_chunks:
        return None
    if HAS_RANK_BM25:
        return BM25Okapi(tokenized_chunks)
    return SimpleBM25(tokenized_chunks)

bm25 = get_bm25(tokenized_chunks)

# ---------------------------
# Gemini LLM call
# ---------------------------
def call_gemini(api_key: str, prompt: str) -> str:
    if not HAS_GEMINI or not api_key:
        return "Gemini LLM unavailable or API key missing."
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gemini-1.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Gemini LLM error: {e}"

# ---------------------------
# Query input & answer
# ---------------------------
query = st.text_input("🔎 Ask a question:")

if query:
    if not bm25:
        st.info("Upload PDFs first.")
    else:
        q_tokens = tokenize(query)
        scores = bm25.get_scores(q_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = ranked[:top_k]
        top_score = scores[top_indices[0]] if top_indices else 0.0
        outside = (top_score < min_score)

        st.subheader("📌 Relevant Passages")
        if not top_indices or all(scores[i] <= 0 for i in top_indices):
            st.markdown("<div class='passage-box'>No strong matches in the PDFs.</div>", unsafe_allow_html=True)
        else:
            for r, idx in enumerate(top_indices, 1):
                passage = all_chunks[idx]
                highlighted = passage
                for term in set(q_tokens):
                    highlighted = re.sub(rf"\b({re.escape(term)})\b", r"**\1**", highlighted, flags=re.IGNORECASE)
                st.markdown(f"<div class='passage-box'><b>{r}.</b> {highlighted} "
                            f"<span class='score-chip'>score: {scores[idx]:.3f}</span></div>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🧠 Answer")

        answer = None
        if not outside:
            answer = all_chunks[top_indices[0]]
            st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)
            st.caption("Answer derived from your PDFs.")
        elif outside and enable_gemini and gemini_api_key_input:
            answer = call_gemini(gemini_api_key_input, query)
            st.markdown(f"<div class='answer-box'>{answer} <span class='badge'>Gemini LLM</span></div>", unsafe_allow_html=True)
            st.caption("No strong PDF match; answered online by Gemini LLM.")
        else:
            st.markdown("<div class='answer-box'>Not in PDFs. Enable Gemini API to get an online answer.</div>", unsafe_allow_html=True)
