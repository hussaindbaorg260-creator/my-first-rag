import re
import time
from collections import defaultdict

import chromadb
import requests
import os
from google import genai
import streamlit as st


# =========================================================
# STREAMLIT PAGE
# =========================================================
st.set_page_config(
    page_title="Enterprise Knowledge Assistant | RAG V6.11.5",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PROFESSIONAL GUI THEME - V6.8
# UI only. RAG backend logic remains unchanged.
# =========================================================
st.markdown("""
<style>
.stApp { background: linear-gradient(180deg,#f4f8fd 0%,#ffffff 48%); }
.block-container { max-width:1180px; padding-top:1.25rem; padding-bottom:3rem; }

[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#071b3d 0%,#0b3269 100%);
    border-right:1px solid rgba(255,255,255,.08);
}
[data-testid="stSidebar"] * { color:#f7fbff; }
[data-testid="stSidebar"] .stButton>button {
    width:100%; min-height:42px; border-radius:10px;
    border:1px solid rgba(255,255,255,.25);
    background:rgba(255,255,255,.10); color:white; font-weight:700;
}
[data-testid="stSidebar"] .stButton>button:hover {
    background:rgba(255,255,255,.18); border-color:rgba(255,255,255,.5);
}

.rag-hero {
    background:linear-gradient(120deg,#071b3d 0%,#0b4da2 55%,#087cc4 100%);
    border-radius:20px; padding:30px 34px; color:white; margin-bottom:20px;
    box-shadow:0 14px 34px rgba(7,27,61,.16);
}
.rag-kicker {
    display:inline-block; padding:5px 11px; border-radius:999px;
    background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.18);
    font-size:.78rem; font-weight:700; letter-spacing:.04em; margin-bottom:12px;
}
.rag-hero h1 { color:white!important; font-size:2.25rem; margin:0 0 8px 0; }
.rag-hero p { color:#dcecff!important; font-size:1.02rem; margin:0; }
.rag-badges { display:flex; flex-wrap:wrap; gap:8px; margin-top:18px; }
.rag-badge {
    padding:7px 11px; border-radius:999px; background:rgba(255,255,255,.11);
    border:1px solid rgba(255,255,255,.18); font-size:.82rem; font-weight:600;
}

.feature-grid {
    display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:20px;
}
.feature-card {
    background:white; border:1px solid #dbe7f5; border-radius:14px;
    padding:15px 16px; box-shadow:0 5px 16px rgba(15,48,89,.05);
}
.feature-title { color:#102a56; font-weight:750; margin:5px 0 3px; }
.feature-copy { color:#65758b; font-size:.82rem; line-height:1.35; }

.welcome-panel {
    background:#edf5ff; border:1px solid #d7e8fb; border-radius:14px;
    padding:14px 18px; margin-bottom:18px; color:#173a68;
}
.sidebar-card {
    background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.12);
    border-radius:12px; padding:13px 14px; margin:10px 0;
}
.sidebar-label { color:#a9c6e9!important; font-size:.73rem; text-transform:uppercase; }
.sidebar-value { color:white!important; font-weight:700; margin-bottom:8px; }
.system-ready {
    background:rgba(27,194,111,.14); border:1px solid rgba(66,224,143,.28);
    border-radius:12px; padding:12px 14px; margin:12px 0; color:#dffff0!important;
}
[data-testid="stChatMessage"] {
    background:white; border:1px solid #e1e9f2; border-radius:15px;
    padding:8px 12px; margin-bottom:10px; box-shadow:0 3px 12px rgba(17,47,82,.04);
}
div[data-testid="stExpander"] {
    border:1px solid #dce6f2; border-radius:12px; background:#fbfdff;
}

.developer-credit {
    margin-top:16px;
    padding-top:13px;
    border-top:1px solid rgba(255,255,255,.18);
    color:#d9eaff;
    font-size:.84rem;
    letter-spacing:.01em;
}
.developer-credit strong {
    color:#ffffff;
    font-size:.94rem;
    letter-spacing:.03em;
}
.footer-credit {
    margin:28px 0 76px 0;
    padding:14px 18px;
    border-top:1px solid #dce6f2;
    text-align:center;
    color:#6b7d92;
    font-size:.82rem;
}
.footer-credit strong { color:#164d8f; }
.evidence-strip {
    display:flex;
    align-items:center;
    gap:10px;
    flex-wrap:wrap;
    margin:10px 0 4px 0;
}
.evidence-badge {
    display:inline-block;
    padding:5px 10px;
    border-radius:999px;
    font-size:.76rem;
    font-weight:800;
    letter-spacing:.04em;
}
.evidence-strong { background:#e8f8ef; color:#137a46; border:1px solid #bce8cf; }
.evidence-partial { background:#fff6df; color:#956400; border:1px solid #f0d68e; }
.evidence-weak { background:#fff0f0; color:#a93b3b; border:1px solid #efc2c2; }
.source-card {
    background:#f8fbff;
    border:1px solid #dce8f5;
    border-left:4px solid #2477c9;
    border-radius:10px;
    padding:9px 12px;
    margin:6px 0;
    color:#294867;
    font-size:.88rem;
}

@media(max-width:900px) {
    .feature-grid { grid-template-columns:repeat(2,1fr); }
    .rag-hero h1 { font-size:1.7rem; }
}

/* V6.9.6.2 FINAL UI POLISH — UI ONLY */
.block-container {max-width:1280px!important;padding-top:2.4rem!important;padding-bottom:7rem!important;}
.rag-hero {border-radius:32px!important;padding:34px 38px 30px!important;margin-top:8px!important;margin-bottom:22px!important;overflow:hidden;position:relative;}
.rag-kicker {padding:8px 16px!important;margin:0 0 20px!important;border-radius:999px!important;font-size:.82rem!important;line-height:1.2!important;white-space:nowrap;}
.rag-hero h1 {font-size:2.45rem!important;line-height:1.12!important;margin:0 0 13px!important;}
.rag-hero p {font-size:1.02rem!important;line-height:1.55!important;max-width:850px;}
.rag-badges {gap:10px!important;margin-top:22px!important;}
.rag-badge {padding:8px 13px!important;border-radius:999px!important;white-space:nowrap;}
.developer-credit {margin-top:22px!important;padding-top:18px!important;font-size:.92rem!important;line-height:1.55!important;}
.developer-credit strong {color:#ffd84d!important;font-size:1.12rem!important;font-weight:800!important;}
[data-testid="stSidebar"] {
    padding-bottom:2rem!important;
}

/* V6.13 - allow long sidebar content such as document uploader to scroll */
[data-testid="stSidebar"] > div:first-child {
    height:100vh!important;
    overflow-y:auto!important;
    overflow-x:hidden!important;
    padding-bottom:4rem!important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] small {color:#d9e9ff!important;opacity:1!important;}
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 {color:#fff!important;}
.sidebar-label {color:#bcd4f3!important;opacity:1!important;}
.sidebar-value {color:#fff!important;opacity:1!important;}
.system-ready,.system-ready * {color:#eafff3!important;opacity:1!important;}
.feature-card {border-radius:18px!important;min-height:118px;}
.welcome-panel {border-radius:18px!important;}
.footer-credit {margin-bottom:90px!important;}
@media(max-width:900px) {
.block-container {padding-top:2rem!important;}
.rag-hero {padding:26px 22px!important;border-radius:24px!important;}
.rag-kicker {white-space:normal;}
.rag-hero h1 {font-size:1.85rem!important;}
}


/* V6.9.6.2 VIEWPORT + CHAT COMPOSER POLISH — UI ONLY */
.block-container {padding-top:1.45rem!important;padding-bottom:8.5rem!important;}
.rag-hero {border-radius:32px!important;padding:24px 38px 22px!important;margin-top:0!important;margin-bottom:16px!important;}
.rag-kicker {margin-bottom:13px!important;padding:7px 14px!important;}
.rag-hero h1 {font-size:2.25rem!important;margin-bottom:8px!important;}
.rag-hero p {line-height:1.4!important;}
.rag-badges {margin-top:14px!important;}
.rag-badge {padding:7px 12px!important;}
.developer-credit {margin-top:15px!important;padding-top:12px!important;line-height:1.4!important;}
.feature-grid {margin-bottom:14px!important;}
.feature-card {min-height:96px!important;padding:12px 15px!important;}
.welcome-panel {padding:11px 16px!important;margin-bottom:88px!important;}
[data-testid="stChatInput"] {border-radius:18px!important;}


/* =========================================================
   V6.9.6.2 TOP KICKER CLIPPING FIX — UI ONLY
   ========================================================= */
.block-container {
    padding-top: 3.15rem !important;
}
.rag-hero {
    margin-top: 0.65rem !important;
}
.rag-kicker {
    position: relative;
    top: 0 !important;
    margin-top: 0 !important;
}


/* =========================================================
   V6.13 - SIDEBAR UPLOAD CONTROL VISIBILITY FIX
   Keep the entire sidebar vertically scrollable so upload,
   preview and Process/Add controls remain reachable.
   ========================================================= */

[data-testid="stSidebar"] {
    overflow: hidden !important;
}

[data-testid="stSidebar"] > div:first-child {
    height: 100vh !important;
    max-height: 100vh !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding-bottom: 8rem !important;
}

/* Prevent the upload preview from consuming excessive space */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    max-width: 100% !important;
}

/* Keep sidebar buttons fully visible */
[data-testid="stSidebar"] .stButton {
    width: 100% !important;
    margin-bottom: 0.75rem !important;
}


/* =========================================================
   V6.13 - PREVIEW EXPANDER FONT VISIBILITY FIX
   ========================================================= */

/* Preview/expander header text */
[data-testid="stSidebar"] [data-testid="stExpander"] summary,
[data-testid="stSidebar"] [data-testid="stExpander"] summary *,
[data-testid="stSidebar"] [data-testid="stExpander"] summary p {
    color: #123B6D !important;
    font-weight: 600 !important;
}

/* Expander arrow/icon */
[data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
    fill: #123B6D !important;
    color: #123B6D !important;
}


/* =========================================================
   V6.13 - PROFESSIONAL SIDEBAR UI
   UI/CSS ONLY - NO RAG OR DATABASE LOGIC CHANGES
   ========================================================= */

/* ---------- File uploader white panel ---------- */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: #F8FBFF !important;
    border: 1px solid #D7E3F4 !important;
    border-radius: 12px !important;
    padding: 0.85rem !important;
}

/* ---------- Upload button ---------- */
[data-testid="stSidebar"]
[data-testid="stFileUploaderDropzone"] button {
    background: #EAF3FF !important;
    color: #123B6D !important;
    border: 1px solid #9FC3EE !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

/* Force button child text/icons to same readable color */
[data-testid="stSidebar"]
[data-testid="stFileUploaderDropzone"] button * {
    color: #123B6D !important;
    opacity: 1 !important;
}

/* ---------- File-size / PDF help text ---------- */
[data-testid="stSidebar"]
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stSidebar"]
[data-testid="stFileUploaderDropzone"] small *,
[data-testid="stSidebar"]
[data-testid="stFileUploaderDropzone"] span {
    color: #526B89 !important;
    opacity: 1 !important;
}

/* ---------- Uploaded filename ---------- */
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {
    background: #F4F8FD !important;
    border-radius: 8px !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] *,
[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"] {
    color: #173B67 !important;
    opacity: 1 !important;
}

/* ---------- Preview expander ---------- */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: #F8FBFF !important;
    border: 1px solid #C7D8ED !important;
    border-radius: 10px !important;
}

[data-testid="stSidebar"] [data-testid="stExpander"] summary,
[data-testid="stSidebar"] [data-testid="stExpander"] summary *,
[data-testid="stSidebar"] [data-testid="stExpander"] summary p {
    color: #123B6D !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
    color: #123B6D !important;
    fill: #123B6D !important;
}

/* Expanded preview content */
[data-testid="stSidebar"] [data-testid="stExpanderDetails"],
[data-testid="stSidebar"] [data-testid="stExpanderDetails"] * {
    color: #173B67 !important;
}

/* ---------- Normal sidebar buttons ---------- */
[data-testid="stSidebar"] .stButton > button {
    border-radius: 9px !important;
    font-weight: 600 !important;
}

/* ---------- Process & Add button ---------- */
[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    color: #FFFFFF !important;
}

/* ---------- Sidebar scrollbar ---------- */
[data-testid="stSidebar"] > div:first-child {
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding-bottom: 6rem !important;
}

/* ---------- Better scrollbar appearance ---------- */
[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar {
    width: 6px;
}

[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-thumb {
    background: rgba(110, 155, 210, 0.55);
    border-radius: 10px;
}

</style>

<div class="rag-hero">
  <div class="rag-kicker">LOCAL ENTERPRISE RAG • V6.11.5</div>
  <h1>🧠 Enterprise Knowledge Assistant</h1>
  <p>Evidence-grounded answers from your enterprise documents, with transparent source references.</p>
  <div class="rag-badges">
    <span class="rag-badge">✓ Multi-PDF RAG</span>
    <span class="rag-badge">✓ Evidence Guard</span>
    <span class="rag-badge">✓ Top-3 Retrieval</span>
    <span class="rag-badge">✓ Local LLM</span>
    <span class="rag-badge">✓ No Auto AI Fallback</span>
  </div>
  <div class="developer-credit">
    👤 &nbsp;This project is developed and designed by<br>
    <strong>MD SHAHID HUSSAIN</strong>
  </div>
</div>

<div class="feature-grid">
  <div class="feature-card">📄<div class="feature-title">Document Grounded</div><div class="feature-copy">Answers based on retrieved PDF evidence.</div></div>
  <div class="feature-card">🛡️<div class="feature-title">Evidence Guard</div><div class="feature-copy">Rejects weak or unsupported evidence.</div></div>
  <div class="feature-card">🎯<div class="feature-title">Top-3 Retrieval</div><div class="feature-copy">Focused context for relevant answers.</div></div>
  <div class="feature-card">🔒<div class="feature-title">Local Processing</div><div class="feature-copy">Gemini provides cloud-based LLM inference.</div></div>
</div>

<div class="welcome-panel">
<b>💡 Ask your knowledge base</b><br>
Try “What is ADOP and its phases?”, “What is Oracle Apps R12.2?” or “What is our Mission and Vision?”
</div>
""", unsafe_allow_html=True)


# =========================================================
# CACHED SYSTEM INITIALIZATION (V6.10)
# =========================================================
@st.cache_resource
def load_system_resources():
    # V7 Lightweight:
    # SentenceTransformer/PyTorch is intentionally not loaded.
    # Embeddings are generated through the Gemini API.

    db_client = chromadb.PersistentClient(
        path="chroma_db_gemini_experimental"
    )

    db_collection = db_client.get_collection(
        name="oracle_multi_pdf_gemini"
    )

    all_data = db_collection.get(
        include=["documents", "metadatas"]
    )

    docs = all_data["documents"]
    metas = all_data["metadatas"]

    return db_collection, docs, metas

collection, all_documents, all_metadatas = load_system_resources()


# =========================================================
# QUERY NORMALIZATION
# =========================================================
def normalize_query(question):
    q = question.strip().lower()

    replacements = {
        "vission": "vision",
        "visssion": "vision",
        "vison": "vision",
        "mision": "mission",
        "misssion": "mission",
        "adopp": "adop",
        "r 12.2": "r12.2",
        "r12 2": "r12.2"
    }

    for wrong, correct in replacements.items():
        q = q.replace(wrong, correct)

    return q


# =========================================================
# EXTRACT KEYWORDS
# =========================================================
def extract_keywords(question):
    words = re.findall(
        r"[a-zA-Z0-9_.-]+",
        question.lower()
    )

    stop_words = {
        "what", "is", "are", "the", "a", "an",
        "of", "to", "in", "for", "and", "our",
        "your", "me", "tell", "about", "please",
        "explain", "describe", "any",
        "it", "its", "this", "that", "these", "those",
        "they", "them", "their"
    }

    keywords = []

    for word in words:
        if word not in stop_words and len(word) >= 2:
            keywords.append(word)

    return list(dict.fromkeys(keywords))


# =========================================================
# EXACT TECHNICAL-TERM MATCHING - V6.9.6
# =========================================================
def technical_term_in_text(term, text):
    """
    Match a technical term as a complete token while preserving punctuation
    inside terms such as r12.2. This prevents short terms such as "adop"
    from matching unrelated longer words such as "adoption" or "adopted".
    """
    term = term.strip()
    if not term:
        return False

    pattern = rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


# =========================================================
# DETERMINISTIC CONCEPT EXPANSION - V6.9
# =========================================================
def get_concept_queries(normalized_question):
    """Preserve the original query and add document-language retrieval concepts."""
    queries = [normalized_question]
    q = normalized_question.lower()

    # Project diagnostic: ADOP phase questions map to the EBS document's
    # terminology "online patching cycle phases".
    if (
        re.search(r"\badop\b", q)
        and re.search(r"\bphase(?:s)?\b|\bcycle\b", q)
    ):
        queries.extend([
            "adop online patching cycle phases",
            "online patching cycle phases",
        ])

    # V6.9.6.1: users may type compact "EBS12.2", while the document's
    # strongest semantic terminology is "Oracle E-Business Suite 12.2".
    #
    # Keep this broad alias expansion OUT of ADOP phase/cycle questions.
    # Those questions already have a proven domain-specific expansion above;
    # adding a broad EBS query can dilute the complementary phase evidence.
    is_adop_phase_question = bool(
        re.search(r"\badop\b", q)
        and re.search(r"\bphase(?:s)?\b|\bcycle\b", q)
    )

    if (
        re.search(r"(?<![A-Za-z0-9_])ebs\s*12\.2(?![A-Za-z0-9_])", q)
        and not is_adop_phase_question
    ):
        queries.append("Oracle E-Business Suite 12.2")

    return list(dict.fromkeys(queries))


# =========================================================
# SEMANTIC SEARCH
# =========================================================
def semantic_search(question):
    retrieval_queries = get_concept_queries(question)
    merged = {}

    for retrieval_query in retrieval_queries:
        response = gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=retrieval_query
        )
        embedding = response.embeddings[0].values

        results = collection.query(
            query_embeddings=[embedding],
            n_results=20,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            semantic_score = 1 / (1 + distance)

            key = (
                metadata.get("source"),
                metadata.get("page"),
                document
            )

            candidate = {
                "document": document,
                "metadata": metadata,
                "semantic_score": semantic_score,
                "keyword_score": 0,
                "source_type": "semantic"
            }

            # Same chunk may be found by more than one retrieval query.
            # Keep the strongest semantic score.
            if (
                key not in merged
                or semantic_score > merged[key]["semantic_score"]
            ):
                merged[key] = candidate

    return list(merged.values())


# =========================================================
# KEYWORD SEARCH
# =========================================================
def keyword_search(keywords):
    candidates = []

    if not keywords:
        return candidates

    for document, metadata in zip(
        all_documents,
        all_metadatas
    ):
        text_lower = document.lower()

        matched = []

        for keyword in keywords:
            if technical_term_in_text(keyword, text_lower):
                matched.append(keyword)

        if not matched:
            continue

        keyword_hits = len(matched)
        keyword_score = keyword_hits * 0.35

        if keyword_hits == len(keywords):
            keyword_score += 0.50

        candidates.append({
            "document": document,
            "metadata": metadata,
            "semantic_score": 0,
            "keyword_score": keyword_score,
            "source_type": "keyword"
        })

    return candidates


# =========================================================
# MERGE CANDIDATES
# =========================================================
def merge_candidates(
    semantic_candidates,
    keyword_candidates
):
    merged = {}

    combined = semantic_candidates + keyword_candidates

    for candidate in combined:
        metadata = candidate["metadata"]

        key = (
            metadata.get("source"),
            metadata.get("page"),
            candidate["document"]
        )

        if key not in merged:
            merged[key] = {
                "document": candidate["document"],
                "metadata": metadata,
                "semantic_score": 0,
                "keyword_score": 0,
                "source_types": set()
            }

        merged[key]["semantic_score"] = max(
            merged[key]["semantic_score"],
            candidate["semantic_score"]
        )

        merged[key]["keyword_score"] = max(
            merged[key]["keyword_score"],
            candidate["keyword_score"]
        )

        merged[key]["source_types"].add(
            candidate["source_type"]
        )

    final_candidates = []

    for candidate in merged.values():
        candidate["final_score"] = (
            candidate["semantic_score"]
            + candidate["keyword_score"]
        )

        final_candidates.append(candidate)

    return final_candidates


# =========================================================
# EVIDENCE COVERAGE-AWARE TOP-3 SELECTION - V6.9.2
# =========================================================
def select_intent_aware_top3(normalized_question, ranked_candidates):
    """
    For ADOP phase/cycle questions, keep Top-3 but maximize complementary
    evidence coverage rather than selecting three near-duplicate chunks.

    Slot 1: direct ADOP anchor.
    Slot 2: strongest chunk covering the main phase sequence.
    Slot 3: chunk that adds the most still-missing phase evidence.

    General questions keep the normal ranked Top-3 behavior.
    """
    q = normalized_question.lower()

    is_adop_phase_question = (
        re.search(r"\badop\b", q)
        and re.search(r"\bphase(?:s)?\b|\bcycle\b", q)
    )

    # Generic definition-aware selection.
    # For simple "What is X?" questions, prefer one strong chunk that
    # actually explains/defines X instead of selecting only mention-heavy
    # chunks. This is generic and does not hardcode ADOP or page numbers.
    definition_match = re.match(
        r"^what\s+(?:is|are)\s+(.+?)[?!.]*$",
        q.strip(),
        flags=re.IGNORECASE
    )

    if definition_match and not is_adop_phase_question:
        subject = definition_match.group(1).strip()
        subject_pattern = re.compile(
            r"\b" + re.escape(subject) + r"\b",
            flags=re.IGNORECASE
        )

        def definition_quality(candidate):
            text = candidate["document"]
            lower_text = text.lower()

            if not subject_pattern.search(text):
                return (0, candidate.get("final_score", 0.0))

            score = 0

            # Strong explicit definition patterns.
            if re.search(
                r"\b" + re.escape(subject)
                + r"\b\s+(?:is|are|means|refers\s+to|stands\s+for)\b",
                text,
                flags=re.IGNORECASE
            ):
                score += 8

            # Strong tool / utility / process descriptions.
            if re.search(
                r"\b" + re.escape(subject)
                + r"\b.{0,80}\b(?:tool|utility|process|system|application|feature)\b",
                text,
                flags=re.IGNORECASE | re.DOTALL
            ):
                score += 6

            if re.search(
                r"\b(?:tool|utility|process|system|application|feature)\b.{0,80}\b"
                + re.escape(subject) + r"\b",
                text,
                flags=re.IGNORECASE | re.DOTALL
            ):
                score += 4

            # Purpose / functional description.
            if re.search(
                r"\b" + re.escape(subject)
                + r"\b.{0,100}\b(?:used\s+to|used\s+for|performs|provides|allows|manages|orchestrates)\b",
                text,
                flags=re.IGNORECASE | re.DOTALL
            ):
                score += 6

            # Useful descriptive construction such as:
            # "Patching is performed by running the adop (...) utility."
            if re.search(
                r"\b(?:performed|running|using)\b.{0,100}\b"
                + re.escape(subject)
                + r"\b",
                text,
                flags=re.IGNORECASE | re.DOTALL
            ):
                score += 5

            # Parenthetical expansion close to the subject is strong
            # definition evidence for acronyms / technical terms.
            if re.search(
                r"\b" + re.escape(subject) + r"\b\s*\([^)]{3,100}\)",
                text,
                flags=re.IGNORECASE
            ):
                score += 7

            # Heading/context hints.
            if "tools" in lower_text or "utilities" in lower_text:
                score += 2

            return (score, candidate.get("final_score", 0.0))

        definition_candidates = sorted(
            ranked_candidates,
            key=definition_quality,
            reverse=True
        )

        if definition_candidates:
            best_definition = definition_candidates[0]
            best_quality = definition_quality(best_definition)[0]

            # Only override normal ranking when genuine definition evidence
            # exists. Otherwise preserve the original Top-3 behavior.
            if best_quality > 0:
                selected = [best_definition]

                for candidate in ranked_candidates:
                    if len(selected) >= 3:
                        break
                    if candidate is not best_definition:
                        selected.append(candidate)

                return selected[:3]

        return ranked_candidates[:3]

    if not is_adop_phase_question:
        return ranked_candidates[:3]

    phase_terms = ["prepare", "apply", "finalize", "cutover", "cleanup"]
    selected = []
    selected_keys = set()

    def candidate_key(candidate):
        return (
            candidate["metadata"].get("source"),
            candidate["metadata"].get("page"),
            candidate["document"]
        )

    def add_candidate(candidate):
        key = candidate_key(candidate)
        if key in selected_keys or len(selected) >= 3:
            return False
        selected.append(candidate)
        selected_keys.add(key)
        return True

    def terms_in(candidate):
        text = candidate["document"].lower()
        return {term for term in phase_terms if term in text}

    # Slot 1: direct ADOP anchor/utility evidence.
    anchor_candidates = []
    for candidate in ranked_candidates:
        text = candidate["document"].lower()
        if (
            "adop" in text
            and (
                "ad online patching" in text
                or "orchestrates the entire patching cycle" in text
                or "adop phase=" in text
            )
        ):
            anchor_candidates.append(candidate)

    if anchor_candidates:
        anchor_candidates.sort(
            key=lambda c: c.get("final_score", 0.0),
            reverse=True
        )
        add_candidate(anchor_candidates[0])

    # Slot 2: choose the chunk with the broadest actual phase-name coverage.
    # Retrieval score breaks ties; page numbers are never hardcoded.
    phase_candidates = [
        candidate
        for candidate in ranked_candidates
        if candidate_key(candidate) not in selected_keys
        and terms_in(candidate)
    ]

    if phase_candidates:
        phase_candidates.sort(
            key=lambda c: (
                len(terms_in(c)),
                "online patching cycle" in c["document"].lower(),
                c.get("final_score", 0.0)
            ),
            reverse=True
        )
        add_candidate(phase_candidates[0])

    # Slot 3: greedily add the candidate that contributes the most phase
    # names not already present in selected evidence.
    covered_terms = set()
    for candidate in selected:
        covered_terms.update(terms_in(candidate))

    remaining_candidates = [
        candidate
        for candidate in ranked_candidates
        if candidate_key(candidate) not in selected_keys
    ]

    if remaining_candidates:
        remaining_candidates.sort(
            key=lambda c: (
                len(terms_in(c) - covered_terms),
                "cleanup" in terms_in(c),
                len(terms_in(c)),
                c.get("final_score", 0.0)
            ),
            reverse=True
        )

        best_complement = remaining_candidates[0]
        if terms_in(best_complement) - covered_terms:
            add_candidate(best_complement)

    # Safety fill for unusual chunking/index states.
    for candidate in ranked_candidates:
        if len(selected) >= 3:
            break
        add_candidate(candidate)

    return selected[:3]


# =========================================================
# HYBRID RETRIEVAL - V6.1
# Evidence-aware PDF routing:
# semantic evidence + exact keyword evidence choose the PDF.
# =========================================================
def retrieve_context(question):
    normalized_question = normalize_query(question)
    keywords = extract_keywords(normalized_question)

    # V6.9: expand retrieval vocabulary deterministically, without an LLM call.
    # The original normalized question remains unchanged for Evidence Guard
    # and final grounded answer generation.
    concept_queries = get_concept_queries(normalized_question)

    semantic_candidates = semantic_search(normalized_question)

    retrieval_keywords = []
    for retrieval_query in concept_queries:
        retrieval_keywords.extend(extract_keywords(retrieval_query))
    retrieval_keywords = list(dict.fromkeys(retrieval_keywords))

    keyword_candidates = keyword_search(retrieval_keywords)

    if not semantic_candidates and not keyword_candidates:
        return [], normalized_question

    # Semantic evidence by PDF.
    semantic_source_scores = defaultdict(list)
    for candidate in semantic_candidates:
        source = candidate["metadata"].get("source", "Unknown")
        semantic_source_scores[source].append(
            candidate["semantic_score"]
        )

    semantic_totals = {}
    for source, scores in semantic_source_scores.items():
        scores.sort(reverse=True)
        semantic_totals[source] = sum(scores[:5])

    # Keyword evidence by PDF.
    keyword_source_best = defaultdict(float)
    keyword_source_terms = defaultdict(set)

    for candidate in keyword_candidates:
        source = candidate["metadata"].get("source", "Unknown")

        keyword_source_best[source] = max(
            keyword_source_best[source],
            candidate["keyword_score"]
        )

        text_lower = candidate["document"].lower()
        for keyword in retrieval_keywords:
            if technical_term_in_text(keyword, text_lower):
                keyword_source_terms[source].add(keyword)

    # Evidence-aware PDF routing.
    all_sources = set(semantic_totals) | set(keyword_source_best)
    routing_scores = {}

    for source in all_sources:
        semantic_total = semantic_totals.get(source, 0.0)
        best_keyword_score = keyword_source_best.get(source, 0.0)

        if retrieval_keywords:
            keyword_coverage = (
                len(keyword_source_terms.get(source, set()))
                / len(retrieval_keywords)
            )
        else:
            keyword_coverage = 0.0

        routing_score = (
            semantic_total
            + best_keyword_score * 1.50
            + keyword_coverage * 1.50
        )

        # Reward complete exact-term coverage.
        if retrieval_keywords and keyword_coverage >= 1.0:
            routing_score += 1.00

        routing_scores[source] = routing_score

    best_source = max(
        routing_scores,
        key=routing_scores.get
    )

    # Keep evidence only from the selected PDF.
    semantic_candidates = [
        candidate
        for candidate in semantic_candidates
        if candidate["metadata"].get("source") == best_source
    ]

    keyword_candidates = [
        candidate
        for candidate in keyword_candidates
        if candidate["metadata"].get("source") == best_source
    ]

    # Merge and rerank chunks inside selected PDF.
    merged_candidates = merge_candidates(
        semantic_candidates,
        keyword_candidates
    )

    if not merged_candidates:
        return [], normalized_question

    for candidate in merged_candidates:
        candidate["final_score"] = (
            candidate["semantic_score"] * 2.0
            + candidate["keyword_score"] * 0.50
        )

        if (
            "semantic" in candidate["source_types"]
            and "keyword" in candidate["source_types"]
        ):
            candidate["final_score"] += 0.20

    merged_candidates.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    selected_candidates = select_intent_aware_top3(
        normalized_question,
        merged_candidates
    )

    return selected_candidates, normalized_question


# =========================================================
# BUILD CONTEXT
# =========================================================
def build_context(candidates):
    context_parts = []

    for candidate in candidates:
        metadata = candidate["metadata"]

        source = metadata.get(
            "source",
            "Unknown"
        )

        page = metadata.get(
            "page",
            "Unknown"
        )

        content = candidate["document"]

        context_parts.append(
            f"""
Source: {source}
PDF Page: {page}

Content:
{content}
"""
        )

    return "\n\n".join(context_parts)


# =========================================================
# GEMINI CLIENT - V7 CLOUD DEPLOYMENT
# =========================================================
@st.cache_resource
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not configured."
        )

    return genai.Client(api_key=api_key)


gemini_client = get_gemini_client()


# =========================================================
# ASK GEMINI - V7 CLOUD DEPLOYMENT
# =========================================================
def ask_llm(question, context):
    prompt = f"""You are a strict document-grounded RAG assistant.

Answer the user's question using ONLY facts explicitly supported by the provided context.

Rules:
1. Give a short, direct answer. Prefer 1 to 3 sentences unless the question clearly requires a list.
2. For "What is X?" questions, state only the clearest definition or description of X explicitly supported by the context.
3. Do not add typical features, modules, examples, background knowledge, or explanations that are not explicitly stated in the context.
4. Do not infer a broad definition from one narrow capability, example, reference, or use case.
5. Do not invent or complete acronym expansions, product features, relationships, versions, dates, or technical details.
6. If the context supports a useful partial answer but not a complete definition, give the supported part and clearly say that the provided context does not establish a complete definition.
7. If the context contains only superficial mentions and does not support a useful answer, say exactly:
"I could not find sufficient information in the provided documents."
8. Do not mention information outside the provided context.
9. Keep the answer concise and factual.
10. When the user asks for a list, phases, steps, stages, components, or similar grouped items, inspect ALL provided context chunks before answering. Combine all distinct items that are explicitly supported anywhere in the context, even when different items appear in different chunks or pages.
11. For such list-style questions, do not stop after the first relevant chunk. Deduplicate repeated items, preserve the document's terminology, and do not invent missing items.
12. If the context explicitly states the expected number of items, use that statement only as a completeness check. Include an item only when its name is explicitly supported somewhere in the provided context. If fewer supported names can be found than the stated count, list only the supported names and clearly say the context does not expose all names.
13. For "What is X?" questions, inspect ALL context chunks for explicit statements about X before deciding that information is insufficient.
14. A useful grounded answer may come from an explicit description, purpose statement, replacement/supersession statement, or version-specific contrast. If the context explicitly explains what X does and also states that another tool replaces X in the user's requested release, answer both facts concisely rather than refusing.
15. Do not treat a version-specific replacement statement as absence of information. Clearly distinguish the older tool from the tool actually used in the requested release, using only the context.

Context:
{context}

User Question:
{question}

Answer:"""
    response = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return response.text.strip()


# =========================================================
# DOCUMENT ANSWER USABILITY - V6.3
# =========================================================
def document_answer_is_usable(answer):
    """
    Treat a document answer as unusable only when it is essentially a refusal.
    A useful grounded answer is preserved even if the model adds a trailing
    caveat such as "I could not find additional details".
    """
    cleaned = " ".join(answer.strip().lower().split())

    if not cleaned:
        return False

    refusal_starts = (
        "i could not find sufficient information",
        "i couldn't find sufficient information",
        "i could not find information",
        "i couldn't find information",
        "there is insufficient information",
        "insufficient information",
        "the provided documents do not contain",
        "the provided context does not contain",
        "not enough information",
    )

    return not cleaned.startswith(refusal_starts)



# =========================================================
# SIMPLE DEFINITION QUESTION DETECTOR - V6.16
# =========================================================
def is_simple_definition_question(question):
    """
    Return True only for direct definition questions such as:
    What is CCNA?
    What is Product Management?
    What is ADPATCH in Oracle Apps 12.2?

    No product or technology name is hardcoded.
    """
    q = normalize_query(question)

    return re.search(
        r"\bwhat\s+is\s+"
        r"[A-Za-z0-9_.+-]+"
        r"(?:\s+[A-Za-z0-9_.+-]+){0,5}"
        r"\s*[?.!]*$",
        q,
        flags=re.IGNORECASE,
    ) is not None


# =========================================================
# DEFINITION ANSWER GROUNDING GUARD - V6.16
# =========================================================
def definition_answer_is_grounded(question, answer, candidates):
    """
    Extra grounding check for simple "What is X?" questions.

    If the model generates a definition, require meaningful answer content
    beyond the requested subject to be supported by the retrieved chunks.

    This prevents model-memory answers from being presented as document facts
    when the retrieved document only mentions the subject superficially.
    """
    q = normalize_query(question)

    match = re.search(
        r"\bwhat\s+is\s+([A-Za-z0-9_.+-]+(?:\s+[A-Za-z0-9_.+-]+){0,5})\s*[?.!]*$",
        q,
        flags=re.IGNORECASE,
    )

    # This guard only handles simple definition questions.
    if not match:
        return True

    subject = match.group(1).strip()
    subject = re.split(
        r"\s+(?:in|for|on|under|within)\s+",
        subject,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()

    if not answer or not candidates:
        return False

    context_text = " ".join(
        candidate.get("document", "")
        for candidate in candidates
    ).lower()

    # If the answer itself is a refusal, normal usability logic handles it.
    if not document_answer_is_usable(answer):
        return False

    answer_words = re.findall(
        r"[A-Za-z0-9_.+-]+",
        answer.lower()
    )

    subject_words = set(
        re.findall(r"[A-Za-z0-9_.+-]+", subject.lower())
    )

    grounding_stop_words = {
        "a", "an", "the", "is", "are", "was", "were",
        "means", "mean", "refers", "refer", "to",
        "stands", "stand", "for", "of", "and", "or",
        "it", "this", "that"
    }

    meaningful_words = []
    for word in answer_words:
        if word in subject_words:
            continue
        if word in grounding_stop_words:
            continue
        if len(word) < 3:
            continue
        meaningful_words.append(word)

    meaningful_words = list(dict.fromkeys(meaningful_words))

    # A claimed definition with no meaningful descriptive content is not useful.
    if not meaningful_words:
        return False

    supported_words = [
        word
        for word in meaningful_words
        if technical_term_in_text(word, context_text)
    ]

    coverage = len(supported_words) / len(meaningful_words)

    # Require most of the substantive definition vocabulary to be evidenced.
    return coverage >= 0.60


# =========================================================
# GROUNDED DEFINITION RECOVERY - V6.11.5
# =========================================================
def recover_grounded_definition(question, candidates):
    """
    Fast deterministic recovery for simple "What is X?" questions.

    V6.11.5:
    - Uses only retrieved document evidence.
    - Reconstructs wrapped bullet records without crossing into the next bullet.
    - Attaches an immediately following Important/Note block when it explicitly
      mentions the requested subject.
    - Prevents neighboring records (for example, another utility bullet) from
      contaminating the recovered answer.
    - No product names, page numbers, or domain-specific answers are hardcoded.
    """
    q = normalize_query(question)

    match = re.search(
        r"\bwhat\s+is\s+([A-Za-z0-9_.+-]+(?:\s+[A-Za-z0-9_.+-]+){0,5})\s*[?.!]*$",
        q,
        flags=re.IGNORECASE,
    )
    if not match or not candidates:
        return None

    subject = match.group(1).strip()
    subject = re.split(r"\s+(?:in|for|on|under|within)\s+", subject, maxsplit=1, flags=re.IGNORECASE)[0]
    subject = subject.strip()

    bullet_re = re.compile(
        r"^\s*[\u2022\u25cf\u25e6\u25aa\u25ab]\s*(.+?)\s*$"
    )
    note_re = re.compile(
        r"^\s*(?:important|note|warning|caution)\s*:\s*(.*)$",
        flags=re.IGNORECASE,
    )

    purpose_patterns = re.compile(
        r"\b(?:applies?|performs?|used\s+(?:to|for)|allows?|provides?|"
        r"manages?|creates?|adds?|removes?|updates?|configures?|orchestrates?)\b",
        flags=re.IGNORECASE,
    )
    relation_patterns = re.compile(
        r"\b(?:instead\s+of|rather\s+than|replaces?|replacement|"
        r"prior\s+to|previous\s+releases?|release|version)\b",
        flags=re.IGNORECASE,
    )

    definition_patterns = re.compile(
        rf"\b{re.escape(subject)}\b\s+"
        r"(?:is|are|means?|refers?\s+to|stands?\s+for)\b",
        flags=re.IGNORECASE,
    )

    evidence = []
    seen = set()

    def normalize_piece(value):
        return " ".join(value.split()).strip()

    def score_piece(value, kind):
        score = 0
        if purpose_patterns.search(value):
            score += 6
        if relation_patterns.search(value):
            score += 6
        if re.search(
            r"\b(?:utility|tool|process|system|application|feature)\b",
            value,
            flags=re.IGNORECASE,
        ):
            score += 2
        if kind == "bullet":
            score += 4
        elif kind == "note":
            score += 3

        words = len(value.split())
        if words <= 45:
            score += 2
        elif words > 80:
            score -= 3
        return score

    def add_evidence(value, kind):
        value = normalize_piece(value)
        if not value or len(value.split()) < 3:
            return
        if not technical_term_in_text(subject, value):
            return

        key = value.lower()
        if key in seen:
            return
        seen.add(key)
        evidence.append((score_piece(value, kind), kind, value))

    def add_linked_bullet_evidence(value):
        """
        Allow a preceding bullet to contribute definition/purpose evidence only
        when an immediately following note explicitly mentions the requested
        subject. This is structural linkage, not a hardcoded alias mapping.
        """
        value = normalize_piece(value)
        if not value or len(value.split()) < 3:
            return
        if not purpose_patterns.search(value):
            return

        key = value.lower()
        if key in seen:
            return
        seen.add(key)
        evidence.append((score_piece(value, "linked_bullet") + 4,
                         "linked_bullet", value))

    for candidate in candidates:
        raw_lines = candidate.get("document", "").splitlines()
        lines = [line.rstrip() for line in raw_lines]

        # Stores only the bullet that ends exactly where the next structural
        # record begins. It is eligible for linkage only if that next record
        # is a subject-explicit Important/Note block.
        preceding_bullet = None
        preceding_bullet_next_index = None

        idx = 0
        while idx < len(lines):
            raw = lines[idx]
            stripped = raw.strip()

            if not stripped:
                idx += 1
                continue

            bullet_match = bullet_re.match(raw)
            note_match = note_re.match(raw)

            if bullet_match:
                # Rebuild only this bullet record. Stop at a new bullet, note,
                # heading-like boundary, or separator.
                parts = [bullet_match.group(1).strip()]
                j = idx + 1

                while j < len(lines):
                    nxt = lines[j].strip()
                    if not nxt:
                        break
                    if bullet_re.match(lines[j]) or note_re.match(lines[j]):
                        break
                    if re.fullmatch(r"-{5,}", nxt):
                        break

                    # Wrapped PDF lines commonly continue a sentence. A short
                    # title-like line without punctuation is treated as a boundary.
                    if (
                        len(nxt.split()) <= 6
			and not parts[-1].rstrip().endswith("-")
                        and not re.search(r"[.,;:)]\s*$", parts[-1])
                        and re.match(r"^[A-Z][A-Za-z0-9 /_.()+-]+$", nxt)
                    ):
                        break

                    parts.append(nxt)
                    j += 1

                bullet_text = " ".join(parts)
                add_evidence(bullet_text, "bullet")

                preceding_bullet = bullet_text
                preceding_bullet_next_index = j

                idx = j
                continue

            if note_match:
                # Rebuild the note until the next bullet or structural boundary.
                parts = []
                first_payload = note_match.group(1).strip()
                if first_payload:
                    parts.append(first_payload)

                j = idx + 1
                while j < len(lines):
                    nxt = lines[j].strip()
                    if not nxt:
                        break
                    if bullet_re.match(lines[j]) or note_re.match(lines[j]):
                        break
                    if re.fullmatch(r"-{5,}", nxt):
                        break
                    parts.append(nxt)
                    j += 1

                note_text = " ".join(parts)
                add_evidence(note_text, "note")

                if (
                    preceding_bullet is not None
                    and preceding_bullet_next_index == idx
                    and technical_term_in_text(subject, note_text)
                ):
                    add_linked_bullet_evidence(preceding_bullet)

                # A note consumes the structural relationship. Never carry the
                # previous bullet forward to a later bullet such as Rapid Clone.
                preceding_bullet = None
                preceding_bullet_next_index = None

                idx = j
                continue

            idx += 1

        # Normal prose can still contain a clean subject-specific statement.
        for sentence in re.split(r"(?<=[.!?])\s+", candidate.get("document", "")):
            sentence = normalize_piece(sentence.strip(" \t-•"))
            if sentence:
                add_evidence(sentence, "sentence")

    # Prefer explicit subject bullet definitions/purpose statements first,
    # then version/replacement notes. Do not simply concatenate arbitrary
    # adjacent text.
    evidence.sort(
        key=lambda item: (
            item[0],
            item[1] == "bullet",
            item[1] == "note",
            -len(item[2]),
        ),
        reverse=True,
    )

    # V6.11.5: choose evidence by ROLE, not only by overall score.
    # A subject-specific purpose statement is the definition anchor.
    # A distinct version/replacement statement is supporting evidence.
    definition_candidates = [
        item for item in evidence
        if definition_patterns.search(item[2])
    ]
    purpose_candidates = [
        item for item in evidence
        if purpose_patterns.search(item[2])
    ]
    relation_candidates = [
        item for item in evidence
        if relation_patterns.search(item[2])
    ]

    primary = None
    support = None

    if definition_candidates:
        definition_candidates.sort(
            key=lambda item: (
                item[1] in ("bullet", "linked_bullet"),
                item[0],
                -len(item[2]),
            ),
            reverse=True,
        )
        primary = definition_candidates[0][2]

    if primary is None and purpose_candidates:
        # A generic purpose statement is acceptable only when the requested
        # subject itself is the grammatical anchor of that statement.
        # This prevents contextual mentions such as:
        # "Create a small lab for X interview practice"
        # from being returned as the definition of X.
        anchored_purpose_candidates = []

        subject_anchor_re = re.compile(
            rf"^\s*(?:the\s+)?{re.escape(subject)}\b"
            rf"(?:\s+(?:tool|utility|process|system|application|feature))?",
            flags=re.IGNORECASE,
        )

        for item in purpose_candidates:
            if subject_anchor_re.search(item[2]):
                anchored_purpose_candidates.append(item)

        if anchored_purpose_candidates:
            anchored_purpose_candidates.sort(
                key=lambda item: (
                    item[1] in ("bullet", "linked_bullet"),
                    item[0],
                    -len(item[2]),
                ),
                reverse=True,
            )
            primary = anchored_purpose_candidates[0][2]

    if relation_candidates:
        relation_candidates.sort(
            key=lambda item: (
                item[1] == "note",
                item[0],
                -len(item[2]),
            ),
            reverse=True,
        )
        for _, _, value in relation_candidates:
            if primary is None or value.lower() not in primary.lower():
                support = value
                break

    # A version/replacement statement alone is still useful when the retrieved
    # evidence does not expose a separate purpose statement.
    if primary is None:
        primary = support
        support = None

    # V6.12.1: If normal bullet/note/sentence evidence exists but does
    # not produce a usable definition, try a generic PDF index-style definition
    # before giving up and allowing the caller to invoke the LLM.
    #
    # Example structure:
    #     technical_term, 1-24
    #     descriptive phrase, 4-15
    #
    # No technical term, product, page, or answer is hardcoded.
    if primary is None:
        index_page_ref_re = re.compile(
            r"^\s*(.+?),\s*(?:\d+-\d+(?:\s*,\s*\d+-\d+)*)\s*$"
        )

        for candidate in candidates:
            index_lines = [
                line.strip()
                for line in candidate.get("document", "").splitlines()
                if line.strip()
            ]

            for idx, line in enumerate(index_lines):
                entry_match = index_page_ref_re.match(line)
                if not entry_match:
                    continue

                entry_label = entry_match.group(1).strip()

                # Require the index anchor to be exactly the requested term.
                if entry_label.lower() != subject.lower():
                    continue

                descriptions = []

                # Only inspect a very small immediately-following window.
                # This limits accidental capture of unrelated index entries.
                for next_line in index_lines[idx + 1: idx + 4]:
                    next_match = index_page_ref_re.match(next_line)
                    if not next_match:
                        break

                    description = next_match.group(1).strip()

                    if not description:
                        break

                    if len(description.split()) > 10:
                        break

                    if description.lower() == subject.lower():
                        continue

                    descriptions.append(description)

                if descriptions:
                    description = descriptions[0].rstrip(".")

                    # Use "a" / "an" generically for cleaner output.
                    article = (
                        "an"
                        if description[:1].lower() in "aeiou"
                        else "a"
                    )

                    primary = (
                        f"{subject} is {article} {description}."
                    )
                    break

            if primary is not None:
                break

    if primary is None:
        return None

    # Generic parenthetical technical-definition recovery.
    #
    # Example structure:
    #     SUBJECT (descriptive expansion) utility/tool
    #
    # If retrieved evidence explicitly contains this structure, prefer a
    # concise grounded definition over a weaker relation/version statement.
    # No product name, technical term, or page number is hardcoded.
    parenthetical_definition = None

    parenthetical_re = re.compile(
        rf"\b{re.escape(subject)}\b\s*"
        r"\(\s*([A-Za-z][A-Za-z0-9 /_.+-]{2,100}?)\s*\)"
        r"\s+(utility|tool|process|system|application|feature)\b",
        flags=re.IGNORECASE,
    )

    for candidate in candidates:
        candidate_text = normalize_piece(
            candidate.get("document", "")
        )

        parenthetical_match = parenthetical_re.search(
            candidate_text
        )

        if not parenthetical_match:
            continue

        expansion = normalize_piece(
            parenthetical_match.group(1)
        )
        object_type = parenthetical_match.group(2).lower()

        # Avoid repeating the requested acronym/term when the
        # parenthetical expansion begins with it.
        expansion_words = expansion.split()
        subject_words = subject.split()

        if (
            expansion_words
            and subject_words
            and expansion_words[0].lower()
            == subject_words[0].lower()
            and len(expansion_words) > 1
        ):
            expansion = " ".join(expansion_words[1:])

        if not expansion:
            continue

        # "utility" and "tool" are equivalent descriptive categories
        # for this concise definition form. Normalize utility -> tool
        # to keep the output natural and consistent.
        definition_type = (
            "tool"
            if object_type in ("utility", "tool")
            else object_type
        )

        article = (
            "an"
            if expansion[:1].lower() in "aeiou"
            else "a"
        )

        parenthetical_definition = (
            f"{subject} is {article} "
            f"{expansion} {definition_type}."
        )

        break

    if parenthetical_definition is not None:
        primary = parenthetical_definition

    selected = [primary]
    if support:
        selected.append(support)

    answer = " ".join(selected)
    answer = re.sub(r"\s+", " ", answer).strip()

    return answer


# =========================================================
# STRUCTURAL EVIDENCE COMPLETENESS VALIDATOR - V6.9.6.2
# =========================================================
def complete_grounded_list_answer(question, answer, candidates):
    """
    Repair incomplete list-style answers from structural evidence in retrieved
    Top-3 chunks. No second LLM call, no page-number hardcoding, and no
    domain-specific list-item names are hardcoded.

    V6.9.6.2 improvement:
    - Do not require every bullet heading to already appear in the LLM answer.
    - Preserve the raw bullet's terminal punctuation before label cleanup.
    - In an explicitly introduced list block, a short bullet with no sentence-
      ending punctuation is treated as a structural heading; sentence-like
      action bullets ending in punctuation are rejected.
    - Continuation headings remain supported when nearby prose explicitly
      refers to "this phase/stage/step/etc.".
    """
    q = normalize_query(question)

    group_match = re.search(
        r"\b(phases?|steps?|stages?|components?|items?|types?|categories?|"
        r"actions?|processes?|features?)\b",
        q,
    )
    if not group_match:
        return answer, []

    if not candidates or not document_answer_is_usable(answer):
        return answer, []

    answer_lower = answer.lower()
    singular_group = {
        "phases": "phase", "phase": "phase",
        "steps": "step", "step": "step",
        "stages": "stage", "stage": "stage",
        "components": "component", "component": "component",
        "items": "item", "item": "item",
        "types": "type", "type": "type",
        "categories": "category", "category": "category",
        "actions": "action", "action": "action",
        "processes": "process", "process": "process",
        "features": "feature", "feature": "feature",
    }.get(group_match.group(1), group_match.group(1).rstrip("s"))

    bullet_re = re.compile(r"^\s*[\u2022\u25cf\u25e6\u25aa\u25ab\-–—*]\s+(.+?)\s*$")

    def clean_label(value):
        value = value.strip()
        value = re.sub(r"^\(?\d{1,2}[\).:-]\s*", "", value)
        value = re.sub(r"^[A-Za-z][\).:-]\s+", "", value)
        return value.strip(" \t:;.-")

    def short_label(value):
        if not value or len(value) > 48:
            return False
        words = re.findall(r"[A-Za-z][A-Za-z0-9_./+-]*", value)
        return 1 <= len(words) <= 4

    def appears_in_answer(label):
        return re.search(
            r"(?<![A-Za-z0-9_])" + re.escape(label.lower()) + r"(?![A-Za-z0-9_])",
            answer_lower,
        ) is not None

    # -----------------------------------------------------
    # A. Find structurally introduced bullet-list blocks.
    # -----------------------------------------------------
    best_block = []

    for candidate in candidates:
        lines = candidate.get("document", "").splitlines()

        for idx, raw_line in enumerate(lines):
            intro = raw_line.strip().lower()

            if not (
                re.search(r"\b(key\s+)?actions?\b", intro)
                or re.search(r"\b(?:various\s+)?stages?\b", intro)
                or re.search(r"\b(?:patching\s+)?cycle\b", intro)
                or re.search(
                    r"\b(?:phases?|steps?|components?|items?|types?|categories?|"
                    r"processes?|features?)\b",
                    intro,
                )
            ):
                continue

            intro_window = " ".join(
                line.strip().lower()
                for line in lines[idx:min(len(lines), idx + 2)]
            )
            if not re.search(
                r"\b(?:follows|following|summari[sz]ed|overview|include|includes|consists?)\b",
                intro_window,
            ):
                continue

            block = []
            started = False

            for following in lines[idx + 1:]:
                match = bullet_re.match(following)
                if match:
                    started = True

                    # IMPORTANT: inspect punctuation BEFORE clean_label() strips it.
                    raw_payload = match.group(1).strip()
                    label = clean_label(raw_payload)

                    # Structural list headings in extracted manuals are commonly short
                    # labels without sentence-ending punctuation. Descriptive/action
                    # bullets are usually sentence-like and end with punctuation.
                    if (
                        short_label(label)
                        and not re.search(r"[.!?;:]\s*$", raw_payload)
                        and "," not in raw_payload
                    ):
                        block.append(label)
                    continue

                if started and following.strip():
                    break

            if len(block) > len(best_block):
                best_block = block

    structural_labels = []
    seen = set()

    for label in best_block:
        key = re.sub(r"\s+", " ", label.lower())
        if key not in seen:
            seen.add(key)
            structural_labels.append(label)

    # Require multiple structurally supported headings before repair.
    if len(structural_labels) < 2:
        return answer, []

    # -----------------------------------------------------
    # B. Find continuation headings across retrieved chunks.
    # -----------------------------------------------------
    continuation_labels = []

    for candidate in candidates:
        lines = candidate.get("document", "").splitlines()

        for idx, raw_line in enumerate(lines):
            raw_candidate = raw_line.strip()
            candidate_label = clean_label(raw_candidate)

            if not short_label(candidate_label):
                continue

            # Standalone continuation headings should not look like sentences.
            if re.search(r"[.!?;:]\s*$", raw_candidate) or "," in raw_candidate:
                continue

            key = re.sub(r"\s+", " ", candidate_label.lower())
            if key in seen:
                continue

            following_text = " ".join(
                line.strip()
                for line in lines[idx + 1:min(len(lines), idx + 4)]
                if line.strip()
            ).lower()

            if not following_text:
                continue

            if not re.search(
                rf"\b(?:this|the)\s+{re.escape(singular_group)}\b",
                following_text,
            ):
                continue

            if len(re.findall(r"[A-Za-z]+", following_text)) < 6:
                continue

            seen.add(key)
            continuation_labels.append(candidate_label)

    evidence_labels = structural_labels + continuation_labels

    if len(evidence_labels) < 3:
        return answer, []

    missing = [
        label for label in evidence_labels
        if not appears_in_answer(label)
    ]

    if not missing:
        return answer, []

    # -----------------------------------------------------
    # C. Rebuild the grounded list from explicit structure.
    # -----------------------------------------------------
    subject = None

    subject_match = re.search(
        r"\b(?:phases?|steps?|stages?|components?|items?|types?|categories?|"
        r"actions?|processes?|features?)\s+of\s+([A-Za-z0-9_.+-]+)",
        q,
    )
    if subject_match:
        subject = subject_match.group(1)

    if not subject:
        combined_match = re.search(
            r"\bwhat\s+is\s+([A-Za-z0-9_.+-]+)\b.*\band\s+its\b",
            q,
        )
        if combined_match:
            subject = combined_match.group(1)

    group_word = group_match.group(1)
    heading = (
        f"The {group_word} of {subject.upper()} are:"
        if subject
        else f"The document-supported {group_word} are:"
    )

    repaired_list = heading + "\n\n" + "\n".join(
        f"- {label}" for label in evidence_labels
    )

    is_combined_definition = bool(
        re.search(r"\bwhat\s+is\b", q)
        and re.search(r"\band\s+its\b", q)
    )

    if not is_combined_definition:
        return repaired_list, missing

    answer_lines = answer.splitlines()
    prefix_lines = []

    for line in answer_lines:
        stripped = line.strip()
        if re.match(r"^\s*[-*•]\s+", line):
            break
        if re.search(
            r"\b(phases?|steps?|stages?|components?|items?|types?|categories?|"
            r"actions?|processes?|features?)\b.*\b(?:are|include|includes)\b",
            stripped.lower(),
        ):
            break
        prefix_lines.append(line)

    prefix = "\n".join(prefix_lines).strip()

    if prefix and document_answer_is_usable(prefix):
        return prefix + "\n\n" + repaired_list, missing

    return repaired_list, missing


# =========================================================
# RETRIEVAL EVIDENCE GUARD
# =========================================================

def recover_grounded_structured_list(question, candidates):
    """
    Generic deterministic recovery for structured list questions.

    Uses only retrieved document evidence.
    No product names, phase names, or page numbers are hardcoded.
    """
    q = question.strip().lower()

    list_intent_patterns = (
        r"\bphases?\b",
        r"\bstages?\b",
        r"\bsteps?\b",
        r"\btypes?\b",
        r"\bcomponents?\b",
        r"\bitems?\b",
    )

    if not any(re.search(pattern, q) for pattern in list_intent_patterns):
        return None

    def clean_item(value):
        return re.sub(
            r"\s+",
            " ",
            value
        ).strip(" •●◦▪▫\t\r\n:-")

    def is_heading_like(value):
        value = clean_item(value)

        if not value:
            return False

        words = value.split()

        if len(words) > 5:
            return False

        if re.search(r"[.;]$", value):
            return False

        explanatory_starts = (
            "the ",
            "a ",
            "an ",
            "this ",
            "these ",
            "it ",
            "is ",
            "are ",
            "can ",
            "will ",
            "should ",
            "executes ",
            "creates ",
            "compiles ",
            "generates ",
            "synchronizes ",
            "patches ",
            "performs ",
        )

        if value.lower().startswith(explanatory_starts):
            return False

        return bool(re.search(r"[A-Za-z]", value))

    recovered = []
    recovered_keys = set()

    def add_item(value):
        value = clean_item(value)

        if not is_heading_like(value):
            return

        key = value.lower()

        if key in recovered_keys:
            return

        recovered_keys.add(key)
        recovered.append(value)

    # --------------------------------------------------------
    # PASS 1:
    # Establish a structured-list anchor using explicit bullets.
    #
    # We require at least two short bullet-like items from the
    # retrieved evidence before considering cross-page headings.
    # --------------------------------------------------------
    for candidate in candidates:
        raw_text = candidate.get("document", "")

        if not re.search(
            r"\b(phases?|stages?|steps?|types?|components?|items?)\b",
            raw_text,
            flags=re.IGNORECASE,
        ):
            continue

        lines = [
            line.strip()
            for line in raw_text.splitlines()
            if line.strip()
        ]

        for line in lines:
            match = re.match(
                r"^[•●◦▪▫]\s*(.+?)\s*$",
                line
            )

            if not match:
                continue

            value = match.group(1).strip()

            if is_heading_like(value):
                add_item(value)

    # Without a real structured-list anchor, do not guess from
    # ordinary document headings.
    if len(recovered) < 2:
        return None

    # --------------------------------------------------------
    # PASS 2:
    # Recover continuation items from other retrieved chunks.
    #
    # A continuation must be a short standalone heading followed
    # by explanatory text. We only enable this after PASS 1 has
    # established that the retrieved evidence actually contains
    # a structured list.
    # --------------------------------------------------------
    for candidate in candidates:
        raw_text = candidate.get("document", "")

        lines = [
            line.strip()
            for line in raw_text.splitlines()
            if line.strip()
        ]

        for idx, line in enumerate(lines[:-1]):
            if not is_heading_like(line):
                continue

            # Do not re-process explicit bullet lines here.
            if re.match(r"^[•●◦▪▫]", line):
                continue

            next_line = lines[idx + 1]

            next_looks_explanatory = (
                len(next_line.split()) >= 6
                or bool(re.search(r"[.:;]$", next_line))
                or next_line.lower().startswith(
                    (
                        "the ",
                        "this ",
                        "these ",
                        "in ",
                        "during ",
                        "after ",
                        "before ",
                    )
                )
            )

            if not next_looks_explanatory:
                continue

            # V6.12.2.2:
            # Reject container/section headings that introduce the
            # structured list itself rather than representing one
            # member of that list.
            #
            # Example structure:
            #   <section heading>
            #   The ... stages ... can be summarized as
            #   follows:
            #   • item
            #
            # This rule is generic and does not depend on any
            # product, phase name, or page number.
            lookahead_text = " ".join(
                lines[idx + 1: min(len(lines), idx + 4)]
            ).lower()

            introduces_list = (
                bool(
                    re.search(
                        r"\b(summarized|summarised|listed|enumerated|"
                        r"outlined|described)\b",
                        lookahead_text
                    )
                )
                and bool(
                    re.search(
                        r"\b(phases?|stages?|steps?|types?|"
                        r"components?|items?|actions?)\b",
                        lookahead_text
                    )
                )
            )

            if introduces_list:
                continue

            # Generic continuation safety:
            # prefer headings whose surrounding text refers to the
            # same list concept requested by the user.
            nearby_start = max(0, idx - 3)
            nearby_end = min(len(lines), idx + 4)

            nearby_text = " ".join(
                lines[nearby_start:nearby_end]
            ).lower()

            concept_supported = any(
                re.search(pattern, nearby_text)
                for pattern in list_intent_patterns
            )

            if concept_supported:
                add_item(line)

    if len(recovered) < 2:
        return None

    recovered = recovered[:12]

    return (
        "The document-supported items are:\n\n"
        + "\n".join(
            f"- {item}"
            for item in recovered
        )
    )

def evaluate_retrieval_evidence(question, candidates):
    """
    Classify retrieval evidence without pretending that embedding similarity
    is a calibrated probability.

    Returns:
        label: STRONG, PARTIAL, or WEAK
        details: diagnostic values for optional debugging
    """

    normalized = normalize_query(question)
    keywords = extract_keywords(normalized)

    if not candidates:
        return "WEAK", {
            "keywords": keywords,
            "matched_keywords": [],
            "coverage": 0.0,
            "top_semantic_score": 0.0,
            "hybrid_hits": 0
        }

    matched_keywords = set()
    top_semantic_score = 0.0
    hybrid_hits = 0

    for candidate in candidates:
        document_lower = candidate["document"].lower()

        for keyword in keywords:
            if technical_term_in_text(keyword, document_lower):
                matched_keywords.add(keyword)

        top_semantic_score = max(
            top_semantic_score,
            candidate.get("semantic_score", 0.0)
        )

        source_types = candidate.get("source_types", set())

        if (
            "semantic" in source_types
            and "keyword" in source_types
        ):
            hybrid_hits += 1

    if keywords:
        coverage = len(matched_keywords) / len(keywords)
    else:
        coverage = 0.0

    # Evidence policy derived from the project's diagnostic tests:
    # - Exact/lexical evidence is a strong signal when combined with retrieval.
    # - Semantic similarity alone is NOT treated as a confidence probability.
    # - Weak evidence bypasses document generation and uses the clearly
    #   labelled General AI fallback.
    #
    # STRONG:
    #   all important query terms are represented in selected evidence,
    #   OR substantial term coverage is supported by semantic/hybrid evidence.
    #
    # PARTIAL:
    #   some lexical evidence exists, but not enough to call it strong.
    #   We still let the grounded LLM perform the final sufficiency check.
    #
    # WEAK:
    #   no important query terms are present in the selected evidence.
    #   The PDF evidence is too weak to justify a grounded answer attempt.

    if coverage >= 1.0:
        label = "STRONG"

    elif (
        coverage >= 0.50
        and (
            top_semantic_score >= 0.40
            or hybrid_hits >= 1
        )
    ):
        label = "STRONG"

    elif coverage > 0.0:
        label = "PARTIAL"

    else:
        label = "WEAK"

    return label, {
        "keywords": keywords,
        "matched_keywords": sorted(matched_keywords),
        "coverage": coverage,
        "top_semantic_score": top_semantic_score,
        "hybrid_hits": hybrid_hits
    }


# =========================================================
# FOLLOW-UP DEPENDENCY DETECTION - V6.2A
# =========================================================
def needs_followup_rewrite(question):
    """
    Rewrite only when the current question contains a reference that
    normally depends on recent conversation. Standalone questions are
    returned unchanged and do not need an extra LLM rewrite call.
    """
    q = question.strip().lower()

    # V6.9.5: protect self-contained questions such as
    # "What is ADOP and its phases?" from unnecessary LLM rewriting.
    if re.search(r"\b[a-zA-Z0-9_.-]+\b(?:\s+[a-zA-Z0-9_.-]+){0,5}\s+and\s+its\b", q):
        return False

    reference_patterns = [
        r"\bit\b",
        r"\bits\b",
        r"\bthis\b",
        r"\bthat\b",
        r"\bthese\b",
        r"\bthose\b",
        r"\bthey\b",
        r"\bthem\b",
        r"\btheir\b",
        r"\bthe above\b",
        r"\bthis process\b",
        r"\bthis utility\b",
        r"\bthis feature\b",
    ]

    return any(
        re.search(pattern, q)
        for pattern in reference_patterns
    )


# =========================================================
# CONVERSATION-AWARE QUERY REWRITING
# =========================================================
def rewrite_followup_question(question, messages):
    """
    V6.3.1: Resolve dependent follow-ups from the immediately preceding
    user question, not from several older conversation topics.
    """
    if not messages:
        return question

    if not needs_followup_rewrite(question):
        return question

    previous_user_question = None
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content", "").strip()
            if content:
                previous_user_question = content
                break

    if not previous_user_question:
        return question

    prompt = f"""
You rewrite a context-dependent follow-up question for a RAG retrieval system.

Use ONLY the immediately preceding user question to resolve the reference.
Never choose an older conversation topic.

Rules:
1. Resolve words such as it, its, this, that, these, those, they, them, and their
   using only the immediately preceding user question.
2. Rewrite the current question as one clear standalone retrieval question.
3. Preserve the exact technical subject from the preceding user question.
4. Do not switch to another technology or older topic.
5. Do not answer the question.
6. Return only the rewritten question.

Immediately Preceding User Question:
{previous_user_question}

Current Follow-up Question:
{question}

Standalone Retrieval Question:
"""

    response = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    rewritten = (response.text or "").strip()
    if not rewritten:
        return question

    return rewritten.splitlines()[0].strip()


# =========================================================
# SESSION STATE
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🧠 RAG V6.11.5")
    st.caption("Enterprise Knowledge Assistant")
    st.markdown("### 📚 Knowledge Base")

    st.markdown(
        f"""
        <div class="sidebar-card">
          <div class="sidebar-label">PDF chunks loaded</div>
          <div class="sidebar-value">{len(all_documents):,}</div>
          <div class="sidebar-label">Vector database</div>
          <div class="sidebar-value">ChromaDB</div>
          <div class="sidebar-label">Embedding model</div>
          <div class="sidebar-value">all-MiniLM-L6-v2</div>
          <div class="sidebar-label">LLM</div>
          <div class="sidebar-value">Llama 3.2 : 3B</div>
        </div>
        <div class="system-ready"><b>● System Ready</b><br>
        <span style="font-size:.78rem;">Local document RAG is active</span></div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # V6.13 - DOCUMENT UPLOAD UI (PHASE 1)
    # UI + validation only. No database write in this phase.
    # =====================================================
    st.markdown("### Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        accept_multiple_files=False,
        help="Phase 1 supports PDF files only."
    )

    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_size_bytes = uploaded_file.size
        file_size_mb = file_size_bytes / (1024 * 1024)

        st.markdown(f"**Selected:** {file_name}")
        st.caption(f"File size: {file_size_mb:.2f} MB")

        # Basic validation
        is_pdf = file_name.lower().endswith(".pdf")
        size_ok = 0 < file_size_bytes <= 25 * 1024 * 1024

        if not is_pdf:
            st.error("Only PDF files are supported in V6.13 Phase 1.")

        elif not size_ok:
            st.error(
                "PDF must be larger than 0 bytes and no more than 25 MB."
            )

        else:
            st.success(
                "PDF validated successfully. Ready for processing."
            )

            # =================================================
            # V6.13 - PHASE 2: SAFE PDF TEXT EXTRACTION
            # Extraction only. No chunking/database write yet.
            # =================================================
            try:
                from io import BytesIO
                from pypdf import PdfReader

                pdf_bytes = uploaded_file.getvalue()

                # =============================================
                # V6.13 - DUPLICATE DOCUMENT PROTECTION
                # Generate a content-based SHA-256 fingerprint.
                # Same file content = same fingerprint even if
                # the filename is changed.
                # =============================================
                import hashlib

                document_hash = hashlib.sha256(
                    pdf_bytes
                ).hexdigest()

                st.session_state[
                    "uploaded_pdf_hash"
                ] = document_hash

                if "processed_document_hashes" not in st.session_state:
                    st.session_state[
                        "processed_document_hashes"
                    ] = set()

                # =============================================
                # V6.13 - PERSISTENT DUPLICATE STATUS
                # Check ChromaDB directly instead of relying only
                # on Streamlit session state.
                # =============================================
                existing_document = collection.get(
                    where={
                        "document_hash": document_hash
                    },
                    include=["metadatas"]
                )

                existing_document_ids = (
                    existing_document.get("ids", [])
                    if existing_document
                    else []
                )

                already_indexed = bool(
                    existing_document_ids
                )

                if already_indexed:
                    st.warning(
                        "Already indexed — duplicate detected."
                    )
                else:
                    st.success(
                        "Duplicate check passed — new document."
                    )

                pdf_reader = PdfReader(BytesIO(pdf_bytes))

                total_pages = len(pdf_reader.pages)
                pages_with_text = 0
                total_characters = 0

                extracted_pages = []

                for page_number, page in enumerate(
                    pdf_reader.pages,
                    start=1
                ):
                    page_text = page.extract_text() or ""
                    page_text = page_text.strip()

                    if page_text:
                        pages_with_text += 1
                        total_characters += len(page_text)

                    extracted_pages.append(
                        {
                            "page": page_number,
                            "text": page_text
                        }
                    )

                if total_characters == 0:
                    st.warning(
                        "PDF opened successfully, but no extractable "
                        "text was found. It may be a scanned/image PDF."
                    )
                else:
                    st.success("PDF text extraction successful.")

                    st.markdown(
                        f"""
                        **Extraction Summary**

                        - Pages detected: **{total_pages}**
                        - Pages with text: **{pages_with_text}**
                        - Characters extracted: **{total_characters:,}**
                        """
                    )

                    # Keep extracted data available for later phases
                    # during the current Streamlit session.
                    st.session_state["uploaded_pdf_pages"] = extracted_pages
                    st.session_state["uploaded_pdf_name"] = file_name

                    # =============================================
                    # V6.13 - PHASE 3: PAGE-AWARE CHUNKING
                    # Chunking only. No embeddings/database write.
                    # =============================================
                    chunk_size = 1000
                    chunk_overlap = 200
                    step_size = chunk_size - chunk_overlap

                    uploaded_chunks = []

                    for page_data in extracted_pages:
                        page_number = page_data["page"]
                        page_text = page_data["text"]

                        if not page_text:
                            continue

                        start = 0

                        while start < len(page_text):
                            end = min(
                                start + chunk_size,
                                len(page_text)
                            )

                            chunk_text = page_text[start:end].strip()

                            if chunk_text:
                                uploaded_chunks.append(
                                    {
                                        "text": chunk_text,
                                        "source": file_name,
                                        "page": page_number
                                    }
                                )

                            if end >= len(page_text):
                                break

                            start += step_size

                    st.session_state[
                        "uploaded_pdf_chunks"
                    ] = uploaded_chunks

                    st.success(
                        "Page-aware chunking successful."
                    )

                    st.markdown(
                        f"""
                        **Chunking Summary**

                        - Chunk size: **{chunk_size} characters**
                        - Overlap: **{chunk_overlap} characters**
                        - Total chunks created: **{len(uploaded_chunks):,}**
                        """
                    )

                    if uploaded_chunks:
                        preview = uploaded_chunks[0]

                        with st.expander(
                            "Preview first chunk"
                        ):
                            st.markdown(
                                f"**Source:** {preview['source']}"
                            )
                            st.markdown(
                                f"**PDF Page:** {preview['page']}"
                            )
                            st.text(
                                preview["text"][:700]
                            )

                st.caption(
                    "Phase 3 complete — text was extracted and "
                    "chunked in memory."
                )

                # =================================================
                # V6.13 - PHASE 4:
                # EMBEDDINGS + PERSISTENT CHROMADB INDEXING
                # =================================================
                if uploaded_chunks:
                    st.markdown("### Add to Knowledge Base")

                    if st.button(
                        "Process & Add to Knowledge Base",
                        key="process_uploaded_pdf",
                        use_container_width=True
                    ):
                        try:
                            # -------------------------------------
                            # 1. Persistent duplicate check
                            # -------------------------------------
                            existing_hash_records = collection.get(
                                where={
                                    "document_hash": document_hash
                                },
                                include=["metadatas"]
                            )

                            existing_ids = (
                                existing_hash_records.get("ids", [])
                                if existing_hash_records
                                else []
                            )

                            if existing_ids:
                                st.warning(
                                    "This document is already indexed "
                                    "in the knowledge base."
                                )

                            else:
                                # ---------------------------------
                                # 2. Prepare documents + metadata
                                # ---------------------------------
                                documents_to_add = [
                                    item["text"]
                                    for item in uploaded_chunks
                                ]

                                metadatas_to_add = [
                                    {
                                        "source": item["source"],
                                        "page": int(item["page"]),
                                        "document_hash": document_hash,
                                        "upload_type": "user_upload"
                                    }
                                    for item in uploaded_chunks
                                ]

                                # Collision-safe deterministic IDs.
                                short_hash = document_hash[:16]

                                ids_to_add = [
                                    f"upload_{short_hash}_{index}"
                                    for index in range(
                                        len(uploaded_chunks)
                                    )
                                ]

                                # ---------------------------------
                                # 3. Generate embeddings using
                                # Gemini Embedding API.
                                # ---------------------------------
                                with st.spinner(
                                    "Generating Gemini embeddings..."
                                ):
                                    embeddings = []

                                    # Process chunks in small batches
                                    # instead of loading a local model.
                                    embedding_batch_size = 20

                                    for batch_start in range(
                                        0,
                                        len(documents_to_add),
                                        embedding_batch_size
                                    ):
                                        batch_documents = documents_to_add[
                                            batch_start:
                                            batch_start + embedding_batch_size
                                        ]

                                        response = (
                                            gemini_client.models.embed_content(
                                                model="gemini-embedding-001",
                                                contents=batch_documents
                                            )
                                        )

                                        batch_embeddings = [
                                            item.values
                                            for item in response.embeddings
                                        ]

                                        if len(batch_embeddings) != len(
                                            batch_documents
                                        ):
                                            raise RuntimeError(
                                                "Gemini embedding count "
                                                "does not match chunk count."
                                            )

                                        if any(
                                            len(vector) != 3072
                                            for vector in batch_embeddings
                                        ):
                                            raise RuntimeError(
                                                "Unexpected Gemini embedding "
                                                "dimension. Expected 3072."
                                            )

                                        embeddings.extend(
                                            batch_embeddings
                                        )

                                    if len(embeddings) != len(
                                        documents_to_add
                                    ):
                                        raise RuntimeError(
                                            "Final embedding count does not "
                                            "match document chunk count."
                                        )

                                # ---------------------------------
                                # 4. Add records to existing
                                # persistent Chroma collection.
                                # ---------------------------------
                                with st.spinner(
                                    "Adding document to knowledge base..."
                                ):
                                    collection.add(
                                        ids=ids_to_add,
                                        documents=documents_to_add,
                                        metadatas=metadatas_to_add,
                                        embeddings=embeddings
                                    )

                                # ---------------------------------
                                # 5. Verify database write
                                # ---------------------------------
                                verified_records = collection.get(
                                    where={
                                        "document_hash": document_hash
                                    },
                                    include=["metadatas"]
                                )

                                verified_count = len(
                                    verified_records.get("ids", [])
                                )

                                if verified_count == len(
                                    uploaded_chunks
                                ):
                                    st.session_state[
                                        "processed_document_hashes"
                                    ].add(document_hash)

                                    st.success(
                                        "Document indexed successfully."
                                    )

                                    st.markdown(
                                        f"""
                                        **Indexing Summary**

                                        - Document: **{file_name}**
                                        - Chunks indexed: **{verified_count:,}**
                                        - Embedding model: **all-MiniLM-L6-v2**
                                        - Vector database: **ChromaDB**
                                        - Status: **Ready for RAG search**
                                        """
                                    )

                                    # Refresh in-memory global corpus
                                    # so keyword/global retrieval can
                                    # see the new document immediately.
                                    refreshed_data = collection.get(
                                        include=[
                                            "documents",
                                            "metadatas"
                                        ]
                                    )

                                    all_documents[:] = (
                                        refreshed_data["documents"]
                                    )

                                    all_metadatas[:] = (
                                        refreshed_data["metadatas"]
                                    )

                                else:
                                    st.error(
                                        "Index verification failed. "
                                        "Please do not upload the "
                                        "document again until the "
                                        "database is inspected."
                                    )

                        except Exception as indexing_error:
                            st.error(
                                "Document indexing failed."
                            )
                            st.caption(str(indexing_error))

                else:
                    st.warning(
                        "No text chunks are available for indexing."
                    )

            except Exception as extraction_error:
                st.error(
                    "PDF validation passed, but text extraction failed."
                )
                st.caption(str(extraction_error))

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Trusted documents • Local inference • Evidence-aware answers")


# =========================================================
# CHAT HISTORY
# =========================================================
for message in st.session_state.messages:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):
            st.markdown("**Sources:**")

            for source in message["sources"]:
                st.write(
                    f"📄 {source['source']} "
                    f"— PDF Page {source['page']}"
                )


st.markdown(
    """
    <div class="footer-credit">
        Enterprise Knowledge Assistant • RAG V6.11.5<br>
        Developed &amp; Designed by <strong>MD SHAHID HUSSAIN</strong>
    </div>
    """,
    unsafe_allow_html=True
)
# =========================================================
# V6.15 - REUSABLE RAG QUERY ENGINE
# =========================================================
def run_rag_query(question, previous_messages=None):
    """
    Run the core document-grounded RAG pipeline without Streamlit UI.

    This function is intended for automated regression/evaluation tests.
    Pass previous_messages=[] for an isolated independent test.
    """
    if previous_messages is None:
        previous_messages = []

    total_start = time.perf_counter()

    timings = {
        "Follow-up Rewrite": 0.0,
        "Retrieval": 0.0,
        "Evidence Guard": 0.0,
        "Context Build": 0.0,
        "LLM Generation": 0.0,
        "Completeness Check": 0.0,
        "General AI": 0.0,
    }

    # 1. Follow-up rewrite
    t = time.perf_counter()
    retrieval_question = rewrite_followup_question(
        question,
        previous_messages
    )
    timings["Follow-up Rewrite"] = time.perf_counter() - t

    # 2. Retrieval
    t = time.perf_counter()
    candidates, normalized_question = retrieve_context(
        retrieval_question
    )
    timings["Retrieval"] = time.perf_counter() - t

    # 3. Evidence Guard
    t = time.perf_counter()
    evidence_label, evidence_details = evaluate_retrieval_evidence(
        normalized_question,
        candidates
    )
    timings["Evidence Guard"] = time.perf_counter() - t

    sources = []
    recovered_answer = None

    # 4. Reject weak/empty evidence
    if not candidates or evidence_label == "WEAK":
        document_answer = (
            "I could not find sufficient information "
            "in the provided documents."
        )
        fallback = True

    else:
        # 5. Deterministic fast recovery
        if evidence_label == "STRONG":
            recovered_answer = recover_grounded_definition(
                normalized_question,
                candidates
            )

            if not recovered_answer:
                recovered_answer = recover_grounded_structured_list(
                    normalized_question,
                    candidates
                )

        if recovered_answer:
            document_answer = recovered_answer
            fallback = False

        elif is_simple_definition_question(normalized_question):
            # V6.16:
            # Retrieval relevance alone does not prove that the document
            # contains a definition. If explicit grounded definition recovery
            # failed, do not let the LLM fill the gap from model memory.
            document_answer = (
                "I could not find sufficient information "
                "in the provided documents."
            )
            fallback = True

        else:
            # 6. Build context
            t = time.perf_counter()
            context = build_context(candidates)
            timings["Context Build"] = time.perf_counter() - t

            # 7. LLM generation via Gemini
            t = time.perf_counter()
            document_answer = ask_llm(
                normalized_question,
                context
            )
            timings["LLM Generation"] = time.perf_counter() - t

            # 8. Completeness validation
            t = time.perf_counter()
            document_answer, completeness_added = complete_grounded_list_answer(
                normalized_question,
                document_answer,
                candidates
            )
            timings["Completeness Check"] = time.perf_counter() - t

            fallback = (
                not document_answer_is_usable(document_answer)
                or not definition_answer_is_grounded(
                    normalized_question,
                    document_answer,
                    candidates
                )
            )

        # 9. Collect unique sources
        if not fallback:
            seen = set()

            for candidate in candidates:
                metadata = candidate["metadata"]
                source = metadata.get("source", "Unknown")
                page = metadata.get("page", "Unknown")

                key = (source, page)

                if key not in seen:
                    seen.add(key)
                    sources.append({
                        "source": source,
                        "page": page
                    })

            sources = sources[:4]

    total_time = time.perf_counter() - total_start

    # 10. Structured result for UI/evaluation
    return {
        "question": question,
        "retrieval_question": retrieval_question,
        "normalized_question": normalized_question,
        "answer": document_answer,
        "evidence_label": evidence_label,
        "evidence_details": evidence_details,
        "sources": sources,
        "fallback": fallback,
        "used_fast_recovery": bool(recovered_answer),
        "used_llm": timings["LLM Generation"] > 0.0,
        "timings": timings,
        "total_time": total_time,
    }

# =========================================================
# V6.15 - AUTOMATED RAG EVALUATION
# =========================================================
with st.expander("🧪 V6.15 RAG Evaluation", expanded=False):
    st.caption(
        "Run automated regression tests against the reusable RAG engine."
    )

    if st.button("Run ADOP Test", key="run_adop_evaluation"):

        with st.spinner("Running evaluation..."):

            result = run_rag_query(
                "What is ADOP?",
                previous_messages=[]
            )

        answer_lower = result["answer"].lower()

        content_pass = (
            "online patching tool" in answer_lower
        )

        evidence_pass = (
            result["evidence_label"] == "STRONG"
        )

        source_pass = (
            len(result["sources"]) > 0
        )

        llm_pass = (
            result["used_llm"] is False
        )

        overall_pass = all([
            content_pass,
            evidence_pass,
            source_pass,
            llm_pass
        ])

        if overall_pass:
            st.success("✅ ADOP Regression Test: PASS")
        else:
            st.error("❌ ADOP Regression Test: FAIL")

        st.write("**Question:** What is ADOP?")
        st.write("**Answer:**", result["answer"])

        st.write(
            {
                "Expected content": content_pass,
                "Evidence STRONG": evidence_pass,
                "Source present": source_pass,
                "LLM bypassed": llm_pass,
                "Total time": round(
                    result["total_time"],
                    2
                )
            }
        )
# =========================================================
# V6.15 - AUTOMATED RAG REGRESSION SUITE
# =========================================================
with st.expander("🧪 V6.15 RAG Evaluation", expanded=False):
    st.caption(
        "Automated regression tests for retrieval, grounding, "
        "citations, hallucination protection and LLM bypass."
    )

    if st.button(
        "Run Full Regression Suite",
        key="run_full_rag_evaluation"
    ):

        test_cases = [
            {
                "name": "ADOP Definition",
                "question": "What is ADOP?",
                "required_terms": [
                    "online patching tool"
                ],
                "evidence": "STRONG",
                "min_sources": 1,
                "max_sources": None,
                "fallback": False,
                "llm": False,
            },
            {
                "name": "ADOP Phases",
                "question": "adop phases?",
                "required_terms": [
                    "prepare",
                    "apply",
                    "finalize",
                    "cutover",
                    "cleanup",
                ],
                "evidence": "STRONG",
                "min_sources": 1,
                "max_sources": None,
                "fallback": False,
                "llm": False,
            },
            {
                "name": "ADPATCH R12.2",
                "question": "What is ADPATCH in oracle apps 12.2?",
                "required_terms": [
                    "adop"
                ],
                "evidence": "STRONG",
                "min_sources": 1,
                "max_sources": None,
                "fallback": False,
                "llm": False,
            },
            {
                "name": "Product Management",
                "question": "What is Product Management?",
                "required_terms": [
                    "product"
                ],
                "evidence": "STRONG",
                "min_sources": 1,
                "max_sources": None,
                "fallback": False,
                "llm": False,
            },
            {
                "name": "Unsupported Question Guard",
                "question": "What is the capital of France?",
                "required_terms": [
                    "could not find sufficient information"
                ],
                "evidence": "WEAK",
                "min_sources": 0,
                "max_sources": 0,
                "fallback": True,
                "llm": False,
            },
            {
                "name": "CCNA Unsupported Definition",
                "question": "What is CCNA?",
                "required_terms": [
                    "could not find sufficient information"
                ],
                "evidence": "STRONG",
                "min_sources": 0,
                "max_sources": 0,
                "fallback": True,
                "llm": False,
            },
        ]

        evaluation_results = []

        progress = st.progress(0)

        for index, test in enumerate(test_cases, start=1):

            with st.spinner(
                f"Running {test['name']}..."
            ):
                result = run_rag_query(
                    test["question"],
                    previous_messages=[]
                )

            answer_lower = result["answer"].lower()

            content_pass = all(
                term.lower() in answer_lower
                for term in test["required_terms"]
            )

            evidence_pass = (
                result["evidence_label"]
                == test["evidence"]
            )

            source_count = len(result["sources"])

            source_pass = (
                source_count >= test["min_sources"]
            )

            if test["max_sources"] is not None:
                source_pass = (
                    source_pass
                    and source_count <= test["max_sources"]
                )

            fallback_pass = (
                result["fallback"]
                == test["fallback"]
            )

            llm_pass = (
                result["used_llm"]
                == test["llm"]
            )

            overall_pass = all([
                content_pass,
                evidence_pass,
                source_pass,
                fallback_pass,
                llm_pass,
            ])

            evaluation_results.append({
                "Test": test["name"],
                "Status": (
                    "PASS"
                    if overall_pass
                    else "FAIL"
                ),
                "Content": content_pass,
                "Evidence": evidence_pass,
                "Sources": source_pass,
                "Fallback": fallback_pass,
                "LLM": llm_pass,
                "Time (sec)": round(
                    result["total_time"],
                    2
                ),
            })

            progress.progress(
                index / len(test_cases)
            )

        passed_tests = sum(
            1
            for item in evaluation_results
            if item["Status"] == "PASS"
        )

        total_tests = len(evaluation_results)

        if passed_tests == total_tests:
            st.success(
                f"✅ Regression Suite PASS: "
                f"{passed_tests}/{total_tests} tests passed."
            )
        else:
            st.error(
                f"❌ Regression Suite FAIL: "
                f"{passed_tests}/{total_tests} tests passed."
            )

        st.dataframe(
            evaluation_results,
            use_container_width=True,
            hide_index=True
        )
# =========================================================
# USER QUESTION
# =========================================================
question = st.chat_input("Ask a question from your enterprise documents...")

if question:
    total_start = time.perf_counter()
    timings = {
        "Follow-up Rewrite": 0.0,
        "Retrieval": 0.0,
        "Evidence Guard": 0.0,
        "Context Build": 0.0,
        "LLM Generation": 0.0,
        "Completeness Check": 0.0,
        "General AI": 0.0,
    }

    previous_messages = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Understanding conversation..."):
            try:
                # V6.18:
                # Main Streamlit chat and automated regression tests now use
                # the same reusable RAG execution pipeline.
                result = run_rag_query(
                    question,
                    previous_messages=previous_messages
                )

                retrieval_question = result["retrieval_question"]
                normalized_question = result["normalized_question"]
                document_answer = result["answer"]
                evidence_label = result["evidence_label"]
                evidence_details = result["evidence_details"]
                sources = result["sources"]
                fallback = result["fallback"]
                timings = result["timings"]
                total_time = result["total_time"]

                if fallback:
                    st.markdown("### 📚 Trusted Document RAG Status")
                    st.markdown(document_answer)
                    st.info(
                        "No automatic General AI answer was generated because the "
                        "uploaded PDFs did not provide sufficient evidence."
                    )

                    saved_answer = (
                        "### 📚 Trusted Document RAG Status\\n\\n" + document_answer +
                        "\\n\\nℹ️ No automatic General AI answer was generated because "
                        "the uploaded PDFs did not provide sufficient evidence."
                    )
                else:
                    st.markdown(document_answer)
                    if sources:
                        st.markdown("**Sources**")
                        for source in sources:
                            st.markdown(
                                f"""
                                <div class="source-card">
                                    📄 <b>{source['source']}</b><br>
                                    <span>PDF Page {source['page']}</span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    saved_answer = document_answer

                # V6.18:
                # total_time comes directly from run_rag_query() so the UI
                # displays the same timing used by automated evaluation.

                st.markdown("---")
                evidence_css = {
                    "STRONG": "evidence-strong",
                    "PARTIAL": "evidence-partial",
                    "WEAK": "evidence-weak",
                }.get(evidence_label, "evidence-partial")

                # V6.17:
                # Separate retrieval relevance from final answer support.
                # A document can be highly relevant to the topic while still
                # lacking enough evidence to answer the exact question.
                answer_support = (
                    "INSUFFICIENT" if fallback else "GROUNDED"
                )

                answer_support_css = (
                    "evidence-weak"
                    if fallback
                    else "evidence-strong"
                )

                evidence_html = (
                    f'<div class="evidence-strip">'
                    f'<span style="color:#65758b;font-size:.84rem;">'
                    f'Retrieval Relevance:</span> '
                    f'<span class="evidence-badge {evidence_css}">'
                    f'{evidence_label}</span> '
                    f'<span style="color:#65758b;font-size:.84rem;">'
                    f'Answer Support:</span> '
                    f'<span class="evidence-badge {answer_support_css}">'
                    f'{answer_support}</span> '
                    f'<span style="color:#65758b;font-size:.84rem;">'
                    f'📎 {len(sources)} source(s) &nbsp;•&nbsp; '
                    f'⏱️ {total_time:.2f} sec</span>'
                    f'</div>'
                )

                st.markdown(
                    evidence_html,
                    unsafe_allow_html=True
                )

                with st.expander("⚙️ Technical Details"):
                    st.code(
                        f"Follow-up Rewrite : {timings['Follow-up Rewrite']:.2f} sec\n"
                        f"Retrieval         : {timings['Retrieval']:.2f} sec\n"
                        f"Evidence Guard    : {timings['Evidence Guard']:.2f} sec\n"
                        f"Context Build     : {timings['Context Build']:.2f} sec\n"
                        f"LLM Generation    : {timings['LLM Generation']:.2f} sec\n"
                        f"Completeness Check: {timings['Completeness Check']:.2f} sec\n"
                        f"General AI        : {timings['General AI']:.2f} sec\n"
                        f"TOTAL             : {total_time:.2f} sec"
                    )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": saved_answer,
                    "sources": sources
                })

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to Gemini. Please check GEMINI_API_KEY and your internet connection.")
            except Exception as e:
                st.error(f"Error: {e}")
