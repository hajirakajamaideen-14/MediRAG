MediRAG: A Retrieval-Augmented Generation System for Evidence-Based Clinical Question Answering






🚀 Project Overview

MediRAG is an AI-powered clinical knowledge assistant built with Retrieval-Augmented Generation (RAG). It answers medical questions strictly from documents, shows cited sources, tracks hallucinations, and provides a LLM vs SLM performance comparison.

It is ideal for academic projects, clinical research, and medical decision support.

🎯 Key Features

Source-Based Answering: Highlighted text, similarity score, document citations.

LLM vs SLM Comparison: Side-by-side evaluation of answer quality, latency, token usage, and hallucinations.

Evaluation Dashboard: Tracks retrieval precision, top-K accuracy, response time, token usage, and hallucination detection.

Professional UI: Dark medical theme, clean typography, left-panel controls, chat interface, right-panel context & sources.

Advanced Options: Hybrid search (BM25 + embeddings), medical safety guards, research mode, confidence thresholds.

🧰 Tech Stack

Backend: Python, Django / FastAPI

Frontend: React.js, Tailwind CSS

Models: GPT-based LLM, Smaller SLM

Vector Database: FAISS / Chroma

PDF Processing: PyPDF2 / PDFMiner

Deployment: Docker / Cloud hosting

⚡ Installation
# Clone the repository
git clone https://github.com/yourusername/MediRAG.git
cd MediRAG

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run backend server
python app.py   # Or `uvicorn main:app --reload` if FastAPI

# Start frontend
cd frontend
npm install
npm start
🖥 Usage

Upload medical PDFs in the left panel.

Select model (LLM / SLM) and Top-K retrieval.

Ask clinical questions in chat.

View grounded answers with highlighted sources, confidence, and similarity scores.

Compare LLM vs SLM performance side-by-side.

Monitor metrics on the evaluation dashboard.

📊 Performance Metrics

Retrieval Precision: Relevance of top-K chunks

Top-K Accuracy: Likelihood correct answer is in top-K

Response Time: Average query latency

Token Usage: Efficiency of model responses

Hallucination Detection: Identifies unsupported statements

🧠 Architecture
User Query
     ↓
Embedding Model
     ↓
Vector Database (FAISS/Chroma)
     ↓
Top-K Retrieval
     ↓
LLM / SLM
     ↓
Grounded Answer + Sources

Optional enhancements: reranking, hybrid search, research mode, safety disclaimers.

📝 Future Enhancements

Multi-language medical document support

Integration with PubMed / Medline

Interactive analytics dashboard

Hospital-specific personalized knowledge base

📚 References

Dense Passage Retrieval (DPR) – Facebook AI

FAISS – Facebook AI Similarity Search

RAG: Retrieval-Augmented Generation – Hugging Face

GPT and Transformer architectures

🏆 Project Highlights

Evidence-based answers

Professional UI

LLM vs SLM benchmarking

Detailed evaluation metrics

Tagline:
"MediRAG: Turning Medical Documents into Reliable Clinical Insights."
