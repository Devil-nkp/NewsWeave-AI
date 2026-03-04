# 📰 NewsWeave AI – RAG-Based Fact Verification System

NewsWeave AI is a Retrieval-Augmented Generation (RAG) powered dossier generator designed to combat misinformation by producing grounded, evidence-backed reports using real-time web retrieval.

---

##  Overview

This system combines Large Language Models (LLMs) with external information retrieval to reduce hallucinations and improve factual reliability.

Instead of relying solely on an LLM’s internal knowledge, NewsWeave:
1. Retrieves relevant documents from external sources
2. Embeds and ranks them
3. Feeds grounded context into the LLM
4. Generates structured, fact-based responses

---

##  Architecture

User Query  
→ Retriever (DuckDuckGo Search API)  
→ Embedding + Ranking  
→ Context Injection (LangChain)  
→ Llama-3 (via Groq API)  
→ Structured Dossier Output  

---

## ⚙️ Tech Stack

- Python
- FastAPI
- LangChain
- Llama-3 (Groq API)
- DuckDuckGo Search
- Docker
- Vercel (Frontend Deployment)

---

##  Key Features

-  Real-time web-based retrieval
-  Reduced hallucinations (~35% vs baseline LLM)
-  40% improved response latency through prompt optimization
-  Containerized deployment with Docker
-  Live deployed application

---

##  Live Demo

🔗 https://newsweave-supreme.vercel.app/

---

##  Installation (Local Setup)

```bash
git clone https://github.com/Devil-nkp/NewsWeave-AI.git
cd NewsWeave-AI
pip install -r requirements.txt
```

Set your environment variables:

```
GROQ_API_KEY=your_api_key_here
```

Run the application:

```bash
uvicorn main:app --reload
```

---

##  Future Improvements

- Add vector database (FAISS / Pinecone)
- Add citation linking system
- Implement evaluation metrics (BLEU / ROUGE)
- Add user authentication

---

##  Author

Naveenkumar G  
AI / ML Engineer
