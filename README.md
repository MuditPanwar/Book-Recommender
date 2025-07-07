# 📚 Book Recommender (WIP)

> A semantic book recommendation system built in Python, powered by OpenAI embeddings and LangChain, with an interactive Gradio frontend.

---

## 🚧 Current Status
> ⚙️ **Under Development**  
> - Core ingestion, embedding, and vector-store pipeline in place  
> - Basic Gradio UI up, but styling & error handling still TODO  
> - No formal evaluation yet—benchmarks forthcoming

---

## 🔍 Project Overview
This project lets users discover new books based on semantic similarity. You give it a book title or description, and it finds the most relevant reads in the “7k Books” dataset.

- **Approach**: Embedding-based retrieval via OpenAI + LangChain  
- **UI**: Lightweight Gradio app for entering queries and displaying top-K results  
- **Data**: “7k Books” metadata & descriptions from Kaggle  

---

## 🛠 Tech Stack & Dependencies
- **Python** ≥3.8  
- **openai** (for embeddings & text APIs)  
- **LangChain** (manages prompt → embedding → retrieval flow)  
- **Gradio** (demo web interface)  
- **pandas**, **numpy** (data loading & preprocessing)  
- **faiss-cpu** (local vector index)  
