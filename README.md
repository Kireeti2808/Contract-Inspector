# 📄 Contract Inspector
Contract Inspector is a Streamlit-based application that uses LangChain, FAISS, and OpenAI to let you query insurance policy documents, contracts, or other uploaded PDFs.
It retrieves relevant clauses and presents concise answers along with source references.

## 🚀 Features
- Gradient-themed UI with smooth animations
- Natural language queries over multiple PDF documents
- Relevant clause extraction with source highlighting
- Secure API key handling using Streamlit Secrets
- FAISS vectorstore for fast document retrieval

## 📂 Project Structure
- `app.py`: Main Streamlit application
- `requirements.txt`: Dependency specifications
- `.gitignore`: Ignores sensitive files like `.streamlit/secrets.toml`
- `faiss_index_openai/`: FAISS index folder containing `index.faiss` and `index.pkl`
- `runtime.txt`: Specifies Python version (3.11)
