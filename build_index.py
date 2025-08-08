# build_index.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import time

# 🔑 Load OpenAI API key
# It's better to load this from environment variables
# For local testing, you might need to load it from a .env file
# For deployment, you will set this in your deployment service's secrets.
from dotenv import load_dotenv
load_dotenv()

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("Please set your OpenAI API key in the environment variables.")

SOURCE_DIR = "source_documents"
INDEX_PATH = "faiss_index_openai"

def build_index():
    """Scans for PDFs, processes them, and saves a new FAISS index."""
    print("Starting to build the FAISS index...")
    docs = []

    # 1. Load documents from the source directory
    for filename in os.listdir(SOURCE_DIR):
        if filename.endswith(".pdf"):
            path = os.path.join(SOURCE_DIR, filename)
            print(f"  -> Loading: {filename}")
            loader = PyPDFLoader(path)
            docs.extend(loader.load())

    if not docs:
        print("No PDF documents found. Exiting.")
        return

    # 2. Split documents into chunks
    print(f"Loaded {len(docs)} pages. Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # 3. Create embeddings and FAISS index
    print(f"Creating embeddings for {len(splits)} chunks...")
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    # 4. Save the FAISS index locally
    print(f"Saving index to: {INDEX_PATH}")
    vectorstore.save_local(INDEX_PATH)

    # 5. Create a timestamp file to signal completion
    with open("last_updated.txt", "w") as f:
        f.write(str(time.time()))

    print("✅ Index built and saved successfully!")

if __name__ == "__main__":
    build_index()
