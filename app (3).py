import os
import subprocess
try:
    import langchain_community
except ImportError:
    subprocess.check_call(["pip", "install", "langchain-community", "faiss-cpu", "langchain-openai", "pypdf", "tiktoken", "sentence-transformers"])

import streamlit as st
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Contract Inspector", page_icon="📄", layout="wide")


if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Please set your OpenAI API key in Streamlit Secrets.")
    st.stop()

@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local("faiss_index_openai", embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()

@st.cache_resource
def make_qa_chain():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

qa_chain = make_qa_chain()

st.markdown("""
<style>
:root {
  --bg-color-1: hsl(0, 80%, 60%);
  --bg-color-2: hsl(120, 80%, 60%);
  --bg-color-3: hsl(240, 80%, 60%);
  --gradient-angle: 135deg;
}
.gradient-bg {
  background: linear-gradient(var(--gradient-angle), var(--bg-color-1), var(--bg-color-2), var(--bg-color-3));
  background-size: 200% 200%;
  animation: gradientShift 20s ease infinite;
}
@keyframes gradientShift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
.custom-box {
  background-color: rgba(17, 24, 39, 0.85);
  padding: 1.5rem;
  border-radius: 1rem;
  box-shadow: 0 6px 24px rgba(0, 0, 0, 0.4);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="gradient-bg" style="padding:2rem;min-height:100vh;">', unsafe_allow_html=True)
st.markdown("<h1 style='text-align:center;color:white;'>📄 CONTRACT INSPECTOR</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:white;'>Analyze policies, contracts, or emails and get clear, detailed answers instantly.</p>", unsafe_allow_html=True)

st.markdown('<div class="custom-box">', unsafe_allow_html=True)
query = st.text_area("Enter your query", placeholder="e.g., '46M, knee surgery, Pune, 3-month policy'")
if st.button("Submit Query"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Analyzing the documents..."):
            result = qa_chain.invoke({"query": query})
            st.subheader("Answer")
            st.write(result["result"])
            with st.expander("Sources"):
                for doc in result["source_documents"]:
                    src = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "Unknown")
                    st.markdown(f"**{src}** — page {page}")
                    st.write(doc.page_content[:400] + "...")
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
