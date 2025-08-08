import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# 📄 Streamlit Page Config
st.set_page_config(page_title="Contract Inspector", page_icon="📄", layout="wide")

# 🔑 Load OpenAI API key from Streamlit Secrets
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Please set your OpenAI API key in Streamlit Secrets.")
    st.stop()

# 📦 Load FAISS vector store
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    return FAISS.load_local("faiss_index_openai", embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()

# 🤖 Make Retrieval QA Chain
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

# 🎨 Gradient CSS
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
  position: fixed; /* Fixed positioning to cover the entire page */
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: -1; /* Background layer below all content */
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
  position: relative; /* Ensure it’s in a new stacking context */
  z-index: 1; /* Elevate above the gradient background */
  color: black; /* Ensure text is readable */
}
.footer {
  text-align: center;
  padding: 1rem;
  color: white;
  font-size: 0.9rem;
  background-color: rgba(0, 0, 0, 0.5);
  margin-top: 1rem;
  position: relative;
  z-index: 1; /* Keep footer above gradient */
}
</style>
""", unsafe_allow_html=True)

# 🖥 UI Layout
st.markdown('<div class="gradient-bg">', unsafe_allow_html=True)  # Background layer
st.markdown('<div style="padding: 2rem;">', unsafe_allow_html=True)  # Content wrapper with padding

st.markdown("<h1 style='text-align:center;color:black;'>📄 CONTRACT INSPECTOR</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:black;'>Analyze policies, contracts, or emails and get clear, detailed answers instantly.</p>", unsafe_allow_html=True)

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
st.markdown('</div>', unsafe_allow_html=True)  # Close custom-box

# 🦶 Footer Layout (stacked below)
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("<p>© 2025 Contract Inspector | Developed by Gale_Ops  | <a href='mailto:your-email@example.com' style='color:white;'>Contact Us</a></p>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close content wrapper
st.markdown('</div>', unsafe_allow_html=True)  # Close gradient-bg
