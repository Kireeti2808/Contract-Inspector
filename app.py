import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
# New, correct import
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# --- 1. STREAMLIT PAGE CONFIG ---
st.set_page_config(page_title="Contract Inspector", page_icon="📄", layout="wide")

# --- 2. LOAD OPENAI API KEY ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Please set your OpenAI API key in Streamlit Secrets.")
    st.stop()

# --- 3. DEFINE THE DESIRED JSON STRUCTURE ---
class ClaimDecision(BaseModel):
    decision: str = Field(description="The final decision, e.g., 'Approved', 'Rejected', or 'Partially Approved'.")
    amount: float = Field(description="The approved payout amount. Should be 0.0 if rejected.")
    justification: str = Field(description="A detailed explanation for the decision, citing specific rules or clauses.")
    source_clauses: list[str] = Field(description="A list of direct quotes from the source documents that support the justification.")

# --- 4. LOAD VECTORSTORE AND SETUP RETRIEVER ---
@st.cache_resource
def load_components():
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    vectorstore = FAISS.load_local("faiss_index_openai", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 10})
    return retriever

retriever = load_components()

# --- 5. CREATE THE ADVANCED RAG CHAIN ---
@st.cache_resource
def make_structured_qa_chain():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Instantiate the Pydantic parser
    parser = PydanticOutputParser(pydantic_object=ClaimDecision)

    # Create a detailed prompt template
    prompt_template = """
    You are an expert insurance claims analyst. Your task is to analyze the user's query and the provided document excerpts to make a decision.
    Base your decision STRICTLY on the context provided.
    
    CONTEXT:
    {context}
    
    QUERY:
    {query}
    
    INSTRUCTIONS:
    Respond with a JSON object formatted according to the following schema. Do not add any text before or after the JSON.
    
    {format_instructions}
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["query", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Create the LCEL chain
    chain = (
        RunnableParallel(
            context=(lambda x: x["query"]) | retriever,
            query=RunnablePassthrough()
        )
        | prompt
        | llm
        | parser
    )
    return chain

qa_chain = make_structured_qa_chain()


# --- 6. STREAMLIT UI ---
# (Your existing CSS and UI code can go here, but I've updated the logic part)
st.markdown("<h1 style='text-align:center;color:white;'>📄 CONTRACT INSPECTOR</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:white;'>Analyze policies, contracts, or emails and get clear, detailed answers instantly.</p>", unsafe_allow_html=True)

st.markdown('<div class="custom-box">', unsafe_allow_html=True)
query = st.text_area("Enter your query", placeholder="e.g., '46M, knee surgery, Pune, 3-month policy'")

if st.button("Submit Query"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Analyzing the documents..."):
            try:
                # Invoke the chain with the query dictionary
                result = qa_chain.invoke({"query": query})

                # Display the structured result
                st.subheader("Decision")
                st.success(f"*Status:* {result.decision}")
                st.info(f"*Amount:* ₹{result.amount:,.2f}")

                st.subheader("Justification")
                st.write(result.justification)

                with st.expander("Supporting Clauses"):
                    for clause in result.source_clauses:
                        st.markdown(f"> {clause}")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.warning("Could not process the query. The documents may not contain relevant information, or the query may be too ambiguous.")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">...</div>', unsafe_allow_html=True) # Your footer code
