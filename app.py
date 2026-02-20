import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="MediRAG", page_icon="🩺", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f4f9ff;
}
h1 {
    color: #0b3d91;
}
.stChatMessage {
    border-radius: 15px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("🩺 MediRAG - Clinical AI Assistant")
st.warning("⚠️ For educational purposes only. Not for medical diagnosis.")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Number of retrieved chunks", 1, 5, 3)
    st.info("Model: LLaMA3 (Local via Ollama)")
    st.success("Embeddings: nomic-embed-text")

# ---------------- LOAD MODELS ----------------
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

llm = OllamaLLM(
    model="phi3",
    temperature=0.3,
    num_predict=300
)


prompt_template = """
You are a professional medical assistant.
Answer ONLY from the given context.
If the answer is not in the context, say "Insufficient information found."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate.from_template(prompt_template)

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["💬 Chat", "📚 Retrieved Context"])

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_context" not in st.session_state:
    st.session_state.last_context = ""

# ---------------- CHAT TAB ----------------
# ---------------- CHAT TAB ----------------
with tab1:
    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Ask a medical question..."):

        # Save user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Retrieve relevant docs
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        st.session_state.last_context = context

        # Format prompt
        final_prompt = prompt.format(context=context, question=question)

        # Stream LLM response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing medical literature..."):
                placeholder = st.empty()  # placeholder for streaming text
                response = ""
                for chunk in llm.stream(final_prompt):
                    response += chunk
                    placeholder.markdown(response)  # update in-place

        # Save assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
# ---------------- CONTEXT TAB ----------------
with tab2:
    st.subheader("Retrieved Knowledge Chunks")
    if st.session_state.last_context:
        st.write(st.session_state.last_context)
    else:
        st.info("No query asked yet.")

