from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate

# Load embedding model
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

# Load FAISS index
vectorstore = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Load LLM
llm = OllamaLLM(model="llama3")

# Prompt Template
prompt_template = """
You are a medical assistant. 
Answer ONLY using the provided context.
If the answer is not in the context, say "Insufficient information found."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate.from_template(prompt_template)

# Chat loop
while True:
    question = input("\nAsk your medical question (or type 'exit'): ")

    if question.lower() == "exit":
        break

    # Retrieve relevant chunks
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Format prompt
    final_prompt = prompt.format(context=context, question=question)

    # Generate answer
    response = llm.invoke(final_prompt)

    print("\nAnswer:\n")
    print(response)
