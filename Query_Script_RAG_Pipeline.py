from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from Keys import KEY1

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector DB
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model,
    collection_name="genai_docs"
)


def retrieve_prompt_with_context(Question):
    ## Retrieve relevant documents
    results = vectorstore.similarity_search(Question, k=3)
    # print(results[0].metadata["chunk_id"])

    context = "\n\n".join([doc.page_content for doc in results]) #join just clean content and gives it to LL
    # results

    ## Prompt
    prompt = f"""
        Answer the question using ONLY the provided context.

        Context:
        {context}

        Question:
        {Question}
        """
    
    return prompt