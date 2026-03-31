from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from Keys import KEY1

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector DB
vectorstore = Chroma(
    persist_directory="./data/chroma_db",
    embedding_function=embedding_model,
    collection_name="genai_docs"
)


def retrieve_prompt_with_context(Question):

    ## Retrieve relevant documents
    # docs = vectorstore.similarity_search(Question, k=3)
    # # print(docs[0].metadata["chunk_id"])

    #more better
    retriever = vectorstore.as_retriever(search_kwargs={"k":3})
    docs = retriever.invoke(Question)

    context = "\n\n".join([doc.page_content for doc in docs]) #join just clean content and gives it to LL

    sources = [
        {
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page")
        }
        for doc in docs
    ]

    ## Prompt
    prompt = f"""
    You are a helpful assistant answering questions from documents.

    Rules:
    - Only use the provided context.
    - If the answer is not in the context, say:
    "I don't know based on the provided documents."

    Context:
    {context}

    Question:
    {Question}

    Answer:
    """
    
    return prompt, sources