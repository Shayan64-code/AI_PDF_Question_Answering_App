import hashlib
import tempfile

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb import PersistentClient


def ingest_pdf(uploaded_file):

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    pdf_docs = loader.load()

    # Add metadata
    for doc in pdf_docs:
        doc.metadata = {
            "source": uploaded_file.name,
            "doc_id": uploaded_file.name,
            "page": doc.metadata.get("page", 0)
        }

    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    split_docs = splitter.split_documents(pdf_docs)

    # Create chunks + SHA256 IDs
    gen_chunks = []
    ids = []

    for chunk in split_docs:

        metadata = chunk.metadata.copy()

        hash_id = hashlib.sha256(
            chunk.page_content.encode()
        ).hexdigest()

        metadata["chunk_id"] = hash_id

        gen_chunks.append(
            Document(
                page_content=chunk.page_content,
                metadata=metadata
            )
        )

        ids.append(hash_id)

    # Embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Connect to Chroma DB
    client = PersistentClient(path="./data/chroma_db")

    vectorstore = Chroma(
        persist_directory="./data/chroma_db",
        embedding_function=embedding_model,
        collection_name="genai_docs"
    )

    # Add documents
    vectorstore.add_documents(
        documents=gen_chunks,
        ids=ids
    )

    return len(gen_chunks)

# vectorstore._collection.count()