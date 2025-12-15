# rag/vector_store.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vectordb = Chroma(
        persist_directory="rag_store",
        embedding_function=embeddings,
    )

    return vectordb.as_retriever(search_kwargs={"k": 3})
