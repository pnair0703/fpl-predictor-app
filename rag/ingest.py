import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


NEWS_SOURCES = [
    "https://www.bbc.com/sport/football/premier-league",
    "https://www.skysports.com/premier-league-news",
]

def scrape_news():
    docs = []

    for url in NEWS_SOURCES:
        html = requests.get(url).text
        soup = BeautifulSoup(html, "html.parser")

        # Extract paragraphs
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text() for p in paragraphs)

        docs.append(text)

    return docs

def ingest_documents():
    raw_docs = scrape_news()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents(raw_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="rag_store"
    )

    db.persist()
    print("Finished building vector store.")

if __name__ == "__main__":
    ingest_documents()
