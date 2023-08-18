import os
from langchain.document_loaders.pubmed import PubMedLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
import pinecone

load_dotenv()

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


def ingest_docs() -> None:
    loader = PubMedLoader(query="covid", load_max_docs=10)
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=64, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} chunks")

    print(f"Inserting {len(documents)} to Pinecone")
    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(documents, embeddings, index_name="pubmed-index")
    print("***** Added to Pinecone Vectorstore. *****")


if __name__ == "__main__":
    ingest_docs()
