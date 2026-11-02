import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from dotenv import load_dotenv

load_dotenv()

def load_documents(file_path: str) -> List[str]:
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        docs = []
        for doc in data:
            if "text" in doc:
                text = doc["text"]
                # Strip the Source: https://... line
                import re
                text = re.sub(r"^Source: https?://[^\s]+\s*", "", text)
                docs.append(text)
        return docs
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON.")
        return []

def create_vector_store(documents: List[str], vector_store_path: str) -> FAISS:
    print(f"Initializing embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print(f"Splitting {len(documents)} documents into smaller chunks...")
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, #  It takes a massive block of text and cuts it into smaller paragraphs of 400 characters each
        chunk_overlap=50, # It also uses an overlap of 50 characters between paragraphs to maintain the context
        length_function=len,
        is_separator_regex=False,
    )
    chunked_documents = text_splitter.create_documents(documents)

    print(f"Generating embeddings for {len(chunked_documents)} chunks. This may take a minute...")
    # Use from_documents since chunked_documents is a list of Document objects
    vector_store = FAISS.from_documents(chunked_documents, embedding=embedding_model)

    print(f"Saving vector store to {vector_store_path}...")
    vector_store.save_local(vector_store_path)
    print(f"FAISS index successfully saved at {vector_store_path}")
    
    return vector_store

if __name__ == "__main__":
    document_file_path = "document.json"
    documents = load_documents(document_file_path)
    
    if documents:
        VECTOR_STORE_PATH = "faiss_index"
        create_vector_store(documents, VECTOR_STORE_PATH)
    else:
        print("No documents found to index. Did you run scraper.py first?")
