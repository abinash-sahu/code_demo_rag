from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings


load_dotenv()  # Load environment variables from .env file
# openai_api_key = os.getenv("OPENAI_API_KEY")
# if not openai_api_key:
#     raise ValueError("OpenAI API Key not found in .env file.")

hf_token = os.getenv("hf_token")
if not hf_token:
    raise ValueError("huggingface Key not found in .env file.")

def load_pdfs_from_folder(folder_path):
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]
    documents = []
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        docs = loader.load()
        documents.extend(docs)
    return documents


def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def create_embeddings_and_store(documents):
    #embeddings = OpenAIEmbeddings(model="text-embedding-3-small",openai_api_key=openai_api_key)  # Replace with OpenAI or SentenceTransformers
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=hf_token, model_name="sentence-transformers/all-MiniLM-l6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def save_faiss_index(vectorstore, index_path="faiss_index"):
    vectorstore.save_local(index_path)


def pipeline(folder_path, index_path="faiss_index"):
    print("Loading PDFs...")
    documents = load_pdfs_from_folder(folder_path)
    print(f"Loaded {len(documents)} documents from PDFs.")
    
    print("Splitting documents...")
    split_docs = split_documents(documents)
    print(f"Split into {len(split_docs)} chunks.")

    print("Creating embeddings and storing in FAISS...")
    vectorstore = create_embeddings_and_store(split_docs)
    save_faiss_index(vectorstore, index_path)
    print(f"FAISS index saved at {index_path}.")


if __name__ == "__main__":
    folder_path = "data"  # Replace with the folder containing your PDFs
    pipeline(folder_path)
