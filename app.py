import streamlit as st
from langchain.vectorstores import FAISS
#from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from dotenv import load_dotenv
import os

# Load FAISS index
INDEX_FOLDER = "faiss_index"
INDEX_FAISS_FILE = "index.faiss"
INDEX_PKL_FILE = "index.pkl"

load_dotenv() 
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API Key not found in .env file.")

hf_token = os.getenv("hf_token")
if not hf_token:
    raise ValueError("huggingface Key not found in .env file.")

def load_faiss_index(index_folder):
    """Load FAISS index from the specified folder."""
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=hf_token, model_name="sentence-transformers/all-MiniLM-l6-v2")
    return FAISS.load_local(folder_path=index_folder,embeddings=embeddings,allow_dangerous_deserialization=True)


# Initialize Streamlit App
st.title("Document Search and Query Response App")

# # Input API Key
# st.sidebar.header("Configuration")
# openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")



# if not openai_api_key:
#     st.warning("Please enter your OpenAI API Key in .env file")
#     st.stop()

# Step 1: User Query Input
query = st.text_input("Enter your query:")

if query:
    try:
        # Step 2: Load the FAISS Index
        st.write("Loading FAISS index...")
        vectorstore = load_faiss_index(INDEX_FOLDER)

        # # Retrieve relevant documents
        # st.write("Searching for relevant documents...")
        # retriever = vectorstore.as_retriever()
        # relevant_docs = retriever.get_relevant_documents(query)

        # Retrieve relevant documents using similarity search
        st.write("Searching for relevant documents...")
        relevant_docs = vectorstore.similarity_search(query)

        if not relevant_docs:
            st.warning("No relevant documents found for the query.")
        else:
            st.write(f"Found {len(relevant_docs)} relevant document(s). Generating response...")

            # Step 3: Generate Response using OpenAI Model
            #llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_api_key)
            llm = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                    openai_api_key=openai_api_key
            )
            knowledge_context = "\n".join([doc.page_content for doc in relevant_docs])
            response = llm.invoke(f"Based on the following documents, answer the query: {query}\n\n{knowledge_context}")

            # Display Response
            st.subheader("Response:")
            st.write(response)

            # Option to Display Relevant Documents
            with st.expander("Relevant Documents:"):
                for i, doc in enumerate(relevant_docs):
                    st.write(f"**Document {i + 1}:**")
                    st.write(doc.page_content)

    except Exception as e:
        st.error(f"An error occurred: {e}")