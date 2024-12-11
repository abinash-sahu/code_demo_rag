import streamlit as st
from langchain.vectorstores import FAISS
#from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Initialize user query history in session state
if "user_queries" not in st.session_state:
    st.session_state.user_queries = []

# Function to add a user query to history
def add_user_query(query):
    st.session_state.user_queries.append(query)

# Sidebar for user query history
st.sidebar.title("Chat History")
if st.session_state.user_queries:
    for query in st.session_state.user_queries:
        st.sidebar.markdown(f"**🟦 You:** {query}")
else:
    st.sidebar.write("No messages yet.")

# Load FAISS index
INDEX_FOLDER = "faiss_index"
INDEX_FAISS_FILE = "index.faiss"
INDEX_PKL_FILE = "index.pkl"

load_dotenv() 
# openai_api_key = os.getenv("OPENAI_API_KEY")
# if not openai_api_key:
#     raise ValueError("OpenAI API Key not found in .env file.")

hf_token = os.getenv("hf_token")
if not hf_token:
    raise ValueError("huggingface Key not found in .env file.")

def load_faiss_index(index_folder):
    """Load FAISS index from the specified folder."""
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=hf_token, model_name="sentence-transformers/all-MiniLM-l6-v2")
    return FAISS.load_local(folder_path=index_folder,embeddings=embeddings,allow_dangerous_deserialization=True)

# Initialize Streamlit App
st.title("Document Search and Query Response App")

# Sidebar for Configuration and Session History
st.sidebar.header("Session and Configuration")

#Sidebar to display documents in 'data' folder
def display_document_list():
    st.sidebar.subheader("Available Documents")
    data_folder = "data"
    try:
        documents = os.listdir(data_folder)
        for doc in documents:
            st.sidebar.write(f"- {doc}")
    except FileNotFoundError:
        st.sidebar.write("No documents found in the 'data' folder.")

display_document_list()

# Step 1: User Query Input
query = st.text_input("Enter your query:")

if st.button("Send"):
    if query.strip():
        add_user_query(query.strip())

        try:
            # Step 2: Load the FAISS Index
            st.write("Loading FAISS index...")
            vectorstore = load_faiss_index(INDEX_FOLDER)

            # Retrieve relevant documents using similarity search
            st.write("Searching for relevant documents...")
            relevant_docs = vectorstore.similarity_search(query)

            if not relevant_docs:
                st.warning("No relevant documents found for the query.")
            else:
                st.write(f"Found {len(relevant_docs)} relevant document(s). Generating response...")

                # Step 3: Generate Response using OpenAI Model
                repo_id="mistralai/Mistral-7B-Instruct-v0.2"
                llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,token=hf_token)

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
