import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Constants
FILE_PATH = "./data/Prince2_2017Edition.pdf"
DB_PATH = "db/chroma"

# Use resource caching for unserializable objects
@st.cache_resource(show_spinner=False)
def get_vector_store():
    loader = PyPDFLoader(FILE_PATH)
    documents = loader.load()
    # Adjusted parameters for fewer, larger chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,        # Increased chunk size
        chunk_overlap=200,      # Reduced overlap
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    return Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=3072
        ),
        persist_directory=DB_PATH
    )

# Lazy initialization of vector store
vector_store = get_vector_store()
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.3, "filter": None}
)

llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    temperature=0.1
)

SYSTEM_TEMPLATE = """
You are an AWS certification expert. Answer questions using ONLY the provided context.
If the answer isn't in the context, say "I don't have information about that."
Keep answers technical but concise.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    ("user", "Question: {question}")
])

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def main():
    st.title("Prince2 (2017) Q&A")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    query = st.chat_input("Ask a question about Prince2 (2017)...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Processing..."):
            response = chain.invoke(query)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == '__main__':
    main()
