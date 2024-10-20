import streamlit as st
import fitz  # PyMuPDF to extract PDF content
import pinecone
from pinecone import Pinecone
from pinecone.core.client.exceptions import NotFoundException
import openai
import os
import logging

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# Initialize Pinecone
def initialize_pinecone():
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        return pc.Index("pdf-doc-index")
    except Exception as e:
        logging.error(f"Failed to initialize Pinecone: {str(e)}")
        st.error("Failed to connect to Pinecone. Please check your configuration.")
        return None

# Initialize OpenAI
def initialize_openai():
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
    except Exception as e:
        logging.error(f"Failed to set OpenAI API key: {str(e)}")
        st.error("Failed to initialize OpenAI. Please check your configuration.")

# Initialize connections
index = initialize_pinecone()
initialize_openai()

# Initialize session state for uploaded documents
if 'uploaded_docs' not in st.session_state:
    st.session_state.uploaded_docs = []

# PDF Extraction Function
def extract_text_from_pdf(uploaded_file):
    try:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {str(e)}")
        st.error("Error extracting text from PDF. Please try again.")

### Embedding Generation Function ###
def generate_embedding(text):
    """Generate embeddings using OpenAI's embedding model."""
    try:
        response = openai.Embedding.create(
            input=[text], engine="text-embedding-ada-002"
        )
        return response['data'][0]['embedding']
    except Exception as e:
        logging.error(f"Error generating embedding: {str(e)}")
        st.error("Error generating embedding. Please try again.")

# Clean-up Function
def cleanup_uploaded_data():
    try:
        index.delete(delete_all=True)
        st.session_state.uploaded_docs.clear()
        st.session_state.uploaded_file = None
        st.success("Uploaded document data cleared successfully!")
    except NotFoundException:
        st.warning("No documents are uploaded, so nothing to clear.")
    except Exception as e:
        logging.error(f"Error clearing uploaded data: {str(e)}")
        st.error("Error clearing uploaded data. Please try again.")

### Generate Answer using OpenAI ###
def generate_answer(query, context):
    """Generate an answer based on the provided context."""
    prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    sys_prompt = (
        "You are a helpful and reliable assistant. Always provide answers strictly based on the given context. "
        "If the context does not contain the relevant information, respond with: 'The information is not available in the provided content.' "
        "Avoid making assumptions or generating answers beyond the content provided."
    )

    try:
        res = openai.ChatCompletion.create(
            model='gpt-4o-mini',
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return res['choices'][0]['message']['content'].strip()
    except Exception as e:
        logging.error(f"Error generating answer: {str(e)}")
        st.error("Error generating answer. Please try again.")


# Streamlit Frontend Layout
st.title("Interactive QA Bot")
st.write("Upload a PDF document and ask questions based on its content.")

# 1. **File Upload Section**
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    document_id = uploaded_file.name
    if document_id not in st.session_state.uploaded_docs:
        with st.spinner("Processing the document..."):
            document_text = extract_text_from_pdf(uploaded_file)
            if document_text:
                # Generate and store embeddings
                embedding = generate_embedding(document_text)
                index.upsert([(uploaded_file.name, embedding, {"text": document_text})])
                st.session_state.uploaded_docs.append(document_id)
                st.success("Document uploaded and indexed successfully!")
            else:
                st.info(f"Unable to read {document_id}")
    else:
        st.info(f"The document '{document_id}' is already uploaded.")

### Query Pinecone for Relevant Chunks ###
def query_pinecone(query, top_k=1):
    """Retrieve the most relevant document chunks."""
    try:
        query_embedding = generate_embedding(query)
        response = index.query(
            vector=query_embedding, top_k=top_k, include_metadata=True
        )
        # Combine the text from matched chunks
        context = "\n\n".join([match['metadata']['text'] for match in response['matches']])
        return context
    except Exception as e:
        logging.error(f"Error querying Pinecone: {str(e)}")
        st.error("Error querying document chunks. Please try again.")

# 2. **Show Number of Uploaded Documents**
uploaded_docs_count = len(st.session_state.uploaded_docs)
st.info(f"Number of Uploaded Documents: {uploaded_docs_count}")

# 3. **Two Side-by-Side Buttons: Expandable Box and Clear Button**
col1, col2 = st.columns([2, 1])

with col1:
    # Expandable box to show uploaded document names
    with st.expander("Show Uploaded Document Names", expanded=False):
        if uploaded_docs_count > 0:
            for doc_name in st.session_state.uploaded_docs:
                st.write(f"- {doc_name}")
        else:
            st.write("No documents uploaded yet.")

with col2:
    # Button to clear uploaded document data
    if st.button("Clear Uploaded document"):
        cleanup_uploaded_data()
         # Clear session state
        st.session_state.clear()

        # Trigger a rerun by setting a flag
        st.session_state["rerun"] = True

# 4. **Question Input Section and Answer Output**
query = st.text_input("Ask a question about the document:")

if query:
    with st.spinner("Generating answer..."):
        context = query_pinecone(query)
        logging.info("Context formed: %s", context)

        if context:  # Proceed if context retrieval was successful
            answer = generate_answer(query, context)

            st.subheader("Answer:")
            st.write(answer)

            st.subheader("Retrieved Document Segments:")
            st.write(context)