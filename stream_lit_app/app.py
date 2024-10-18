import streamlit as st
import fitz  # PyMuPDF to extract PDF content
import pinecone
from pinecone import Pinecone
import openai  # Use Cohere if you prefer
import os
import logging


# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "pdf-doc-index"
index = pc.Index(index_name)

# Set OpenAI API key (or Cohere)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to extract text from a PDF
def extract_text_from_pdf(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
    return text

# Function to generate embeddings using OpenAI's model (or switch to Cohere)
def generate_embedding(text):
    response = openai.Embedding.create(
        input=[text],
        engine="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

# Function to query Pinecone index
def query_pinecone(query, top_k=3):
    query_embedding = generate_embedding(query)
    response = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return response

# Function to generate answer using OpenAI GPT-3 (or use Cohere)
def generate_answer(query, context):
    prompt = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n" 
        f"{context}"

        f"\n\nQuestion: {query}\nAnswer:"
    )
    # instructions
    sys_prompt = "You are a helpful assistant that always answers questions based on the context provided. Do not make up the answers."
    # query text-davinci-003
    res = openai.ChatCompletion.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return res['choices'][0]['message']['content'].strip()

# Clean-up function to delete uploaded document from Pinecone
def cleanup_uploaded_data(doc_id):
    index.delete(ids=[doc_id])

# Streamlit frontend
st.title("Interactive QA Bot")
st.write("Upload a PDF document and ask questions based on its content.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing the document..."):
        # Extract text from the uploaded PDF
        document_text = extract_text_from_pdf(uploaded_file)
    
        # Generate and store embeddings
        embedding = generate_embedding(document_text)

        document_id = uploaded_file.name
        logging.info("Document id:%s\n", document_id)
        index.upsert([(uploaded_file.name, embedding, {"text": document_text})])
        
        st.success("Document uploaded and indexed successfully!")

    query = st.text_input("Ask a question about the document:")
    
    if query:
        with st.spinner("Generating answer..."):
            # Query Pinecone for relevant content
            response = query_pinecone(query)
            logging.info("The response data:%s\n", response)
            context = "\n".join([match['metadata']['text'] for match in response['matches']])
            logging.info("The context formed:%s\n", context)
            
            # Generate the final answer
            answer = generate_answer(query, context)
            
            st.subheader("Answer:")
            st.write(answer)

            st.subheader("Retrieved Document Segments:")
            for match in response['matches']:
                st.write(match['metadata']['text'])
    
    if st.button("Clear Uploaded Document Data"):
        cleanup_uploaded_data(document_id)
        st.success("Uploaded document data cleared successfully!")

