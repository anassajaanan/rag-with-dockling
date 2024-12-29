import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add this near the top of your file, after imports
if not os.getenv("OPENAI_API_KEY"):
    st.error("OpenAI API key not found. Please add it to your .env file.")
    st.stop()

# Make sure OPENAI_API_KEY is set for the application
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

import gc
import tempfile
import uuid
import fitz  # PyMuPDF for PDF handling

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.node_parser import SentenceSplitter

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

@st.cache_resource
def load_llm():
    llm = OpenAI(model="gpt-4", temperature=0)
    return llm

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

def display_pdf(file):
    print("\n\n ############### Displaying the File ###############")
    print(file)
    st.markdown("### PDF Preview")
    # Replace st.pdf_viewer with pdf_viewer component
    binary_data = file.getvalue()
    pdf_viewer(binary_data)

# Initialize PDF file reference in session state
if "pdf_ref" not in st.session_state:
    st.session_state.pdf_ref = None

with st.sidebar:
    st.header(f"Add your documents!")
    
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type=["pdf"])

    if uploaded_file:
        # Store PDF reference in session state
        st.session_state.pdf_ref = uploaded_file
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")

                if file_key not in st.session_state.get('file_cache', {}):

                    if os.path.exists(temp_dir):
                            reader = DoclingReader()
                            loader = SimpleDirectoryReader(
                                input_dir=temp_dir,
                                file_extractor={".pdf": reader},
                            )
                    else:    
                        st.error('Could not find the file you uploaded, please check again...')
                        st.stop()
                    
                    docs = loader.load_data()

                    # setup llm & embedding model
                    llm = load_llm()
                    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
                    # Creating an index over loaded data
                    Settings.embed_model = embed_model
                    # Modify the node parser configuration based on official MarkdownNodeParser implementation
                    node_parser = MarkdownNodeParser.from_defaults(
                        include_metadata=True,
                        include_prev_next_rel=True,
                    )
                    
                    # If you need to control chunk size, use the TextSplitter
                    text_splitter = SentenceSplitter(
                        chunk_size=1024,
                        chunk_overlap=200
                    )
                    
                    # Combine both parsers in the transformations
                    index = VectorStoreIndex.from_documents(
                        documents=docs, 
                        transformations=[node_parser, text_splitter], 
                        show_progress=True
                    )

                    # Create the query engine, where we use a cohere reranker on the fetched nodes
                    Settings.llm = llm
                    query_engine = index.as_query_engine(streaming=True)

                    # ====== Customise prompt template ======
                    qa_prompt_tmpl_str = (
                        "You are an expert document analyst specializing in extracting and analyzing information from PDF documents. "
                        "Your task is to provide accurate, detailed answers based on the provided PDF content.\n"
                        "---------------------\n"
                        "Context information from the PDF:\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "Given the context information above, please follow these guidelines:\n"
                        "1. Provide specific answers based on the document content\n"
                        "2. Reference specific sections or pages when relevant\n"
                        "3. If information spans multiple sections, synthesize it coherently\n"
                        "4. Maintain the original document's terminology and context\n"
                        "5. If you cannot find specific information in the document, clearly state: 'I cannot find this information in the provided document'\n"
                        "6. If the context is ambiguous, explain the limitations of the available information\n"
                        "7. Format your response in a clear, structured manner\n\n"
                        "Query: {query_str}\n"
                        "Answer: "
                    )

                    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

                    # Update the query engine with the new prompt
                    query_engine.update_prompts(
                        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                    )
                    
                    st.session_state.file_cache[file_key] = query_engine
                else:
                    query_engine = st.session_state.file_cache[file_key]

                # Inform the user that the file is processed and Display the PDF uploaded
                st.success("Ready to Chat!")
                # Use session state reference for display
                if st.session_state.pdf_ref:
                    display_pdf(st.session_state.pdf_ref)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()     

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"RAG over Excel using Dockling 🐥 &  OpenAI 🤖")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Simulate stream of response with milliseconds delay
        streaming_response = query_engine.query(prompt)
        
        
        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        # full_response = query_engine.query(prompt)

        message_placeholder.markdown(full_response)
        # st.session_state.context = ctx

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})