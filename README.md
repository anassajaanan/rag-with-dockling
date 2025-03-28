# PDF RAG Assistant with Dockling and Grok LAMA 3.1

A Streamlit-based Retrieval-Augmented Generation (RAG) application that allows you to upload PDF documents and chat with them using Grok's LAMA 3.1 model and Dockling for enhanced document parsing.

## Features

- **Upload and Process PDF Documents**: Seamlessly upload PDF files and have them indexed for intelligent querying
- **Interactive PDF Preview**: Built-in PDF viewer to reference documents while chatting
- **AI-Powered Document Analysis**: Leverages Grok's LAMA 3.1 for intelligent responses based on document content
- **Enhanced Document Parsing**: Uses Dockling for advanced PDF extraction capabilities
- **Streaming Responses**: Real-time response generation with a typing effect for better user experience
- **Session Management**: Maintains chat history and document context within the session

## Technical Implementation

### Core Technologies

- **Streamlit**: Powers the interactive web interface
- **LlamaIndex**: Provides the RAG framework for document indexing and retrieval
- **Grok LAMA 3.1**: Handles understanding and generating natural language responses
- **Dockling**: Specializes in parsing complex document formats like PDFs
- **PyMuPDF (fitz)**: Handles PDF file operations and rendering

### Architecture Overview

1. **Document Ingestion Pipeline**:
   - User uploads a PDF document through the Streamlit interface
   - Document is saved to a temporary directory and processed by Dockling
   - LlamaIndex creates a vector index from the extracted content

2. **Indexing and Retrieval**:
   - Text is split into semantic chunks using sentence splitters
   - Document structure is preserved using markdown node parsing
   - Grok LAMA 3.1 embeddings convert text chunks into vector representations
   - Vector store enables semantic search against the document content

3. **Query Processing**:
   - User inputs queries via the chat interface
   - Custom prompt template guides the LLM to provide document-specific responses
   - Retrieved document chunks provide context for the LLM
   - Response is streamed back to the user in real-time

4. **Session Management**:
   - Application maintains state across interactions
   - Document indexes are cached to improve performance
   - Chat history is preserved within the session

## Code Structure

The application consists of several key components:

### Environment Setup
```python
# Load environment variables and validate Grok API key
load_dotenv()
if not os.getenv("GROK_API_KEY"):
    st.error("Grok API key not found. Please add it to your .env file.")
    st.stop()
```

### LLM Configuration
```python
@st.cache_resource
def load_llm():
    llm = HuggingFaceInferenceAPI(model_name="Grok-1/LAMA-3.1", api_key=os.getenv("GROK_API_KEY"), temperature=0)
    return llm
```

### Document Processing
```python
# Document indexing with Dockling and LlamaIndex
reader = DoclingReader()
loader = SimpleDirectoryReader(
    input_dir=temp_dir,
    file_extractor={".pdf": reader},
)
docs = loader.load_data()

# Advanced document parsing configuration
node_parser = MarkdownNodeParser.from_defaults(
    include_metadata=True,
    include_prev_next_rel=True,
)
text_splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=200
)
```

### Customized RAG Prompt
```python
# Expert document analyst prompt template
qa_prompt_tmpl_str = (
    "You are an expert document analyst specializing in extracting and analyzing information from PDF documents. "
    "Your task is to provide accurate, detailed answers based on the provided PDF content.\n"
    # ... detailed instructions for the AI ...
)
```

### Chat Interface
```python
# Streamlit chat interface with streaming responses
for chunk in streaming_response.response_gen:
    full_response += chunk
    message_placeholder.markdown(full_response + "▌")
```

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/rag-with-dockling.git
   cd rag-with-dockling
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create a `.env` file with your Grok API key**:
   ```
   GROK_API_KEY=your_grok_api_key_here
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Access the application**:
   Open your browser and go to `http://localhost:8501`

## Usage

1. Upload your PDF document using the file uploader in the sidebar
2. Wait for the document to be processed and indexed
3. Start chatting with your document by entering questions in the chat input
4. View the AI responses based on the content of your document
5. Use the "Clear ↺" button to reset the chat and start a new conversation

## Requirements

The application requires Python 3.8+ and the packages listed in `requirements.txt`.

## License

[MIT License](LICENSE)

## Acknowledgments

- [LlamaIndex](https://www.llamaindex.ai/) for the RAG framework
- [Dockling](https://docling.ai/) for enhanced document parsing
- [Grok](https://grok.x.ai/) for the LAMA 3.1 model
- [Streamlit](https://streamlit.io/) for the web interface