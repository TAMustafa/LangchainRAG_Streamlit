# RAG Q&A Application

A Streamlit-based application that leverages LangChain and OpenAI to create an intelligent RAG Q&A system based on provided data. The application processes PDF documentation and provides accurate, context-aware responses to user questions.

## Features

- PDF document processing and vectorization
- Semantic search with similarity threshold filtering
- Interactive web interface built with Streamlit
- Context-aware responses using GPT-4 Turbo
- Environment-based configuration
- Vector store persistence for improved performance

## Prerequisites

- Python 3.8+
- OpenAI API key
- Required Python packages (see Installation section)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd aws-certification-qa
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install streamlit langchain-openai langchain-community langchain-core chromadb python-dotenv
```

4. Create a `.env` file in the project root and add your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

## Project Structure

```
aws-certification-qa/
├── data/
│   └── Data_Engineer.pdf
├── db/
│   └── chroma/
├── .env
├── app.py
└── README.md
```

## Configuration

The application uses several key configurations:

- **File Path**: PDF document location (`./data/Data_Engineer.pdf`)
- **Database Path**: Vector store location (`db/chroma`)
- **Embedding Model**: OpenAI text-embedding-3-large with 3072 dimensions
- **LLM Model**: GPT-4 Turbo Preview with temperature 0.1
- **Chunk Settings**: 1000 characters with 200 character overlap

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically `http://localhost:8501`)

3. Enter your AWS certification-related question in the text input field

4. Click "Submit" to receive an AI-generated response based on the provided documentation

## How It Works

1. **Document Processing**:
   - Loads PDF document using PyPDFLoader
   - Splits content into manageable chunks using RecursiveCharacterTextSplitter
   - Creates embeddings using OpenAI's text-embedding-3-large model

2. **Question Processing**:
   - Uses similarity search to find relevant document chunks
   - Applies a similarity score threshold of 0.3
   - Retrieves up to 5 most relevant chunks

3. **Response Generation**:
   - Combines retrieved context with the user's question
   - Processes through GPT-4 Turbo with system prompt
   - Returns context-aware response

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

MIT

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [LangChain](https://python.langchain.com/)
- Uses [OpenAI](https://openai.com/) models
- Vector storage by [Chroma](https://www.trychroma.com/)
