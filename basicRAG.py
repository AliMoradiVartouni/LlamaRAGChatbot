import os
import re
import torch
from langchain_community.document_loaders import PDFPlumberLoader
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.core import StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from transformers import AutoTokenizer, BitsAndBytesConfig
import chromadb
from langchain_text_splitters import TokenTextSplitter

import warnings
warnings.filterwarnings("ignore", message="Field .* has conflict with protected namespace 'model_'")

class RAGChat:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.context_window = 2048  # TinyLlama's max context window
        # Calculate safe token limits
        self.max_new_tokens = 512  # Leave room for context
        self.chunk_size = 1024  # Half of context window
        self.chunk_overlap = 50
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_environment()
        self.initialize_models()
        self.setup_query_engine()

    def process_document(self, text: str) -> str:
        """Clean document by removing Q&A format markers and extra content"""
        # Remove page labels and file paths
        text = re.sub(r'file_path:.*?\n', '', text)
        text = re.sub(r'page_label:.*?\n', '', text)

        # Extract only relevant information, not full Q&A list
        text = re.sub(r'Question:\s*\d+[-\)]?\s*', '', text)
        text = re.sub(r'Answer:\s*', '', text)

        return text.strip()

    def setup_environment(self):
        """Setup necessary directories and environment variables"""
        os.environ['TRANSFORMERS_CACHE'] = './hf_cache'
        os.environ['HF_CACHE'] = './hf_cache'

        for dir_path in ['./rag_documents', './hf_cache']:
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

    def initialize_models(self):
        """Initialize LLM, embeddings, and tokenizer"""
        # Quantization config
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Initialize embedding model
        self.embed_model = HuggingFaceEmbedding(
            # model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_name= self.model_dir,
            cache_folder='./hf_cache'
        )

        # Initialize LLM
        # hf_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        hf_model_name = self.model_dir

        # Setup tokenizer and settings
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_model_name,
            cache_dir='./hf_cache',
            padding_side = 'right'
        )
        # Then set its pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Then initialize model with the tokenizer config
        self.llm_model = HuggingFaceLLM(
            model_name=hf_model_name,
            tokenizer_name=hf_model_name,
            tokenizer=self.tokenizer,  # Pass the configured tokenizer
            context_window=self.context_window,
            max_new_tokens=self.max_new_tokens,
            model_kwargs={
                "quantization_config": quantization_config,
                "pad_token_id": self.tokenizer.pad_token_id  # Pass pad token ID
            },
            generate_kwargs={
                "top_k": 50,
                "pad_token_id": self.tokenizer.pad_token_id  # Here too
            },
            device_map="cuda",
        )

        # Configure Llama-index settings
        Settings.tokenizer = self.tokenizer
        Settings.llm = self.llm_model
        Settings.embed_model = self.embed_model
        Settings.context_window = self.context_window
        Settings.chunk_size = self.chunk_size
        Settings.chunk_overlap = self.chunk_overlap

    def setup_query_engine(self):
        """Setup the RAG pipeline and query engine"""
        # Load documents
        reader = SimpleDirectoryReader('./rag_documents/')
        documents = reader.load_data()

        # Setup document processing pipeline
        parser = LangchainNodeParser(
            TokenTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        )
        nodes = parser.get_nodes_from_documents(documents)

        # Setup vector store
        db = chromadb.PersistentClient(path="./chroma_db_basic_minilm")
        chroma_collection = db.get_or_create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create index
        self.index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=self.embed_model)

        # Setup retriever and response synthesizer
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=5,
        )

        # Configure prompt template
        qa_template = PromptTemplate(
            "Using the information below, provide a single, concise answer without repetition.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Give a clear, step-by-step answer without repeating information.\n"
            "Question: {query_str}\n"
            "Answer: ",
            prompt_type=PromptType.QUESTION_ANSWER
        )

        response_synthesizer = get_response_synthesizer(
            llm=self.llm_model,
            response_mode="compact",  # as string instead of enum
            text_qa_template=qa_template
        )

        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

    def get_response(self, message: str) -> str:
        """Generate response for user query"""
        try:
            response = self.query_engine.query(message)
            # Extract only the direct answer, remove additional Q&A pairs
            answer = str(response).split('Question:')[0].strip()
            return answer
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I encountered an error processing your question. Please try rephrasing it."


# Function to maintain compatibility with original code
def responsechat(model_dir: str):
    return RAGChat(model_dir)