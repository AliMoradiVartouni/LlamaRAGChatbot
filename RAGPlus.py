import os
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from transformers import pipeline


class RAGChat:
    def __init__(self, model_dir: str, retriever):
        self.model_dir = model_dir
        self.retriever = retriever
        self.setup_environment()
        self.initialize_models()
        self.setup_chain()

    def setup_environment(self):
        """Setup necessary directories and environment variables"""
        os.environ['TRANSFORMERS_CACHE'] = './hf_cache'
        os.environ['HF_CACHE'] = './hf_cache'
        for dir_path in ['./rag_documents', './hf_cache']:
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

    def initialize_models(self):
        """Initialize LLM and tokenizer"""
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            cache_dir='./hf_cache',
            padding_side='right'
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Setup pipeline
        pipe = pipeline(
            "text-generation",
            # "question-answering",
            model=self.model_dir,
            tokenizer=self.tokenizer,
            max_length=1024,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.15,
            return_full_text=False,
            torch_dtype=torch.float16,  # Add this for faster inference
            device_map="auto"
        )

        # Initialize LLM
        self.llm = HuggingFacePipeline(pipeline=pipe)

    def setup_chain(self):
        """Setup the RAG chain"""
        # Create prompt template
        prompt_template = """Use the following pieces of context to answer the question.
        If you don't know the answer based on the context, just say you don't know.
        Try to be brief and direct in your answer.

        Context: {context}

        Question: {question}

        Answer:"""

        PROMPT = ChatPromptTemplate.from_template(prompt_template)

        # Create chain
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={
                "prompt": PROMPT
            },
            return_source_documents=False
        )

    def get_response(self, query: str) -> str:
        """Generate response using the RAG chain"""
        try:
            # Get response from chain
            result = self.chain({"query": query})
            return result["result"]
        except Exception as e:
            return f"Error generating response: {str(e)}"


# Function to maintain compatibility with original code
def responsechat(model_dir: str):
    from RAGRetriever import DocumentRetriever

    # Initialize DocumentRetriever
    doc_retriever = DocumentRetriever(
        model_dir=model_dir,
        persist_dir="./chroma_db"
    )

    # Load and process documents
    doc_directory = "/home/ali/moradi/Conference_Content"
    doc_retriever.load_documents(doc_directory)

    # Get retriever
    retriever_instance = doc_retriever.get_retriever()

    # Initialize RAGChat
    return RAGChat(model_dir=model_dir, retriever=retriever_instance)