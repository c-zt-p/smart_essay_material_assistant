# core/rag_manager.py
import asyncio
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import functools # Import functools
import jieba # For ChineseBM25Retriever
import itertools # For ChineseBM25Retriever

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings as LlamaSettings,
    QueryBundle,
    PromptTemplate,
    Document as LlamaDocument, # Rename to avoid conflict with our Document model if any
    ServiceContext # Often useful, though LlamaSettings handles much now
)
from llama_index.core.schema import NodeWithScore, TextNode, BaseNode, IndexNode # Added BaseNode, IndexNode
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever # Base BM25
from llama_index.core.vector_stores.utils import node_to_metadata_dict, metadata_dict_to_node # For BM25
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K


from .settings_manager import settings, ModelSetting
import logging

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define TEST_FLY_PROJECT_DIR or make stopwords_path configurable
# For now, let's assume it's relative to this file or a known project root.
# If rag_chat_app is your root:
STOPWORDS_PATH = Path(__file__).parent.parent / "data" / "stopwords.txt"
if not STOPWORDS_PATH.exists():
    logger.warning(f"Stopwords file not found at {STOPWORDS_PATH}. ChineseBM25Retriever might not work optimally.")
    # You might want to create a dummy empty file or handle this more gracefully
    # For now, let's ensure the directory exists for potential creation by user
    STOPWORDS_PATH.parent.mkdir(parents=True, exist_ok=True)


class ChineseBM25Retriever(BM25Retriever):
    """A BM25 retriever that uses the BM25 algorithm to retrieve nodes, customized for Chinese."""
    def __init__(
        self,
        nodes: List[BaseNode], # Changed from Optional to required for our use case
        tokenizer: Optional[Any] = None, # Allow custom tokenizer, defaults to jieba
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        self._nodes = nodes # Store nodes for re-indexing if tokenizer changes, though not implemented here
        
        # Load Chinese stopwords
        self.stop_words = set()
        if STOPWORDS_PATH.exists():
            with open(STOPWORDS_PATH, encoding='utf-8') as f:
                for line in f:
                    self.stop_words.add(line.strip())
        else:
            logger.warning(f"Stopwords file missing at {STOPWORDS_PATH}. Proceeding without stopwords for BM25.")

        # Default Chinese tokenizer using jieba
        if tokenizer is None:
            def jieba_tokenizer(text: str) -> List[str]:
                return [
                    word for word in jieba.cut_for_search(text) 
                    if word not in self.stop_words and word.strip()
                ]
            tokenizer = jieba_tokenizer
        
        super().__init__(
            nodes=nodes, 
            similarity_top_k=similarity_top_k, 
            verbose=verbose, 
            **kwargs
        )


class RAGManager:
    def __init__(self):
        self.rag_db_base_path = Path(settings.rag_db_path)
        self.rag_db_base_path.mkdir(parents=True, exist_ok=True)
        self._configure_llama_index_settings() # Initial configuration
        self.current_db_name: Optional[str] = None
        self.index: Optional[VectorStoreIndex] = None
        self.query_engine: Optional[RetrieverQueryEngine] = None
        self.retriever: Optional[QueryFusionRetriever] = None # Store the retriever

    def _configure_llama_index_settings(self, chat_model_name: Optional[str] = None, embedding_model_name: Optional[str] = None):
        chat_model_config: Optional[ModelSetting] = settings.get_chat_model(chat_model_name)
        embedding_model_config: Optional[ModelSetting] = settings.get_embedding_model(embedding_model_name)

        if not chat_model_config:
            raise ValueError(f"Chat model configuration '{chat_model_name or 'default'}' not found.")
        if not embedding_model_config:
            raise ValueError(f"Embedding model configuration '{embedding_model_name or 'default'}' not found.")

        LlamaSettings.llm = OpenAILike(
            model=chat_model_config.model_name,
            api_base=chat_model_config.model_url,
            api_key=chat_model_config.api_key,
            is_chat_model=True,
            max_tokens=chat_model_config.token_limit
        )
        LlamaSettings.embed_model = OpenAIEmbedding(
            model_name=embedding_model_config.model_name,
            api_base=embedding_model_config.model_url,
            api_key=embedding_model_config.api_key,
        )
        LlamaSettings.chunk_size = 512
        LlamaSettings.chunk_overlap = 64
        LlamaSettings.node_parser = SentenceSplitter(
            chunk_size=LlamaSettings.chunk_size,
            chunk_overlap=LlamaSettings.chunk_overlap,
            # callback_manager=LlamaSettings.callback_manager # If you use a global callback manager
        )
        # ServiceContext is deprecated, but useful to bundle. LlamaSettings now preferred.
        # LlamaSettings.service_context = ServiceContext.from_defaults(
        #     llm=LlamaSettings.llm,
        #     embed_model=LlamaSettings.embed_model,
        #     node_parser=LlamaSettings.node_parser
        # )
        logger.info(f"LlamaIndex re-configured with LLM: {chat_model_config.name} and Embed: {embedding_model_config.name}")

    def get_db_path(self, db_name: str) -> Path:
        return self.rag_db_base_path / db_name

    def list_rag_dbs(self) -> List[str]:
        return [p.name for p in self.rag_db_base_path.iterdir() if p.is_dir() and (p / "docstore.json").exists()]


    async def create_rag_db(self, db_name: str, file_paths: List[str], progress_callback=None) -> bool:
        db_path = self.get_db_path(db_name)
        if db_path.exists():
            logger.warning(f"RAG DB '{db_name}' already exists.")
            if progress_callback: progress_callback(-1, f"Error: DB '{db_name}' already exists.")
            return False

        db_path.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Creating RAG DB '{db_name}' with files: {file_paths}")
            if progress_callback: progress_callback(0.1, "Loading documents...")
            
            loop = asyncio.get_event_loop()
            # SimpleDirectoryReader.load_data() is synchronous
            documents = await loop.run_in_executor(None, SimpleDirectoryReader(input_files=file_paths).load_data)
            
            if not documents:
                logger.error("No documents loaded.")
                if progress_callback: progress_callback(-1, "Error: No documents loaded.")
                return False

            if progress_callback: progress_callback(0.3, "Parsing nodes...")
            node_parser = LlamaSettings.node_parser # Get from global settings
            nodes = node_parser.get_nodes_from_documents(documents, show_progress=False)
            # Add fake page_label if not present, similar to LocalSetupNode
            for node in nodes:
                if "page_label" not in node.metadata:
                    node.metadata["page_label"] = "N/A" # Or some default like 0 or 1


            if progress_callback: progress_callback(0.5, "Building VectorStoreIndex...")
            # VectorStoreIndex.from_documents is synchronous
            # It will use LlamaSettings.embed_model and LlamaSettings.node_parser implicitly
            self.index = await loop.run_in_executor(None, VectorStoreIndex, nodes)
            
            if progress_callback: progress_callback(0.9, "Persisting index...")
            self.index.storage_context.persist(persist_dir=str(db_path))
            
            logger.info(f"RAG DB '{db_name}' created and persisted at {db_path}")
            if progress_callback: progress_callback(1.0, "DB Created Successfully!")
            return True
        except Exception as e:
            logger.error(f"Error creating RAG DB '{db_name}': {e}", exc_info=True)
            if progress_callback: progress_callback(-1, f"Error creating DB: {e}")
            return False

    async def _setup_retriever_and_query_engine(self, index: VectorStoreIndex, similarity_top_k: int = 3):
        """Helper to build retriever and query engine from an index."""
        logger.info(f"Setting up retriever and query engine. Top K: {similarity_top_k}")
        
        # 1. Vector Retriever
        vector_retriever = index.as_retriever(similarity_top_k=similarity_top_k)

        # 2. BM25 Retriever (using our ChineseBM25Retriever)
        # We need all nodes from the docstore for BM25
        all_nodes = list(index.docstore.docs.values())
        if not all_nodes:
            logger.warning("No nodes in docstore for BM25 retriever. BM25 will be skipped.")
            bm25_retriever = None
        else:
            try:
                bm25_retriever = ChineseBM25Retriever(
                    nodes=all_nodes,
                    similarity_top_k=similarity_top_k,
                )
            except Exception as e:
                logger.error(f"Failed to initialize ChineseBM25Retriever: {e}. BM25 will be skipped.", exc_info=True)
                bm25_retriever = None
        
        retrievers_list = [vector_retriever]
        if bm25_retriever:
            retrievers_list.append(bm25_retriever)

        if len(retrievers_list) > 1:
            self.retriever = QueryFusionRetriever(
                retrievers=retrievers_list,
                similarity_top_k=similarity_top_k, # Reranked top_k
                num_queries=3,  # Generate N-1 new queries. (e.g., 3 for 2 new queries)
                mode="reciprocal_rerank",
                use_async=True, # Important for aretrieve
                verbose=True,
                # query_gen_prompt="...", # Optional: custom prompt for query generation
            )
        else:
            self.retriever = vector_retriever # Fallback to just vector if BM25 fails

        # query_engine uses LlamaSettings.llm by default
        self.query_engine = RetrieverQueryEngine.from_args(
            self.retriever,
            streaming=True, # Enable streaming from query engine if LLM supports it
            # service_context=LlamaSettings.service_context # Deprecated, uses LlamaSettings
        )
        logger.info("Retriever and query engine set up.")


    async def load_rag_db(self, db_name: str, progress_callback=None) -> bool:
        db_path = self.get_db_path(db_name)
        if not db_path.exists() or not (db_path / "docstore.json").exists():
            logger.error(f"RAG DB '{db_name}' not found or incomplete at {db_path}")
            if progress_callback: progress_callback(-1, "DB not found or incomplete.")
            return False
        try:
            if progress_callback: progress_callback(0.1, "Loading index from storage...")
            loop = asyncio.get_event_loop()

            load_storage_context_with_args = functools.partial(
                StorageContext.from_defaults, 
                persist_dir=str(db_path)
            )
            storage_context = await loop.run_in_executor(None, load_storage_context_with_args)
            
            # load_index_from_storage will use LlamaSettings.embed_model and LlamaSettings.llm
            self.index = await loop.run_in_executor(
                None,
                load_index_from_storage,
                storage_context
            )

            if self.index is None:
                logger.error(f"Index for '{db_name}' could not be loaded.")
                if progress_callback: progress_callback(-1, f"Error: Failed to load index.")
                return False

            if progress_callback: progress_callback(0.5, "Setting up retriever and query engine...")
            # Determine similarity_top_k, ensuring it's not more than docs available
            docstore_size = len(self.index.docstore.docs)
            similarity_top_k = min(DEFAULT_SIMILARITY_TOP_K, docstore_size) if docstore_size > 0 else 1
            
            await self._setup_retriever_and_query_engine(self.index, similarity_top_k)

            self.current_db_name = db_name
            logger.info(f"RAG DB '{db_name}' loaded successfully.")
            if progress_callback: progress_callback(1.0, f"DB '{db_name}' Loaded.")
            return True
        except Exception as e:
            logger.error(f"Error loading RAG DB '{db_name}': {e}", exc_info=True)
            if progress_callback: progress_callback(-1, f"Error loading DB: {e}")
            return False


    async def query_rag(self, query_text: str, history: List[Dict[str, str]] = None) -> Tuple[str, List[NodeWithScore]]:
        if not self.retriever: # Check retriever instead of query_engine for node fetching
            logger.error("query_rag called but no RAG retriever is loaded.")
            raise ValueError("No RAG DB loaded or retriever not initialized. Please select/create a DB.")

        # The query_engine might do its own history management if configured.
        # For now, retriever.aretrieve typically takes the direct query.
        # If history needs to be part of retrieval, a condense_question step might be needed first.
        logger.info(f"Retrieving nodes for query: {query_text}")
        nodes_with_score: List[NodeWithScore] = await self.retriever.aretrieve(query_text)

        if not nodes_with_score:
            logger.info(f"No relevant documents found for query: {query_text}")
            context_str = "No relevant context was found in the knowledge base for your query."
        else:
            logger.info(f"Retrieved {len(nodes_with_score)} nodes.")
            context_str = "\n\n".join([
                f"Source (File: {Path(n.node.metadata.get('file_name', 'N/A')).name}, "
                f"Page: {n.node.metadata.get('page_label', 'N/A')}, Score: {n.score:.2f}):\n"
                f"{n.node.get_text()[:700]}..." # Limit context per node
                if len(n.node.get_text()) > 700 else
                f"Source (File: {Path(n.node.metadata.get('file_name', 'N/A')).name}, "
                f"Page: {n.node.metadata.get('page_label', 'N/A')}, Score: {n.score:.2f}):\n"
                f"{n.node.get_text()}"
                for n in nodes_with_score # QueryFusionRetriever already handles top_k
            ])
        
        # This template is for the LLM to generate the final answer
        # The history will be prepended by the LLMClient
        qa_template_str = (
            "Based on the following context from a knowledge base AND the previous conversation history, "
            "please answer the user's question. "
            "If the context is not relevant or insufficient, state that you cannot answer based on the "
            "provided documents and try to answer generally if possible, or ask for clarification.\n"
            "---------------------\n"
            "Context from Knowledge Base:\n{context_str}\n"
            "---------------------\n"
            "User's Question: {query_text}"
            # "Your Answer:" # LLM will append its answer here
        )
        
        final_query_for_llm = qa_template_str.format(context_str=context_str, query_text=query_text)
        
        return final_query_for_llm, nodes_with_score