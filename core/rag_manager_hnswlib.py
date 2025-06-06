import asyncio
from pathlib import Path
from re import L
from typing import List, Optional, Tuple, Dict, Any, Callable
import functools
import jieba
import itertools

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings as LlamaSettings,
)
from llama_index.vector_stores.hnswlib import HnswlibVectorStore
from llama_index.core.schema import NodeWithScore, TextNode, BaseNode, IndexNode
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from .arborparser_spliter import ArborParserNodeParser

from .settings_manager import settings, ModelSetting
import logging

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 停用词文件路径
STOPWORDS_PATH = Path(__file__).parent.parent / "data" / "stopwords.txt"
if not STOPWORDS_PATH.exists():
    logger.warning(f"Stopwords file not found at {STOPWORDS_PATH}. ChineseBM25Retriever might not work optimally.")
    STOPWORDS_PATH.parent.mkdir(parents=True, exist_ok=True)

SIMILARITY_TOP_K = 4

class ChineseBM25Retriever(BM25Retriever):
    """A BM25 retriever customized for Chinese text retrieval."""
    
    def __init__(
        self,
        nodes: List[BaseNode],
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        self._nodes = nodes
        
        # 加载中文停用词
        self.stop_words = set()
        if STOPWORDS_PATH.exists():
            with open(STOPWORDS_PATH, encoding='utf-8') as f:
                for line in f:
                    self.stop_words.add(line.strip())
        else:
            logger.warning(f"Stopwords file missing at {STOPWORDS_PATH}. Proceeding without stopwords for BM25.")

        # 默认使用jieba分词器
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
        self._configure_llama_index_settings()
        self.current_db_name: Optional[str] = None
        self.index: Optional[VectorStoreIndex] = None
        self.query_engine: Optional[RetrieverQueryEngine] = None
        self.retriever: Optional[QueryFusionRetriever] = None

    def _configure_llama_index_settings(self, chat_model_name: Optional[str] = None, embedding_model_name: Optional[str] = None):
        """配置LlamaIndex的全局设置"""
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
        
        # 使用ArborParser作为默认的节点解析器
        LlamaSettings.chunk_size = 512
        LlamaSettings.chunk_overlap = 64
        LlamaSettings.node_parser = ArborParserNodeParser(
            chunk_size=LlamaSettings.chunk_size,
            chunk_overlap=LlamaSettings.chunk_overlap,
            merge_threshold=5000
        )
        
        logger.info(f"LlamaIndex re-configured with LLM: {chat_model_config.name} and Embed: {embedding_model_config.name}")

    def get_db_path(self, db_name: str) -> Path:
        """获取数据库路径"""
        return self.rag_db_base_path / db_name

    def list_rag_dbs(self) -> List[str]:
        """列出所有可用的RAG数据库"""
        return [p.name for p in self.rag_db_base_path.iterdir() if p.is_dir() and (p / "docstore.json").exists()]

    async def create_rag_db(self, db_name: str, file_paths: List[str], progress_callback=None) -> bool:
        """
        创建新的RAG数据库
        
        Args:
            db_name: 数据库名称
            file_paths: 文件路径列表
            progress_callback: 进度回调函数
            
        Returns:
            是否成功创建
        """
        db_path = self.get_db_path(db_name)
        if db_path.exists():
            logger.warning(f"RAG DB '{db_name}' already exists.")
            if progress_callback: 
                progress_callback(-1, f"Error: DB '{db_name}' already exists.")
            return False

        db_path.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Creating RAG DB '{db_name}' with files: {file_paths}")
            if progress_callback: 
                progress_callback(0.1, "Loading documents...")
            
            loop = asyncio.get_event_loop()
            documents = await loop.run_in_executor(None, 
                                                 SimpleDirectoryReader(input_files=file_paths).load_data)
            
            if not documents:
                logger.error("No documents loaded.")
                if progress_callback: 
                    progress_callback(-1, "Error: No documents loaded.")
                return False

            if progress_callback: 
                progress_callback(0.3, "Parsing nodes with ArborParser...")
            
            node_parser = LlamaSettings.node_parser
            nodes = node_parser.get_nodes_from_documents(documents, show_progress=False)
            logger.info(f"Parsed {len(nodes)} nodes with ArborParser.")

            if progress_callback: 
                progress_callback(0.5, "Building Hnswlib VectorStore...")
            
            # 获取一个示例文本的嵌入向量
            sample_embedding = LlamaSettings.embed_model.get_text_embedding("sample text")
            dimension = len(sample_embedding)
            hnswlib_vector_store = HnswlibVectorStore.from_params(
                space="ip",
                dimension=dimension,
                #dimension=LlamaSettings.embed_model._model.get_sentence_embedding_dimension(),
                max_elements=10000,  # 增加最大元素数量以适应更多节点
            )
            
            if progress_callback: 
                progress_callback(0.6, "Getting Storage Context...")
            
            hnswlib_storage_context = StorageContext.from_defaults(
                vector_store=hnswlib_vector_store
            )
            
            if progress_callback: 
                progress_callback(0.7, "Building Index...")
            
            self.index = VectorStoreIndex(
                nodes,
                storage_context=hnswlib_storage_context,
            )
            
            #self.index = VectorStoreIndex.from_documents(
            #    nodes,
            #    storage_context=hnswlib_storage_context,
            #    show_progress=False,
            #)
            #self.index = await loop.run_in_executor(None, VectorStoreIndex, nodes)
            
            
            if progress_callback: 
                progress_callback(0.9, "Persisting index...")
            
            self.index.storage_context.persist(persist_dir=str(db_path))
            
            logger.info(f"RAG DB '{db_name}' created and persisted at {db_path}")
            if progress_callback: 
                progress_callback(1.0, "DB Created Successfully!")
            return True
        except Exception as e:
            logger.error(f"Error creating RAG DB '{db_name}': {e}", exc_info=True)
            if progress_callback: 
                progress_callback(-1, f"Error creating DB: {e}")
            return False

    async def _setup_retriever_and_query_engine(self, index: VectorStoreIndex, similarity_top_k: int = 4):
        """
        设置检索器和查询引擎
        
        Args:
            index: 向量存储索引
            similarity_top_k: 检索结果数量
        """
        logger.info(f"Setting up retriever and query engine. Top K: {similarity_top_k}")
        
        # 1. 向量检索器
        vector_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        
        # 2. BM25检索器（使用ChineseBM25Retriever）
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
        
        # 组合检索器
        retrievers_list = [vector_retriever]
        if bm25_retriever:
            retrievers_list.append(bm25_retriever)

        # 设置融合检索器或单一检索器
        if len(retrievers_list) > 1:
            self.retriever = QueryFusionRetriever(
                retrievers=retrievers_list,
                similarity_top_k=similarity_top_k,
                num_queries=3,  # 生成2个新的查询
                mode="reciprocal_rerank",
                use_async=True,
                verbose=True,
            )
        else:
            self.retriever = vector_retriever  # 如果BM25失败则回退到向量检索

        # 设置查询引擎
        self.query_engine = RetrieverQueryEngine.from_args(
            self.retriever,
            streaming=True,  # 启用流式响应
        )
        logger.info("Retriever and query engine set up.")

    async def load_rag_db(self, db_name: str, progress_callback=None) -> bool:
        """
        加载现有的RAG数据库
        
        Args:
            db_name: 数据库名称
            progress_callback: 进度回调函数
            
        Returns:
            是否成功加载
        """
        db_path = self.get_db_path(db_name)
        if not db_path.exists() or not (db_path / "docstore.json").exists():
            logger.error(f"RAG DB '{db_name}' not found or incomplete at {db_path}")
            if progress_callback: 
                progress_callback(-1, "DB not found or incomplete.")
            return False
            
        try:
            if progress_callback: 
                progress_callback(0.1, "Loading index from storage...")
            
            loop = asyncio.get_event_loop()

            # 加载存储上下文
            load_storage_context_with_args = functools.partial(
                StorageContext.from_defaults, 
                persist_dir=str(db_path),
                vector_store=HnswlibVectorStore.from_persist_dir(str(db_path))
            )
            storage_context = await loop.run_in_executor(None, load_storage_context_with_args)
            
            # 加载索引
            self.index = await loop.run_in_executor(
                None,
                load_index_from_storage,
                storage_context
            )

            if self.index is None:
                logger.error(f"Index for '{db_name}' could not be loaded.")
                if progress_callback: 
                    progress_callback(-1, f"Error: Failed to load index.")
                return False

            if progress_callback: 
                progress_callback(0.5, "Setting up retriever and query engine...")
            
            # 确定similarity_top_k，确保不超过可用文档数
            docstore_size = len(self.index.docstore.docs)
            similarity_top_k = min(SIMILARITY_TOP_K, docstore_size) if docstore_size > 0 else 1
            
            await self._setup_retriever_and_query_engine(self.index, similarity_top_k)

            self.current_db_name = db_name
            logger.info(f"RAG DB '{db_name}' loaded successfully.")
            if progress_callback: 
                progress_callback(1.0, f"DB '{db_name}' Loaded.")
            return True
            
        except Exception as e:
            logger.error(f"Error loading RAG DB '{db_name}': {e}", exc_info=True)
            if progress_callback: 
                progress_callback(-1, f"Error loading DB: {e}")
            return False

    async def query_rag(self, query_text: str, history: List[Dict[str, str]] = None) -> Tuple[str, List[NodeWithScore]]:
        """
        查询RAG数据库
        
        Args:
            query_text: 查询文本
            history: 对话历史
            
        Returns:
            包含上下文的查询提示和检索到的节点
        """
        if not self.retriever:
            logger.error("query_rag called but no RAG retriever is loaded.")
            raise ValueError("No RAG DB loaded or retriever not initialized. Please select/create a DB.")
        logger.info(f"Retrieving nodes for query: {query_text}")
        nodes_with_score: List[NodeWithScore] = await self.retriever.aretrieve(query_text)
        if not nodes_with_score:
            logger.info(f"No relevant documents found for query: {query_text}")
            context_str = "No relevant context was found in the knowledge base for your query."
        else:
            logger.info(f"Retrieved {len(nodes_with_score)} nodes.")
            context_str = "\n\n".join([
                f"Source (File: {Path(n.node.metadata.get('file_name', 'N/A')).name}, "
                f"Section: {n.node.metadata.get('title_path', 'N/A')}, "
                f"Page: {n.node.metadata.get('page_label', 'N/A')}, Score: {n.score:.2f}):\n"
                f"{n.node.get_text()[:700]}..." if len(n.node.get_text()) > 700 else
                f"Source (File: {Path(n.node.metadata.get('file_name', 'N/A')).name}, "
                f"Section: {n.node.metadata.get('title_path', 'N/A')}, "
                f"Page: {n.node.metadata.get('page_label', 'N/A')}, Score: {n.score:.2f}):\n"
                f"{n.node.get_text()}"
                for n in nodes_with_score
            ])
        
        # 为LLM准备带有上下文的查询模板
        qa_template_str = (
            "Based on the following context from a knowledge base AND the previous conversation history, "
            "please answer the user's question. "
            "---------------------\n"
            "Context from Knowledge Base:\n{context_str}\n"
            "---------------------\n"
            "User's Question: {query_text}"
        )
        
        final_query_for_llm = qa_template_str.format(context_str=context_str, query_text=query_text)
        
        return final_query_for_llm, nodes_with_score
