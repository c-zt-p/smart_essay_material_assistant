# arborparser imports
from arborparser.tree import TreeBuilder, TreeExporter, TreeNode
from arborparser.chain import ChainParser
from arborparser.pattern import (
    CHINESE_CHAPTER_PATTERN_BUILDER,
    NUMERIC_DOT_PATTERN_BUILDER,
    ENGLISH_CHAPTER_PATTERN_BUILDER,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeWithScore, TextNode, BaseNode, IndexNode
from typing import Tuple, Dict, Any, List

class ArborParserNodeParser:
    """使用ArborParser进行文档解析和分块的自定义解析器。"""
    
    def __init__(
        self, 
        chunk_size: int = 1024, 
        chunk_overlap: int = 80,
        merge_threshold: int = 5000
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.merge_threshold = merge_threshold
        self.sentence_splitter = SentenceSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        
        # 定义章节模式
        self.chinese_chapter_pattern = CHINESE_CHAPTER_PATTERN_BUILDER.modify(
            prefix_regex=r"[\#\s]*第?",
        ).build()
        
        self.english_chapter_pattern = ENGLISH_CHAPTER_PATTERN_BUILDER.modify(
            prefix_regex=r"[\#\s]*Chapter\s",
        ).build()
        
        self.sector_pattern = NUMERIC_DOT_PATTERN_BUILDER.modify(
            prefix_regex=r"[\#\s]*",
            suffix_regex=r"[\.\s]*",
            min_level=2,
        ).build()
        
        self.patterns = [
            self.chinese_chapter_pattern,
            self.english_chapter_pattern,
            self.sector_pattern,
        ]

    def merge_deep_nodes(self, node: TreeNode) -> None:
        """
        递归合并节点，使每个节点长度尽可能接近阈值。
        
        Args:
            node: 要处理的节点
        """
        # 首先递归处理所有子节点
        for child in node.children:
            self.merge_deep_nodes(child)

        def mergable(node1: TreeNode, node2: TreeNode, is_pre_node_parent: bool) -> bool:
            if (not is_pre_node_parent) and len(node1.children) > 0:
                return False
            if len(node2.children) > 0:
                return False
            return len(node1.content) + len(node2.content) < self.merge_threshold

        idx = -1  # -1表示父节点
        while idx < len(node.children) - 1:
            is_pre_node_parent = (idx == -1)
            if is_pre_node_parent:
                pre_node = node
            else:
                pre_node = node.children[idx]
            next_node = node.children[idx + 1]
            if mergable(pre_node, next_node, is_pre_node_parent):
                pre_node.concat_node(next_node)
                node.children.pop(idx + 1)
            else:
                idx += 1

    def split_node_content(self, node: TreeNode) -> List[str]:
        """
        使用SentenceSplitter对节点内容进行分块
        
        Args:
            node: 要分块的节点
        
        Returns:
            分块后的文本列表
        """
        if node.content:
            return self.sentence_splitter.split_text(node.content)
        return []

    def collect_chunks_with_path(self, node: TreeNode, title_path: List[str] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """
        递归收集树中所有节点的分块，并附带标题路径
        
        Args:
            node: 当前处理的节点
            title_path: 当前节点的标题路径
            
        Returns:
            包含文本块和元数据的元组列表
        """
        if title_path is None:
            title_path = []
        
        # 更新当前节点的标题路径
        current_path = title_path.copy()
        if node.title.strip():
            current_path.append(node.title.strip())
        
        # 当前节点的分块
        chunks = []
        if node.content:
            text_chunks = self.split_node_content(node)
            # 为每个分块添加标题路径元数据
            for chunk in text_chunks:
                chunks.append((
                    chunk, 
                    {
                        "title_path": " > ".join(current_path) if current_path else "Root",
                    }
                ))
        
        # 递归收集所有子节点的分块
        for child in node.children:
            chunks.extend(self.collect_chunks_with_path(child, current_path))
        
        return chunks

    def parse_text(self, text: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        解析文本并返回分块列表及其元数据
        
        Args:
            text: 要解析的文本
            
        Returns:
            包含文本块和元数据的元组列表
        """
        # 解析文本结构
        parser = ChainParser(self.patterns)
        chain = parser.parse_to_chain(text)
        
        # 构建树
        builder = TreeBuilder()
        tree = builder.build_tree(chain)
        
        # 合并小节点
        pre_content = tree.get_full_content()
        self.merge_deep_nodes(tree)
        post_content = tree.get_full_content()
        
        # 验证合并过程没有丢失内容
        assert pre_content == post_content
        
        # 收集所有分块及其标题路径
        chunks_with_metadata = self.collect_chunks_with_path(tree)
        assert len(chunks_with_metadata) > 0
        
        return chunks_with_metadata

    def get_nodes_from_documents(self, documents, show_progress=False):
        """
        从文档列表中获取节点
        
        Args:
            documents: 文档列表
            show_progress: 是否显示进度
        Returns:
            解析后的TextNode节点列表
        """
        nodes = []
        
        for doc in documents:
            # 使用ArborParser解析文档并获取带有标题路径的分块
            chunks_with_metadata = self.parse_text(doc.text)
            
            # 创建节点
            for i, (chunk, title_metadata) in enumerate(chunks_with_metadata):
                node = TextNode(
                    text=chunk,
                    metadata={
                        "file_name": doc.metadata.get("file_name", ""),
                        "file_path": doc.metadata.get("file_path", ""),
                        "chunk_id": i,
                        "title_path": title_metadata["title_path"],
                    }
                )
                nodes.append(node)
                
        return nodes
