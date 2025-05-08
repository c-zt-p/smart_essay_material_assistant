from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, 
                              QFrame, QPushButton, QDialog)
from PySide6.QtCore import Qt, Signal
from qfluentwidgets import (CardWidget, TitleLabel, BodyLabel, StrongBodyLabel, 
                           FluentIcon, ToolButton, HyperlinkButton, SubtitleLabel, 
                           CaptionLabel, TransparentPushButton, MessageBoxBase,
                           Flyout, FlyoutViewBase, FlyoutAnimationType)
from llama_index.core.schema import NodeWithScore
from .icons import Icon
from pathlib import Path
from typing import List, Optional


class SourceDetailView(FlyoutViewBase):
    """Flyout view to display full source content."""
    
    def __init__(self, title: str, content: str, parent=None):
        super().__init__(parent)
        self.vBoxLayout = QVBoxLayout(self)
        self.vBoxLayout.setContentsMargins(20, 16, 20, 16)
        self.vBoxLayout.setSpacing(12)
        
        # Title
        self.titleLabel = SubtitleLabel(title, self)
        
        # Content scroll area for long texts
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setMinimumWidth(500)
        self.scroll_area.setMinimumHeight(350)
        
        # Content container
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        
        self.contentLabel = BodyLabel(content, content_widget)
        self.contentLabel.setWordWrap(True)
        content_layout.addWidget(self.contentLabel)
        
        self.scroll_area.setWidget(content_widget)
        
        # Add components
        self.vBoxLayout.addWidget(self.titleLabel)
        self.vBoxLayout.addWidget(self.scroll_area)


class SourceItem(QFrame):
    """Widget representing a single source item that can be clicked to show details."""
    clicked = Signal(str, str, object)  # title, full_content, source_widget
    
    def __init__(self, number: int, source: NodeWithScore, parent=None):
        super().__init__(parent)
        self.setObjectName(f"SourceItem{number}")
        self.setCursor(Qt.PointingHandCursor)
        
        # Get metadata
        self.metadata = getattr(source.node, 'metadata', {})
        self.title = self.metadata.get('title', self.metadata.get('file_name', f"Source {number}"))
        self.full_content = source.node.text
        self.number = number
        self.score = getattr(source, 'score', None)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the UI components."""
        item_layout = QVBoxLayout(self)
        item_layout.setContentsMargins(8, 8, 8, 8)
        
        # Header with title and score
        header_layout = QHBoxLayout()
        
        # Title with expand indicator
        title_layout = QHBoxLayout()
        title_label = BodyLabel(f"{self.number}. {self.title}", self)
        title_label.setObjectName("SourceTitle")
        expand_button = TransparentPushButton(self)
        expand_button.setFixedSize(20, 20)
        
        title_layout.addWidget(title_label)
        title_layout.addWidget(expand_button)
        
        header_layout.addLayout(title_layout)
        header_layout.addStretch(1)
        
        # Add score indicator if available
        if self.score is not None:
            score_text = f"Relevance: {self.score:.2f}"
            score_label = CaptionLabel(score_text, self)
            header_layout.addWidget(score_label)
        
        item_layout.addLayout(header_layout)
        
        # Add source text content (trimmed)
        content = self.full_content
        if len(content) > 100:
            content = content[:97] + "..."
        
        content_label = BodyLabel(content, self)
        content_label.setWordWrap(True)
        item_layout.addWidget(content_label)
        
        # Style the item frame
        self.setStyleSheet("""
            QFrame[objectName^="SourceItem"] {
                background-color: rgba(255, 255, 255, 0.8);
                border-radius: 6px;
                border: 1px solid rgba(200, 200, 220, 0.8);
            }
            
            QFrame[objectName^="SourceItem"]:hover {
                background-color: rgba(240, 245, 255, 0.9);
                border: 1px solid rgba(180, 180, 220, 1.0);
            }
            
            #SourceTitle {
                font-weight: bold;
            }
        """)
    
    def mousePressEvent(self, event):
        """Handle mouse press events to show full content dialog."""
        self.clicked.emit(self.title, self.full_content, self)
        super().mousePressEvent(event)


class RAGSourceWidget(QFrame):
    """Widget to display RAG sources below AI messages."""
    
    def __init__(self, sources: List[NodeWithScore], parent=None):
        super().__init__(parent)
        self.sources = sources
        self.setObjectName("RAGSourceFrame")
        self.flyout_instance: Optional[QWidget] = None
        
        self._setup_ui()
        self._populate_sources()
        
        # Set style
        self.setStyleSheet("""
            #RAGSourceFrame {
                background-color: rgba(240, 240, 250, 0.7);
                border-radius: 8px;
                margin: 4px 10px;
            }
            
            #SourceTitle {
                color: #555555;
                font-weight: bold;
            }
        """)
    
    def _setup_ui(self):
        """Set up the UI components."""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 8, 10, 8)
        self.main_layout.setSpacing(5)
        
        # Title
        self.title_label = SubtitleLabel("Sources", self)
        self.title_label.setObjectName("SourceTitle")
        self.main_layout.addWidget(self.title_label)
        
        # Container for source items
        self.sources_container = QWidget(self)
        self.sources_layout = QVBoxLayout(self.sources_container)
        self.sources_layout.setContentsMargins(5, 0, 5, 0)
        self.sources_layout.setSpacing(8)
        
        self.main_layout.addWidget(self.sources_container)
    
    def _populate_sources(self):
        """Populate the widget with source information."""
        if not self.sources:
            self.title_label.setText("No relevant sources found")
            return
            
        self.title_label.setText(f"Sources ({len(self.sources)})")
        
        for idx, source in enumerate(self.sources):
            source_item = SourceItem(idx + 1, source, self.sources_container)
            source_item.clicked.connect(self._show_source_detail)
            self.sources_layout.addWidget(source_item)
    
    def _show_source_detail(self, title: str, content: str, source_item: QWidget):
        """Show flyout with full source content."""
        # Create source detail view
        view = SourceDetailView(title, content)
        
        # Show flyout
        self.flyout_instance = Flyout.make(
            view, 
            target=source_item,
            parent=self.window(),
            aniType=FlyoutAnimationType.FADE_IN
        )
        