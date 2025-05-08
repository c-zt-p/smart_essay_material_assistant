from PySide6.QtCore import Qt, QSize, Signal, QTimer
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QSizePolicy
from PySide6.QtGui import QFontMetrics, QTextOption
from qfluentwidgets import IconWidget, IndeterminateProgressBar, TextEdit, BodyLabel
from markdown import markdown
from .icons import Icon


class ChatBubble(TextEdit):
    """Chat bubble widget for displaying formatted message content."""
    heightChanged = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        
        # Critical: Fixed sizing policies to avoid expanding beyond content
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        self.document().contentsChanged.connect(self._schedule_adjustments)
        self.setStyleSheet("TextEdit { border: none; background-color: transparent; padding: 0px; }")
        self._max_bubble_width_ratio = 0.80
        
        # Add extra property to avoid recursive sizing issues
        self._currently_adjusting = False
    
    def _schedule_adjustments(self):
        """Schedule geometry adjustment after content changes."""
        if not self._currently_adjusting:
            QTimer.singleShot(0, self._adjust_height)
    
    def _adjust_height(self):
        """Adjust height based on content."""
        self._currently_adjusting = True
        
        # Calculate document height with current width
        doc = self.document()
        doc.setTextWidth(self.width())
        doc_height = doc.size().height()
        
        # Add padding and set fixed height
        self.setFixedHeight(int(doc_height) + 5)
        
        self._currently_adjusting = False
        self.heightChanged.emit()
    
    def resizeEvent(self, event):
        """Handle resize events to adjust height when width changes."""
        super().resizeEvent(event)
        # Only adjust height on width changes to prevent loops
        if event.oldSize().width() != event.size().width():
            self._adjust_height()
    
    def setMarkdown(self, md_text: str):
        """Convert markdown to formatted HTML and display it."""
        html = markdown(md_text, extensions=['fenced_code', 'tables', 'nl2br'])
        
        # Style HTML elements
        html = html.replace("<ul>", "<ul style='margin-left: 15px; list-style-type: disc;'>")
        html = html.replace("<ol>", "<ol style='margin-left: 15px; list-style-type: decimal;'>")
        html = html.replace("<li>", "<li style='margin-bottom: 3px;'>")
        html = html.replace("<pre>", "<pre style='background-color: rgba(200, 200, 200, 0.2); "
                                    "padding: 6px; border-radius: 4px; "
                                    "font-family: \"Courier New\", Courier, monospace; "
                                    "white-space: pre-wrap; word-wrap: break-word;'>")
        html = html.replace("<code>", "<code style='font-family: \"Courier New\", Courier, monospace;'>")
        
        super().setHtml(html)
    
    def appendMarkdownChunk(self, md_text_chunk: str):
        """Append plain text chunk to existing content."""
        current_plain_text = self.toPlainText()
        new_text = current_plain_text + md_text_chunk
        super().setPlainText(new_text)
    
    def sizeHint(self) -> QSize:
        """Calculate the preferred size based on content."""
        fm = self.fontMetrics()
        doc = self.document()
        
        # Determine available width for text wrapping
        parent_width = self.parentWidget().width() if self.parentWidget() else 300
        
        # Calculate actual available width considering layout margins
        available_text_width = parent_width
        parent_layout = self.parentWidget().layout() if self.parentWidget() else None
        if parent_layout and hasattr(parent_layout, 'contentsMargins'):
            margins = parent_layout.contentsMargins()
            available_text_width -= (margins.left() + margins.right())
        
        # Add a safety margin
        available_text_width = max(available_text_width, fm.horizontalAdvance("WWWW"))
        
        # Set text width for calculation
        doc.setTextWidth(available_text_width)
        content_height = doc.size().height() + 5  # Add padding
        
        # Use normal width, height will be fixed separately
        preferred_width = int(parent_width * self._max_bubble_width_ratio)
        
        return QSize(preferred_width, int(content_height))
    
    def minimumSizeHint(self) -> QSize:
        """Provide minimum size hint."""
        fm = self.fontMetrics()
        min_h = fm.height() + 5
        min_w = fm.horizontalAdvance("...")
        return QSize(min_w, min_h)


class MessageUI(QWidget):
    """Widget representing a single message in the chat interface."""
    contentHeightChanged = Signal()
    
    def __init__(self, sender_type: str, text: str = "", is_streaming: bool = False, parent=None):
        super().__init__(parent)
        
        if sender_type == "system":
            self.setVisible(False)
            self.deleteLater()
            return
        
        self.sender_type = sender_type
        self.is_streaming = is_streaming
        
        # Critical: Use Fixed vertical policy instead of Preferred
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        self._init_ui()
        self._setup_layouts()
        self._connect_signals()
        
        if text:
            self.chat_bubble.setMarkdown(text)
        self.set_streaming(is_streaming)
    
    def _init_ui(self):
        """Initialize UI components."""
        # Avatar
        self.avatar_widget = IconWidget(parent=self)
        self.avatar_widget.setFixedSize(32, 32)
        
        # Bubble container
        self.bubble_container = QFrame(self)
        self.bubble_container.setObjectName("MessageBubbleContainer")
        self.bubble_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        # Sender label
        self.sender_label = BodyLabel("", self.bubble_container)
        self.sender_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.sender_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        # Style sender label based on sender type
        label_style = "BodyLabel {{ color: {}; padding-top: 0px; margin-bottom: 0px; }}"
        color = "rgb(50,50,50)" if self.sender_type == "user" else "rgb(90,90,90)"
        self.sender_label.setStyleSheet(label_style.format(color))
        
        # Chat bubble
        self.chat_bubble = ChatBubble(self.bubble_container)
        
        # Progress bar for AI responses
        self.progress_bar = IndeterminateProgressBar(self.bubble_container)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setVisible(False)
    
    def _setup_layouts(self):
        """Set up widget layouts."""
        # Main layout
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(10, 3, 10, 3)
        self.layout.setSpacing(8)
        
        # Bubble layout
        self.bubble_layout = QVBoxLayout(self.bubble_container)
        self.bubble_layout.setContentsMargins(10, 5, 10, 5)
        self.bubble_layout.setSpacing(3)
        
        # Add widgets to bubble layout
        self.bubble_layout.addWidget(self.sender_label)
        self.bubble_layout.addWidget(self.chat_bubble)
        if self.sender_type == "ai":
            self.bubble_layout.addWidget(self.progress_bar)
        
        # Set bubble style based on sender type
        bubble_style_sheet = """
            QFrame#MessageBubbleContainer {{
                background-color: {bg_color};
                border-radius: 10px;
            }}
        """
        
        # Configure layout based on sender type
        bubble_stretch = 7
        space_stretch = 1
        
        if self.sender_type == "user":
            self.avatar_widget.setIcon(Icon.USER)
            self.sender_label.setText("You")
            self.layout.addStretch(space_stretch)
            self.layout.addWidget(self.bubble_container, bubble_stretch)
            self.layout.addWidget(self.avatar_widget, 0, Qt.AlignmentFlag.AlignTop)
            bg_color = "rgb(215, 235, 255)"
        else:  # AI
            self.avatar_widget.setIcon(Icon.ROBOT)
            self.sender_label.setText("AI Assistant")
            self.layout.addWidget(self.avatar_widget, 0, Qt.AlignmentFlag.AlignTop)
            self.layout.addWidget(self.bubble_container, bubble_stretch)
            self.layout.addStretch(space_stretch)
            bg_color = "rgb(242, 242, 242)"
        
        self.bubble_container.setStyleSheet(bubble_style_sheet.format(bg_color=bg_color))
    
    def _connect_signals(self):
        """Connect widget signals."""
        self.chat_bubble.heightChanged.connect(self._on_bubble_height_changed)
    
    def _on_bubble_height_changed(self):
        """Handle bubble height changes."""
        # Adjust the message widget height to fit the bubble
        self.updateGeometry()
        self.contentHeightChanged.emit()
    
    def set_text(self, text: str):
        """Set message text content."""
        if not self.isVisible():
            return
        self.chat_bubble.setMarkdown(text)
    
    def append_text_chunk(self, chunk: str):
        """Append text chunk to existing message."""
        if not self.isVisible():
            return
        self.chat_bubble.appendMarkdownChunk(chunk)
    
    def set_streaming(self, streaming: bool):
        """Set streaming status for AI messages."""
        if not self.isVisible():
            return
        self.is_streaming = streaming
        if self.sender_type == "ai":
            self.progress_bar.setVisible(streaming)
            if streaming:
                self.progress_bar.start()
            else:
                self.progress_bar.stop()
    
    def get_text(self) -> str:
        """Get plain text content of the message."""
        if not self.isVisible():
            return ""
        return self.chat_bubble.toPlainText()
    
    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)
        # Update bubble height when width changes
        self.chat_bubble._adjust_height()
