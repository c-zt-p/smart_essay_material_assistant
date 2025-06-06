import asyncio
from pathlib import Path
from typing import List, Dict, Optional

from PySide6.QtCore import Qt, Signal, QThread, Slot, QTimer, QEvent
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QSpacerItem, QSizePolicy,
    QFileDialog, QApplication, QPlainTextEdit
)
from PySide6.QtGui import QColor
from qfluentwidgets import (
    FluentWindow, SingleDirectionScrollArea, LineEdit, PrimaryPushButton,
    MessageBoxBase, ComboBox, InfoBar, InfoBarPosition,
    BodyLabel, ProgressBar, SubtitleLabel, CaptionLabel,
    NavigationItemPosition
)
from llama_index.core.schema import NodeWithScore

try:
    import assets_rc
except ImportError:
    print("Warning: assets_rc.py not found. Icons might not load. Run: pyside6-rcc assets.qrc -o assets_rc.py")

from .icons import Icon
from .message_widgets import MessageUI
from .source_widget import RAGSourceWidget
from core.settings_manager import settings
from core.llm_client import LLMClient
#from core.rag_manager import RAGManager
from core.rag_manager_hnswlib import RAGManager


class AsyncRunner(QThread):
    """Helper QThread to run asyncio tasks"""
    task_completed = Signal(object)
    task_failed = Signal(str)
    progress_updated = Signal(float, str)

    def __init__(self, coro_callable, expects_progress_callback: bool = False, parent=None):
        super().__init__(parent)
        self.coro_callable = coro_callable
        self.expects_progress_callback = expects_progress_callback
        self.loop = None

    def run(self):
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            actual_coro_to_run = None
            if self.expects_progress_callback:
                actual_coro_to_run = self.coro_callable(progress_callback=self._emit_progress)
            else:
                actual_coro_to_run = self.coro_callable()

            if not asyncio.iscoroutine(actual_coro_to_run):
                raise TypeError(f"The callable provided to AsyncRunner did not return a coroutine. Got: {type(actual_coro_to_run)}")

            result = self.loop.run_until_complete(actual_coro_to_run)
            self.task_completed.emit(result)
        except Exception as e:
            self.task_failed.emit(str(e))
        finally:
            if self.loop:
                self.loop.close()

    @Slot(float, str)
    def _emit_progress(self, value, message):
        self.progress_updated.emit(value, message)


# Custom Event classes for thread-safe UI updates from async stream
class _StreamChunkEvent(QEvent):
    EVENT_TYPE = QEvent.Type(QEvent.User + 1)
    def __init__(self, chunk: str):
        super().__init__(_StreamChunkEvent.EVENT_TYPE)
        self.chunk = chunk


class _StreamCompleteEvent(QEvent):
    EVENT_TYPE = QEvent.Type(QEvent.User + 2)
    def __init__(self, full_response: str):
        super().__init__(_StreamCompleteEvent.EVENT_TYPE)
        self.full_response = full_response


class _StreamErrorEvent(QEvent):
    EVENT_TYPE = QEvent.Type(QEvent.User + 3)
    def __init__(self, error_message: str):
        super().__init__(_StreamErrorEvent.EVENT_TYPE)
        self.error_message = error_message


class MultilineInput(QPlainTextEdit):
    """A multiline input widget that supports sending messages with Enter or Ctrl+Enter."""
    messageSent = Signal()

    def __init__(self, parent=None, send_on_enter: bool = True):
        super().__init__(parent)
        self.send_on_enter = send_on_enter
        self.setPlaceholderText("Type your message here...")
        self.setMaximumHeight(100)  # Start with a reasonable height

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return and not event.modifiers() & Qt.ShiftModifier and self.send_on_enter:
            self.messageSent.emit()
            return
        elif event.key() == Qt.Key_Return and event.modifiers() & Qt.ControlModifier and not self.send_on_enter:
            self.messageSent.emit()
            return
        elif event.key() == Qt.Key_Return and event.modifiers() & Qt.ShiftModifier:
            super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)


class CreateDBDialog(MessageBoxBase):
    """Custom message box for getting RAG DB name"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.titleLabel = SubtitleLabel('New RAG Database Name', self)
        self.contentLabel = BodyLabel("Enter a unique name for the new RAG database:", self)
        self.dbNameLineEdit = LineEdit(self)

        self.dbNameLineEdit.setPlaceholderText("e.g., project_docs_v1")
        self.dbNameLineEdit.setClearButtonEnabled(True)

        self.warningLabel = CaptionLabel("Database name cannot be empty.", self)
        self.warningLabel.setTextColor(QColor(207, 16, 18), QColor(255, 77, 79))
        self.warningLabel.setHidden(True)

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.contentLabel)
        self.viewLayout.addWidget(self.dbNameLineEdit)
        self.viewLayout.addWidget(self.warningLabel)

        self.widget.setMinimumWidth(380)
        self.yesButton.setText("Create")
        self.cancelButton.setText("Cancel")

        self.dbNameLineEdit.textChanged.connect(lambda: self.warningLabel.setHidden(True))

    def get_db_name(self) -> str:
        return self.dbNameLineEdit.text().strip()

    def validate(self) -> bool:
        """Override to validate the input before closing."""
        db_name = self.get_db_name()
        if not db_name:
            self.warningLabel.setText("Database name cannot be empty.")
            self.warningLabel.setHidden(False)
            return False
        self.warningLabel.setHidden(True)
        return True


class ChatInterfaceWidget(QWidget):
    """Widget encapsulating the entire chat UI."""
    def __init__(self, llm_client: LLMClient, rag_manager: RAGManager, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("ChatInterfaceWidgetUniqueName")

        self.llm_client = llm_client
        self.rag_manager = rag_manager
        self.chat_history: List[Dict[str, str]] = []
        self.current_ai_message_widget: Optional[MessageUI] = None
        self.is_ai_replying = False
        self.current_rag_sources: List[NodeWithScore] = []
        self._initial_load_complete = False # Flag to manage initial setup behavior

        self._setup_ui()
        self._setup_connections()
        self._load_initial_data()

        # Welcome message
        QTimer.singleShot(100, lambda: self._add_message("system", "欢迎来到作文素材库！请选择一个作文素材库，或者新建一个。然后开始你的问答吧！"))

    def _setup_ui(self):
        """Initialize the UI components."""
        self.main_layout = QVBoxLayout(self)

        self.controls_frame = QFrame(self)
        self.controls_layout = QHBoxLayout(self.controls_frame)
        self._setup_controls()
        self.main_layout.addWidget(self.controls_frame)

        self.progress_container = QWidget(self)
        progress_layout = QHBoxLayout(self.progress_container)
        progress_layout.setContentsMargins(10, 0, 10, 0)

        self.rag_progress_label = BodyLabel("", self)
        self.rag_progress_bar = ProgressBar(self)
        self.rag_progress_bar.setFixedHeight(8)

        progress_layout.addWidget(self.rag_progress_label)
        progress_layout.addWidget(self.rag_progress_bar)

        self.progress_container.setVisible(False)
        self.main_layout.addWidget(self.progress_container)

        self.message_scroll_area = SingleDirectionScrollArea(self, orient=Qt.Vertical)
        self.message_scroll_area.setWidgetResizable(True)
        self.message_scroll_area.setFrameShape(QFrame.NoFrame)

        self.message_history_widget = QWidget()
        self.message_history_layout = QVBoxLayout(self.message_history_widget)
        self.message_history_layout.setAlignment(Qt.AlignTop)
        self.message_history_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        self.message_scroll_area.setWidget(self.message_history_widget)
        self.main_layout.addWidget(self.message_scroll_area, 1)

        self.input_frame = QFrame(self)
        self.input_layout = QHBoxLayout(self.input_frame)
        self.input_layout.setContentsMargins(10, 5, 10, 5)

        self.chat_input = MultilineInput(self)
        self.send_button = PrimaryPushButton(Icon.SEND, "Send", self)

        self.input_layout.addWidget(self.chat_input)
        self.input_layout.addWidget(self.send_button)

        self.main_layout.addWidget(self.input_frame)

    def _setup_controls(self):
        """Set up the control widgets."""
        self.controls_layout.setContentsMargins(10, 5, 10, 5)

        self.controls_layout.addWidget(BodyLabel("对话模型:", self))
        self.chat_model_selector = ComboBox(self)
        for model_cfg in settings.chat_models:
            self.chat_model_selector.addItem(model_cfg.name, userData=model_cfg.name)
        current_chat_model_name = self.llm_client.model_config.name
        self.chat_model_selector.setCurrentText(current_chat_model_name)
        self.controls_layout.addWidget(self.chat_model_selector)

        self.controls_layout.addSpacerItem(QSpacerItem(20, 1, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed))

        self.controls_layout.addWidget(BodyLabel("作文素材库:", self))
        self.rag_db_selector = ComboBox(self)
        self.rag_db_selector.addItem("无素材库（纯聊天模式）", userData=None)
        self.controls_layout.addWidget(self.rag_db_selector)

        self.create_rag_db_button = PrimaryPushButton(Icon.DATABASE, "新建素材库", self)
        self.controls_layout.addWidget(self.create_rag_db_button)
        
        self.open_dify_button = PrimaryPushButton(Icon.ROBOT, "呼叫作文素材小帮手", self)
        self.controls_layout.addWidget(self.open_dify_button)


        self.controls_layout.addStretch(1)

    def _setup_connections(self):
        """Connect signals and slots."""
        self.chat_model_selector.currentTextChanged.connect(self._on_chat_model_changed)
        self.rag_db_selector.currentTextChanged.connect(self._on_rag_db_selected_via_combobox)
        self.create_rag_db_button.clicked.connect(self._on_create_rag_db)
        self.open_dify_button.clicked.connect(self._on_open_dify)
        self.chat_input.messageSent.connect(self._on_send_message)
        self.send_button.clicked.connect(self._on_send_message)

    def _load_initial_data(self):
        """Load initial data for the UI."""
        self._update_rag_db_selector() # Populates the list

        # Always default to "None (Direct Chat)" on initial load.
        if self.rag_db_selector.count() > 0:
            self.rag_db_selector.setCurrentIndex(0)
        
        self._initial_load_complete = True

    def _update_rag_db_selector(self):
        """Update the RAG database selector with available databases."""
        self.rag_db_selector.blockSignals(True)
        current_selection_data = self.rag_db_selector.currentData()
        self.rag_db_selector.clear()
        self.rag_db_selector.addItem("无素材库（纯聊天模式）", userData=None)

        dbs = self.rag_manager.list_rag_dbs()
        for db_name in dbs:
            self.rag_db_selector.addItem(db_name, userData=db_name)

        found_previous = False
        if current_selection_data:
            for i in range(self.rag_db_selector.count()):
                if self.rag_db_selector.itemData(i) == current_selection_data:
                    self.rag_db_selector.setCurrentIndex(i)
                    found_previous = True
                    break
        
        if not found_previous: # If previous selection not found or no previous selection
            if dbs and self.rag_db_selector.currentIndex() !=0 :
                 pass
            else:
                 self.rag_db_selector.setCurrentIndex(0) # Select "None" by default


        self.rag_db_selector.blockSignals(False)

    @Slot(str)
    def _on_chat_model_changed(self, model_name: str):
        """Handle chat model selection change."""
        try:
            self.llm_client = LLMClient(model_name)
            self.rag_manager._configure_llama_index_settings(chat_model_name=model_name)
            InfoBar.success(
                "Success",
                f"对话模型切换为 {model_name}",
                parent=self.window(),
                duration=2000,
                position=InfoBarPosition.TOP
            )
        except ValueError as e:
            InfoBar.error(
                "Error",
                str(e),
                parent=self.window(),
                duration=3000,
                position=InfoBarPosition.TOP
            )
            self.chat_model_selector.setCurrentText(self.llm_client.model_config.name)

    @Slot(str)
    def _on_rag_db_selected_via_combobox(self, text_not_used):
        """Handle RAG database selection."""
        db_name = self.rag_db_selector.currentData()
        if db_name: # An actual RAG DB is selected
            self._handle_rag_db_load(db_name)
        else: # "None (Direct Chat)" is selected
            switched_from_active_rag = self.rag_manager.query_engine is not None
            
            self.rag_manager.current_db_name = None
            self.rag_manager.query_engine = None
            
            # Only show info bar if RAG was actively being used and now it's not,
            # and it's not the very first call during initialization.
            if switched_from_active_rag and self._initial_load_complete:
                InfoBar.info(
                    "Info",
                    "纯聊天模式开启，素材库下线",
                    parent=self.window(),
                    duration=2000,
                    position=InfoBarPosition.TOP_RIGHT
                )

    def _handle_rag_db_load(self, db_name: str):
        """Load the selected RAG database."""
        self.progress_container.setVisible(True)
        self.rag_progress_bar.setVal(0)
        self.rag_progress_label.setText(f"加载素材库: {db_name}...")

        coro_func = self.rag_manager.load_rag_db
        self.db_loader_thread = AsyncRunner(
            lambda progress_callback: coro_func(db_name, progress_callback),
            expects_progress_callback=True
        )
        self.db_loader_thread.task_completed.connect(self._on_rag_db_load_completed)
        self.db_loader_thread.task_failed.connect(self._on_rag_db_op_failed)
        self.db_loader_thread.progress_updated.connect(self._update_rag_progress)
        self.db_loader_thread.start()

    @Slot(object)
    def _on_rag_db_load_completed(self, success: bool):
        """Handle RAG database load completion."""
        self.progress_container.setVisible(False)
        if success:
            db_name = self.rag_manager.current_db_name
            InfoBar.success(
                "Success",
                f"素材库 '{db_name}' 加载成功.",
                parent=self.window(),
                duration=3000,
                position=InfoBarPosition.TOP_RIGHT
            )
        else:
            # If load failed, revert selection to "None" in combobox
            self.rag_db_selector.setCurrentIndex(0) # This will trigger _on_rag_db_selected_via_combobox

    @Slot()
    def _on_create_rag_db(self):
        """Handle creation of a new RAG database."""
        file_dialog = QFileDialog(self.window())
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("Documents (*.txt *.md *.pdf *.docx *.json *.html *.htm)")
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if not file_paths:
                return

            create_db_dialog = CreateDBDialog(self.window())

            if create_db_dialog.exec():
                db_name = create_db_dialog.get_db_name()

                if db_name in self.rag_manager.list_rag_dbs():
                    InfoBar.warning(
                        "Name Exists",
                        f"素材库 '{db_name}' 已经存在，请另起雅名",
                        parent=self.window(),
                        duration=3000,
                        position=InfoBarPosition.TOP_RIGHT
                    )
                    return

                self.progress_container.setVisible(True)
                self.rag_progress_bar.setVal(0)
                self.rag_progress_label.setText(f"创建素材库: {db_name}...")

                coro_func = self.rag_manager.create_rag_db
                self.db_creator_thread = AsyncRunner(
                    lambda progress_callback: coro_func(db_name, file_paths, progress_callback),
                    expects_progress_callback=True
                )
                self.db_creator_thread.task_completed.connect(self._on_rag_db_create_completed)
                self.db_creator_thread.task_failed.connect(self._on_rag_db_op_failed)
                self.db_creator_thread.progress_updated.connect(self._update_rag_progress)
                self.db_creator_thread.start()
                
    def _on_open_dify(self):
        """Open Dify web interface"""
        import webbrowser
        webbrowser.open("https://udify.app/chat/qdmzJb4iNrPXGVnJ")

    @Slot(object)
    def _on_rag_db_create_completed(self, success: bool):
        """Handle RAG database creation completion."""
        self.progress_container.setVisible(False)
        if success:
            try:
                # Attempt to get the name from the progress label if it's reliable
                label_text = self.rag_progress_label.text()
                if "创建素材库: " in label_text and "..." in label_text:
                     last_created_db_name = label_text.split("创建素材库: ")[1].replace("...", "")
                else: # Fallback if parsing fails
                    # This might be tricky if the DB name isn't directly passed back
                    # For now, use the RAG manager's current DB if it was set by create_rag_db
                    last_created_db_name = self.rag_manager.current_db_name or "Newly Created DB"
            except Exception:
                last_created_db_name = self.rag_manager.current_db_name or "Newly Created DB"


            InfoBar.success(
                "Success",
                f"素材库 '{last_created_db_name}' 已创建",
                parent=self.window(),
                duration=3000,
                position=InfoBarPosition.TOP_RIGHT
            )
            self._update_rag_db_selector() # Reload DB list
            # Select the newly created DB
            for i in range(self.rag_db_selector.count()):
                if self.rag_db_selector.itemData(i) == last_created_db_name:
                    self.rag_db_selector.setCurrentIndex(i) # This will trigger load via _on_rag_db_selected
                    break

    @Slot(str)
    def _on_rag_db_op_failed(self, error_message: str):
        """Handle RAG database operation failure."""
        self.progress_container.setVisible(False)
        InfoBar.error(
            "RAG Operation Failed",
            error_message,
            parent=self.window(),
            duration=5000,
            position=InfoBarPosition.TOP_RIGHT
        )
        self._update_rag_db_selector() # Refresh DB list, current selection might change

    @Slot(float, str)
    def _update_rag_progress(self, value: float, message: str):
        """Update progress bar and label during RAG operations."""
        self.rag_progress_label.setText(message)
        if value >= 0:
            self.rag_progress_bar.setVal(int(value * 100))
        if value == 1.0 and "Error" not in message:
            QTimer.singleShot(1500, lambda: self.progress_container.setVisible(False))
        elif "Error" in message or value == -1: # Explicit error signal or error in message
            self.progress_container.setVisible(False)


    def _add_message(self, sender_type: str, text: str, is_streaming: bool = False) -> MessageUI:
        """Add a new message to the chat history UI."""
        if self.message_history_layout.count() > 0:
            spacer = self.message_history_layout.itemAt(self.message_history_layout.count() - 1)
            if isinstance(spacer, QSpacerItem):
                self.message_history_layout.removeItem(spacer)

        message_widget = MessageUI(sender_type, text, is_streaming, self.message_history_widget)
        self.message_history_layout.addWidget(message_widget)

        if sender_type == "ai" and not is_streaming and self.current_rag_sources:
            sources_widget = RAGSourceWidget(self.current_rag_sources, self.message_history_widget)
            self.message_history_layout.addWidget(sources_widget)

        self.message_history_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        QTimer.singleShot(0, lambda: self.message_scroll_area.verticalScrollBar().setValue(
            self.message_scroll_area.verticalScrollBar().maximum()
        ))
        return message_widget

    @Slot()
    def _on_send_message(self):
        """Handle sending a new user message."""
        if self.is_ai_replying:
            InfoBar.warning("Busy", "AI is currently replying.", parent=self.window(), duration=2000)
            return

        user_query = self.chat_input.toPlainText().strip()
        if not user_query:
            return

        self._add_message("user", user_query)
        self.chat_history.append({"role": "user", "content": user_query})
        self.chat_input.clear()
        self.is_ai_replying = True
        self.send_button.setEnabled(False)

        self.current_ai_message_widget = self._add_message("ai", "", is_streaming=True)
        
        # RAG is used if a DB is selected in the combobox AND its query_engine is loaded
        use_rag = self.rag_db_selector.currentData() is not None and \
                  self.rag_manager.query_engine is not None

        if use_rag:
            self.current_rag_sources = []
            coro_func = self.rag_manager.query_rag
            self.rag_query_thread = AsyncRunner(
                lambda: coro_func(user_query, self.chat_history[:-1]),
                expects_progress_callback=False
            )
            self.rag_query_thread.task_completed.connect(self._on_rag_query_completed)
            self.rag_query_thread.task_failed.connect(self._on_ai_reply_failed)
            self.rag_query_thread.start()
        else:
            self.current_rag_sources = [] # Ensure no old sources are displayed
            self._start_direct_chat_stream(user_query, self.chat_history[:-1])

    def _start_direct_chat_stream(self, query: str, history: List[Dict[str, str]]):
        """Start streaming chat response from LLM."""
        async def stream_and_emit():
            full_response = ""
            try:
                async for chunk in self.llm_client.stream_chat(query, history):
                    QApplication.instance().postEvent(self, _StreamChunkEvent(chunk))
                    full_response += chunk
                QApplication.instance().postEvent(self, _StreamCompleteEvent(full_response))
            except Exception as e:
                QApplication.instance().postEvent(self, _StreamErrorEvent(str(e)))

        class StreamerThread(QThread):
            def __init__(self, coro_fn, parent_widget):
                super().__init__(parent_widget)
                self.coro_fn = coro_fn
            def run(self):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.coro_fn())
                loop.close()

        self.streamer_thread_instance = StreamerThread(stream_and_emit, self)
        self.streamer_thread_instance.start()

    def customEvent(self, event: QEvent):
        """Handle custom events for stream updates."""
        if event.type() == _StreamChunkEvent.EVENT_TYPE:
            self._on_ai_reply_chunk_received(event.chunk)
        elif event.type() == _StreamCompleteEvent.EVENT_TYPE:
            self._on_ai_reply_finished(event.full_response)
        elif event.type() == _StreamErrorEvent.EVENT_TYPE:
            self._on_ai_reply_failed(event.error_message)
        else:
            super().customEvent(event)

    @Slot(tuple)
    def _on_rag_query_completed(self, result: tuple):
        """Handle RAG query completion."""
        final_query_for_llm, source_nodes = result
        self.current_rag_sources = source_nodes
        self._start_direct_chat_stream(final_query_for_llm, self.chat_history[:-1])

    @Slot(str)
    def _on_ai_reply_chunk_received(self, chunk: str):
        """Handle receiving a chunk of the AI's response."""
        if self.current_ai_message_widget:
            self.current_ai_message_widget.append_text_chunk(chunk)
            QTimer.singleShot(0, lambda: self.message_scroll_area.verticalScrollBar().setValue(
                self.message_scroll_area.verticalScrollBar().maximum()
            ))

    @Slot(str)
    def _on_ai_reply_finished(self, full_response: str):
        """Handle completion of the AI's response."""
        if self.current_ai_message_widget:
            self.current_ai_message_widget.set_text(full_response)
            self.current_ai_message_widget.set_streaming(False)

            if self.current_rag_sources:
                ai_msg_idx = self.message_history_layout.indexOf(self.current_ai_message_widget)
                if ai_msg_idx != -1: # Check if widget is found
                    sources_widget = RAGSourceWidget(self.current_rag_sources, self.message_history_widget)
                    self.message_history_layout.insertWidget(ai_msg_idx + 1, sources_widget)
                else: # Fallback if index not found (should not happen)
                    sources_widget = RAGSourceWidget(self.current_rag_sources, self.message_history_widget)
                    self.message_history_layout.addWidget(sources_widget)


        self.chat_history.append({"role": "assistant", "content": full_response})
        self.is_ai_replying = False
        self.send_button.setEnabled(True)
        self.current_ai_message_widget = None

        QTimer.singleShot(0, lambda: self.message_scroll_area.verticalScrollBar().setValue(
            self.message_scroll_area.verticalScrollBar().maximum()
        ))


    @Slot(str)
    def _on_ai_reply_failed(self, error_message: str):
        """Handle failure of the AI's response."""
        if self.current_ai_message_widget:
            self.current_ai_message_widget.set_text(f"Error: {error_message}")
            self.current_ai_message_widget.set_streaming(False)
        else:
            self._add_message("system", f"Error: {error_message}")

        InfoBar.error(
            "AI Error",
            error_message,
            parent=self.window(),
            duration=5000,
            position=InfoBarPosition.TOP_RIGHT
        )
        self.is_ai_replying = False
        self.send_button.setEnabled(True)
        self.current_ai_message_widget = None
        self.current_rag_sources = []


class ChatWindow(FluentWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("素材库搜搜")
        
        # Setup Core Components
        self.llm_client = LLMClient()
        self.rag_manager = RAGManager()

        # Create the main chat interface widget
        self.chat_interface = ChatInterfaceWidget(self.llm_client, self.rag_manager, self)
        
        self.initNavigation()
        self.initWindow()

    def initNavigation(self):
        """Initialize navigation panel."""
        self.addSubInterface(
            self.chat_interface,
            Icon.SEND,
            'Chat',
            NavigationItemPosition.TOP
        )

    def initWindow(self):
        """Initialize window properties."""
        self.resize(1000, 750)
        try:
            app_icon = Icon.ROBOT.icon()
            if not app_icon.isNull():
                self.setWindowIcon(app_icon)
            else:
                print("Warning: Custom app icon is null. Check icon definition and resources.")
        except Exception as e:
            print(f"Warning: Could not set custom window icon: {e}")

    def closeEvent(self, event):
        """Handle window close event."""
        super().closeEvent(event)

            
