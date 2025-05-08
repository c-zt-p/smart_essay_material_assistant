# main.py
import sys
from PySide6.QtWidgets import QApplication
from qfluentwidgets import setTheme, Theme
from ui.chat_window import ChatWindow

# Set font for the entire application (optional, but good for consistency)
from PySide6.QtGui import QFont
QApplication.setFont(QFont("Segoe UI", 9)) # Example for Windows

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Apply Fluent theme (Light or Dark)
    setTheme(Theme.LIGHT)
    # setTheme(Theme.DARK)

    window = ChatWindow()
    window.show()
    sys.exit(app.exec())