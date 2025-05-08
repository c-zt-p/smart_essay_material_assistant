# ui/icons.py
# ui/icons.py
from enum import Enum
from qfluentwidgets import FluentIconBase, Theme
# from PySide6.QtGui import QColor # 保留 QColor 以备将来使用 .icon(color=...) 或 .colored(...)

class Icon(FluentIconBase, Enum):
    """ 
    Custom icons.
    Each enum member represents an icon. The value of the enum member
    is the filename stem (without extension or theme suffix).
    """

    SEND = "Send"
    ROBOT = "Robot"
    USER = "User"
    SYSTEM = "System"
    UPLOAD = "Upload"
    DATABASE = "Database"
    DOCUMENT = "Document" # For source documents

    def path(self, theme=Theme.AUTO) -> str:
        """
        Returns the resource path to the icon's SVG file.
        
        Parameters
        ----------
        theme : Theme
            The current theme. This implementation ignores the theme
            as the icons are assumed to be single-color or designed
            to work with both light and dark themes if getIconColor() is used
            by the QFluentWidgets components themselves when rendering SVGs.
            If you had separate icons for light/dark themes, you would use
            getIconColor(theme) to construct the path.
        """
        # self.value will be the string like "Send", "Robot", etc.
        return f':/assets/icons/{self.value}.svg'

# 使用示例 (在其他 QFluentWidgets 组件中):
# from qfluentwidgets import PushButton
# button = PushButton(Icon.SEND)
#
# 或者如果需要 QIcon 实例:
# q_icon = Icon.SEND.icon() # 获取默认主题的 QIcon
# q_icon_light = Icon.SEND.icon(Theme.LIGHT)
# q_icon_dark_blue = Icon.SEND.icon(Theme.DARK, color=QColor("blue"))


# Create resource file: assets.qrc
# <?xml version="1.0"?>
# <RCC>
#   <qresource prefix="assets">
#     <file>icons/Send.svg</file>
#     <file>icons/Robot.svg</file>
#     <file>icons/User.svg</file>
#     <file>icons/System.svg</file>
#     <file>icons/Upload.svg</file>
#     <file>icons/Database.svg</file>
#     <file>icons/Document.svg</file>
#   </qresource>
# </RCC>
#
# Then compile it: pyside6-rcc assets.qrc -o assets_rc.py
# Make sure assets_rc.py is imported in main.py or chat_window.py
