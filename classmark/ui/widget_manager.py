"""
Created on 19. 12. 2018
Module for widget manager.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

import os
from enum import Enum
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile, QObject
from PySide2.QtGui import QPixmap, QIcon
from builtins import isinstance

    
class IconName(Enum):
    """
    There are some frequently used icon names.
    """
    
    PROPERTIES="gear.svg"
    
class WidgetManager(object):
    """
    This is WidgetManager class. That class has some additional features
    that are used frequently inside that app. Is builded for managing widget that
    are loaded from files.
    """

    UI_FOLDER=os.path.dirname(os.path.realpath(__file__))
    """Absolute path to UI folder."""
    
    TEAMPLATE_FOLDER=os.path.join(UI_FOLDER, "templates")
    """Absolute path to templates folder."""
    
    ICON_FOLDER=os.path.join(UI_FOLDER, "icons")
    """Absolute path to icons folder."""
    
    
    def __init__(self):
        """
        Initializes manager.

        """
        self._widget=None
        self._qObject=QObject()
        
    @property
    def tr(self):
        """
        Get the application translator.
        """
        
        return self._qObject.tr

    @property
    def widget(self):
        """
        Get manager's widget.
        """
        return self._widget
    
    @classmethod
    def loadIcon(cls, iconName):
        """
        Load icon from file.
        
        :param iconName:  Name of the icon (with extension).
        :type iconName: string | IconName
        :return: Loaded icon.
        :rtype:  QIcon
        """
        icon = QIcon()
        if isinstance(iconName, IconName):
            iconName=iconName.value
            
        icon.addPixmap(QPixmap(os.path.join(cls.ICON_FOLDER, iconName)))
        return icon
    
    @classmethod
    def _loadTemplate(cls, template, parent=None):
        """
        Loads template from a file.
        
        :param template: Name of the template.
        :type template: string
        :param parent: Parent widget
        :type parent: QWidget
        :return: Loaded template.
        """
        loader = QUiLoader()
        loader.setWorkingDirectory(cls.TEAMPLATE_FOLDER)
        file = QFile(os.path.join(cls.TEAMPLATE_FOLDER, template + ".ui"))
        file.open(QFile.ReadOnly)
        template=loader.load(file, parent)
        file.close()

        return template

    
    