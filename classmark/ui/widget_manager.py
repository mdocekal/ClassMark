"""
Created on 19. 12. 2018
Module for widget manager.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

import os
from enum import Enum
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile, QObject, Qt
from PySide2.QtGui import QPixmap, QIcon
from PySide2.QtWidgets import QWidget, QComboBox, QCheckBox,QLineEdit,QHBoxLayout, QVBoxLayout, QLabel
from builtins import isinstance
from typing import List
from ..core.plugins import PluginAttribute

    
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

    

class AttributesWidgetManager(WidgetManager):
    """
    This manager could help you create attributes widget for your plugin, if you do not want
    to do it on your own from scratch.
    """
    
    def __init__(self, attributes:List[PluginAttribute], parent=None):
        """
        Initialization of manager.
        
        :param attributes: Attributes that are manageable by user.
        :type attributes: List[PluginAttribute]
        :param parent: Parent widget.
        :type parent: QWidget
        """
        super().__init__()
        self._attributes=attributes
        
        #Create widget for given attributes.
        self._widget=QWidget(parent)
        
        mainLayout=QVBoxLayout(self._widget)
        mainLayout.setAlignment(Qt.AlignTop)
        if len(self._attributes)==0:
            mainLayout.addWidget(QLabel(self.tr("No attributes."),self._widget))
        else:
            for a in self._attributes:
                #create layout for this attribute
                mainLayout.addWidget({
                    PluginAttribute.PluginAttributeType.CHECKABLE:self._createCheckableType,
                    PluginAttribute.PluginAttributeType.VALUE:self._createValueType,
                    PluginAttribute.PluginAttributeType.SELECTABLE:self._createSelectableType,
                    PluginAttribute.PluginAttributeType.GROUP_VALUE:self._createGroupValueType,
                    }[a.type](a))

        self._widget.setLayout(mainLayout)
        
    def _createWidget(self, a, inputW:QWidget, vertical=True):
        """
        Creates widget for given attribute and input.
        
        :param a:The attribute.
        :type a: PluginAttribute
        :param inputW: Input that is used for value manipulation.
        :type inputW: QWidget
        :param vertical: True means QVBoxLayout and Fale means QHBoxLayout.
        :type vertical: bool
        :return: Widget for attribute.
        :rtype: QWidget
        """
        w=QWidget(self._widget)
        inputW.setParent(w)
        if vertical:
            layout=QVBoxLayout(w)
            layout.setContentsMargins(0, 5, 5, 0)
        else:
            layout=QHBoxLayout(w)
            layout.setMargin(0)
        
        w.setLayout(layout)

        label=QLabel(a.name+":", self._widget)
        label.setBuddy(inputW)
        
        
        layout.addWidget(label)
        layout.addWidget(inputW)
        
        return w
        

    def _createCheckableType(self, a:PluginAttribute):
        """
        Creates widget for attribute of CHECKABLE type.
        
        :param a: The attribute.
        :type a: PluginAttribute
        :return: Widget for attribute.
        :rtype: QWidget
        """

        inputW=QCheckBox()
        inputW.setCheckState(Qt.Checked if a.value else Qt.Unchecked)
        inputW.stateChanged.connect(a.setValue)
        
        return self._createWidget(a, inputW, False)
        
        
    def _createValueType(self, a:PluginAttribute):
        """
        Creates widget for attribute of VALUE type.
        
        :param a: The attribute.
        :type a: PluginAttribute
        :return: Widget for attribute.
        :rtype: QWidget
        """

        inputW=QLineEdit()
        if a.value is not None:
            inputW.setText(str(a.value))
        inputW.textChanged.connect(a.setValueBind(inputW.setText))

        return self._createWidget(a, inputW)
    
    def _createSelectableType(self, a:PluginAttribute):
        """
        Creates widget for attribute of SELECTABLE type.
        
        :param a: The attribute.
        :type a: PluginAttribute
        :return: Layout for attribute.
        :rtype: QLayout
        """
        inputW=QComboBox()
        inputW.addItems(a.selVals)
        if a.value is not None:
            inputW.setCurrentText(str(a.value))
        inputW.currentTextChanged.connect(a.setValueBind(inputW.setCurrentText))

        return self._createWidget(a, inputW)
        
    def _createGroupValueType(self, a:PluginAttribute):
        """
        Creates widget for attribute of GROUP_VALUE type.
        
        :param a: The attribute.
        :type a: PluginAttribute
        :return: Widget for attribute.
        :rtype: QWidget
        """
        
        inputW=QWidget()
        layout=QVBoxLayout(inputW)
        layout.setContentsMargins(5, 0, 5, 5)
        inputW.setLayout(layout)
        
        for v in a.value:
            inputW=QLineEdit()
            if v is not None:
                inputW.setText(str(v))
            inputW.textChanged.connect(a.setValueBind(inputW.setText))
        
        return self._createWidget(a, inputW)
        