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
from PySide2.QtWidgets import QWidget, QComboBox, QCheckBox,QLineEdit,QHBoxLayout, QVBoxLayout, QLabel,\
    QPushButton
from builtins import isinstance
from typing import List
from ..core.plugins import PluginAttribute

class IconName(Enum):
    """
    There are some frequently used icon names.
    """
    
    PROPERTIES="gear.svg"
    PLUS="plus.svg"
    MINUS="minus.svg"
    
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
    
    class WNormalize(QWidget):
        """
        Widget for normalization settings.
        Settings are stored
        """
        
        def __init__(self, a, parent=None):
            """
            Initialization of normalization widget.
            
            :param a:The attribute.
            :type a: PluginAttribute
            :param parent: Parent widget.
            :type parent: QWidget
            """
            super().__init__(parent)
            self._a=a
            
        
    
    class WGroup(QWidget):
        """
        Group widget.
        """
        
        def __init__(self, a, parent=None):
            """
            Initialization of group.
            
            :param a:The attribute.
            :type a: PluginAttribute
            :param parent: Parent widget.
            :type parent: QWidget
            """
            super().__init__(parent)
            self._a=a

            layout=QVBoxLayout(self)
            layout.setContentsMargins(5, 0, 5, 5)
            self.setLayout(layout)
            
            self.inputW=QWidget(self)
            self.inputW.setLayout(QVBoxLayout(self.inputW))
            layout.addWidget(self.inputW)
            
            for i,v in enumerate(a.value):
                self._addItemWidget(i, v)

            
            #add buttons for adding and removing items
            buttonsLayoutWidget=QWidget(self.inputW)
            layout.addWidget(buttonsLayoutWidget)
            buttonsLayout=QHBoxLayout(buttonsLayoutWidget)
            buttonsLayoutWidget.setLayout(buttonsLayout)
            
            plusButton=QPushButton(buttonsLayoutWidget)
            plusButton.setIcon(WidgetManager.loadIcon(IconName.PLUS))
            plusButton.clicked.connect(self.appendItem)
            buttonsLayout.addWidget(plusButton)
    
            minusButton=QPushButton(buttonsLayoutWidget)
            minusButton.setIcon(WidgetManager.loadIcon(IconName.MINUS))
            minusButton.clicked.connect(self.popItem)
            buttonsLayout.addWidget(minusButton)
                
        def _addItemWidget(self, i, v):
            """
            Create item widgets and append them to layout.

            :param i: Position of item in group.
            :type i: int
            :param v: Item value.
            :type v: Any
            """
            
            actInp=QLineEdit(self.inputW)
            if self._a.groupItemLabel:
                #ok we have label for items
                label=QLabel(self._a.groupItemLabel.format(i+1)+":", self.inputW)
                label.setTextInteractionFlags(Qt.TextSelectableByMouse|Qt.TextSelectableByKeyboard)
                label.setBuddy(actInp)
                self.inputW.layout().addWidget(label)
                
            if v is not None:
                actInp.setText(str(v))
            actInp.textChanged.connect(self._a.setValueBind(actInp.setText,i))

            self.inputW.layout().addWidget(actInp)
            
            
        def appendItem(self):
            """
            Append new item.
            """
            self._addItemWidget(len(self._a.value),None)
            self._a.value.append(None)
        
        def popItem(self):
            """
            Remove item from the end of group.
            """
            if self._a.value is not None and len(self._a.value)>0:
                self._a.value.pop()
                #remove input and label
                for i in range(2):
                    child = self.inputW.layout().takeAt(self.inputW.layout().count()-1)
                    child.widget().deleteLater()
    
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
            label=QLabel(self.tr("No attributes."),self._widget)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse|Qt.TextSelectableByKeyboard)
            mainLayout.addWidget(label)
        else:
            for a in self._attributes:
                #create layout for this attribute
                mainLayout.addWidget({
                    PluginAttribute.PluginAttributeType.CHECKABLE:self._createCheckableType,
                    PluginAttribute.PluginAttributeType.VALUE:self._createValueType,
                    PluginAttribute.PluginAttributeType.SELECTABLE:self._createSelectableType,
                    PluginAttribute.PluginAttributeType.GROUP_VALUE:self._createGroupValueType,
                    PluginAttribute.PluginAttributeType.SELECTABLE_PLUGIN:self._createSelectablePluginType,
                    }[a.type](a))

        self._widget.setLayout(mainLayout)
        
    def _createWidget(self, a, inputW:QWidget, vertical=True):
        """
        Creates widget for given attribute and input.
        
        :param a: The attribute.
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
        label.setTextInteractionFlags(Qt.TextSelectableByMouse|Qt.TextSelectableByKeyboard)
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
    
        return self._createWidget(a, self.WGroup(a))
    
    
    def _createSelectablePluginType(self, a:PluginAttribute):
        """
        Creates widget for attribute of NORMALIZE type.
        
        :param a: The attribute.
        :type a: PluginAttribute
        :return: Widget for attribute.
        :rtype: QWidget
        """
        
        attrW=QWidget(self._widget)

        attrLayout=QVBoxLayout(attrW)
        attrLayout.setMargin(0)
        attrW.setLayout(attrLayout)
        
        inputW=QComboBox()
        inputW.addItems([None if n is None else n.getName() for n in a.selVals])
        
        inputW.currentTextChanged.connect(self._pluginSelected(a, attrLayout))
        
        if a.value is not None:
            inputW.setCurrentText(str(a.value.getName()))
        
        resW=self._createWidget(a, inputW)
        resW.layout().addWidget(attrW)
        return resW
    
    def _pluginSelected(self, a, attrLayout):
        """
        Partial for new plugin select event.
        
        :param a: The attribute for which we are setting plugin.
        :type a: PluginAttribute
        :param attrLayout: Layout where the attributes of selected plugin will be added.
        :type attrLayout: QLayout
        :return: Callable that sets the new plugin and hides old.
        :rtype: Callable[[str],None]
        """
        
        def perform(pluginName):
            #remove old
            child=attrLayout.takeAt(0)
            while child:
                child.widget().deleteLater()
                child=attrLayout.takeAt(0)
                
            if pluginName is None or pluginName=="":
                a.setValue(None)
            else:
                for x in a.selVals:
                    if x is None:
                        continue
                        
                    if x.getName()==pluginName:
                        if a.value is None or pluginName!=a.value.getName():
                            #create the new plugin and set it
                            a.setValue(x())
        
                        hasOwnWidget=a.value.getAttributesWidget(attrLayout)
                
                        if hasOwnWidget is not None:
                            attrLayout.addWidget(hasOwnWidget)
                        else:
                            if len(a.value.getAttributes())>0:
                                self.manager=AttributesWidgetManager(a.value.getAttributes(), attrLayout.parentWidget())
                                attrLayout.addWidget(self.manager.widget)
                        
                        break                
        return perform
        