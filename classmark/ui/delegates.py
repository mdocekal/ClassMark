"""
Created on 21. 2. 2019
This module contains Qt delegates.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from PySide2.QtWidgets import QStyledItemDelegate, QRadioButton, QComboBox
from PySide2.QtCore import Qt
from typing import List

class RadioButtonDelegate(QStyledItemDelegate):
    """
    Delegate for radio button.
    Pass value True if radio button should be checked. False otherwise.
    """
    def paint(self, painter, option, index):
        #we want to always show the editor
        self.parent().openPersistentEditor(index)

    def createEditor(self, parent, option, index):
        """
        Providing an editor.
        
        :param parent: Parent widget.
        :type parent: QWidget
        :param option: Option.
        :type option: QStyleOptionViewItem
        :param index: The index in view.
        :type index: QModelIndex
        """
        radioButton = QRadioButton(parent)
        radioButton.clicked.connect(self.commitAndCloseEditor)
        return radioButton

    def setEditorData(self, editor, index):
        """
        Set data from model to editor.
        
        :param editor: The created editor.
        :type editor: QRadioButton
        :param index: The index in view.
        :type index: QModelIndex
        """
        editor.setChecked(index.data(Qt.EditRole))


    def setModelData(self, editor, model, index):
        """
        Set data from editor to model.
        
        :param editor: The created editor.
        :type editor: QRadioButton
        :param model: The model that wants to know the value.
        :type model: QAbstractItemModel
        :param index: The index in view.
        :type index: QModelIndex
        """
        model.setData(index, editor.isChecked(), Qt.EditRole);
        
    def commitAndCloseEditor(self):
        """
        Finish editing and inform model that editing is finished.
        """
        
        self.closeEditor.emit(self.sender())
        
        
    def updateEditorGeometry(self, editor, option, index):
        """
        Update actual geometry of created editor.
        
        :param editor: The created editor.
        :type editor: QRadioButton
        :param option: Option.
        :type option: QStyleOptionViewItem
        :param index: The index in view.
        :type index: QModelIndex
        """
        
        editor.setGeometry(option.rect)
        
class ComboBoxDelegate(QStyledItemDelegate):
    """
    Delegate for combo box.
    """
    
    def __init__(self, parent, items:List[str]):
        """
        Initialization of combobox.
        
        :param parent: Parent of that combobox.
        :type parent: PySide2.QtCore.QObject
        :param items: Options that can be selected by user.
        :type items: List[str]
        """
        super().__init__(parent)
        self._items=items
    
    def paint(self, painter, option, index):
        #we want to always show the editor
        self.parent().openPersistentEditor(index)
        
        
    def createEditor(self, parent, option, index):
        """
        Providing an editor.
        
        :param parent: Parent widget.
        :type parent: QWidget
        :param option: Option.
        :type option: QStyleOptionViewItem
        :param index: The index in view.
        :type index: QModelIndex
        """
        comboBox = QComboBox(parent)
        comboBox.addItems(self._items)
        comboBox.currentTextChanged.connect(self.commitAndCloseEditor)
        return comboBox
    
    def setEditorData(self, editor, index):
        """
        Set data from model to editor.
        
        :param editor: The created editor.
        :type editor: QComboBox
        :param index: The index in view.
        :type index: QModelIndex
        """
        editor.setCurrentText(index.data(Qt.EditRole))

    def setModelData(self, editor, model, index):
        """
        Set data from editor to model.
        
        :param editor: The created editor.
        :type editor: QComboBox
        :param model: The model that wants to know the value.
        :type model: QAbstractItemModel
        :param index: The index in view.
        :type index: QModelIndex
        """
        model.setData(index, editor.currentText(), Qt.EditRole);
        
    def commitAndCloseEditor(self):
        """
        Finish editing and inform model that editing is finished.
        """
        
        self.closeEditor.emit(self.sender())
        
        
    def updateEditorGeometry(self, editor, option, index):
        """
        Update actual geometry of created editor.
        
        :param editor: The created editor.
        :type editor: QComboBox
        :param option: Option.
        :type option: QStyleOptionViewItem
        :param index: The index in view.
        :type index: QModelIndex
        """
        
        editor.setGeometry(option.rect)
    
