"""
Created on 20. 12. 2018
Module for experiment section of the application.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from .widget_manager import WidgetManager, IconName
from .section_router import SectionRouter
from PySide2.QtWidgets import QFileDialog, QHeaderView, QPushButton
from PySide2.QtCore import Qt, QModelIndex
from ..core.experiment import Experiment
from .delegates import RadioButtonDelegate, ComboBoxDelegate
from .models import TableDataAttributesModel
from typing import Callable
from functools import partial


class ClassifierRowWidgetManager(WidgetManager):
    """
    Widget manager of row with selection of classifier for testing.
    
    About user choice of classifier this widget informs Experiment object directly, but
    if you want to catch remove row and show properties events, than register your callback
    with appropriate (registerRemoveEvent,registerPropertiesEvent) methods.
    
    """
    TEMPLATE="classifier_row"
    """Corresponding template name."""

    def __init__(self, experiment:Experiment, parent=None):
        """
        Creates classifier row widget.

        :param experiment: This manager will use that experiment for storing information
            about selection of classifier.
        :type experiment: Experiment
        :param parent: Parent widget.
        :type parent: QWidget
        """
        super().__init__()
        self._widget=self._loadTemplate(self.TEMPLATE, parent)
        self._experiment=experiment
        
        #lets create own classifier slot
        self.classifierSlot=self._experiment.newClassifierSlot()
        
        #let's fill the combo box
        self._widget.classifiersComboBox.addItems([c.getName() for c in self._experiment.availableClassifiers.values()])
        self._clsNameToCls={c.getName():c for c in self._experiment.availableClassifiers.values()}
        self._onChange()
        
        #register events
        self._widget.removeClassifierButton.clicked.connect(self._onRemove)
        self._widget.classifiersComboBox.currentTextChanged.connect(self._onChange)
        
        self._removeCallback=None   #registered remove event callback
        
    def __eq__(self, other):
        if not isinstance(other, __class__):
            return False
        return self.classifierSlot==other.classifierSlot #two different managers can not share same slot
    
    def __hash__(self):
        return hash(self.classifierSlot)
    
    def registerRemoveEvent(self, callback):
        """
        Provided callback will be called when user wants to remove this row.
        
        :param callback: Method that will be called. That method must have one 
            parameter where this manager will be passed.
        :type callback: Callable[[ClassifierRowWidgetManager],None]
        """
        self._removeCallback=callback
        
    def _onChange(self):
        """
        Classifier change.
        """
        
        self.classifierSlot.classifier=self._clsNameToCls[self._widget.classifiersComboBox.currentText()]
        
        
    def _onRemove(self):
        """
        Remove event occurred.
        """
        self._experiment.removeClassifierSlot(self.classifierSlot)
        if self._removeCallback is not None:
            self._removeCallback(self)
        
        
    def registerPropertiesEvent(self, callback:Callable):
        """
        Provided callback will be called when user wants to show classifiers properties.
        :param callback: Method that will be called.
        :type callback: Callable
        """
        self._widget.propertiesButton.clicked.connect(callback)

class ExperimentSection(WidgetManager):
    """
    Experiment section manager class.
    """

    TEMPLATE="experiment"
    """Corresponding template name."""

    def __init__(self, sectionRouter:SectionRouter, parent=None, load=None):
        """
        Initializes experiment section.
        
        :param sectionRouter: Router for in app navigation.
        :type sectionRouter: SectionRouter
        :param parent: Parent widget
        :type parent: QWidget
        :param load: Path to file containing experiment configuration.
            None means that new experiment should be loaded.
        :type load: string|None
        """
        
        super().__init__()
        self._widget=self._loadTemplate(self.TEMPLATE, parent)
        self._router=sectionRouter
        #create new or load saved experiment
        self._experiment=Experiment(load)
                
        self._initData()
        self._initCls()
        
    def _initData(self):
        """
        Initialization of data tab.
        """
            
        #register click events
        self._widget.buttonChooseData.clicked.connect(self.loadDataset)
        
        #set cell span for features extractor header
        self._widget.tableDataAttributes.setSpan (-1,TableDataAttributesModel.COLL_FEATURE_EXTRACTION,1, 2)
        
        #assign model to table view
        self._widget.tableDataAttributes.setModel(TableDataAttributesModel(self._widget, 
                                                                           self._experiment))
        #set delegates
        self._widget.tableDataAttributes.setItemDelegateForColumn(TableDataAttributesModel.COLL_LABEL, 
                                                                  RadioButtonDelegate(self._widget.tableDataAttributes))
        
        self._widget.tableDataAttributes.setItemDelegateForColumn(TableDataAttributesModel.COLL_FEATURE_EXTRACTION, 
                                                                  ComboBoxDelegate(self._widget.tableDataAttributes,
                                                                    [n for n in self._experiment.featuresExt]))
        self._setPropertiesButtonsToDataAttrTable()
        
        #set resize modes
        self._setSecResModeForDataAttrTable()
        
    def _initCls(self):
        """
        Initialization of classifiers tab.
        """
        
        #register click events
        self._widget.buttonAddClassifierOption.clicked.connect(self._addClassifierOption)
        
        #set alignment to classifier layout
        self._widget.testClassifiersLayout.setAlignment(Qt.AlignTop)
        
        self._classifiersRowsManagers=[]
        
        #add one classifier option
        self._addClassifierOption()
        
        
    def _addClassifierOption(self):
        """
        Add one new classifier option to UI.
        """
        
        #lets create option row manager
        manager=ClassifierRowWidgetManager(self._experiment, self._widget)
        #register events
        manager.registerRemoveEvent(self._removeClassifierOption)
        self._classifiersRowsManagers.append(manager)
        self._widget.testClassifiersLayout.addWidget(manager.widget);
        
    def _removeClassifierOption(self, manager:ClassifierRowWidgetManager):
        """
        Remove classifier option from UI.
        
        :param manager: Manger of classifier row.
        :type manager:ClassifierRowWidgetManager
        """
        self._classifiersRowsManagers.remove(manager)
        manager.widget.deleteLater()
        
        
    def loadExperiment(self, load):
        """
        Loads given experiment.
        
        :param load: Path to file containing experiment configuration.
        :type load: string
        """
        pass
    
    def loadDataset(self):
        """
        Selection of data set.
        """
        file=QFileDialog.getOpenFileName(self._widget, self.tr("Load dataset"), "~", self.tr("CSV files (*.csv)"))
        if file[0]:
            #use selected a file
            self._experiment.loadDataset(file[0])
            self._widget.pathToData.setText(file[0])
            self._widget.tableDataAttributes.setModel(TableDataAttributesModel(self._widget, self._experiment))
            self._setPropertiesButtonsToDataAttrTable()
            
    def _setPropertiesButtonsToDataAttrTable(self):
        """
        Sets properties button for features extraction methods in data attribute table.
        """

        for row in range(self._widget.tableDataAttributes.model().rowCount()):
            button= QPushButton()
            button.setIcon(self.loadIcon(IconName.PROPERTIES))
            button.clicked.connect(self._fExtPropClicked(row))

            self._widget.tableDataAttributes.setIndexWidget(
                self._widget.tableDataAttributes.model().index(row, TableDataAttributesModel.COLL_FEATURE_EXTRACTION_PROPERITES),button)

        
    def _fExtPropClicked(self, row):        
        """
        Button for showing properties of features extractor
        was clicked.
        
        :param row: Button was clicked on that row.
        :type row: int
        """
        return partial(print, row)
        
    def _showFeaturesExtractorProperties(self):
        pass
            
    def _setSecResModeForDataAttrTable(self):
        """
        Sets section resize mode for data attribute table.
        """
        
        self._widget.tableDataAttributes.horizontalHeader().setSectionResizeMode(TableDataAttributesModel.COLL_ATTRIBUTE_NAME,QHeaderView.Stretch);
        self._widget.tableDataAttributes.horizontalHeader().setSectionResizeMode(TableDataAttributesModel.COLL_USE,QHeaderView.ResizeMode.ResizeToContents);
        self._widget.tableDataAttributes.horizontalHeader().setSectionResizeMode(TableDataAttributesModel.COLL_PATH,QHeaderView.ResizeMode.ResizeToContents);
        self._widget.tableDataAttributes.horizontalHeader().setSectionResizeMode(TableDataAttributesModel.COLL_LABEL,QHeaderView.ResizeMode.ResizeToContents);
        self._widget.tableDataAttributes.horizontalHeader().setSectionResizeMode(TableDataAttributesModel.COLL_FEATURE_EXTRACTION,QHeaderView.ResizeMode.ResizeToContents);
        self._widget.tableDataAttributes.horizontalHeader().setSectionResizeMode(TableDataAttributesModel.COLL_FEATURE_EXTRACTION_PROPERITES,QHeaderView.ResizeMode.ResizeToContents);
        



