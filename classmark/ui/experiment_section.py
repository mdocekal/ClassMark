"""
Created on 20. 12. 2018
Module for experiment section of the application.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from .widget_manager import WidgetManager
from .section_router import SectionRouter
from PySide2.QtWidgets import QFileDialog, QHeaderView, QHBoxLayout, QComboBox
from ..core.experiment import Experiment
from .delegates import RadioButtonDelegate, ComboBoxDelegate
from .models import TableDataAttributesModel
from ..core.plugins import CLASSIFIERS, FEATURE_EXTRACTORS

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
        
        #assign model to table view
        self._widget.tableDataAttributes.setModel(TableDataAttributesModel(self._widget, 
                                                                           self._experiment))
        #set delegates
        self._widget.tableDataAttributes.setItemDelegateForColumn(TableDataAttributesModel.COLL_LABEL, 
                                                                  RadioButtonDelegate(self._widget.tableDataAttributes))
        
        self._widget.tableDataAttributes.setItemDelegateForColumn(TableDataAttributesModel.COLL_FEATURE_EXTRACTION, 
                                                                  ComboBoxDelegate(self._widget.tableDataAttributes,
                                                                    [n for n in self._experiment.featuresExt]))
        #set resize modes
        self._setSecResModeForDataAttrTable()
        
    def _initCls(self):
        """
        Initialization of classifiers tab.
        """
        
        #register click events
        self._widget.buttonAddClassifierOption.clicked.connect(self.loadDataset)
        
        #add one classifier option
        self._addClassifierOption()
        
    def _addClassifierOption(self):
        """
        Add one new classifier option to UI.
        """
        
        #lets create option row
        
        comBox= QComboBox()
        comBox.addItems([cls.getName() for cls in CLASSIFIERS.values()])
        row = QHBoxLayout();
        row.addWidget(comBox)
        self._widget.testClassifiersLayout.addLayout(row);
        
        
        
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
            
            
    def _setSecResModeForDataAttrTable(self):
        """
        Sets section resize mode for data attribute table.
        """
        
        self._widget.tableDataAttributes.horizontalHeader().setSectionResizeMode(TableDataAttributesModel.COLL_ATTRIBUTE_NAME,QHeaderView.Stretch);
        self._widget.tableDataAttributes.horizontalHeader().setSectionResizeMode(TableDataAttributesModel.COLL_USE,QHeaderView.ResizeMode.ResizeToContents);
        self._widget.tableDataAttributes.horizontalHeader().setSectionResizeMode(TableDataAttributesModel.COLL_PATH,QHeaderView.ResizeMode.ResizeToContents);
        self._widget.tableDataAttributes.horizontalHeader().setSectionResizeMode(TableDataAttributesModel.COLL_LABEL,QHeaderView.ResizeMode.ResizeToContents);
        self._widget.tableDataAttributes.horizontalHeader().setSectionResizeMode(TableDataAttributesModel.COLL_FEATURE_EXTRACTION,QHeaderView.ResizeMode.ResizeToContents);



