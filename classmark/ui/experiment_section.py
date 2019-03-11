"""
Created on 20. 12. 2018
Module for experiment section of the application.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from .widget_manager import WidgetManager, IconName, AttributesWidgetManager
from .section_router import SectionRouter
from PySide2.QtWidgets import QFileDialog, QHeaderView, QPushButton
from PySide2.QtCore import Qt
from ..core.experiment import Experiment, ExperimentRunner
from ..core.plugins import Classifier
from .delegates import RadioButtonDelegate, ComboBoxDelegate
from .models import TableDataAttributesModel
from typing import Callable
from functools import partial
import copy
from enum import Enum


class ClassifierRowWidgetManager(WidgetManager):
    """
    TODO: TRANSFORM TO ORDINARY WIDGET
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
        self._removeCallback=None   #registered remove event callback
        self._changeCallback=None   #registered change event callback
        self._propertyCallback=None   #registered property event callback
        
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
        self._widget.propertiesButton.clicked.connect(self._onProperty)
        self._widget.classifiersComboBox.currentTextChanged.connect(self._onChange)
        

        
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
        
            
    def registerPropertiesEvent(self, callback:Callable):
        """
        Provided callback will be called when user wants to show classifiers properties.
        
        :param callback: Method that will be called. That method must have one 
            parameter where selected classifier will be passed.
        :type callback: Callable[[Classifier],None]
        """
        self._propertyCallback=callback
        
    def registerChangeEvent(self, callback:Callable):
        """
        Provided callback will be called when user changed classifier.
        
        :param callback: Method that will be called. That method must have one 
            parameter where newly selected classifier will be passed.
        :type callback: Callable[[Classifier],None]
        """
        self._changeCallback=callback
        
    def _onChange(self):
        """
        Classifier change.
        """
        
        self.classifierSlot.classifier=self._clsNameToCls[self._widget.classifiersComboBox.currentText()]()
        if self._changeCallback is not None:
            self._changeCallback(self.classifierSlot.classifier)
        
        
    def _onRemove(self):
        """
        Remove event occurred.
        """
        self._experiment.removeClassifierSlot(self.classifierSlot)
        if self._removeCallback is not None:
            self._removeCallback(self)
            
    def _onProperty(self):
        """
        User wants to show properties.
        """
        
        if self._propertyCallback is not None:
            self._propertyCallback(self.classifierSlot.classifier)

        

class ExperimentSection(WidgetManager):
    """
    Experiment section manager class.
    """

    TEMPLATE="experiment"
    """Corresponding template name."""
    
    class ResultPage(Enum):
        """
        Pages in result tab.
        """
        PAGE_NO_RESULTS=0
        PAGE_RUNNING=1
        PAGE_RESULTS=2


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
        
        #experiment start events
        self._widget.startExperimentButton.clicked.connect(self._experimentStarts)
        
    def _experimentStarts(self):
        """
        Starts the experiment.
        """
        #create runner for that experiment
        self._experimentRunner=ExperimentRunner(copy.deepcopy(self._experiment))
        self._experimentRunner.finished.connect(self._experimentFinished)
        self._experimentRunner.numberOfSteps.connect(self._widget.experimentProgressBar.setMaximum)
        self._experimentRunner.step.connect(self._incExperimentProgressBar)
        self._experimentRunner.actInfo.connect(self._widget.experimentActInfo.setText)
        
        #set the progress bar
        self._widget.experimentProgressBar.setValue(0)
        
        #change the page
        self._widget.resultTabPager.setCurrentIndex(self.ResultPage.PAGE_RUNNING.value)
        
        self._experimentRunner.start()
        
    def _incExperimentProgressBar(self):
        """
        Increases progress bar for running experiment.
        """
        self._widget.experimentProgressBar.setValue(self._widget.experimentProgressBar.value()+1)
        
    def _experimentFinished(self):
        #show the data and classifiers tabs
        
        #TODO: CHANGE TO RESULTS
        self._widget.resultTabPager.setCurrentIndex(self.ResultPage.PAGE_NO_RESULTS.value)
        
        
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
                                                                           self._experiment,
                                                                           self._showFeaturesExtractorProperties))
        #set delegates
        self._widget.tableDataAttributes.setItemDelegateForColumn(TableDataAttributesModel.COLL_LABEL, 
                                                                  RadioButtonDelegate(self._widget.tableDataAttributes))
        
        self._widget.tableDataAttributes.setItemDelegateForColumn(TableDataAttributesModel.COLL_FEATURE_EXTRACTION, 
                                                                  ComboBoxDelegate(self._widget.tableDataAttributes,
                                                                    [n for n in self._experiment.featuresExt]))
        self._setPropertiesButtonsToDataAttrTable()
        
        #set resize modes
        self._setSecResModeForDataAttrTable()
        
        #hide the plugin attributes
        self._widget.dataPluginAttributesHeader.hide()
        self._widget.dataAttributesScrollArea.hide()
        
        #init validators
        self._widget.comboBoxValidation.addItems([v.getName() for v in self._experiment.availableEvaluationMethods])
        
        self._widget.comboBoxValidation.currentTextChanged.connect(self._experiment.setEvaluationMethod)
        self._widget.comboBoxValidation.currentTextChanged.connect(self._showEvaluationMethodProperties)
        self._widget.validationPropertiesButton.clicked.connect(self._showEvaluationMethodProperties)
        
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
        
        #hide the plugin attributes header
        self._hideClassifierProperties()
        
    def _hideClassifierProperties(self):
        """
        Hides UI section of classifier properties.
        """
        self._widget.classifierPluginAttributesHeader.hide()
        self._widget.classifierAttributesScrollArea.hide()
        
    def _addClassifierOption(self):
        """
        Add one new classifier option to UI.
        """
        
        #lets create option row manager
        manager=ClassifierRowWidgetManager(self._experiment, self._widget)
        #register events
        manager.registerRemoveEvent(self._removeClassifierOption)
        manager.registerChangeEvent(self._classifierPropertiesEvent)
        manager.registerPropertiesEvent(self._classifierPropertiesEvent)
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

        self._hideClassifierProperties()
        
    def _classifierPropertiesEvent(self, classifier:Classifier):
        """
        Show properties for given classifier.
        
        :param classifier: The classifier which attributes you want to show.
        :type classifier: Classifier
        """
        #TODO: _classifierPropertiesEvent _showFeaturesExtractorProperties is quite the same code
        #    maybe they can share a method
        
        #remove old
        child=self._widget.classifierPluginAttributesContent.takeAt(0)
        while child:
            child.widget().deleteLater()
            child=self._widget.classifierPluginAttributesContent.takeAt(0)

        #set the header
        self._widget.classifierPluginAttributesHeader.show()
        self._widget.classifierAttributesScrollArea.show()
        self._widget.classifierPluginNameShownAttributes.setText(classifier.getName())
        
        
        hasOwnWidget=classifier.getAttributesWidget(self._widget.classifierPluginAttributesWidget)
        
        if hasOwnWidget is not None:
            self._widget.classifierPluginAttributesContent.addWidget(hasOwnWidget)
        else:
            self.manager=AttributesWidgetManager(classifier.getAttributes(), self._widget.classifierPluginAttributesWidget)
            self._widget.classifierPluginAttributesContent.addWidget(self.manager.widget)
        
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
            self._widget.tableDataAttributes.setModel(TableDataAttributesModel(self._widget, 
                                                                               self._experiment, 
                                                                               self._showFeaturesExtractorProperties))
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
        return partial(self._showFeaturesExtractorProperties, row)
        
    def _showFeaturesExtractorProperties(self, row):
        """
        Show properties on row.
        
        :param row: Row number.
        :type row: int
        """
        #remove old
        plugin=self._experiment.getAttributeSetting(self._experiment.dataset.attributes[row], 
                    self._experiment.AttributeSettings.FEATURE_EXTRACTOR)
        
        self._showPropertiesInDataSection(plugin, plugin.getName()+"\n["+self._experiment.dataset.attributes[row]+"]")
        
    def _showEvaluationMethodProperties(self):
        """
        Show properties of actually selected evaluation method.
        """
        self._showPropertiesInDataSection(self._experiment.evaluationMethod, self._experiment.evaluationMethod.getName())
            
    def _showPropertiesInDataSection(self, plugin, name):
        """
        Shows plugins attributes.
        
        :param plugin: The plugin which attributes should be shown.
        :type plugin: Plugin
        :param name: Name that will be shown in header.
        :type name: str
        """
        #remove old
        child=self._widget.dataPluginAttributesContent.takeAt(0)
        while child:
            child.widget().deleteLater()
            child=self._widget.dataPluginAttributesContent.takeAt(0)

        #set the header
        self._widget.dataPluginAttributesHeader.show()
        self._widget.dataAttributesScrollArea.show()
        self._widget.dataPluginNameShownAttributes.setText(name)
        
        
        hasOwnWidget=plugin.getAttributesWidget(self._widget.dataPluginAttributesWidget)
        
        if hasOwnWidget is not None:
            self._widget.dataPluginAttributesContent.addWidget(hasOwnWidget)
        else:
            self.manager=AttributesWidgetManager(plugin.getAttributes(), self._widget.dataPluginAttributesWidget)
            self._widget.dataPluginAttributesContent.addWidget(self.manager.widget)
            
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
        



