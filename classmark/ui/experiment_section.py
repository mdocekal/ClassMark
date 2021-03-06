"""
Created on 20. 12. 2018
Module for experiment section of the application.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from .widget_manager import WidgetManager, IconName, AttributesWidgetManager
from .section_router import SectionRouter
from PySide2.QtWidgets import QFileDialog, QHeaderView, QPushButton, QMessageBox,\
    QWidget, QTableView
from PySide2.QtCore import Qt

from ..core.experiment import Experiment, PluginSlot, ExperimentRunner, ExperimentStatsRunner, ExperimentDataStatistics
from ..core.plugins import Plugin, Classifier, FeatureExtractor
from ..core.selection import FeaturesSelector
from ..core.validation import Validator
from .delegates import RadioButtonDelegate, ComboBoxDelegate
from .models import TableDataAttributesModel, TableClassStatsModel, TableAttributesStatsModel, \
    TableSummarizationResultsModel, TableConfusionMatrixModel, TableResultsModel, saveTableModelAsCsv
from ..core.results import Results
from typing import Callable, Dict
from functools import partial
from enum import Enum
from abc import ABC, abstractmethod

class PluginRowWidgetManager(ABC,WidgetManager):
    """
    Widget manager of row with selection of plugin.
    """
    TEMPLATE="plugin_row"
    """Corresponding template name."""
    
    def __init__(self, slot:PluginSlot, plugins:Dict[str, Plugin], parent=None):
        """
        Creates classifier row widget.

        :param slot: Plugin slot.
        :type slot: PluginSlot
        :param pluginNames: Plugins that could be selected.
            Key is plugin name that should be shown to user and value is
            plugin itself.
        :type pluginNames: Dict[str, Plugin]
        :param parent: Parent widget.
        :type parent: QWidget
        """
        
        super().__init__()
        self._removeCallback=None   #registered remove event callback
        self._changeCallback=None   #registered change event callback
        self._propertyCallback=None   #registered property event callback
        
        self._widget=self._loadTemplate(self.TEMPLATE, parent)

        self.pluginSlot=slot
        
        #let's fill the combo box
        self._widget.comboBox.addItems([p.getName() for p in plugins.values()])
        self._pluginNameToCls=plugins
        
        
        if self.pluginSlot.plugin is not None:
            #set index of actual plugin
            index = self._widget.comboBox.findText(self.pluginSlot.plugin.getName())
            self._widget.comboBox.setCurrentIndex(index)
        else:
            #it's new empty slot
            #Let's call onChage method that create new plugin.
            self._onChange()
        
        #register events
        #Unfortunately it is necessary to make this callbacks partials, because
        #without it invalid methods were called (mismatch between subclasses [do no why]).
        self._widget.removeButton.clicked.connect(partial(self.__class__._onRemove,self))
        self._widget.propertiesButton.clicked.connect(partial(self.__class__._onProperty,self))
        self._widget.comboBox.currentTextChanged.connect(partial(self.__class__._onChange,self))
    
    def __eq__(self, other):
        if not isinstance(other, __class__):
            return False
        return self.pluginSlot==other.pluginSlot #two different managers can not share same slot
    
    def __hash__(self):
        return hash(self.pluginSlot)


    def registerRemoveEvent(self, callback):
        """
        Provided callback will be called when user wants to remove this row.
        
        :param callback: Method that will be called. That method must have one 
            parameter where this manager will be passed.
        :type callback: Callable[[PluginRowWidgetManager],None]
        """
        self._removeCallback=callback
        
    def registerPropertiesEvent(self, callback:Callable):
        """
        Provided callback will be called when user wants to show plugins properties.
        
        :param callback: Method that will be called. That method must have one 
            parameter where selected plugin will be passed.
        :type callback: Callable[[Plugin],None]
        """
        
        self._propertyCallback=callback
        
    def registerChangeEvent(self, callback:Callable):
        """
        Provided callback will be called when user changed plugin.
        
        :param callback: Method that will be called. That method must have one 
            parameter where newly selected plugin will be passed.
        :type callback: Callable[[Plugin],None]
        """
        self._changeCallback=callback
        
    def _onChange(self, *args,**dargs):
        """
        Plugin change.
        """

        self.pluginSlot.plugin=self._pluginNameToCls[self._widget.comboBox.currentText()]()
        if self._changeCallback is not None:
            self._changeCallback(self.pluginSlot.plugin)
        
    @abstractmethod
    def _onRemove(self):
        """
        Remove event occurred.
        """
        pass
            
    def _onProperty(self):
        """
        User wants to show properties.
        """

        if self._propertyCallback is not None:
            self._propertyCallback(self.pluginSlot.plugin)
    
class ClassifierRowWidgetManager(PluginRowWidgetManager):
    """
    Widget manager of row with selection of classifier for testing.
    
    About user choice of classifier this widget informs Experiment object directly, but
    if you want to catch remove row and show properties events, than register your callback
    with appropriate (registerRemoveEvent,registerPropertiesEvent) methods.
    
    """
    

    def __init__(self, experiment:Experiment, slot:PluginSlot=None, parent=None):
        """
        Creates classifier row widget.

        :param experiment: This manager will use that experiment for storing information
            about selection of classifier.
        :type experiment: Experiment
        :param slot: Use given slot. If None than new slot is created.
        :type slot: PluginSlot | None
        :param parent: Parent widget.
        :type parent: QWidget
        """
        self._experiment=experiment
        if slot is None:
            slot=self._experiment.newClassifierSlot()
        super().__init__(slot,
                         {c.getName():c for c in self._experiment.availableClassifiers.values()},
                         parent)

    def _onRemove(self):
        """
        Remove event occurred.
        """
        self._experiment.removeClassifierSlot(self.pluginSlot)
        if self._removeCallback is not None:
            self._removeCallback(self)



class FeaturesSelectorRowWidgetManager(PluginRowWidgetManager):
    """
    Widget manager of row with selection of features selector.
    
    This widget informs about user choice of features selector Experiment object directly, but
    if you want to catch remove row and show properties events, than register your callback
    with appropriate (registerRemoveEvent,registerPropertiesEvent) methods.
    
    """

    def __init__(self, experiment:Experiment, slot:PluginSlot=None, parent=None):
        """
        Creates classifier row widget.

        :param experiment: This manager will use that experiment for storing information
            about selection of classifier.
        :type experiment: Experiment
        :param slot: Use given slot. If None than new slot is created.
        :type slot: PluginSlot | None
        :param parent: Parent widget.
        :type parent: QWidget
        """
        self._experiment=experiment
        if slot is None:
            slot=self._experiment.newFeaturesSelectorSlot()
        super().__init__(slot,
                         {c.getName():c for c in self._experiment.availableFeatureSelectors},
                         parent)

    
    def _onRemove(self):
        """
        Remove event occurred.
        """
        self._experiment.removeFeaturesSelectorSlot(self.pluginSlot)

        if self._removeCallback is not None:
            self._removeCallback(self)



class ExperimentSectionDataStatsTabWatcher(WidgetManager):
    """
    Watches changes in data stats tab and performs action that are needed to update UI.
    """
    
    def __init__(self, widget, experiment:Experiment):
        """
        Initialization of the watcher.
        
        :param widget: UI
        :type widget: QWidget
        :param experiment: The experiment we are working on.
        :type experiment: Experiment
        """
        super().__init__()
        self._widget=widget
        self._experiment=experiment
        #lets observe experiment changes
        self._experiment.registerObserver("NEW_DATA_STATS", self._newStats)
        
        self._actStats=None
        
    def _newStats(self):
        """
        Experiment is using new stats.
        """
        #We need to register new observer.
        if self._actStats is not None:
            #but first unregister the old one
            self._actStats.unregisterObserver("SAMPLES_CHANGED",self._samplesChanged)
            
        if self._experiment.dataStats is not None:
            self._actStats=self._experiment.dataStats
            self._actStats.registerObserver("SAMPLES_CHANGED",self._samplesChanged)
        
        self._allChange()
    
    def _allChange(self):
        """
        Update all.
        """
        #assign models to table views
        self._widget.dataStatsClassTable.setModel(TableClassStatsModel(self._widget, self._experiment))
        self._widget.dataStatsAttributesTable.setModel(TableAttributesStatsModel(self._widget, self._experiment))
        
        self._setSecResModeForDataStatsTables()
        
        self._samplesChanged()
        
    def _samplesChanged(self):
        """
        Update thinks correlated with samples.
        """
        if self._experiment.dataStats is None:
            self._widget.numOfSamplesLabel.setText(self.tr("Loading"))
            self._widget.numOfClassesLabel.setText(self.tr("Loading"))
            self._widget.numOfSelAttrLabel.setText(self.tr("Loading"))
            self._widget.maxSamplesInClassLabel.setText(self.tr("Loading"))
            self._widget.classWithMaxSamplesLabel.setText(self.tr("Loading"))
            self._widget.minSamplesInClassLabel.setText(self.tr("Loading"))
            self._widget.classWithMinSamplesLabel.setText(self.tr("Loading"))
            self._widget.avgNumberOfClassSamplesLabel.setText(self.tr("Loading"))
            self._widget.classSamplesStandardDeviationLabel.setText(self.tr("Loading"))
        else:
            self._widget.numOfSamplesLabel.setText(str(self._experiment.dataStats.numberOfSamples))
            self._widget.numOfClassesLabel.setText(str(len(self._experiment.dataStats.activeClasses)))
            self._widget.numOfSelAttrLabel.setText(str(len(self._experiment.dataStats.attributes)))
            
            maxVal,maxC=self._experiment.dataStats.maxSamplesInClass
            self._widget.maxSamplesInClassLabel.setText(str(maxVal))
            self._widget.classWithMaxSamplesLabel.setText(str(maxC))
            
            minVal,minC=self._experiment.dataStats.minSamplesInClass
            self._widget.minSamplesInClassLabel.setText(str(minVal))
            self._widget.classWithMinSamplesLabel.setText(str(minC))
            
            self._widget.avgNumberOfClassSamplesLabel.setText(str(self._experiment.dataStats.AVGSamplesInClass))
            self._widget.classSamplesStandardDeviationLabel.setText(str(self._experiment.dataStats.SDSamplesInClass))
              
        
    def _setSecResModeForDataStatsTables(self):
        """
        Sets section resize mode for data stats tables.
        """
        self._widget.dataStatsClassTable.horizontalHeader().setSectionResizeMode(TableClassStatsModel.COLL_CLASS_NAME,QHeaderView.Stretch);
        self._widget.dataStatsClassTable.horizontalHeader().setSectionResizeMode(TableClassStatsModel.COLL_SAMPLES,QHeaderView.ResizeMode.ResizeToContents);
        self._widget.dataStatsClassTable.horizontalHeader().setSectionResizeMode(TableClassStatsModel.COLL_SAMPLES_ORIG,QHeaderView.ResizeMode.ResizeToContents);
        self._widget.dataStatsClassTable.horizontalHeader().setSectionResizeMode(TableClassStatsModel.COLL_USE,QHeaderView.ResizeMode.ResizeToContents);
        
        self._widget.dataStatsAttributesTable.horizontalHeader().setSectionResizeMode(TableAttributesStatsModel.COLL_ATTRIBUTE_NAME,QHeaderView.Stretch);
        self._widget.dataStatsAttributesTable.horizontalHeader().setSectionResizeMode(TableAttributesStatsModel.COLL_NUM_OF_FEATURES,QHeaderView.ResizeMode.ResizeToContents);
        self._widget.dataStatsAttributesTable.horizontalHeader().setSectionResizeMode(TableAttributesStatsModel.COLL_FEATURES_SD,QHeaderView.ResizeMode.ResizeToContents);

    
class ResultsPageManager(WidgetManager):
    """
    Manager for results page, but just for showing part.
    """
    
    class ResultShowPage(Enum):
        """
        Pages in result tab.
        """
        PAGE_SUMMARIZATION=0
        PAGE_LOG=1
        PAGE_CONF_MAT=2
        PAGE_RESULT=3
        
    def __init__(self,widget:QWidget,parent:QWidget=None):
        """
        Initializes manager.
        
        :param widget: Results page QWidget.
        :type widget: Qwidget
        :param parent: Parent widget
        :type parent: QWidget
        """
        
        super().__init__()
        self._widget=widget
        self._parent=parent
        
        
        #changing views
        self._widget.resultsSummarizationButton.clicked.connect(partial(self.changePage,self.ResultShowPage.PAGE_SUMMARIZATION))
        self._widget.resultsLogButton.clicked.connect(partial(self.changePage,self.ResultShowPage.PAGE_LOG))
        
        #register save buttons events
        self._widget.summarizationSaveButton.clicked.connect(self._saveSummarizationAsCsv)
        self._widget.saveClassificatorCunfusionMatrixButton.clicked.connect(self._saveConfusionMatrixAsCsv)
        self._widget.saveClassificationResultsButton.clicked.connect(self._saveClassificationResultsAsCsv)

    def changePage(self, page):
        """
        Changes page.
        
        :param page: Page you want to show
        :type page: ResultShowPage
        """
        self._widget.resultsContentStackedWidget.setCurrentIndex(page.value)
        
    def showResults(self, results:Results):
        """
        Show given results.
        
        :param results: Results that should be shown.
        :type results: Results
        """
        self._actShowingResults=results
        self._widget.resultsContentStackedWidget.setCurrentIndex(self.ResultShowPage.PAGE_SUMMARIZATION.value)
        #assign summarization model to table view
        self._widget.resultSummarizationTable.setModel(TableSummarizationResultsModel(self._widget, results))
            

        for i in range(TableSummarizationResultsModel.NUM_COL):
            self._widget.resultSummarizationTable.horizontalHeader().setSectionResizeMode(i,QHeaderView.ResizeMode.ResizeToContents);
        
                
        #assign log
        self._widget.resultLog.setText(results.log)
        
        #add buttons for confusion matrices and results
        
        #remove the old ones
        self.removeChildWidgetFromLayout(self._widget.resultsConfMatLayout)
        self.removeChildWidgetFromLayout(self._widget.resultsClassificationLayout)
        #add new
        for c in results.classifiers:
            #confusion matrix
            button=QPushButton()
            button.setText(c.getName())
            button.clicked.connect(partial(self.showConfusionMatrix,c))
            self._widget.resultsConfMatLayout.addWidget(button)
            
            #results
            button=QPushButton()
            button.setText(c.getName())
            button.clicked.connect(partial(self.showPredResultTable,c))
            self._widget.resultsClassificationLayout.addWidget(button)
            
    def _saveSummarizationAsCsv(self):
        """
        Saves summarization results as csv.
        """
        self._saveTableAsCsv(self._widget.resultSummarizationTable, self.tr("Save summarized results."))
                
    def _saveConfusionMatrixAsCsv(self):
        """
        Saves confusion matrix as csv.
        """
        self._saveTableAsCsv(self._widget.resultConfussionMatrixTable, self.tr("Save confusion matrix."))
    
    def _saveClassificationResultsAsCsv(self):
        """
        Saves classification result as csv.
        """
        self._saveTableAsCsv(self._widget.resultClassificationResultTable, self.tr("Save experiment results."))
    
    def _saveTableAsCsv(self, table:QTableView, title:str):
        """
        Saves given table to csv.
        
        :param table: Table you want to save.
        :type table: QTableView
        :param title: Title for file selection window.
        :type title: str
        """
        
        #get model
        if table.model():
            #let user choose path
            file=QFileDialog.getSaveFileName(self._widget, title, ".csv", self.tr("Any files (*)"))
            file=file[0]
            if file:
                saveTableModelAsCsv(table.model(),file)
            
    def showConfusionMatrix(self, classifier:Classifier):
        """
        Shows confussion matrix for given classifier.
        
        You must first show results with showResults method, else this is just empty operation.
        
        :param classifier: For that classifier confusion matrix will be shown.
        :type classifier: Classifier
        """
        if self._actShowingResults is None:
            #there is nothing there
            return
        
        #assign model to table view
        model=TableConfusionMatrixModel(self._widget, self._actShowingResults, classifier)
        self._widget.resultConfussionMatrixTable.setModel(model)
            

        for i in range(model.columnCount()):
            self._widget.resultConfussionMatrixTable.horizontalHeader().setSectionResizeMode(i,QHeaderView.ResizeMode.ResizeToContents);
            
        #change title
        self._widget.confusionMatrixForClsLabel.setText(classifier.getName())

        self.changePage(self.ResultShowPage.PAGE_CONF_MAT)
            
    def showPredResultTable(self, classifier:Classifier):
        """
        Shows classification results for given classifier.
        
        You must first show results with showResults method, else this is just empty operation.
        
        :param classifier: For that classifier results will be shown.
        :type classifier: Classifier
        """
        if self._actShowingResults is None:
            #there is nothing there
            return
        
        #assign model to table view
        model=TableResultsModel(self._widget, self._actShowingResults, classifier)
        self._widget.resultClassificationResultTable.setModel(model)
 
        for i in range(model.columnCount()):
            self._widget.resultClassificationResultTable.horizontalHeader().setSectionResizeMode(i,QHeaderView.ResizeMode.ResizeToContents);
            
        #change title
        self._widget.resultsForClsLabel.setText(classifier.getName())
        
        self.changePage(self.ResultShowPage.PAGE_RESULT)
            
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
        
    class DataStatsPage(Enum):
        """
        Pages in data stats tab.
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
        self._router=sectionRouter
        self._parent=parent
        self.loadExperiment(load)
        
        self.__dataStatsStartHaveStats=False    #flag that is used when will call background process for data stats
        
    @property
    def experiment(self):
        """
        Actually used experiment.
        
        :return: Experiment
        :rtype: Experiment 
        """
        return self._experiment
 
    def saveExperiment(self, filePath):
        """
        Save experiment to given file.
        
        :param filePath: File where the experiment should be saved.
        :type filePath: str
        """
        self._experiment.save(filePath)
        
         
    def loadExperiment(self, load):
        """
        Loads given experiment.
        
        :param load: Path to file containing experiment configuration.
            None means that new experiment should be loaded.
        :type load: string|None
        :raise ExperimentLoadException: When experiment couldn't be loaded.
        """
        
        #create new or load saved experiment
        loaded=Experiment(load) #could raise exception
        self._experiment=loaded
        
        self._widget=self._loadTemplate(self.TEMPLATE, self._parent)
        
        #register observable for experiment data statistics

        self._experiment.registerObserver("NEW_DATA_STATS",self._dataStatsChanged)
        
        self._initData()
        self._initDataStats()
        self._initFeatSel()
        self._initCls()
        self._initRes()
        
    def _experimentStarts(self):
        """
        Starts the experiment.
        """
        if self._experiment.label is None:
            msgBox=QMessageBox();
            msgBox.setText(self.tr("Select the label first."));
            msgBox.exec();
            return
        
        self._experiment.results=None   #remove old
        #create runner for that experiment
        
        self._experimentRunner=ExperimentRunner(self._experiment)
        self._experimentRunner.finished.connect(self._experimentFinished)
        self._experimentRunner.numberOfSteps.connect(self._widget.experimentProgressBar.setMaximum)
        self._experimentRunner.step.connect(self._incExperimentProgressBar)
        self._experimentRunner.actInfo.connect(self._widget.experimentActInfo.setText)
        self._experimentRunner.error.connect(self._showErrorMessageInBox)
        self._experimentRunner.result.connect(partial(self._experiment.setResults))
        self._experimentRunner.log.connect(self._widget.logView.append)
        
        #clear the act info
        self._widget.experimentActInfo.setText("")
        #clear the log
        self._widget.logView.setText("")
        
        #set the progress bar
        self._widget.experimentProgressBar.setValue(0)
        
        #change the page
        self._widget.resultTabPager.setCurrentIndex(self.ResultPage.PAGE_RUNNING.value)
        
        self._experimentRunner.start()
        
        
    def _experimentStops(self):
        """
        Experiment was stopped by the user.
        """
        self._experimentRunner.requestInterruption()
        
    def _incExperimentProgressBar(self):
        """
        Increases progress bar for running experiment.
        """
        self._widget.experimentProgressBar.setValue(self._widget.experimentProgressBar.value()+1)
        
        
    def _experimentFinished(self):
        """
        Show the data and classifiers tabs
        """
        if self._experiment.results is None:
            self._widget.resultTabPager.setCurrentIndex(self.ResultPage.PAGE_NO_RESULTS.value)
        else:
            self._widget.resultTabPager.setCurrentIndex(self.ResultPage.PAGE_RESULTS.value)

            #save the log
            self._experiment.results.log=self._widget.logView.toPlainText()
            self._resultPageManager.showResults(self._experiment.results)
            self._widget.logView.setText("")    #clear the log
            
        
    def _dataStatsStart(self):
        """
        Starts statistics calculation.
        """
        if self._experiment.label is None:
            msgBox=QMessageBox();
            msgBox.setText(self.tr("Select the label first."));
            msgBox.exec();
            return
        
        
        self.__dataStatsStartHaveStats=False    #mark that we actual get the stats

        #create runner for 
        self._statsRunner=ExperimentStatsRunner(self._experiment)
        self._statsRunner.finished.connect(self._dataStatsFinished)
        self._statsRunner.numberOfSteps.connect(self._widget.dataStatsRunProgressBar.setMaximum)
        self._statsRunner.step.connect(self._incDataStatsProgressBar)
        self._statsRunner.actInfo.connect(self._widget.dataStatsRunActInfo.setText)
        self._statsRunner.error.connect(self._showErrorMessageInBox)
        self._statsRunner.calcStatsResult.connect(self._newDataStatsResults)
        
        #clear the act info
        self._widget.dataStatsRunActInfo.setText("")
        
        #set the progress bar
        self._widget.dataStatsRunProgressBar.setValue(0)
        
        #change the page
        self._widget.dataStatsPager.setCurrentIndex(self.DataStatsPage.PAGE_RUNNING.value)
        
        self._statsRunner.start()
        
    def _dataStatsClear(self):
        """
        Clears data stats.
        """
        self._experiment.setDataStats(None)
        self._widget.dataStatsPager.setCurrentIndex(self.DataStatsPage.PAGE_NO_RESULTS.value)
            
    def _dataStatsStop(self):
        """
        Stops statistics calculation.
        """
        self._statsRunner.requestInterruption()

    def _incDataStatsProgressBar(self):
        """
        Increases progress bar for running experiment.
        """
        self._widget.dataStatsRunProgressBar.setValue(self._widget.dataStatsRunProgressBar.value()+1)
        
    def _newDataStatsResults(self, stats:ExperimentDataStatistics):
        """
        We get new data stats.
        
        :param stats: The new stats.
        :type stats: ExperimentDataStatistics
        """

        self._experiment.setDataStats(stats, True)
        self.__dataStatsStartHaveStats=True
        
    def _dataStatsChanged(self):
        """
        Data stats changed.
        """
        if self.experiment.dataStats is None:
            #in this method we just hiding (when user changes the label)
            #showing results is _dataStatsFinished responsibility
            self._widget.dataStatsPager.setCurrentIndex(self.DataStatsPage.PAGE_NO_RESULTS.value)
        
            
        
        
    def _dataStatsFinished(self):
        """
        Background worker just finished with data stats counting.
        """
        
        #show the data stats tab
        if not self.__dataStatsStartHaveStats or self.experiment.dataStats is None:
            #we couldn't show data stats
            #we do not havy any or data stats finished without result (error)
            self._widget.dataStatsPager.setCurrentIndex(self.DataStatsPage.PAGE_NO_RESULTS.value)
        else:
            self._widget.dataStatsPager.setCurrentIndex(self.DataStatsPage.PAGE_RESULTS.value)
            
        
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
        self._hideDataProperties()
        
        #init validators

        self._widget.comboBoxValidation.addItems([v.getName() for v in self._experiment.availableEvaluationMethods])
        
        #assign loaded data (if exists)
        if self._experiment.dataset is not None:
            self._widget.pathToData.setText(self._experiment.dataset.filePath)
            
        if self._experiment.evaluationMethod is not None:
            index = self._widget.comboBoxValidation.findText(self._experiment.evaluationMethod.getName())
            self._widget.comboBoxValidation.setCurrentIndex(index)
        
        #connect events for validator
        self._widget.comboBoxValidation.currentTextChanged.connect(self._experiment.setEvaluationMethod)
        self._widget.comboBoxValidation.currentTextChanged.connect(self._showEvaluationMethodProperties)
        self._widget.validationPropertiesButton.clicked.connect(self._showEvaluationMethodProperties)
        
    def _hideDataProperties(self):
        """
        Hides UI section of plugin properties that are shown on the right side of data tab..
        """
        self._widget.dataPluginAttributesHeader.hide()
        self._widget.dataAttributesScrollArea.hide()
        
    def _initDataStats(self):
        """
        Initialization of data stats tab.
        """
        #experiment start events
        self._widget.startDataStatisticsCalculationButton.clicked.connect(self._dataStatsStart)
        #experiment stop event
        self._widget.stopDataStatsRunButton.clicked.connect(self._dataStatsStop)
        
        #add watcher for updating the view
        self._dataStatsTabWatcher=ExperimentSectionDataStatsTabWatcher(self._widget, self._experiment)

        self._widget.refreshDataStatisticsButton.clicked.connect(self._dataStatsStart)
        
        self._widget.saveNewDataSetButton.clicked.connect(self._saveAsNewDataset)
        
        
        
    def _initCls(self):
        """
        Initialization of classifiers tab.
        """
        
        #register click events
        self._widget.buttonAddClassifierOption.clicked.connect(partial(self._addClassifierOption,None))
        
        #set alignment to classifier layout
        self._widget.testClassifiersLayout.setAlignment(Qt.AlignTop)
        
        self._classifiersRowsManagers=[]
        
        if len(self._experiment.classifiersSlots)>0:
            for slot in self._experiment.classifiersSlots:
                self._addClassifierOption(slot)
        else:
            #add one classifier option
            self._addClassifierOption(None)
        
        #hide the plugin attributes header
        self._hideClassifierProperties()
        
    def _initFeatSel(self):
        """
        Initialization of features selection tab.
        """
        #register click events
        self._widget.buttonAddFeaturesSelectorOption.clicked.connect(partial(self._addFeaturesSelectorOption,None))
        
        #set alignment to layout
        self._widget.featuresSelectorsLayout.setAlignment(Qt.AlignTop)
        self._featuresSelectorRowsManagers=[]
        
        #hide the plugin attributes header
        self._hideFeaturesSelectorProperties()
        

        for slot in self._experiment.featuresSelectorsSlots:
            self._addFeaturesSelectorOption(slot)
        
    def _hideFeaturesSelectorProperties(self):
        """
        Hides UI section of classifier properties.
        """
        self._widget.featureSelectorAttributesHeader.hide()
        self._widget.featureSelectorAttributesScrollArea.hide()
        
    def _addFeaturesSelectorOption(self, slot:PluginSlot=None):
        """
        Add one new classifier option to UI.
        
        :param slot: Uses provided slot. (optionally)
        :type slot:PluginSlot|None
        """

        #lets create option row manager
        manager=FeaturesSelectorRowWidgetManager(self._experiment, slot, None)
        #register events
        manager.registerRemoveEvent(self._removeFeaturesSelectorOption)
        manager.registerChangeEvent(self._featuresSelectorPropertiesEvent)
        manager.registerPropertiesEvent(self._featuresSelectorPropertiesEvent)
        
        self._featuresSelectorRowsManagers.append(manager)
        self._widget.featuresSelectorsLayout.addWidget(manager.widget)

    def _removeFeaturesSelectorOption(self, manager:FeaturesSelectorRowWidgetManager):
        """
        Remove features selector option from UI.
        
        :param manager: Manger of features selector row.
        :type manager:FeaturesSelectorRowWidgetManager
        """
        self._featuresSelectorRowsManagers.remove(manager)
        manager.widget.deleteLater()

        self._hideFeaturesSelectorProperties()
        
    def _featuresSelectorPropertiesEvent(self, featuresSelector:FeaturesSelector):
        """
        Show properties for given features selector.
        
        :param featuresSelector: The features selector which attributes you want to show.
        :type featuresSelector: FeaturesSelector
        """
        
        self._showPluginAttributes(featuresSelector)
        
    def _initRes(self):
        """
        Initialization of results tab.
        """
        
        self._resultPageManager=ResultsPageManager(self._widget, self._parent)
        #experiment start events
        self._widget.startExperimentButton.clicked.connect(self._experimentStarts)
        self._widget.runNewExperiment.clicked.connect(self._experimentStarts)
        #experiment stop event
        self._widget.stopExperimentButton.clicked.connect(self._experimentStops)
        
        if self.experiment.results is not None:
            #we have loaded results
            #let's show them
            self._widget.resultTabPager.setCurrentIndex(self.ResultPage.PAGE_RESULTS.value)
            self._resultPageManager.showResults(self._experiment.results)
            
    def _hideClassifierProperties(self):
        """
        Hides UI section of classifier properties.
        """
        self._widget.classifierPluginAttributesHeader.hide()
        self._widget.classifierAttributesScrollArea.hide()
        
    def _addClassifierOption(self, slot:PluginSlot=None):
        """
        Add one new classifier option to UI.
        
        :param slot: Uses provided slot. (optionally)
        :type slot:PluginSlot|None
        """
        
        #lets create option row manager
        manager=ClassifierRowWidgetManager(self._experiment, slot, None)
        #register events
        manager.registerRemoveEvent(self._removeClassifierOption)
        manager.registerChangeEvent(self._classifierPropertiesEvent)
        manager.registerPropertiesEvent(self._classifierPropertiesEvent)
        self._classifiersRowsManagers.append(manager)
        self._widget.testClassifiersLayout.addWidget(manager.widget)
        
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
        
        self._showPluginAttributes(classifier)
    
    def _saveAsNewDataset(self):
        """
        Saves subset of current data set to its own file.
        Subset is created from selected attributes and selected samples.
        """
        if self._experiment.label is None:
            msgBox=QMessageBox();
            msgBox.setText(self.tr("Select the label first."));
            msgBox.exec();
            return
            
        filePath=QFileDialog.getSaveFileName(self._widget, self.tr("Save dataset"), ".csv", self.tr("CSV files (*.csv)"))[0]
        if filePath:
            self._experiment.useDataSubset()
            self._experiment.dataset.save(filePath, self._experiment.attributesThatShouldBeUsed())
        
    
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
            
            self._dataStatsClear()
            self._hideDataProperties()
            
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
        
        self._showPluginAttributes(plugin, plugin.getName()+"\n["+self._experiment.dataset.attributes[row]+"]")
        
    def _showEvaluationMethodProperties(self):
        """
        Show properties of actually selected evaluation method.
        """
        self._showPluginAttributes(self._experiment.evaluationMethod)
            

    def _showPluginAttributes(self, plugin:Plugin, name:str=None):
        """
        Shows attributes of given plugin in UI.
        According to Plugin type the appropriate section is used.

        
        :param plugin: Plugin that you want to show.
        :type plugin: Classifier | FeaturesSelector | FeaturesExtractor | Validator
        :param name: Name of plugin that should be shown in header.
            If no name is given than plugin name is determined from Plugin itself.
        :type name: str
        """
        
        if isinstance(plugin, Classifier):
            header=self._widget.classifierPluginAttributesHeader
            subHeader=self._widget.classifierPluginNameShownAttributes
            content=self._widget.classifierPluginAttributesContent
            scrollArea=self._widget.classifierAttributesScrollArea
            attrWidget=self._widget.classifierPluginAttributesWidget
            
        elif isinstance(plugin, FeaturesSelector):
            header=self._widget.featureSelectorAttributesHeader
            subHeader=self._widget.featureSelectorNameShownAttributes
            content=self._widget.featureSelectorPluginAttributesContent
            scrollArea=self._widget.featureSelectorAttributesScrollArea
            attrWidget=self._widget.featureSelectorPluginAttributesWidget
        
        elif isinstance(plugin, FeatureExtractor) or isinstance(plugin, Validator):
            header=self._widget.dataPluginAttributesHeader
            subHeader=self._widget.dataPluginNameShownAttributes
            content=self._widget.dataPluginAttributesContent
            scrollArea=self._widget.dataAttributesScrollArea
            attrWidget=self._widget.dataPluginAttributesWidget
            
        #remove old childs
        self.removeChildWidgetFromLayout(content)
        
        #set the header
        header.show()
        scrollArea.show()
        subHeader.setText(plugin.getName() if name is None else name)
        
        hasOwnWidget=plugin.getAttributesWidget(attrWidget)
        
        #set the content
        if hasOwnWidget is not None:
            content.addWidget(hasOwnWidget)
        else:
            self.manager=AttributesWidgetManager(plugin.getAttributes(), attrWidget)
            content.addWidget(self.manager.widget)
        
    
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
        



