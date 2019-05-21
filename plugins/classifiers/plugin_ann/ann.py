"""
Created on 24. 4. 2019
Artificial Neural Networks classifier plugin for ClassMark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from classmark.core.plugins import Classifier, PluginAttribute, Plugin, PluginAttributeIntChecker,\
    PluginAttributeFloatChecker
from classmark.core.preprocessing import BaseNormalizer, NormalizerPlugin,\
    MinMaxScalerPlugin,StandardScalerPlugin, RobustScalerPlugin
    
from keras.models import Model
from keras.layers import Dense, Input
from keras import optimizers
from keras.callbacks import Callback
import tensorflow as tf
import numpy as np


USED_ACCURACY="sparse_categorical_accuracy"

class Layer(Plugin):
    
    def __init__(self, neurons:int=1, activation:str='relu'):
        """
        Layer initialization.
        
        :param neurons: Number of neurons in layer.
        :type neurons: int
        :param activation: activation function
        :type activation: str
        """
        
        self.neurons=PluginAttribute("Neurons", PluginAttribute.PluginAttributeType.VALUE, PluginAttributeIntChecker(minV=1))
        self.neurons.value=neurons
        
        self.activation=PluginAttribute("Activation function", PluginAttribute.PluginAttributeType.SELECTABLE, str,
                                        ["relu", "sigmoid", "softmax"])
        self.activation.value=activation
        
    @staticmethod
    def getName():
        return "Layer"
    
    @staticmethod
    def getNameAbbreviation():
        return "L"
 
    @staticmethod
    def getInfo():
        return ""
    

    
class EpochLogger(Callback):
    
    def __init__(self, logger):
        """
        Initialization of logger.
        
        :param logger: Logger that will be used.
        :type logger: Logger
        """
        super().__init__()
        self._logger=logger

    def on_epoch_end(self, epoch, logs={}):
        self._logger.log("Epoch {}\tLoss: {}, Accuracy: {}".format(epoch+1,logs.get('loss'), logs.get(USED_ACCURACY)))


class ANN(Classifier):
    """
    Artificial Neural Networks classifier.
    """

    def __init__(self, normalizer:BaseNormalizer=None, randomSeed:int=None, epochs:int=10, batchSize:int=32, learningRate:float=0.001, 
                 gpu:bool=True, outLactFun:str="softmax", log:bool=True):
        """
        Classifier initialization.
        
        :param normalizer: Normalizer used for input data. If none than normalization/scalling is omitted.
        :type normalizer: None | BaseNormalizer
        :param randomSeed: If not None than fixed seed is used.
        :type randomSeed: int
        :param epochs: Number of training epochs.
        :type epochs: int
        :param batchSize: Number of samples processed before weights are updated.
        :type batchSize: int
        :param learningRate: How big step we do when we learn (in direction of gradient).
        :type learningRate: int
        :param gpu: Should GPU be used?
        :type gpu: bool
        :param outLactFun: Activation function for output layer
        :type outLactFun: str
        :param log: Should log or not?
        :type log: bool
        """
        
        normalizer=MinMaxScalerPlugin(-1,1)
        #object
        self._normalizer=PluginAttribute("Normalize", PluginAttribute.PluginAttributeType.SELECTABLE_PLUGIN, None,
                                         [None, NormalizerPlugin, MinMaxScalerPlugin, StandardScalerPlugin, RobustScalerPlugin])
        self._normalizer.value=normalizer
        
        self._randomSeed=PluginAttribute("Random seed", PluginAttribute.PluginAttributeType.VALUE, PluginAttributeIntChecker(couldBeNone=True))
        self._randomSeed.value=randomSeed
        
        self._epochs=PluginAttribute("Epochs", PluginAttribute.PluginAttributeType.VALUE, PluginAttributeIntChecker(minV=1))
        self._epochs.value=epochs
        
        self._batchSize=PluginAttribute("Batch size", PluginAttribute.PluginAttributeType.VALUE, PluginAttributeIntChecker(minV=1))
        self._batchSize.value=batchSize
        
        self._learningRate=PluginAttribute("Learning rate", PluginAttribute.PluginAttributeType.VALUE, PluginAttributeFloatChecker())
        self._learningRate.value=learningRate

        self._gpu=PluginAttribute("GPU", PluginAttribute.PluginAttributeType.CHECKABLE, bool)
        self._gpu.value=gpu
        
        self._outLactFun=PluginAttribute("Output layer activation function", PluginAttribute.PluginAttributeType.SELECTABLE, str,
                                        ["relu", "sigmoid", "softmax"])
        self._outLactFun.value=outLactFun
        
        self._log=PluginAttribute("Log epoch", PluginAttribute.PluginAttributeType.CHECKABLE, bool)
        self._log.value=log
        
        self._hiddenLayers=PluginAttribute("Hidden layers", PluginAttribute.PluginAttributeType.GROUP_PLUGINS, Layer)
        self._hiddenLayers.groupItemLabel="Hidden layer {}"
        
        
        
        self._CUDA_VISIBLE_DEVICES_CACHED=None  #to remember initial state when GPU is switched off
        
    @staticmethod
    def getName():
        return "Artificial Neural Networks"
    
    @staticmethod
    def getNameAbbreviation():
        return "ANN"
 
    @staticmethod
    def getInfo():
        return ""
    
    def _selectDevice(self):
        """
        Selects gpu/cpu by user priority.
        Ofcourse if GPU option exists.
        """
        
        GPU=tf.test.gpu_device_name()

        if GPU and self._gpu.value:
            #we have gpu and user wants it
            return GPU
        else:
            return '/cpu:0'
    
    def train(self, data, labels):
        
        if self._normalizer.value is not None:
            data=self._normalizer.value.fitTransform(data)
        
        
        
        with tf.device(self._selectDevice()):
        
            if self._randomSeed is not None:
                #fix random seed
                np.random.seed(self._randomSeed.value)
            
            numberOfClasses=np.unique(labels).shape[0]
            
            
            #add input layer
            inputL=outputL=Input(shape=(data.shape[1],))
            
            if self._hiddenLayers.value:
                #we have hidden layers
    
                for i in range(0, len(self._hiddenLayers.value)): #add rest of hidden layer
                    layer=self._hiddenLayers.value[i]
                    #add dense layer
                    outputL=Dense(layer.neurons.value, activation=layer.activation.value)(outputL)
    
            #add output layer
            #must have same number of neurons as the number of classes
            outputL=Dense(numberOfClasses, activation=self._outLactFun.value)(outputL)
                
            #create model
            self._cls=Model(inputs=inputL, outputs=outputL)
    
    
            #compile model
            #sparse_categorical_crossentropy because we will receive labels as integers
            self._cls.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=self._learningRate.value), metrics=[USED_ACCURACY])
            
            
            #and train
            self._cls.fit(data, labels, epochs=self._epochs.value, batch_size=self._batchSize.value, verbose=0, 
                          callbacks=[EpochLogger(self._logger)] if self._log.value else [])
    
    def classify(self, data):
        if self._normalizer.value is not None:
            data=self._normalizer.value.transform(data)
        
        with tf.device(self._selectDevice()):

            return np.argmax(self._cls.predict(data), axis=1)   #argmax because class probabilities is what we are getting.