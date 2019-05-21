#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
""""
Created on 4. 3. 2019
Naive Bayes classifier plugin for ClassMark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from distutils.core import setup

setup(name='ClassMarkPluginClassifierNBC',
    version='1.0dev',
    description='Naive Bayes classifier plugin for ClassMark.',
    author='Martin Dočekal',
    entry_points={'classmark.plugins.classifiers': 'nbc = nbc:NaiveBayesClassifier'},
    install_requires=[
        'scipy==1.2.0',
        'scikit_learn==0.21.1',
    ]
)