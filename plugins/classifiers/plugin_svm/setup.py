#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
""""
Created on 4. 3. 2019
SVM classifier plugin for ClassMark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from distutils.core import setup

setup(name='ClassMarkPluginClassifierSVM',
    version='1.0',
    description='SVM classifier plugin for ClassMark.',
    author='Martin Dočekal',
    packages=["svm"],
    entry_points={'classmark.plugins.classifiers': 'svm = svm.svm:SVM'},
    install_requires=[
        'scikit_learn>=0.20'
    ]
)