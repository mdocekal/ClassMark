#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
""""
Created on 4. 3. 2019
KNN classifier plugin for ClassMark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from distutils.core import setup

setup(name='ClassMarkPluginClassifierKNN',
    version='1.0',
    description='KNN classifier plugin for ClassMark.',
    author='Martin Dočekal',
    packages=["knn"],
    entry_points={'classmark.plugins.classifiers': 'knn = knn.knn:KNN'},
    install_requires=[
        'scikit_learn>=0.20'
    ]
)