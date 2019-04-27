#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
""""
Created on 4. 3. 2019
Artificial Neural Networks classifier plugin for ClassMark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from distutils.core import setup

setup(name='ClassMarkPluginClassifierANN',
    version='1.0dev',
    description='Artificial Neural Networks classifier plugin for ClassMark.',
    author='Martin Dočekal',
    entry_points={'classmark.plugins.classifiers': 'ann = ann:ANN'}
)