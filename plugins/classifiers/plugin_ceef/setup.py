#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
""""
Created on 4. 3. 2019
SVM classifier plugin for ClassMark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from distutils.core import setup

setup(name='ClassMarkPluginClassifierCEEF',
    version='1.0dev',
    description='CEEF classifier plugin for ClassMark.',
    author='Martin Dočekal',
    entry_points={'classmark.plugins.classifiers': 'ceef = ceef.ceef:CEEF'},
    install_requires=[
        'scipy==1.2.0',
        'typing==3.6.6',
        'numpy==1.16.3',
        'scikit_learn==0.21.1',
    ]
)