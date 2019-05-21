#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
""""
Created on 4. 3. 2019
Feature extractor plugin for ClassMark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from distutils.core import setup

setup(name='ClassMarkPluginFeatureExtractorHOG',
    version='1.0dev',
    description='HOG feature extractor plugin for ClassMark.',
    author='Martin Dočekal',
    entry_points={'classmark.plugins.features_extractors': 'hog = hog:HOG'},
    install_requires=[
        'scikit_image==0.15.0',
        'scipy==1.2.0',
        'skimage==0.0',
    ]

)