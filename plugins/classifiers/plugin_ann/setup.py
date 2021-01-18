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
    version='1.0',
    description='Artificial Neural Networks classifier plugin for ClassMark.',
    author='Martin Dočekal',
    packages=["ann"],
    entry_points={'classmark.plugins.classifiers': 'ann = ann.ann:ANN'},
    extras_require={
        "tf": ['tensorflow>=2.4.0'],
        "tf_gpu": ['tensorflow-gpu>=2.4.0'],
    },
    install_requires=[
        'numpy>=1.16',
        'Keras>=2.2',
    ]
)