#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
""""
Created on 28. 2. 2019
ClassMark is a classifier benchmark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from distutils.core import setup

setup(name='ClassMark',
    version='1.0.0',
    description='ClassMark is a benchmark for classifiers.',
    author='Martin Dočekal',
    packages=['classmark', 'classmark.core', 'classmark.ui', 'classmark.data'],
    package_data={'classmark': ['ui/icons/*','ui/languages/*','ui/templates/*','core/data/*']},
    data_files=[('examples', ['examples/test.csv'])],
    entry_points={
        'gui_scripts': [
            'classmark = classmark.__main__:main'
        ]
    },
    install_requires=[
        'pandas>=0.23',
        'scikit_image>=0.15',
        'setuptools>=39.0',
        'scipy>=1.2',
        'numpy>=1.16',
        'typing>=3.6',
        'PySide2>=5.12',
        'scikit_learn>=0.20'
    ]
)
