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
    version='1.0',
    description='ClassMark is a benchmark for classifiers.',
    author='Martin Dočekal',
    packages=['classmark'],
    package_data={'classmark': ['ui/icons/*','ui/languages/*','ui/templates/*','core/data/*']},
    data_files=[('examples', ['examples/test.csv'])],
    entry_points={
        'gui_scripts': [
            'classmark = classmark.__main__:main'
        ]
    },
    install_requires=[
        'pandas>=0.23.1',
        'scikit_image>=0.15.0',
        'setuptools>=39.0.1',
        'scipy>=1.2.0',
        'numpy>=1.16.3',
        'typing>=3.6.6',
        'PySide2>=5.12.3',
        'skimage>=0.0',
        'scikit_learn>=0.21.1'
    ]
)
