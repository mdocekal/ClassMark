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
    version='1.0dev',
    description='ClassMark is a benchmark for classifiers.',
    author='Martin Dočekal',
    packages=['classmark'],
    package_data={'classmark': ['ui/icons/*','languages/*','templates/*']},
    data_files=[('examples', ['examples/test.csv'])],
    entry_points={
        'gui_scripts': [
            'classmark = classmark.__main__:main'
        ]
    },
    install_requires=[
        'PySide2>=5',
        'typing>=3'
    ]
)
