#!/bin/sh
pip3 install --editable .
pip3 install --editable plugins/classifiers/plugin_svm/
pip3 install --editable plugins/classifiers/plugin_ceef/
pip3 install --editable plugins/features_extractors/plugin_hashing/
pip3 install --editable plugins/features_extractors/plugin_hog/
pip3 install --editable plugins/features_extractors/plugin_pass/
pip3 install --editable plugins/features_extractors/plugin_tfidf/