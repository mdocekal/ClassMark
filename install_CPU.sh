#!/bin/sh
pip3 install .
pip3 install plugins/classifiers/plugin_svm/
pip3 install plugins/classifiers/plugin_ceef/
pip3 install plugins/classifiers/plugin_knn/
pip3 install plugins/classifiers/plugin_nbc/
pip3 install plugins/classifiers/plugin_dtc/
pip3 install plugins/classifiers/plugin_ann/[tf]
pip3 install plugins/features_extractors/plugin_hashing/
pip3 install plugins/features_extractors/plugin_hog/
pip3 install plugins/features_extractors/plugin_pass/
pip3 install plugins/features_extractors/plugin_tfidf/