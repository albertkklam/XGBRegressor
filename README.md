# XGBRegressor

## Overview
A simple implementation to regression problems using Python 2.7, scikit-learn, and XGBoost. Bulk of code from [Complete Guide to Parameter Tuning in XGBoost](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)

[xgbRegressor](../XGBRegressor/xgbRegressor) is a general purpose script for model training using XGBoost. It contains:

* Functions to preprocess a data file into the necessary train and test set dataframes for XGBoost
* Functions to convert categorical variables into dummies or dense vectors, and convert string values into Python compatible strings
* Additional user functionality that allows notification updates to be sent to a user's chosen Slack channel, so that you know when your model has finished training
* Implementation of sequential hyperparameter grid search via the scikit-learn API
* Implementation of early stopping via the Learning API

## Installing XGBoost for Python
Follow instructions [here](https://github.com/dmlc/xgboost/tree/master/python-package)

## Resources

Here are some additional resources if you are looking to explore XGBoost and its various APIs more extensively:

1. [Introduction to Boosted Trees and the XGBoost algorithm](http://xgboost.readthedocs.io/en/latest/model.html)
2. [The Python API documentation for XGBoost](http://xgboost.readthedocs.io/en/latest/python/python_api.html)
3. [Complete Guide to Parameter Tuning in XGBoost](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)
4. [scikit-learn's Gradient Boosting Classifer documentation](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
5. [scikit-learn's GridSearchCV documentation](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
6. [Tong He's XGBoost presentation](https://www.slideshare.net/ShangxuanZhang/xgboost)
