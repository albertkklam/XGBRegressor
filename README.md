# XGBRegressor
A simple implementation to regression problems using Python 2.7, scikit-learn, and XGBoost. Bulk of code from [Complete Guide to Parameter Tuning in XGBoost] (https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)

* [xgbRegressor](../XGBRegressor/xgbRegressor) is a script that preprocesses a data file into the necessary train and test set dataframes for XGBoost. It includes functions to convert categorical variables into dummies or dense vectors, and convert string values into Python compatible strings. There is additional user functionality that allows notification updates to be sent to a user's chosen Slack channel, so that you know when your model has finished training.

Here are some additional resources if you are looking to explore XGBoost and its scikit-learn API more extensively:

1. [Introduction to Boosted Trees and the XGBoost algorithm] (http://xgboost.readthedocs.io/en/latest/model.html)
2. [The Python API documentation for XGBoost] (http://xgboost.readthedocs.io/en/latest/python/python_api.html)
3. [Complete Guide to Parameter Tuning in XGBoost] (https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)
4. [scikit-learn's Gradient Boosting Classifer documentation](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
5. [scikit-learn's GridSearchCV documentation] (http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
