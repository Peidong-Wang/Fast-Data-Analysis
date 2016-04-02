## Fast-Data-Analysis

This is a workspace building fast data analysis toolkits.

The first paradigm is a **RandomForestClassifier** for an artifical dataset quoted from Andrew Cross.[random-forests-in-python-with-scikit-learn](http://www.agcross.com/2015/02/random-forests-in-python-with-scikit-learn/)

The second paradigm is for a general data analysis model for real-world datasets. Its main framework is from Sunil Ray.[Build a Predictive Model in 10 Minutes (using Python)](http://www.analyticsvidhya.com/blog/2015/09/build-predictive-model-10-minutes-python/) Some revisions are made based on Sunil's framework.

The intended datasets used in paradigm 2 are train.csv and test.csv, both from the SF Salaries dataset on Kaggle. The original dataset was divided into train.csv and test.csv from the 105000th data sample.

In the latest version of paradigm 2, the datasets are train_r.csv and test_r.csv. Both of these two datasets contains only 200 data samples. In another word, train_r.csv contains data sample from 1st to 199th, test_r.csv from 200th to 399th.

>The np.asarray( ) function in paradigm 2 is memory consuming, leading to the use of datasets train_r.csv and test_r.csv. 

In conclusion, paradigm 1 and 2 are both runnable now, althoough paradigm 2 may be memory consuming while running. Further modification to paradigm 2, especially the np.asarray( ) part, will be done if time permits.
