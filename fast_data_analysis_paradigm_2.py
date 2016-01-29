#using SF Salaries database
#step 1, Import required libraries and read test and train data set. Append both.
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train["Type"] = "Train"
test["Type"] = "Test"
fullData = pd.concat([train, test], axis = 0)

#step 3 (there is no step 2), View the column names / summary of the dataset
# print fullData.columns
# print fullData.head(10)
# print fullData.describe()

#step 4, Identify the a) ID variables b)  Target variables c) Categorical Variables d) Numerical Variables e) Other Variables
ID_col = ["Id"]
target_col = ["TotalPayBenefits"]
cat_cols = ["EmployeeName", "JobTitle", "Notes", "Agency", "Status", "Benefits", "BasePay"]
other_col = ["Type"]
num_cols = list(set(list(fullData.columns)) - set(cat_cols) - set(ID_col) - set(target_col) - set(other_col))

#step 5, Identify the variables with missing values and create a flag for those
num_cat_cols = num_cols + cat_cols
for var in num_cat_cols:
    if fullData[var].isnull().any() == True:
        print var
        fullData[var + "_NA"] = fullData[var].isnull() * 1

#step 6, Impute Missing values
fullData[num_cols] = fullData[num_cols].fillna(value = fullData[num_cols].mean(), inplace = True)
fullData[cat_cols] = fullData[cat_cols].fillna(value = -9999)

#step 7, Create a label encoders for categorical variables and split the data set to train & test, further split the train data set to Train and Validate
for var in cat_cols:
    number = LabelEncoder()
    fullData[var] = number.fit_transform(fullData[var].astype("str"))

train = fullData[fullData["Type"] == "Train"]
test = fullData[fullData["Type"] == "Test"]

train["is_train"] = np.random.uniform(0, 1, len(train)) <= .75
Train, Validate = train[train["is_train"] == True], train[train["is_train"] == False]

print fullData.columns
print fullData.head(10)
print fullData.describe()

# for var in train.columns:
#     if fullData[var].isnull().any() == True:
#         print var

#step 8, Pass the imputed and dummy (missing values flags) variables into the modelling process. I am using random forest to predict the class
features = list(set(list(fullData.columns)) - set(ID_col) - set(target_col) - set(other_col))

x_train = Train[list(features)].values
y_train = Train["TotalPayBenefits"].values
x_validate = Validate[list(features)].values
y_validate = Validate["TotalPayBenefits"].values
x_test = test[list(features)].values

random.seed(100)
rf = RandomForestClassifier(n_estimators = 1000)
rf.fit(x_train,y_train)

#step 9, Check performance and make predictions
status = rf.predict_proba(x_validate)
fpr, tpr, _ = roc_curve(y_validate, status[:, 1])
roc_auc = auc(gpr, tpr)
print roc_auc

# final_status = rf.predict_proba(x_test)
# test["Account.Status"]=final_status[:,1]
# test.to_csv('C:/Users/Analytics Vidhya/Desktop/model_output.csv',columns=['REF_NO','Account.Status'])


















