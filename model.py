from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
import joblib
warnings.filterwarnings('ignore')

df = pd.read_csv('Crop_recommendation.csv')

# df.head()

# df.tail()

# df.size

# df.shape

# df.columns

# df['label'].unique()

# df.dtypes

# df['label'].value_counts()

# sns.heatmap(df.corr(),annot=True)

features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'soil_moisture']]
target = df['label']
labels = df['label']

# Initializing empty lists to append all model's name and corresponding name
acc = []
model = []

# Splitting into train and test data

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)

# from sklearn.tree import DecisionTreeClassifier

# DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)

# DecisionTree.fit(Xtrain,Ytrain)

# predicted_values = DecisionTree.predict(Xtest)
# x = metrics.accuracy_score(Ytest, predicted_values)
# acc.append(x)
# model.append('Decision Tree')
# print("DecisionTrees's Accuracy is: ", x*100)

# print(classification_report(Ytest,predicted_values))

from sklearn.model_selection import cross_val_score

# Cross validation score (Decision Tree)
# score = cross_val_score(DecisionTree, features, target,cv=5)

# score

# from sklearn.naive_bayes import GaussianNB

# NaiveBayes = GaussianNB()

# NaiveBayes.fit(Xtrain,Ytrain)

# predicted_values = NaiveBayes.predict(Xtest)
# x = metrics.accuracy_score(Ytest, predicted_values)
# acc.append(x)
# model.append('Naive Bayes')
# print("Naive Bayes's Accuracy is: ", x)

# print(classification_report(Ytest,predicted_values))

# score = cross_val_score(NaiveBayes,features,target,cv=5)
# score

# from sklearn.svm import SVC

# SVM = SVC(gamma='auto')

# SVM.fit(Xtrain,Ytrain)

# predicted_values = SVM.predict(Xtest)

# x = metrics.accuracy_score(Ytest, predicted_values)
# acc.append(x)
# model.append('SVM')
# print("SVM's Accuracy is: ", x)

# print(classification_report(Ytest,predicted_values))

# # Cross validation score (SVM)
# score = cross_val_score(SVM,features,target,cv=5)
# score

# from sklearn.linear_model import LogisticRegression

# LogReg = LogisticRegression(random_state=2)

# LogReg.fit(Xtrain,Ytrain)

# predicted_values = LogReg.predict(Xtest)

# x = metrics.accuracy_score(Ytest, predicted_values)
# acc.append(x)
# model.append('Logistic Regression')
# print("Logistic Regression's Accuracy is: ", x)

# print(classification_report(Ytest,predicted_values))

# # Cross validation score (Logistic Regression)
# score = cross_val_score(LogReg,features,target,cv=5)
# score

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain,Ytrain)

predicted_values = RF.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('RF')
print("RF's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))

# Cross validation score (Random Forest)
score = cross_val_score(RF,features,target,cv=5)
score

# import xgboost as xgb
# XB = xgb.XGBClassifier()
# XB.fit(Xtrain,Ytrain)

# predicted_values = XB.predict(Xtest)

# x = metrics.accuracy_score(Ytest, predicted_values)
# acc.append(x)
# model.append('XGBoost')
# print("XGBoost's Accuracy is: ", x)

# print(classification_report(Ytest,predicted_values))

# # Cross validation score (XGBoost)
# score = cross_val_score(XB,features,target,cv=5)
# score

# plt.figure(figsize=[10,5],dpi = 100)
# plt.title('Accuracy Comparison')
# plt.xlabel('Accuracy')
# plt.ylabel('Algorithm')
# sns.barplot(x = acc,y = model,palette='dark')

# accuracy_models = dict(zip(model, acc))
# for k, v in accuracy_models.items():
#     print (k, '-->', v)

# data = np.array([[90, 42, 43, 20.8, 82.0, 6.5, 202.9]])
# prediction = RF.predict(data)
# print(prediction)

filename = 'finalized_model.sav'
joblib.dump(RF, filename)
print("Model Dumped")