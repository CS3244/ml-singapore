# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 17:00:39 2021

@author: limzi
"""

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

df = pd.read_csv("C:/Users/limzi/Desktop/NUS/AY2021 Sem 2/CS3244/DDOS machine learning project/TrafficLabelling 2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

df = df.drop(df.columns[[0, 1, 2, 3, 4, 5, 6, 62]], axis=1)
df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
le = LabelEncoder()
df[df.columns[-1]] = le.fit_transform(df[df.columns[-1]])

ax = sns.countplot(x = df[df.columns[-1]], data = df)


y = df[df.columns[-1]].values.reshape(-1,1)
X = df[df.columns[:-1]]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
qda = QuadraticDiscriminantAnalysis()

qda.fit(X_train, y_train.ravel())

y_prob_qda = qda.predict_proba(X_test)[:,1]
y_pred_qda = np.where(y_prob_qda > 0.5,1,0)

confusionMatrix = confusion_matrix(y_test, y_pred_qda)
print(confusionMatrix)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob_qda)
roc_auc_qda = auc(false_positive_rate, true_positive_rate)

def plot_roc(roc_auc):
    plt.figure(figsize = (10,10))
    plt.title("Receiver Operating Characteristic")
    plt.plot(false_positive_rate, true_positive_rate, color = "red", label = 'AUC = %0.2f'% roc_auc_qda)
    plt.legend(loc = 'lower right')
    plt.plot([0,1], [0,1], linestyle= "--")
    plt.axis('tight')
    plt.ylabel("True Postive Rate")
    plt.xlabel("False Positive Rate")

plot_roc(roc_auc_qda)
