# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 01:43:35 2019

@author: SURAJ BHADHORIYA
"""
#load libraies
import numpy as np
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.cross_validation import train_test_split
import pandas as pd

#read data
df=pd.read_csv("C:/Users/SURAJ BHADHORIYA/Desktop/loan_dataset.csv")
#print the names of col.
col=df.columns
print(col)
#make label
df['safe_loans']=df['bad_loans'].apply(lambda s:+1 if s==0 else -1)
print(df['safe_loans'])
#find the +ive & -ive % of loan
pos_loan=len(df[df['safe_loans']==1])
neg_loan=len(df[df['safe_loans']==-1])
pos=(pos_loan*100)/122607
neg=(neg_loan*100)/122607
print("positive loan %",pos)
print("negative loan %",neg)
#put all feature together
feature=['grade','sub_grade','short_emp','emp_length_num','home_ownership','dti','purpose',
         'term','last_delinq_none','last_major_derog_none','revol_util','total_rec_late_fee']
label=['safe_loans']
#make new dataframe where only feature and label append
loan=df[feature+label]

#make one hot encoding on dataframe
loan1=pd.get_dummies(loan)

#make feature one hot encoading
x=pd.get_dummies(loan[feature])
#make label
y=loan['safe_loans']
#split dataset
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#apply DecisionTreeClassifier model
clf=DecisionTreeClassifier(max_depth=2)
clf.fit(X_train,y_train)
#accuracy
accuracy=clf.score(X_test,y_test)
print("accuracy =",accuracy)

acc=clf.score(X_train,y_train)
print("accuracy =",acc)

feature1=x.columns
print(feature1)
label1=['faulty','not_faulty']

#making decision tree
from IPython.display import Image
from sklearn.externals.six import StringIO
import pydotplus
dot_data=StringIO()
export_graphviz(clf,
                     out_file=dot_data,
                     feature_names=feature1,
                     class_names=label1,
                     filled=True)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())





