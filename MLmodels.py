import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import scikitplot as skplt
import seaborn as sns
df=pd.read_csv('/Users/Yeshwanth/Desktop/v.csv');
col=['Speed','Blink Rate']
X=df[col]
y=df['Accident']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf= LogisticRegression()
clf1= RandomForestClassifier(max_depth=2,random_state=0)
clf2=KNeighborsClassifier(n_neighbors=3)
clf3 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(6,2), random_state=1)
clf.fit(X_train,y_train)
clf1.fit(X_train,y_train)
clf2.fit(X_train,y_train)
clf3.fit(X_train,y_train)
pred=clf.predict(X_test)
pred1=clf1.predict(X_test)
pred2=clf2.predict(X_test)
pred3=clf3.predict(X_test)
print("The accuracy using logistic regression is : "+str(accuracy_score(y_test, pred)))
print()
print()
print("The F-1 using logistic regression is : "+str(f1_score(y_test, pred,average='weighted')))
print()
print("----------------------------------------------------------------------------")
print("----------------------------------------------------------------------------")
print()
print("The accuracy using Random Forests is : "+str(accuracy_score(y_test, pred1)))
print()
print()
print("The F-1 using Random Forests is : "+str(f1_score(y_test, pred1,average='weighted')))
print()
print("----------------------------------------------------------------------------")
print("----------------------------------------------------------------------------")
print()
print("The accuracy using K-Nearest Neighbours is : "+str(accuracy_score(y_test, pred2)))
print()
print()
print("The F-1 using K-Nearest Neighbours is : "+str(f1_score(y_test, pred2,average='weighted')))
print()
print("----------------------------------------------------------------------------")
print("----------------------------------------------------------------------------")
print()
print("The accuracy using Neural network is : "+str(accuracy_score(y_test, pred3)))
print()
print()
print("The F-1 using Neural network is : "+str(f1_score(y_test, pred3,average='weighted')))
print()
print("----------------------------------------------------------------------------")
print("----------------------------------------------------------------------------")
print()
ax = sns.regplot(x="Speed", y="Accident", data=df)