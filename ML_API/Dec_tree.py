#implementing the decision tree algorith to train the models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
df=pd.read_csv('kyphosis.csv')
x=df.drop('Kyphosis',axis=1)
y=df['Kyphosis']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.3, random_state=0)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test,y_pred))
joblib.dump(model,'model.joblib')