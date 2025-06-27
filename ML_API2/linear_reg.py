#importing data and the required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
warnings.filterwarnings("ignore")

df=pd.read_csv("StudentsPerformance.csv")

#THIS IS WHERE YOU TRAIN THE MODEL
# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV


import warnings
warnings.filterwarnings('ignore')

#SPLITING OF THE X AND Y VARIABLES
X = df.drop(columns="math score",axis=1)
y = df["math score"]


#creating transformers before predicting the model
num_features = X.select_dtypes(exclude="object").columns
cat_features = X.select_dtypes(include="object").columns

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer

numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder()
#creating transformers before predicting the model
num_features = X.select_dtypes(exclude="object").columns
cat_features = X.select_dtypes(include="object").columns

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer

numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder()

preprocessor = ColumnTransformer( [ ("OneHotEncoder", oh_transformer, cat_features), ("StandardScaler", numeric_transformer, num_features), ] )
X = preprocessor.fit_transform(X)

#seperating the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape, X_test.shape

#we are going to create an instance of the linear regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression() # Add parentheses here to create an instance
model.fit(X_train, y_train)
print('Model fitted successfully')

#creating an evaluation function
def evaluate_model(true, predicted):
   print(mae = mean_absolute_error(true, predicted)) 
   print(mse = mean_squared_error(true, predicted))
   print(rmse = np.sqrt(mean_squared_error(true, predicted)))
   print(r2_square = r2_score(true, predicted)) 
joblib.dump(model, 'model.joblib')

