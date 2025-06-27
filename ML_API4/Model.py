#we are going to import the necessary libraries for data analysis and visualization
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

#we are going to read the dataset from the csv file
housing = pd.read_csv('Housing.csv')  

data=pd.read_csv('Housing.csv')
print(data.head())

#we are going to set the indepenndent and the dependent variables
X = data.drop('price', axis=1)  # Independent variables
y = data['price']  # Dependent variable

#we are going to see the data correlations graphically
#sns.pairplot(data)
#plt.show()

# Identify non-numeric columns
non_numeric_cols = X.select_dtypes(exclude=np.number).columns

# Perform one-hot encoding on the non-numeric columns
# drop_first=True to avoid multicollinearity
X = pd.get_dummies(X, columns=non_numeric_cols, drop_first=True)


#we are going to plot the scatter plot of area vs price
#we are going to set the dependent and the independent variables

x=data['area']
y=data['price']
plt.scatter(x, y, color='blue', alpha=0.5)
plt.title('Area vs Price')
plt.xlabel('area')
plt.ylabel('price')
plt.show()


#we are going to plot the heatmap of the data correlations
numeric_data = data.select_dtypes(include=np.number)
sns.heatmap(numeric_data.corr(),annot=True)
plt.show()


#we are going to devide the values into trina nd test size
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
print("the data has been splitted successfully")


#we are going to import the linear regression model from sklearn
from sklearn.linear_model import LinearRegression
#we are now going to train our model using the linear_regression algorithm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
model=LinearRegression()
model.fit(X_train,y_train)

#we are going to predict the values using the trained model 

#we are going to see our model prediction
y_pred=model.predict(X_test)
print(y_pred)

#we are going to check the accuracy of our model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

#we are going to plot the predicted values vs actual values

#we are going to visualize our model
plt.scatter(y_test,y_pred,color='red')
plt.plot(y_test,y_test,color='blue')
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual vs Predicted Charges')
plt.show()

import joblib
joblib.dump(model, 'linear_model.pkl')