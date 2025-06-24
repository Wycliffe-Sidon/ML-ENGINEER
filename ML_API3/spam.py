
#we are goimg to detect whether an email is spam or not
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import joblib
#We are going to load the dataset to our google colab
try:
  data=pd.read_csv('email.csv')
  print('The file has been loaded successfully!')
  data.head()
 
except:
  print('The file has not been loaded successfully!')
  
  #We are going to assign the variable the target variables and the independent variables
x = data.iloc[:, 1] 
# Select the last column as the target variable (y)
y = data.iloc[:, 0]

# Preprocess the text data using CountVectorizer
# This converts text into a matrix of token counts
vectorizer = CountVectorizer()
x_processed = vectorizer.fit_transform(x)
# Encode the target variable if it is categorical ('spam', 'ham')
# Convert 'ham' to 0 and 'spam' to 1
label_encoder = LabelEncoder()
y_processed = label_encoder.fit_transform(y)

#We are now going to train our dataset 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_processed,y_processed,test_size=0.2,random_state=0)
print(x_train.shape)
print(x_test.shape)
print('modell splitted sucessfully')

#we are now going to train our model using logistic regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
print('model trained sucessfully')

#we are now going to test our model
y_pred=lr.predict(x_test)
print(y_pred)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Spam Detection')
plt.show()

#we are going to save our model in the folder we have just created
joblib.dump(lr, 'lr.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')