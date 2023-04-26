# Diabetes-Prediction
Importing the Dependencies


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Data Collection and Analysis

# Diabetic Dataset


diabetes_dataset=pd.read_csv('/content/diabetes.csv') # copy path of csv file

# printing first 5 rows of dataset
diabetes_dataset.head()

# number of rows and columns in the dataset
diabetes_dataset.shape

diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()

# 0 --> Non-diabetic

# 1 --> Diabetic


diabetes_dataset.groupby('Outcome').mean()

# Separating the Data and Labels
X=diabetes_dataset.drop(columns='Outcome', axis=1)
Y=diabetes_dataset['Outcome']

print(X)
print(Y)

Data Standardization

scalar = StandardScaler()

scalar.fit(X)

standardized_data=scalar.transform(X)

print(standardized_data)

X = standardized_data
Y=diabetes_dataset['Outcome']

print(X,Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

Training The Model

classifier = svm.SVC(kernel = 'linear')

classifier.fit(X_train, Y_train)

# Model Evaluation

# Accuracy Score

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print("Accuracy score of training data: ",training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)

print("Accuracy score of test data: ",test_data_accuracy)

# Making a Predictive System 

input_data= (10,168,74,0,0,38,0.537,34)
# Changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# standardize the input data
standard_data = scalar.transform(input_data_reshaped)
print(standard_data)
prediction= classifier.predict(standard_data)
print(prediction)
if (prediction[0] == 0):
  print("The person is not diabetic")
else:
  print("The person is diabetic")
