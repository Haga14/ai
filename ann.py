# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Deep_Learning_A_Z/Churn_Modelling.csv')
x_Test_New = pd.read_csv('Deep_Learning_A_Z/Churn_ModellingTest.csv')

x_Test_New = x_Test_New.iloc[:, 3:13].values

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer , make_column_transformer

preprocessor = make_column_transformer((OneHotEncoder(categories = 'auto'), [1,2]), remainder ='passthrough')
X = preprocessor.fit_transform(X)
X = X[: , 1:]
X = np.delete(X,2,1)


x_Test_New = preprocessor.fit_transform(x_Test_New)
x_Test_New = x_Test_New[:, 1:]
x_Test_New = np.delete(x_Test_New,2,1)
##X = X[:, 1:]
# Encoding Y data
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
##X_test = sc.transform(X_test)
x_Test_New = sc.transform(x_Test_New)

import keras

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the out layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid', ))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Ftting thet ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs =  100)

# Fitting classifier to the Training set
# Create classifier here


# Predicting the Test set results
y_pred = classifier.predict(x_Test_New)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)