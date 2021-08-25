# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Reading data from remote link
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")

#Preparing the data
X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values  

#train test split
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 

#training the algo
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  

#Fitting model with trainig data
regressor.fit(X_train, y_train) 

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2]]))