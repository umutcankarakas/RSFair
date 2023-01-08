from math import sqrt
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

df = pd.read_csv('cleaned_train') #i = "sex" aka sensitive feature, 8th index

input_df = df.iloc[:, :-1]
output_df = df.iloc[:,-1:]

X = np.array(input_df)
Y = np.array(output_df)

model = DecisionTreeClassifier()
model.fit(X, Y)
cvs = cross_val_score(model, X, Y, scoring='accuracy')

trial_count = 1000
disc_count = 0
row_count = np.shape(X)[0]
indexes = random.sample(range(row_count), trial_count)

for index in indexes:
  original_prediction = model.predict([X[index,:]])
  row_data = np.copy(X[index,:])
  row_data[8] = np.absolute(row_data[8]-1)
  modified_prediction = model.predict([row_data])
  if(original_prediction != modified_prediction): 
    disc_count += 1

disc_percentage = disc_count/trial_count * 100