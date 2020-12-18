# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 18:25:54 2020

@author: pradeep
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data =pd.read_csv('C:/Users/pradeep/Downloads/predict-iphone-price-main/predict-iphone-price-main/iphone_price.csv') 
head = data.head
print(data)

plt.scatter(data['version'], data['price'])
plt.show()

model = LinearRegression()
model.fit(data[['version']], data[['price']])
print(model.predict([[10]]))
