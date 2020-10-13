################################################################################
#
# File: project5
# Author: Michael Bechtel
# Date: October 12, 2020
# Class: EECS 731
# Description: Use time series models to predict the future demand of a product 
#               given its previous demands from a dataset
# 
################################################################################

# Python imports
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor

# Create sklearn tools used later
encoder = LabelEncoder()
tscv = TimeSeriesSplit(n_splits=5)

# Create Time Series models
linearReg_model = LinearRegression()
gradBoost_model = GradientBoostingClassifier()
neuralNetwork_model = MLPRegressor(max_iter=100000)

# Read the raw dataset
demand_list = pd.read_csv("../data/raw/demand.csv").dropna()

# Find the most commonly ordered product
product = demand_list["Product_Code"].value_counts().keys()[0]
product_list = demand_list[demand_list["Product_Code"]==product]
product_list = product_list.loc[:,("Date","Order_Demand")]

# Separate all dates into separate Year, Month and Day columns
date_lists = product_list["Date"].str.split("/", expand=True)
years = date_lists[0]
months = date_lists[1].str.zfill(2)
days = date_lists[2].str.zfill(2)

# Create columns for the time frames to be tested: Year+Month and Month+Day
year_month = pd.to_numeric(years+months)
month_day = pd.to_numeric(months+days)

# Convert orders column to a numeric type
#   Need to remove parenthesis found in some entries
orders = product_list["Order_Demand"].str.replace(r"\(","")
orders = orders.str.replace(r"\)","")
orders = pd.to_numeric(orders)

# Create a new dataframe the desired featuers and orders
product_dataset = pd.DataFrame({"Year":years,"Month":months,"Day":days,"Year+Month":year_month,"Month+Day":month_day,"Order":orders})
product_dataset.to_csv("../data/processed/product_dataset.csv")

################################################################################
# Timeframe 1: Year+Month
################################################################################
print()
print("########################################")
print("Year+Month => Orders")
print("########################################")

# Get the total order amounts for each Year+Month combination
ym_dataset = product_dataset.groupby("Year+Month", as_index=False).agg({"Order":"sum"}) 

# Separate the summed dataset into features and orders
#   Both are encoded to reduce the numbers sent to the models 
ym_feature = ym_dataset["Year+Month"]
ym_feature = encoder.fit_transform(ym_feature).reshape(-1,1)
ym_order = ym_dataset["Order"]
ym_order = encoder.fit_transform(ym_order)

# Perform Linear Regression
LRresults = cross_val_score(linearReg_model, ym_feature, ym_order.ravel(), cv=tscv, scoring='r2')
print("Linear Regression: {:.2f} ({:.2f})".format(LRresults.mean(), LRresults.std()))

# Perform Gradient Boosting
GBresults = cross_val_score(gradBoost_model, ym_feature, ym_order.ravel(), cv=tscv, scoring='r2')
print("Gradient Boosting: {:.2f} ({:.2f})".format(GBresults.mean(), GBresults.std()))

# Perform Neural Network (MLP)
NNresults = cross_val_score(neuralNetwork_model, ym_feature, ym_order.ravel(), cv=tscv, scoring='r2')
print("Neural Network: {:.2f} ({:.2f})".format(NNresults.mean(), NNresults.std()))

# Print the validation results for each model
x_vals = range(len(LRresults))
_,graphs = plt.subplots(1,3)

graphs[0].plot(x_vals, LRresults)
graphs[0].set_title("Linear Regression")
graphs[0].set_xticks(x_vals)

graphs[1].plot(x_vals, GBresults)
graphs[1].set_title("Gradient Boosting")

graphs[2].plot(x_vals, NNresults)
graphs[2].set_title("Neural Network")

plt.gcf().set_size_inches((16.0,8.0), forward=False)
plt.savefig("../visualizations/timeframe1.png", bbox_inches='tight', dpi=100)


################################################################################
# Timeframe 2: Month+Day
################################################################################
print()
print("########################################")
print("Month+Day => Orders")
print("########################################")

# Get the total order amounts for each Month+Day combination
md_dataset = product_dataset.groupby("Month+Day", as_index=False).agg({"Order":"sum"}) 

# Separate the summed dataset into features and orders
#   Both are encoded to reduce the numbers sent to the models 
md_feature = md_dataset["Month+Day"]
md_feature = encoder.fit_transform(md_feature).reshape(-1,1)
md_order = md_dataset["Order"]
md_order = encoder.fit_transform(md_order)

# Perform Linear Regression
LRresults = cross_val_score(linearReg_model, md_feature, md_order.ravel(), cv=tscv, scoring='r2')
print("Linear Regression: {:.2f} ({:.2f})".format(LRresults.mean(), LRresults.std()))

# Perform Gradient Boosting
GBresults = cross_val_score(gradBoost_model, md_feature, md_order.ravel(), cv=tscv, scoring='r2')
print("Gradient Boosting: {:.2f} ({:.2f})".format(GBresults.mean(), GBresults.std()))

# Perform Neural Network (MLP)
NNresults = cross_val_score(neuralNetwork_model, md_feature, md_order.ravel(), cv=tscv, scoring='r2')
print("Neural Network: {:.2f} ({:.2f})".format(NNresults.mean(), NNresults.std()))

# Print the validation results for each model
x_vals = range(len(LRresults))
_,graphs = plt.subplots(1,3)

graphs[0].plot(x_vals, LRresults)
graphs[0].set_title("Linear Regression")

graphs[1].plot(x_vals, GBresults)
graphs[1].set_title("Gradient Boosting")

graphs[2].plot(x_vals, NNresults)
graphs[2].set_title("Neural Network")

plt.gcf().set_size_inches((16.0,8.0), forward=False)
plt.savefig("../visualizations/timeframe2.png", bbox_inches='tight', dpi=100)