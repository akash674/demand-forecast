# Databricks notebook source
import numpy as np
import pandas as pd

# Data Visualization
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.express as px
#import bokeh

# For Analysis and Forecasting
from scipy import stats

# Others
import datetime
import os
import pickle
import requests

# COMMAND ----------

# Loading the Data
df = pd.read_csv(r"/Volumes/demandforecast987/default/demand_forecasting/Historical Product Demand.csv")

df.head(5)

# COMMAND ----------

# finding the percentage of missing value
print("Number of attributes with null vaules: ", df.isnull().any().sum())
print("Percentage of missing values: ",df.isnull().any(axis=1).sum()/len(df)*100)

# COMMAND ----------

# Dropping the missing values
df.dropna(axis=0, how="any", inplace=True)

# COMMAND ----------

#Changing the datatype to datetime
df["Date"] = pd.to_datetime(df['Date'])
df["Order_Demand"] = df["Order_Demand"].str.replace("(", "")
df["Order_Demand"] = df["Order_Demand"].str.replace(")", "")
#Changing the datatype to float
df["Order_Demand"] = df["Order_Demand"].astype(float)

# COMMAND ----------

df = df.sort_values(by=['Date', 'Product_Code'])
df = df.set_index('Date')
df.head()

# COMMAND ----------

category_yearly_demand = df.groupby([df.index.year, 'Product_Category'])['Order_Demand'].mean()

# COMMAND ----------

padded_category_data = {}
for category, category_data in category_yearly_demand.groupby(level = 'Product_Category'):
#     print(f"Category: {category}")
    padded_category_data[category] = [0 for _ in range(7)]
    for year, total_demand in category_data.items():
        index = ((year[0] - 2010) % 7) - 1
        padded_category_data[category][index] = total_demand

fig = plt.figure(figsize=(12, 25))
rows, cols = 11, 3
x = [2011, 2012, 2013, 2014, 2015, 2016, 2017]

for title, data in padded_category_data.items():
    # Create subplots in the grid
    ax = fig.add_subplot(rows, cols, int(title[-2:]))
    # Plotting data on the current subplot
    ax.plot(x, data)
    ax.set_title(title)
plt.tight_layout()
fig.suptitle("Yearly Average Demand for all the Product Categories", y=1.02)

# COMMAND ----------

padded_yearly_categories = {}
for year, year_data in category_yearly_demand.groupby(level = 'Date'):
#     print(f"Category: {category}")
    padded_yearly_categories[year] = [0 for _ in range(33)]
    for category, total_demand in year_data.items():
#         print(category)
        index = (int(category[1][-2:]) % 33) - 1
        padded_yearly_categories[year][index] = total_demand

x = [i+1 for i in range(33)]
rows = len(padded_yearly_categories)
cols = 1

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 20))

colors = plt.cm.viridis(np.linspace(0, 1, len(x)))
for i, (year, data) in enumerate(padded_yearly_categories.items()):
    # Calculate the row and column indices for the subplot
    
    # Create a bar plot in the current subplot
    bars = axes[i].bar(x, data, color=colors)
    axes[i].bar_label(bars, labels=x, fontsize = 8)
    
    # Set the category title as the subplot title
    axes[i].set_title(year)
    
    # Hide only the y-axis scales (ticks)
#     axes[row_idx, col_idx].get_yaxis().set_visible(False)

plt.tight_layout()
fig.suptitle("Yearwise Average Demand of all Product Categories", y=1.01)
plt.show()

# COMMAND ----------

warehouse_yearly_demand = df.groupby([df.index.year, 'Warehouse'])['Order_Demand'].mean()

# COMMAND ----------

demand_data = {}
warehouses = []
years = [2011 + i for i in range(7)]
for warehouse, warehouse_data in warehouse_yearly_demand.groupby(level='Warehouse'):
    warehouses.append(warehouse)
    demand_data[warehouse] = [0 for i in range(7)]
    for year, year_data in warehouse_data.items():
        index = ((year[0] - 2010) % 7) - 1
        demand_data[warehouse][index] = year_data
# print(demand_data)

# Determine the number of warehouses and the number of years
num_warehouses = len(warehouses)
num_years = len(years)

# Set the width of the bars
bar_width = 0.15
# Create a figure
fig, ax = plt.subplots(figsize=(12, 8))

# Define the index for the x-axis
x = np.arange(num_years)

# Create a grouped bar chart
for i, warehouse in enumerate(warehouses):
    # Offset the x-position for each warehouse
    x_pos = x + i * bar_width
    
    # Plot the demand values for the current warehouse
    ax.bar(x_pos, demand_data[warehouse], width=bar_width, label=warehouse)

# Set x-axis labels and tick positions
ax.set_xticks(x + (num_warehouses - 1) * bar_width / 2)
ax.set_xticklabels(years)
# Set labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Demand')
ax.set_title('Yearly Average Demand by Warehouse')

# Add a legend to distinguish the warehouses
ax.legend()

# Show the chart
plt.show()
