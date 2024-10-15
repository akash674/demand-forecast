# Databricks notebook source
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression

# from sklearn.metrics import mean_squared_error, r2_score

# # Load dataset
# df = pd.read_csv("/Volumes/demandforecast987/default/demand_forecasting/Historical Product Demand.csv")

# # Step 2: Preprocess the data
# df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# # Clean 'Order_Demand' column (remove non-numeric and negative values)
# df['Order_Demand'] = pd.to_numeric(df['Order_Demand'], errors='coerce')
# df = df[df['Order_Demand'] > 0]  # Keep only positive values

# # Step 3: Aggregate data by month and year for trend analysis
# df['Year'] = df['Date'].dt.year
# df['Month'] = df['Date'].dt.month

# # Aggregate order demand
# monthly_demand = df.groupby(['Product_Category', 'Warehouse', 'Year', 'Month'])['Order_Demand'].sum().reset_index()

# # Step 4: Prepare the data for the model
# # Create features and labels
# X = monthly_demand[['Year', 'Month']]
# y = monthly_demand['Order_Demand']

# # Optional: Normalize the features (Year and Month) for better model performance
# # X['Year'] = (X['Year'] - X['Year'].mean()) / X['Year'].std()
# # X['Month'] = (X['Month'] - X['Month'].mean()) / X['Month'].std()

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 5: Train a Linear Regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Step 6: Predict and evaluate
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')
# print(f'R-squared: {r2}')

# # Optional: Plot the actual vs predicted values
# plt.scatter(y_test, y_pred)
# plt.xlabel('Actual Demand')
# plt.ylabel('Predicted Demand')
# plt.title('Actual vs Predicted Demand')
# plt.show()





# COMMAND ----------

# # Import necessary libraries for time series forecasting

# import pandas as pd


# # Let's first inspect the uploaded file to understand its structure and data before we proceed with the code for demand forecasting.
# # Load the CSV file
# file_path = '/Volumes/demandforecast987/default/demand_forecasting/Historical Product Demand.csv'
# data = pd.read_csv(file_path)

# # Display the first few rows of the data
# data.head()

# COMMAND ----------


# # Handling non-numeric values in 'Order_Demand' by converting them to NaN and then filling with 0 or appropriate values
# data['Order_Demand'] = pd.to_numeric(data['Order_Demand'].str.replace(r"[()]", "", regex=True), errors='coerce')

# # Fill missing values (if any) with 0 (since these indicate no demand)
# data['Order_Demand'].fillna(0, inplace=True)

# # Now proceed with the previous steps: convert 'Date' to datetime and aggregate data
# data['Date'] = pd.to_datetime(data['Date'])

# # Aggregate the data to get the total demand per day
# daily_demand = data.groupby('Date')['Order_Demand'].sum().reset_index()

# # Set 'Date' as the index for time series analysis
# daily_demand.set_index('Date', inplace=True)

# # Display the cleaned and aggregated data
# daily_demand.head()





# COMMAND ----------

# # Import necessary libraries for time series forecasting
# from statsmodels.tsa.arima.model import ARIMA
# import matplotlib.pyplot as plt

# # Fit an ARIMA model on the aggregated demand data
# # (Using a simple order of (5, 1, 0) for ARIMA, but this can be fine-tuned)
# model = ARIMA(daily_demand, order=(5, 1, 0))
# model_fit = model.fit()

# # Forecast the next 30 days
# forecast = model_fit.forecast(steps=30)

# print(forecast)

# # Plot the historical demand and forecasted demand
# plt.figure(figsize=(10,6))
# plt.plot(daily_demand, label='Historical Demand')
# plt.plot(forecast.index, forecast, label='Forecasted Demand', color='red')
# plt.title('Historical and Forecasted Demand')
# plt.xlabel('Date')
# plt.ylabel('Order Demand')
# plt.legend()
# plt.grid(True)
# plt.show()


# COMMAND ----------

# MAGIC %md ### Demand Forecasting

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample data
file_path = '/Volumes/demandforecast987/default/demand_forecasting/Historical Product Demand.csv'
df = pd.read_csv(file_path)

# Handling non-numeric values in 'Order_Demand' by converting them to NaN and then filling with 0
df['Order_Demand'] = pd.to_numeric(df['Order_Demand'].str.replace(r"[()]", "", regex=True), errors='coerce').fillna(0)

df['Date'] = pd.to_datetime(df['Date'])

# Convert date to ordinal for ML model
df['Date_Ordinal'] = df['Date'].map(pd.Timestamp.toordinal)

# Features (X) and target (y)
X = df[['Date_Ordinal']]
y = df['Order_Demand']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future demand
y_pred = model.predict(X_test)
print(y_pred)

# Display predictions alongside the actual data
df_test = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
display(df_test)
# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(df_test.index, df_test['Actual'], label='Actual', color='blue')
plt.scatter(df_test.index, df_test['Predicted'], label='Predicted', color='red')

plt.title('Actual vs Predicted Order Demand')
plt.xlabel('Index')
plt.ylabel('Order Demand')
plt.legend()
plt.grid(True)
plt.show()



# COMMAND ----------

import joblib

modelpath = '/Volumes/demandforecast987/default/demand_forecasting/demand_forecasting_model'
joblib.dump(model, modelpath)



# COMMAND ----------

# MAGIC %md ### Inventory Optimization

# COMMAND ----------

# from statsmodels.tsa.arima.model import ARIMA

# # Ensure 'Date' column is converted to datetime and set as index
# df['Date'] = pd.to_datetime(df['Date'])
# df.set_index('Date', inplace=True)

# # Fit ARIMA model on the 'Order_Demand'
# model = ARIMA(df['Order_Demand'], order=(5,1,0))  # (p,d,q) parameters can be tuned based on data
# model_fit = model.fit()

# # Forecast demand for the next 3 periods
# forecast = model_fit.forecast(steps=3)

# # Print forecasted values (can be used for inventory optimization)
# print(forecast)

# COMMAND ----------

# MAGIC %md ### Sales Projections

# COMMAND ----------

# from sklearn.ensemble import RandomForestRegressor

# # Create feature matrix X and target vector y
# X = df[['Date_Ordinal']]  # Using date as a feature
# y = df['Order_Demand']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Random Forest Regressor
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)

# # Predict future sales
# y_pred = rf_model.predict(X_test)

# # Print predicted vs actual values
# df_test = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# print(df_test)

