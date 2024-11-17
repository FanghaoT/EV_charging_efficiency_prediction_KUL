import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os


EV = 'EV1'

mode='DC'

filename = f'{EV}_{mode}.csv'

rawdata = pd.read_csv(os.path.join('data_all',filename))
 
# Features and target variable
X = rawdata[['AC power', 'SOCave292','BMSmaxPackTemperature']]
# y = rawdata['DC power']
y = rawdata['eff']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999)


# Create and train the model

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)

                                     
            
# Create and train the model
model = RandomForestRegressor(n_estimators=10, random_state=36)

#Use all data for training
# model.fit(X, y)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
# y_pred = model.predict(X)

# # Evaluation
# mse = mean_squared_error(y, y_pred)
# r2 = r2_score(y, y_pred)

# print(f'Mean Squared Error: {mse}')
# print(f'R² Score: {r2}')

# new_data = pd.DataFrame({
#     'SOC': [80],
#     'AC Power': [10],
#     'Temperature': [32]
# })
# predicted_efficiency = model.predict(new_data)
# print(f'Predicted Efficiency: {predicted_efficiency[0]}')


# # Predictions for the entire dataset
# y_pred_full = model.predict(X)

# # Create a DataFrame with actual and predicted values
# plot_data = rawdata.copy()
# plot_data['Predicted Efficiency'] = y_pred_full

# Creating a DataFrame for inputs, measurements, and predictions
df = pd.DataFrame(X_test)
df['y_meas'] = y_test
df['y_pred'] = y_pred
print(df)

# Calculate errors
df['MAE'] = np.abs(df['y_meas'] - df['y_pred'])
df['MSE'] = (df['y_meas'] - df['y_pred']) ** 2
df['RMSE'] = np.sqrt(df['MSE'])
df['MAPE'] = np.abs((df['y_meas'] - df['y_pred']) / (df['y_meas'] + 1e-8)) * 100
df['R2'] = r2_score(df['y_meas'], df['y_pred'])

# Print mean of each metric
print(f"Mean MAE: {df['MAE'].mean():.10f}")
print(f"Mean MSE: {df['MSE'].mean():.10f}")
print(f"Mean RMSE: {df['RMSE'].mean():.10f}")
print(f"Mean MAPE: {df['MAPE'].mean():.10f}")
print(f"R² Score: {df['R2'].mean():.10f}")

# Save the DataFrame to a CSV file
csv_file_path = f'random_forest_metrics_{EV}_{mode}.csv'
df.to_csv(csv_file_path, index=False)
print(f"Saved full data with measurements and predictions to '{csv_file_path}'")

# Generate scatter plot for actual vs. predicted values
plt.scatter(df['y_meas'], df['y_pred'], color='blue', alpha=0.5)
# plt.title('Actual vs. Predicted Values\nR² Score: {:.4f}'.format(df['R2'][0]))
plt.xlabel('Actual Efficiency [%]')
plt.ylabel('Predicted Efficiency [%]')
plt.grid(True)
plt.axis('equal')
plt.plot([df['y_meas'].min(), df['y_meas'].max()], [df['y_meas'].min(), df['y_meas'].max()], 'k--')  # Diagonal line
plt.savefig(f'random_forest_scatter_{EV}_{mode}.png')
plt.show()