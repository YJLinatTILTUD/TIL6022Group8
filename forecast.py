import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# Load data
file_path = r"data/Air/OutraEU_Air_quarterly_5coutries.xlsx"
df_extraeu = pd.read_excel(file_path, sheet_name='Sheet 1', skiprows=9, index_col=0)
df_extraeu.dropna(how='any', inplace=True)

# Preprocess data
time_series = df_extraeu.T.loc[:, 'Germany']  # Germany for example
train = time_series[(time_series.index < '2018-Q1')]  # training set
test = time_series[time_series.index >= '2020-Q1']  # test

# ADF Test to Check for model assumption: Stationarity
result = adfuller(train.dropna())  # Ensure no NaN values
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Convert the training data to an array
train_array = np.asarray(train, dtype=float)
test_array = np.asarray(test, dtype=float)

# Define ranges for non-seasonal p and q
non_seasonal_p_values = range(0, 4)  # Adjust as needed
non_seasonal_q_values = range(0, 4)  # Adjust as needed

# Define ranges for seasonal p and q
seasonal_p_values = range(0, 4)  # Adjust as needed
seasonal_q_values = range(0, 4)  # Adjust as needed

# Initialize lists to store results
non_seasonal_results = []
seasonal_results = []

# Loop over combinations of non-seasonal p and q values
for p in non_seasonal_p_values:
    for q in non_seasonal_q_values:
        model_nonseasonal = ARIMA(train_array, order=(p, 1, q))
        result_nonseasonal = model_nonseasonal.fit()
        non_seasonal_results.append((p, q, result_nonseasonal.aic, result_nonseasonal))
        # Loop over combinations of seasonal p and q values
        for sp in seasonal_p_values:
            for sq in seasonal_q_values:
                model_seasonal = ARIMA(train_array, order=(p, 1, q), seasonal_order=(sp, 1, sq, 4))
                result_seasonal = model_seasonal.fit()
                seasonal_results.append((p, q, result_seasonal.aic, sp, sq, result_seasonal))
print("Done")

# Display the training results
non_seasonal_df = pd.DataFrame(non_seasonal_results, columns=['p', 'q', 'AIC', 'Model']).sort_values(by='AIC')
seasonal_df = pd.DataFrame(seasonal_results, columns=['p', 'q', 'AIC', 'sp', 'sq', 'Model']).sort_values(by='AIC')

print("Non-seasonal Model AICs:")
print(non_seasonal_df.loc[:, ['p', 'q', 'AIC']])

print("\nSeasonal Model AICs:")
print(seasonal_df.loc[:, ['p', 'q', 'AIC', 'sp', 'sq']])

# Combine results and find the model with the smallest AIC
all_results = non_seasonal_results + seasonal_results
best_model_info = min(all_results, key=lambda x: x[2])  # Get the model with the smallest AIC
best_model_aic = best_model_info[2]
best_model = best_model_info[-1]
print(f"The best model has p={best_model_info[0]}, q={best_model_info[1]}, sp={best_model_info[3]}, sq={best_model_info[4]} and AIC={best_model_aic}")

# Forecast future values using the best model
test_array = np.insert(test_array, 0, train_array[-1])
forecast_index = np.insert(test.index.values, 0, train.index[-1])  # Set forecast index to align with test data
forecast_steps = len(test)  # Number of steps to forecast, matching the test set length
forecast = best_model.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean  # Forecasted mean values
forecast_mean = np.insert(forecast_mean, 0, train_array[-1])
forecast_ci = pd.DataFrame(forecast.conf_int(), index=forecast_index[0:-1], columns=['lower', 'upper'])  # Confidence intervals

# Plot the forecast
plt.figure(figsize=(12, 6))
plt.plot(train.index, train_array, label='Training Data')
plt.plot(forecast_index, test_array, label='Actual Data', color='green')
plt.plot(forecast_index, forecast_mean, label='Forecast', color='blue')
plt.fill_between(forecast_index[0:-1], forecast_ci['lower'], forecast_ci['upper'], color='blue', alpha=0.2)
plt.xlabel('Time')
plt.ylabel('Volume of Goods Transported')
plt.title('Best ARIMA Model Forecast of Air Transport Volume')
plt.xticks(rotation=90)
plt.legend()
plt.show()

# Calculate yearly difference and cumulative sum
difference = test_array[1:] - forecast_mean[1:]  # Calculate yearly difference excluding first aligned value
difference_df = pd.DataFrame({'Year': forecast_index[1:].year, 'Difference': difference})
yearly_difference = difference_df.groupby('Year')['Difference'].sum()
cumulative_difference = yearly_difference.cumsum()

# Find the first year where cumulative difference is greater than 0 (indicating recovery)
recovery_year = cumulative_difference[cumulative_difference > 0].index.min()

# Display the cumulative difference and recovery information
print("Yearly Difference between Actual and Forecasted Values:")
print(yearly_difference)
print("\nCumulative Difference by Year:")
print(cumulative_difference)

if recovery_year:
    print(f"The transport volume is considered to have recovered by the year {recovery_year}.")
else:
    print("The transport volume has not shown recovery within the forecast period.")

# Plot cumulative difference over years
plt.figure(figsize=(10, 6))
plt.plot(cumulative_difference.index, cumulative_difference.values, marker='o', linestyle='-')
plt.axhline(0, color='red', linestyle='--', label='Recovery Threshold')
plt.xlabel('Year')
plt.ylabel('Cumulative Difference (Actual - Forecast)')
plt.title('Cumulative Difference between Actual and Forecasted Transport Volume')
plt.legend()

plt.show()
