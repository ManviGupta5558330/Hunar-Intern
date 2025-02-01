import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Generate synthetic weather data
np.random.seed(0)
n_sample = 1000

humidity = np.random.uniform(0, 1000, n_sample)
pressure = np.random.uniform(980, 1050, n_sample)
wind_speed = np.random.uniform(0, 30, n_sample)
temperature = 20 + 0.5 * humidity - 0.02 * pressure + 0.1 * wind_speed + np.random.normal(0, 2, n_sample)

# Create a dataframe from generated data
weather_data = pd.DataFrame({'Humidity': humidity, 'pressure': pressure,
                             'wind_speed': wind_speed, 'temperature': temperature})

# Visualize the data
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.scatter(weather_data['Humidity'], weather_data['temperature'], alpha=0.5)
plt.xlabel('Humidity')
plt.ylabel('Temperature')
plt.title('Humidity vs Temperature')

plt.subplot(2, 2, 2)
plt.scatter(weather_data['pressure'], weather_data['temperature'], alpha=0.5)
plt.xlabel('Pressure')
plt.ylabel('Temperature')
plt.title('Pressure vs Temperature')

plt.subplot(2, 2, 3)
plt.scatter(weather_data['wind_speed'], weather_data['temperature'], alpha=0.5)
plt.xlabel('Wind Speed')
plt.ylabel('Temperature')
plt.title('Wind Speed vs Temperature')

# Split data into feature (x) and target (y)
x = weather_data[['Humidity', 'pressure', 'wind_speed']]
y = weather_data['temperature']

# Split the data into training and testing data set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Make prediction on the test data
y_pred = model.predict(x_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')

# Visualizing the model prediction vs actual value
plt.subplot(2, 2, 4)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.title('Actual vs Predicted Temperature')

# Show the plot
plt.tight_layout()
plt.show()

# Now you can use the trained model to make predictions on new data
new_data = pd.DataFrame({'Humidity': [65], 'pressure': [1005], 'wind_speed': [15]})
prediction = model.predict(new_data)
print(f'Predicted Temperature: {prediction[0]}')



