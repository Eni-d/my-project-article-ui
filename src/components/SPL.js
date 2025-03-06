import React from "react";

export default function SPL() {
  return (
    <div className="min-h-screen bg-gray-100 p-6 flex flex-col items-center">
        <h1 className="text-4xl font-bold text-gray-900 mb-6 text-center">
          Predicting Solar Panel Lifespan Using Decision Tree Regression
        </h1>
      <div className="lg:w-auto w-96 bg-white shadow-lg rounded-2xl p-8">
        <p className="text-lg text-gray-700 leading-relaxed mb-4">
          Solar energy is one of the most sustainable sources of power. However, the efficiency of solar panels degrades over time, affecting their lifespan. This article explores the use of Decision Tree Regression to predict the lifespan of solar panels based on solar radiation data.
        </p>
        
        <h2 className="text-2xl font-semibold mt-6 mb-2">What is a Decision Tree?</h2>
        <p className="text-lg text-gray-700 leading-relaxed">
          A Decision Tree is a supervised learning algorithm used for classification and regression tasks. It splits the data into subsets based on feature values, forming a tree-like structure where each node represents a decision rule, and the branches lead to possible outcomes.
        </p>

        <h2 className="text-2xl font-semibold mt-6 mb-2">How Decision Trees Work:</h2>
        <ul className="list-disc pl-6 text-lg text-gray-700">
          <li><strong>Root Node</strong>: The starting point of the tree, containing the entire dataset.</li>
          <li><strong>Splitting</strong>: The dataset is divided based on a decision rule.</li>
          <li><strong>Decision Nodes</strong>: Intermediate nodes that further split the data.</li>
          <li><strong>Leaf Nodes</strong>: The final nodes that provide the predicted output.</li>
        </ul>

        <h2 className="text-2xl font-semibold mt-6 mb-2">Step 1: Importing Required Libraries</h2>
      <pre className="bg-gray-900 text-white p-4 rounded-lg overflow-x-auto text-sm">
        import pandas as pd{"\n"}
        from sklearn.model_selection import train_test_split{"\n"}
        from sklearn.preprocessing import StandardScaler{"\n"}
        from sklearn.tree import DecisionTreeRegressor{"\n"}
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score{"\n"}
        import numpy as np
      </pre>
      <p className="text-lg mt-2 mb-5">These libraries help with data manipulation, machine learning, and performance evaluation.</p>

      <h1 className="text-2xl font-semibold mb-4">Solar Panel Degradation Assumption</h1>
      <p className="mb-4">
        Solar panels naturally degrade over time due to environmental factors such as UV exposure, temperature fluctuations, and mechanical wear.
        To estimate their lifespan, we assume a <strong>degradation rate of 0.7% per year</strong> and an <strong>initial efficiency of 100%</strong>.
      </p>
      <h2 className="text-xl font-semibold mb-2">How Degradation Works</h2>
      <p className="mb-4">
        Each year, the panel loses <strong>0.7% of its current efficiency</strong>, meaning the decline is <strong>exponential, not linear</strong>. The efficiency after each year is calculated as:
      </p>
      <pre className="bg-gray-900 text-white p-4 rounded-md overflow-x-auto">
        Efficiency_new = Efficiency_current - (Efficiency_current * Degradation Rate)
      </pre>
      <h2 className="text-xl font-semibold mt-4 mb-2">Lifespan Calculation</h2>
      <p className="mb-4">
        We determine the lifespan by iterating over the years, reducing the efficiency each time by <strong>0.7%</strong>, until it falls below 70%.
        The total number of years taken to reach this point is the <strong>predicted lifespan</strong>.
      </p>
      <h2 className="text-xl font-semibold mt-4 mb-2">Why 70%?</h2>
      <p>
        Most manufacturers and industry standards consider <strong>70% efficiency</strong> as the end-of-life threshold because power output becomes significantly reduced, making it less cost-effective.
      </p>

      <h2 className="text-2xl font-semibold mt-6 mb-2">Step 2: Load and Preprocess Data and Apply Degradation Assumptions</h2>
      <pre className="bg-gray-900 text-white p-4 rounded-md overflow-auto">
        <code className="text-sm">
{`# Load the dataset
df = pd.read_csv("/content/DataSetF.csv", delimiter=";", engine="python")

# Handling missing values
df.dropna(inplace=True)

# Convert date column if it exists
if "DATE" in df.columns:
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

# Assume a degradation rate of 0.7% per year and an initial panel efficiency of 100%
def estimate_lifespan(degradation_rate=0.007, threshold=70):
    years = 0
    efficiency = 100
    while efficiency > threshold:
        efficiency -= efficiency * degradation_rate
        years += 1
    return years

# Create an estimated lifespan column
df["Lifespan"] = estimate_lifespan()

# Use only "SOLAR_RADIATION" as the predictor
if "SOLAR_RADIATION" not in df.columns:
    raise ValueError("Column 'SOLAR_RADIATION' not found in dataset.")

X = df[["SOLAR_RADIATION"]]  # Select only solar radiation as input
y = df["Lifespan"]           # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the feature using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)`}
        </code>
      </pre>

        <h2 className="text-2xl font-semibold mt-6 mb-2">Model Training and Evaluation</h2>
        <p className="text-lg text-gray-700 leading-relaxed">
          The model is trained using only <strong>solar radiation</strong> as the predictor. The dataset is split into training and testing sets, and the solar radiation values are standardized. A Decision Tree Regressor is then trained on the scaled data.
        </p>

        <h2 className="text-2xl font-semibold mt-6 mb-2">Step 3: Training, Predicting and Evaluating the Model</h2>
        <pre className="bg-gray-900 text-white p-4 rounded-lg overflow-x-auto">
        <code className="text-sm">
          {`# Train the Decision Tree model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on test data
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Predict lifespan based on a given solar radiation value
sample_solar_radiation = np.array([[df["SOLAR_RADIATION"].mean()]])  # Use mean solar radiation value
sample_solar_radiation_scaled = scaler.transform(sample_solar_radiation)  # Scale input
predicted_lifespan = model.predict(sample_solar_radiation_scaled)[0]

# Print the expected lifespan based on solar radiation
print(f"Predicted Solar Panel Lifespan (Based on Solar Radiation Only): **{predicted_lifespan:.2f} years**")
`}
        </code>
      </pre>
        
        <h2 className="text-2xl font-semibold mt-6 mb-2">Performance Metrics:</h2>
        <ul className="list-disc pl-6 text-lg text-gray-700">
          <li><strong>Mean Absolute Error (MAE)</strong>: Measures the average absolute difference between actual and predicted values.</li>
          <li><strong>Mean Squared Error (MSE)</strong>: Measures the squared difference between actual and predicted values.</li>
          <li><strong>Root Mean Squared Error (RMSE)</strong>: The square root of MSE, providing a more interpretable error metric.</li>
          <li><strong>R-squared (R²)</strong>: Indicates how well the model explains the variance in the data.</li>
        </ul>
        
        <h2 className="text-2xl font-semibold mt-6 mb-2">Integrating Real-Time Weather Data</h2>
        <p className="text-lg text-gray-700 leading-relaxed">
          The model is further enhanced by incorporating real-time weather data using the OpenWeather API. The amount of cloudiness is used to estimate solar radiation, which is then fed into the trained Decision Tree model to predict the expected lifespan of a solar panel in a given location.
        </p>

        <h2 className="text-2xl font-semibold mt-6 mb-2">Step 4: Fetching Real-Time Weather Data</h2>
      <pre className="bg-gray-900 text-white p-4 rounded-lg overflow-x-auto text-sm">
          <code className="text-sm">
            {`
#Predict with weather data

import requests

#Open Weather API key 70b5d279fca69840e6e8ad414ba87f03

def get_weather(api_key, city):
  base_url = "http://api.openweathermap.org/data/2.5/weather?"
  complete_url = f"{base_url}appid={api_key}&q={city}"
  response = requests.get(complete_url)
  if response.status_code == 200:
    return response.json()
  else:
    print(f"Error: {response.status_code}")
    return None

weather = get_weather('70b5d279fca69840e6e8ad414ba87f03','Benin')
solar_irradiance = np.array(100 *(1 - ((weather["clouds"]["all"]) / 100) ))
predicted_lifespan = model.predict(solar_irradiance.reshape(1, -1))
print(f"Predicted Life span from weather data {predicted_lifespan}")
            `}
          </code>
        </pre>

        <h2 className="text-2xl font-semibold mt-6 mb-2">Conclusion</h2>
        <p className="text-lg text-gray-700 leading-relaxed">
          By using Decision Tree Regression, we can effectively estimate solar panel lifespan based on historical solar radiation data and real-time weather information. This approach helps in making informed decisions about solar panel maintenance and replacement, ensuring sustainable energy production.
        </p>

        <h2 className="text-2xl font-semibold mt-6 mb-2">Future Improvements</h2>
        <ul className="list-disc pl-6 text-lg text-gray-700">
          <li>Incorporating additional environmental factors like temperature and humidity.</li>
          <li>Experimenting with other regression models such as Random Forest or Neural Networks.</li>
          <li>Deploying the model as a web-based tool for easy accessibility.</li>
        </ul>
      </div>
    </div>
  );
}
