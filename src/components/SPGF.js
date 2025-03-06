import { motion } from "framer-motion";
import { Link } from "react-router-dom";

export default function SPGF() {
  return (
    <div className="min-h-screen bg-gray-100 p-6 flex flex-col items-center overflow-x-hidden">
      <motion.h1 
        className="lg:text-4xl text-3xl font-bold text-gray-900 mb-6 text-center"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        Solar Power Generation Forecast Using Machine Learning
      </motion.h1>
      <div className="lg:w-auto w-96 bg-white shadow-lg rounded-2xl p-8">
        <motion.p 
          className="text-lg text-gray-700 leading-relaxed"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          Solar energy is one of the most sustainable and renewable sources of power. With the increasing global energy demand, optimizing and predicting solar energy output has become essential. This article explores the use of <strong>Linear Regression</strong> for predicting solar power generation based on <strong>solar radiation</strong> data. By analyzing past trends and integrating real-time weather data, we can create an accurate forecasting model that helps in energy management and optimization.
        </motion.p>

        <h2 className="text-2xl font-semibold mt-6 mb-2">Step 1: Importing Required Libraries</h2>
      <pre className="bg-gray-900 text-white p-4 rounded-lg overflow-x-auto">
        import pandas as pd{"\n"}
        from sklearn.model_selection import train_test_split{"\n"}
        from sklearn.preprocessing import StandardScaler{"\n"}
        from sklearn.linear_model import LinearRegression{"\n"}
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score{"\n"}
        import numpy as np
      </pre>
      <p className="text-lg mt-2">These libraries help with data manipulation, machine learning, and performance evaluation.</p>

        <motion.h2 
          className="text-3xl font-semibold text-gray-800 mt-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
        >
          Dataset and Preprocessing
        </motion.h2>
        <p className="text-lg text-gray-700 mt-2">
          The dataset consists of key parameters such as solar radiation, cloud coverage, and energy output. Proper preprocessing is necessary to clean the data, handle missing values, and convert relevant columns to numeric formats. Once the data is cleaned, it is split into training and testing sets to ensure proper evaluation of the model.
        </p>

        <h2 className="text-2xl font-semibold mt-6 mb-2">Step 2: Loading and Preprocessing Data</h2>
      <pre className="bg-gray-900 text-white p-4 rounded-lg overflow-x-auto">
        df = pd.read_csv("/content/DataSetF.csv", delimiter=';') {"\n"}
        df["ENERGY"] = pd.to_numeric(df["ENERGY"], errors='coerce'){"\n"}
        df["CLOUDY"] = pd.to_numeric(df["CLOUDY"], errors='coerce'){"\n"}
        df["SOLAR_RADIATION"] = pd.to_numeric(df["SOLAR_RADIATION"], errors='coerce')
      </pre>
      <p className="text-lg mt-2">The dataset is loaded, and necessary columns are converted into numeric format.</p>

        <motion.h2 
          className="text-3xl font-semibold text-gray-800 mt-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
        >
          Theory Behind Linear Regression
        </motion.h2>
        <p className="text-lg text-gray-700 mt-2">
          Linear Regression is one of the most fundamental algorithms in machine learning. It establishes a relationship between an independent variable (solar radiation) and a dependent variable (energy output) by fitting a linear equation to the observed data. The equation of a simple linear regression model is:
        </p>
        <p className="text-lg font-mono bg-gray-900 text-white p-4 rounded-lg mt-4">
          Y = mX + b
        </p>
        <p className="text-lg text-gray-700 mt-2">
          Where:
          <ul className="list-disc ml-6 mt-2">
            <li>Y is the dependent variable (energy output)</li>
            <li>X is the independent variable (solar radiation)</li>
            <li>m is the slope of the line (determining how much Y changes for each unit change in X)</li>
            <li>b is the intercept (the baseline energy output when solar radiation is zero)</li>
          </ul>
        </p>

        <h2 className="text-2xl font-semibold mt-6 mb-2">Step 3: Training the Model</h2>
      <pre className="bg-gray-900 text-white p-4 rounded-lg overflow-x-auto">
        X = df[["SOLAR_RADIATION"]]{"\n"}
        y = df["ENERGY"]{"\n"}
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42){"\n"}
        scaler = StandardScaler(){"\n"}
        X_train_scaled = scaler.fit_transform(X_train){"\n"}
        X_test_scaled = scaler.transform(X_test){"\n"}
        model = LinearRegression(){"\n"}
        model.fit(X_train_scaled, y_train)
      </pre>
      <p className="text-lg mt-2">The data is split into training and testing sets, scaled, and used to train a linear regression model.</p>

        <motion.h2 
          className="text-3xl font-semibold text-gray-800 mt-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
        >
          Model Evaluation
        </motion.h2>
        <p className="text-lg text-gray-700 mt-2">
          To assess the performance of our model, we use evaluation metrics such as:
          <ul className="list-disc ml-6 mt-2">
            <li><strong>Mean Absolute Error (MAE)</strong> - Measures the average absolute difference between predicted and actual values.</li>
            <li><strong>Mean Squared Error (MSE)</strong> - Computes the average squared difference, giving higher weight to larger errors.</li>
            <li><strong>Root Mean Squared Error (RMSE)</strong> - The square root of MSE, providing an interpretable measure of error.</li>
            <li><strong>R-squared (R²)</strong> - Indicates how well the model explains the variance in the target variable.</li>
          </ul>
        </p>

        <h2 className="text-2xl font-semibold mt-6 mb-2">Step 4: Predicting and Evaluating the Model</h2>
      <pre className="bg-gray-900 text-white p-4 rounded-lg overflow-x-auto">
        y_pred = model.predict(X_test_scaled){"\n"}
        mae = mean_absolute_error(y_test, y_pred){"\n"}
        mse = mean_squared_error(y_test, y_pred){"\n"}
        rmse = np.sqrt(mse){"\n"}
        r2 = r2_score(y_test, y_pred)
      </pre>
      <p className="text-lg mt-2">The model's performance is evaluated using metrics like MAE, MSE, RMSE, and R².</p>

        <motion.h2 
          className="text-3xl font-semibold text-gray-800 mt-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.0 }}
        >
          Real-Time Prediction Using Weather Data
        </motion.h2>
        <p className="text-lg text-gray-700 mt-2">
          By integrating <strong>OpenWeather API</strong>, we retrieve real-time weather data, particularly cloud coverage. Using this data, we estimate solar radiation and input it into our trained model to predict energy output dynamically. This integration allows us to forecast solar energy generation for different cities and optimize energy distribution efficiently.
        </p>

        <h2 className="text-2xl font-semibold mt-6 mb-2">Step 5: Fetching Real-Time Weather Data</h2>
      <pre className="bg-gray-900 text-white p-4 rounded-lg overflow-x-auto text-sm">
          <code>
            {`import requests

def get_weather(api_key, city):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}appid={api_key}&q={city}"
    response = requests.get(complete_url)
    return response.json() if response.status_code == 200 else None`}
          </code>
        </pre>
      <p className="text-lg mt-2">The OpenWeather API fetches real-time cloudiness data for energy output prediction.</p>

      <h2 className="text-2xl font-semibold mt-6 mb-2">Step 6: Predicting Energy Output</h2>
      <pre className="bg-gray-900 text-white p-4 rounded-lg overflow-x-auto">
        weather = get_weather('YOUR_API_KEY', 'Benin'){"\n"}
        solar_irradiance = np.array(100 * (1 - (weather["clouds"]["all"] / 100))){"\n"}
        print(model.predict(solar_irradiance.reshape(1, -1)))
      </pre>
      <p className="text-lg mt-2">This code converts cloudiness data to solar irradiance and predicts energy output.</p>

        <motion.h2 
          className="text-3xl font-semibold text-gray-800 mt-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2 }}
        >
          Future Enhancements
        </motion.h2>
        <p className="text-lg text-gray-700 mt-2">
          While linear regression provides a simple yet effective approach, more advanced models such as <strong>neural networks</strong> or <strong>gradient boosting</strong> could be implemented for better accuracy. Additionally, incorporating more weather features like temperature, wind speed, and humidity can further improve predictions.
        </p>

        <motion.h2 
          className="text-3xl font-semibold text-gray-800 mt-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.4 }}
        >
          Conclusion
        </motion.h2>
        <p className="text-lg text-gray-700 mt-2">
          Predicting solar power generation using machine learning enables better energy planning and management. Our model demonstrates that solar radiation strongly correlates with energy output, and integrating real-time weather data enhances predictive capabilities. With future improvements, such models can play a vital role in the transition to renewable energy sources.
        </p>
      </div>
    </div>
  );
}
