import { motion } from "framer-motion";
import { Link } from "react-router-dom";

export default function HomePage2() {
  return (
    <div className="min-h-screen bg-gray-100 p-6 flex flex-col items-center">
      <motion.h1 
        className="lg:text-5xl text-3xl font-bold text-gray-900 mb-6 text-center"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        Solar Power Generation Forecast Using Machine Learning
      </motion.h1>
      <div className="max-w-4xl bg-white shadow-lg rounded-2xl p-8">
        <motion.p 
          className="text-lg text-gray-700 leading-relaxed"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          Solar energy is one of the most sustainable and renewable sources of power. With the increasing global energy demand, optimizing and predicting solar energy output has become essential. This article explores the use of <strong>Linear Regression</strong> for predicting solar power generation based on <strong>solar radiation</strong> data. By analyzing past trends and integrating real-time weather data, we can create an accurate forecasting model that helps in energy management and optimization.
        </motion.p>

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
          <motion.button 
            whileHover={{ scale: 1.1 }} 
            whileTap={{ scale: 0.9 }} 
            className="px-6 py-2 mt-3 mb-3 bg-blue-600 text-white rounded-lg"
            >
            <Link to='https://www.kaggle.com/code/hnatyukmu/solar-power-generation-forecast-with-99-auc/input'>Get dataset</Link>
          </motion.button>
        </p>

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
        <p className="text-lg font-mono bg-gray-200 p-4 rounded-lg mt-4">
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
