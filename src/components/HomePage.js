import React from "react";
import "tailwindcss/tailwind.css";

const HomePage = () => {
  return (
    <div className="min-h-screen bg-gray-100 text-gray-900 p-6 flex flex-col items-center">
      <div className="max-w-3xl bg-white shadow-lg rounded-2xl p-8">
        <h1 className="text-4xl font-bold text-blue-600 mb-4">Machine Learning for Studying Solar Panel Conditions</h1>
        <p className="text-lg text-gray-700 mb-6">
          The increasing demand for renewable energy has placed a significant focus on optimizing solar panel efficiency and maintenance.
          This project explores the use of machine learning to monitor and predict solar panel conditions.
        </p>

        <h2 className="text-2xl font-semibold text-blue-500 mb-2">Problem Statement</h2>
        <p className="text-gray-700 mb-4">
          Solar panel performance is influenced by multiple factors such as aging, dust accumulation, and fluctuating weather conditions.
          Traditional maintenance methods rely on manual inspections, which are often inefficient and costly.
        </p>

        <h2 className="text-2xl font-semibold text-blue-500 mb-2">Objectives</h2>
        <ul className="list-disc list-inside text-gray-700 mb-4">
          <li>Develop machine learning models to analyze solar panel conditions.</li>
          <li>Predict energy output based on environmental factors.</li>
          <li>Implement real-time monitoring using weather APIs.</li>
          <li>Enhance predictive maintenance strategies for solar infrastructure.</li>
        </ul>

        <h2 className="text-2xl font-semibold text-blue-500 mb-2">Methodology</h2>
        <p className="text-gray-700 mb-4">
          The project utilized multiple machine learning models, including:
        </p>
        <ul className="list-disc list-inside text-gray-700 mb-4">
          <li><strong>Linear Regression:</strong> Finds relationships between environmental variables and energy output.</li>
          <li><strong>Decision Tree Regression:</strong> Captures non-linear dependencies, enhancing predictive accuracy.</li>
          <li><strong>Weather Data Integration:</strong> Uses real-time weather data from APIs to improve predictions.</li>
          <li><strong>Backend Development:</strong> Implements a FastAPI-based system with MongoDB for data storage.</li>
        </ul>

        <h2 className="text-2xl font-semibold text-blue-500 mb-2">Datasets</h2>
        <p className="text-gray-700 mb-4">The dataset contains the following columns:</p>
        <ul className="list-disc list-inside text-gray-700 mb-4">
          <li><b>Timestamp</b>: Date and time of the record.</li>
          <li><b>AirTemperature</b>: Ambient temperature (°C).</li>
          <li><b>RelativeHumidity</b>: Percentage of humidity.</li>
          <li><b>DewPointTemperature</b></li>
          <li><b>ApparentTemperature</b></li>
          <li><b>WindSpeed</b></li>
          <li><b>WindDirection</b></li>
        </ul>
        <h2 className="font-bold text-xl">Solar Irradiance Calculation</h2>
        <p class="formula">
            <b>GHI</b> = GHI<sub>clear</sub> × (1 − 0.75 C<sup>3</sup>) × (1 − 0.1 × RH) × (1 + 0.02 × (T − 25))
        </p>
        
        <h3>Where:</h3>
        <ul>
            <li><b>GHI<sub>clear</sub></b> = 1000 W/m² (approximate midday clear-sky irradiance).</li>
            <li><b>RH</b> = Relative Humidity (normalized 0-1).</li>
            <li><b>T</b> = Air Temperature (°C).</li>
        </ul>

        <h2 className="text-2xl font-semibold text-blue-500 mb-2 mt-5">Results and Analysis</h2>
        <p className="text-gray-700 mb-4">
          - Temperature and humidity significantly affect solar panel output.
          - Decision trees provided higher accuracy compared to simple regression models.
          - Real-time weather data enhanced model performance, making predictions more reliable.
        </p>

        <h2 className="text-2xl font-semibold text-blue-500 mb-2">Future Work</h2>
        <p className="text-gray-700 mb-4">
          Future improvements include expanding datasets, implementing ensemble models, and deploying real-time monitoring solutions in the cloud.
        </p>
      </div>
    </div>
  );
};

export default HomePage;
