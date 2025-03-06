import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import HomePage from "./components/HomePage";
import HomePage2 from "./components/HomePage2";

const App = () => {
  return (
    <div>
      <Router>
        <Routes>
          {/* <Route path="/" element={<HomePage />} /> */}
          <Route path="/" element={<HomePage2 />} />
        </Routes>
      </Router>
    </div>
    
  );
};

export default App;
