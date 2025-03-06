import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import SPGF from "./components/SPGF";
import SPL from "./components/SPL";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";

const App = () => {
  return (
    <div>
      <Router>
        <Navbar />
        <Routes>
          <Route path="/" element={<SPGF />} />
          <Route path="/spl" element={<SPL />} />
        </Routes>
        <Footer />
      </Router>
    </div>
    
  );
};

export default App;
