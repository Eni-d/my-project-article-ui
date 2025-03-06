import { useState } from "react";
import { Menu, X } from "lucide-react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <nav className="bg-gray-900 text-white p-4 shadow-md">
      <div className="container mx-auto flex justify-between items-center">
        <h1 className="text-white text-xl font-medium mb-2">ML Models For Studying Solar Panel Conditions</h1>
        
        <div className="hidden md:flex space-x-6">
          <Link to="/" className="text-white hover:text-gray-300">Solar Power Generation Forecast</Link>
          <Link to="/spl" className="text-white hover:text-gray-300">Solar Panel Lifespan</Link>
          <motion.button 
            whileHover={{ scale: 1.1 }} 
            whileTap={{ scale: 0.9 }} 
            className="px-6 py-2 bg-blue-600 text-white rounded-lg"
            >
            <Link to='https://www.kaggle.com/code/hnatyukmu/solar-power-generation-forecast-with-99-auc/input'>Get dataset</Link>
          </motion.button>
        </div>
        
        <button className="md:hidden text-white" onClick={() => setIsOpen(!isOpen)}>
          {isOpen ? <X size={28} /> : <Menu size={28} />}
        </button>
      </div>
      
      {isOpen && (
        <div className="md:hidden flex flex-col items-center bg-gray-900 text-white py-4 space-y-4">
          <Link to="/" className="text-white hover:text-gray-300">Solar Power Generation Forecast</Link>
          <Link to="/spl" className="text-white hover:text-gray-300">Solar Panel Lifespan</Link>
          <motion.button 
            whileHover={{ scale: 1.1 }} 
            whileTap={{ scale: 0.9 }} 
            className="px-6 py-2 mt-3 mb-3 bg-blue-600 text-white rounded-lg"
            >
            <Link to='https://www.kaggle.com/code/hnatyukmu/solar-power-generation-forecast-with-99-auc/input'>Get dataset</Link>
          </motion.button>
        </div>
      )}
    </nav>
  );
}
