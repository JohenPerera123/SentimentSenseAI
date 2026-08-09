import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Analyze from './pages/Analyze';
import Compare from './pages/Compare';
import Performance from './pages/Performance';
import About from './pages/About';

function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<Analyze />} />
        <Route path="/compare" element={<Compare />} />
        <Route path="/performance" element={<Performance />} />
        <Route path="/about" element={<About />} />
      </Routes>
    </Router>
  );
}

export default App;
