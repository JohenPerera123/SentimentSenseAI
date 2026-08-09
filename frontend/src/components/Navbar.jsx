import React from 'react';
import { NavLink } from 'react-router-dom';
import { Activity } from 'lucide-react';

const Navbar = () => {
  return (
    <nav className="glass-nav">
      <div className="nav-brand">
        <Activity size={28} color="#4F46E5" />
        SentimentScope
      </div>
      <div className="nav-links">
        <NavLink to="/" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>Analyze</NavLink>
        <NavLink to="/compare" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>Compare</NavLink>
        <NavLink to="/performance" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>Performance</NavLink>
        <NavLink to="/about" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>About</NavLink>
      </div>
    </nav>
  );
};

export default Navbar;
