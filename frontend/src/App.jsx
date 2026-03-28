import React from "react";
import { BrowserRouter as Router, Routes, Route, NavLink } from "react-router-dom";
import Home from "./pages/Home";
import Dashboard from "./pages/Dashboard";

function Navbar() {
  const linkClass = ({ isActive }) =>
    `px-4 py-2 rounded-lg text-sm font-medium transition-all ${
      isActive
        ? "bg-brand-600 text-white"
        : "text-gray-400 hover:text-white hover:bg-gray-800"
    }`;

  return (
    <nav className="bg-gray-900 border-b border-gray-800 px-6 py-3 flex items-center gap-4">
      <span className="text-brand-500 font-bold text-lg mr-4 flex items-center gap-2">
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17H3a2 2 0 01-2-2V5a2 2 0 012-2h14a2 2 0 012 2v10a2 2 0 01-2 2h-2"
          />
        </svg>
        Deepdetect AI
      </span>
      <NavLink to="/" end className={linkClass}>
        Detector
      </NavLink>
      <NavLink to="/dashboard" className={linkClass}>
        Dashboard
      </NavLink>
    </nav>
  );
}

export default function App() {
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <Router>
        <Navbar />
        <main className="max-w-5xl mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/dashboard" element={<Dashboard />} />
          </Routes>
        </main>
      </Router>
    </div>
  );
}
