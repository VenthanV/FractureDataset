import React from "react";
import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import SingleUpload from "./pages/SingleUpload.jsx";
import BatchUpload  from "./pages/BatchUpload.jsx";
import ModelStats   from "./pages/ModelStats.jsx";

const navStyle = {
  display: "flex",
  gap: "1.5rem",
  padding: "1rem 2rem",
  background: "#1a365d",
  alignItems: "center",
};

const logoStyle = {
  color: "#fff",
  fontWeight: 700,
  fontSize: "1.1rem",
  marginRight: "auto",
  textDecoration: "none",
};

const linkStyle = ({ isActive }) => ({
  color: isActive ? "#90cdf4" : "#cbd5e0",
  textDecoration: "none",
  fontWeight: isActive ? 600 : 400,
  fontSize: "0.95rem",
  paddingBottom: "2px",
  borderBottom: isActive ? "2px solid #90cdf4" : "2px solid transparent",
  transition: "color 0.15s",
});

export default function App() {
  return (
    <BrowserRouter>
      <nav style={navStyle}>
        <span style={logoStyle}>Fracture Detection</span>
        <NavLink to="/"      style={linkStyle}>Einzelbild</NavLink>
        <NavLink to="/batch" style={linkStyle}>Batch</NavLink>
        <NavLink to="/stats" style={linkStyle}>Modell-Metriken</NavLink>
      </nav>

      <main style={{ maxWidth: 960, margin: "0 auto", padding: "2rem 1rem" }}>
        <Routes>
          <Route path="/"      element={<SingleUpload />} />
          <Route path="/batch" element={<BatchUpload />} />
          <Route path="/stats" element={<ModelStats />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}
