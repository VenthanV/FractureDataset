import React from "react";

const wrapStyle = {
  display: "flex",
  alignItems: "center",
  gap: "0.75rem",
  padding: "0.75rem 1rem",
  background: "#ebf4ff",
  borderRadius: 8,
  border: "1px solid #bee3f8",
};

const labelStyle = {
  fontSize: "0.85rem",
  color: "#2c5282",
  fontWeight: 600,
  whiteSpace: "nowrap",
};

const valueStyle = {
  fontSize: "0.9rem",
  fontWeight: 700,
  color: "#1a365d",
  minWidth: 36,
  textAlign: "right",
};

/**
 * ThresholdSlider — decision threshold control.
 *
 * Props:
 *   value    {number}   current threshold (0.0 – 1.0)
 *   onChange {function} called with new number value
 */
export default function ThresholdSlider({ value, onChange }) {
  return (
    <div style={wrapStyle}>
      <span style={labelStyle}>Entscheidungsschwelle</span>
      <input
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{ flex: 1, accentColor: "#3182ce" }}
      />
      <span style={valueStyle}>{value.toFixed(2)}</span>
    </div>
  );
}
