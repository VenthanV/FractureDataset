import React from "react";
import ThresholdSlider from "./ThresholdSlider.jsx";

const debugToggleStyle = {
  display: "inline-flex",
  alignItems: "center",
  gap: "0.35rem",
  fontSize: "0.78rem",
  color: "#718096",
  cursor: "pointer",
  border: "1px solid #e2e8f0",
  borderRadius: 6,
  padding: "0.25rem 0.6rem",
  background: "#f7fafc",
  userSelect: "none",
};

/**
 * ThresholdStatusBar — threshold display, optimal badge, debug panel, reset button.
 *
 * Props:
 *   threshold      {number}
 *   optimalThresh  {number|null}
 *   isOptimal      {boolean}
 *   debugOpen      {boolean}
 *   onDebugToggle  {() => void}
 *   onThresholdChange {(value: number) => void}
 *   onReset        {() => void}  — called when user clicks reset to optimal
 */
export default function ThresholdStatusBar({
  threshold,
  optimalThresh,
  isOptimal,
  debugOpen,
  onDebugToggle,
  onThresholdChange,
  onReset,
}) {
  return (
    <>
      <div style={{
        display: "flex", alignItems: "center", gap: "0.75rem",
        marginBottom: "1rem", flexWrap: "wrap",
      }}>
        <span style={{ fontSize: "0.88rem", color: "#4a5568" }}>
          Entscheidungsschwelle:{" "}
          <strong style={{ color: "#1a365d" }}>{threshold.toFixed(3)}</strong>
          {isOptimal && (
            <span style={{
              marginLeft: 6, fontSize: "0.75rem", background: "#c6f6d5",
              color: "#276749", borderRadius: 10, padding: "1px 7px", fontWeight: 600,
            }}>
              optimal (Youden's J)
            </span>
          )}
        </span>

        <button style={debugToggleStyle} onClick={onDebugToggle}>
          ⚙ Debug {debugOpen ? "▲" : "▼"}
        </button>

        {!isOptimal && optimalThresh !== null && (
          <button
            style={{ ...debugToggleStyle, color: "#2b6cb0", borderColor: "#bee3f8", background: "#ebf4ff" }}
            onClick={onReset}
          >
            ↺ Auf {optimalThresh.toFixed(3)} zurücksetzen
          </button>
        )}
      </div>

      {debugOpen && (
        <div style={{
          padding: "0.75rem", background: "#f7fafc",
          border: "1px solid #e2e8f0", borderRadius: 8,
          marginBottom: "1rem",
        }}>
          <ThresholdSlider value={threshold} onChange={onThresholdChange} />
          {optimalThresh !== null && (
            <p style={{ fontSize: "0.75rem", color: "#718096", marginTop: "0.4rem", marginBottom: 0 }}>
              Optimaler Schwellenwert (Youden's J): <strong>{optimalThresh.toFixed(3)}</strong>
              {" "}— niedrigerer Wert = mehr Sensitivität, höherer = mehr Spezifität
            </p>
          )}
        </div>
      )}
    </>
  );
}
