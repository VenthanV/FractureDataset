import React from "react";
import { isUncertain } from "../utils/prediction.js";

const cardStyle = {
  borderRadius: 10,
  padding: "1.25rem 1.5rem",
  border: "1px solid #e2e8f0",
  background: "#fff",
  boxShadow: "0 1px 4px rgba(0,0,0,0.07)",
};

const badgeStyle = (isFracture) => ({
  display: "inline-block",
  padding: "0.3rem 1rem",
  borderRadius: 20,
  fontWeight: 700,
  fontSize: "1rem",
  letterSpacing: "0.04em",
  background: isFracture ? "#fed7d7" : "#c6f6d5",
  color: isFracture ? "#c53030" : "#276749",
  marginBottom: "0.75rem",
});

const barWrapStyle = {
  background: "#e2e8f0",
  borderRadius: 6,
  height: 14,
  overflow: "hidden",
  marginTop: "0.35rem",
};

const barFillStyle = (prob, isFracture) => ({
  height: "100%",
  width: `${(prob * 100).toFixed(1)}%`,
  background: isFracture
    ? `linear-gradient(90deg, #fc8181, #e53e3e)`
    : `linear-gradient(90deg, #68d391, #38a169)`,
  borderRadius: 6,
  transition: "width 0.4s ease",
});

const metaStyle = {
  fontSize: "0.82rem",
  color: "#718096",
  marginTop: "0.6rem",
};

/**
 * PredictionCard — displays label badge, probability bar, and metadata.
 *
 * Props:
 *   result   {object}  API PredictionResponse
 *   filename {string}
 */
export default function PredictionCard({ result, filename }) {
  if (!result) return null;

  const isFracture  = result.label === "fracture";
  const pct         = (result.probability * 100).toFixed(1);
  const uncertain   = isUncertain(result.probability, result.threshold_used);

  return (
    <div style={cardStyle}>
      <div style={badgeStyle(isFracture)}>
        {isFracture ? "FRAKTUR" : "NORMAL"}
      </div>

      <div>
        <div style={{ fontSize: "0.85rem", color: "#4a5568", marginBottom: 2 }}>
          P(Fraktur): <strong>{pct}%</strong>
        </div>
        <div style={barWrapStyle}>
          <div style={barFillStyle(result.probability, isFracture)} />
        </div>
      </div>

      {uncertain && (
        <div style={{
          marginTop: "0.75rem",
          padding: "0.55rem 0.85rem",
          borderRadius: 8,
          background: "#fffbeb",
          border: "1px solid #f6e05e",
          display: "flex",
          alignItems: "flex-start",
          gap: "0.5rem",
        }}>
          <span style={{ fontSize: "1rem", lineHeight: 1.4 }}>⚠</span>
          <div>
            <div style={{ fontSize: "0.82rem", fontWeight: 700, color: "#744210" }}>
              Unsichere Vorhersage
            </div>
            <div style={{ fontSize: "0.78rem", color: "#92400e", marginTop: 2 }}>
              P(Fraktur) liegt innerhalb von ±5% des Schwellenwerts ({result.threshold_used.toFixed(3)}).
              Manuelle Überprüfung durch Radiologen empfohlen.
            </div>
          </div>
        </div>
      )}

      <div style={metaStyle}>
        Schwellenwert: {result.threshold_used.toFixed(2)}
        {filename && <> &nbsp;·&nbsp; {filename}</>}
      </div>
    </div>
  );
}
