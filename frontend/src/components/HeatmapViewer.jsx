import React, { useState } from "react";

const labelStyle = {
  fontSize: "0.78rem",
  fontWeight: 600,
  color: "#4a5568",
  marginBottom: "0.35rem",
  textAlign: "center",
};

const imgStyle = {
  width: "100%",
  borderRadius: 8,
  border: "1px solid #e2e8f0",
  objectFit: "contain",
  background: "#000",
  display: "block",
};


/**
 * HeatmapViewer — side-by-side comparison of original and Grad-CAM overlay.
 *
 * Props:
 *   originalSrc  {string}  object URL of the uploaded file
 *   heatmapSrc   {string}  base64 data-URI from the API (gradcam_image)
 */
export default function HeatmapViewer({ originalSrc, heatmapSrc }) {
  const [opacity, setOpacity] = useState(80);

  if (!originalSrc) return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
      {/* Opacity slider — only shown when heatmap is available */}
      {heatmapSrc && (
        <div style={{
          display: "flex", alignItems: "center", gap: "0.75rem",
          padding: "0.5rem 0.75rem",
          background: "#f7fafc", borderRadius: 8, border: "1px solid #e2e8f0",
        }}>
          <span style={{ fontSize: "0.78rem", color: "#4a5568", whiteSpace: "nowrap" }}>
            Overlay-Intensität
          </span>
          <input
            type="range"
            min={0}
            max={100}
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
            style={{ flex: 1, accentColor: "#3182ce" }}
          />
          <span style={{ fontSize: "0.78rem", color: "#2c5282", fontWeight: 600, minWidth: 36 }}>
            {opacity}%
          </span>
        </div>
      )}

      {/* Side-by-side images */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem" }}>
        {/* Left: Original */}
        <div>
          <p style={labelStyle}>Original</p>
          <img src={originalSrc} alt="Original" style={imgStyle} />
        </div>

        {/* Right: Grad-CAM overlay (backend already blended original+heatmap at 480×480) */}
        <div>
          <p style={labelStyle}>Grad-CAM Overlay</p>
          {heatmapSrc ? (
            <img
              src={heatmapSrc}
              alt="Heatmap"
              style={{ ...imgStyle, opacity: opacity / 100, transition: "opacity 0.15s" }}
            />
          ) : (
            <div style={{
              ...imgStyle,
              display: "flex", alignItems: "center", justifyContent: "center",
              minHeight: 120, color: "#a0aec0", fontSize: "0.82rem",
            }}>
              Kein Overlay verfügbar
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
