import React, { useState } from "react";

const MODES = ["Original", "Heatmap", "Blend"];

const containerStyle = {
  display: "flex",
  flexDirection: "column",
  gap: "0.75rem",
};

const tabRowStyle = {
  display: "flex",
  gap: "0.5rem",
};

const tabStyle = (active) => ({
  padding: "0.35rem 0.85rem",
  borderRadius: 6,
  border: "1px solid #bee3f8",
  background: active ? "#3182ce" : "#ebf4ff",
  color: active ? "#fff" : "#2c5282",
  cursor: "pointer",
  fontSize: "0.82rem",
  fontWeight: active ? 600 : 400,
  transition: "background 0.15s",
});

const imgStyle = {
  width: "100%",
  borderRadius: 8,
  border: "1px solid #e2e8f0",
  objectFit: "contain",
  maxHeight: 400,
  background: "#000",
};

/**
 * HeatmapViewer — toggles between original, heatmap overlay, and blend.
 *
 * Props:
 *   originalSrc  {string}  object URL of the uploaded file
 *   heatmapSrc   {string}  base64 data-URI from the API (gradcam_image)
 */
export default function HeatmapViewer({ originalSrc, heatmapSrc }) {
  const [mode, setMode] = useState("Heatmap");

  const src =
    mode === "Original"
      ? originalSrc
      : mode === "Heatmap"
      ? heatmapSrc || originalSrc
      : heatmapSrc || originalSrc; // blend falls back to heatmap

  if (!originalSrc) return null;

  return (
    <div style={containerStyle}>
      <div style={tabRowStyle}>
        {MODES.map((m) => (
          <button key={m} style={tabStyle(mode === m)} onClick={() => setMode(m)}>
            {m}
          </button>
        ))}
      </div>
      <img
        src={src}
        alt={`${mode} view`}
        style={imgStyle}
      />
    </div>
  );
}
