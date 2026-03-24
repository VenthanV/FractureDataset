import React, { useState, useCallback, useRef, useEffect } from "react";
import { predictSingle, getModelStats } from "../api/client.js";
import ThresholdSlider from "../components/ThresholdSlider.jsx";
import PredictionCard  from "../components/PredictionCard.jsx";
import HeatmapViewer   from "../components/HeatmapViewer.jsx";

const dropZoneStyle = (active) => ({
  border: `2px dashed ${active ? "#3182ce" : "#90cdf4"}`,
  borderRadius: 12,
  padding: "2.5rem 1rem",
  textAlign: "center",
  cursor: "pointer",
  background: active ? "#ebf8ff" : "#f0f9ff",
  color: "#2c5282",
  transition: "all 0.2s",
  marginBottom: "1rem",
});

const headingStyle = {
  fontSize: "1.4rem",
  fontWeight: 700,
  marginBottom: "1.25rem",
  color: "#1a365d",
};

const errorStyle = {
  background: "#fff5f5",
  border: "1px solid #fc8181",
  borderRadius: 8,
  padding: "0.75rem 1rem",
  color: "#c53030",
  fontSize: "0.88rem",
  marginTop: "0.75rem",
};

const spinnerStyle = {
  display: "inline-block",
  width: 20,
  height: 20,
  border: "3px solid #bee3f8",
  borderTopColor: "#3182ce",
  borderRadius: "50%",
  animation: "spin 0.7s linear infinite",
  marginRight: 8,
  verticalAlign: "middle",
};

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

export default function SingleUpload() {
  const [threshold, setThreshold]       = useState(0.5);
  const [optimalThresh, setOptimalThresh] = useState(null);
  const [debugOpen, setDebugOpen]       = useState(false);
  const [file, setFile]                 = useState(null);
  const [previewUrl, setPreviewUrl]     = useState(null);
  const [result, setResult]             = useState(null);
  const [loading, setLoading]           = useState(false);
  const [error, setError]               = useState(null);
  const [dragging, setDragging]         = useState(false);
  const inputRef = useRef(null);

  // Load optimal threshold once on mount
  useEffect(() => {
    getModelStats()
      .then((stats) => {
        const t = stats.optimal_threshold ?? 0.5;
        setOptimalThresh(t);
        setThreshold(t);
      })
      .catch(() => {});   // silently fall back to 0.5
  }, []);

  function handleFile(f) {
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
    setPreviewUrl(URL.createObjectURL(f));
  }

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, []);

  async function runPredict(thresh) {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await predictSingle(file, thresh);
      setResult(res);
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || "Unbekannter Fehler");
    } finally {
      setLoading(false);
    }
  }

  async function handleThresholdChange(val) {
    setThreshold(val);
    if (result) await runPredict(val);
  }

  const isOptimal = optimalThresh !== null && Math.abs(threshold - optimalThresh) < 0.001;

  return (
    <div>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      <h1 style={headingStyle}>Einzelbild-Analyse</h1>

      {/* Threshold status bar — always visible */}
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

        <button style={debugToggleStyle} onClick={() => setDebugOpen((o) => !o)}>
          ⚙ Debug {debugOpen ? "▲" : "▼"}
        </button>

        {!isOptimal && optimalThresh !== null && (
          <button
            style={{ ...debugToggleStyle, color: "#2b6cb0", borderColor: "#bee3f8", background: "#ebf4ff" }}
            onClick={() => handleThresholdChange(optimalThresh)}
          >
            ↺ Auf {optimalThresh.toFixed(3)} zurücksetzen
          </button>
        )}
      </div>

      {/* Collapsible debug slider */}
      {debugOpen && (
        <div style={{
          padding: "0.75rem", background: "#f7fafc",
          border: "1px solid #e2e8f0", borderRadius: 8,
          marginBottom: "1rem",
        }}>
          <ThresholdSlider value={threshold} onChange={handleThresholdChange} />
          {optimalThresh !== null && (
            <p style={{ fontSize: "0.75rem", color: "#718096", marginTop: "0.4rem", marginBottom: 0 }}>
              Optimaler Schwellenwert (Youden's J): <strong>{optimalThresh.toFixed(3)}</strong>
              {" "}— niedrigerer Wert = mehr Sensitivität, höherer = mehr Spezifität
            </p>
          )}
        </div>
      )}

      <div>
        <div
          style={dropZoneStyle(dragging)}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          {file
            ? <span>Datei: <strong>{file.name}</strong> &nbsp;(klicken zum Wechseln)</span>
            : <span>Röntgenbild hier ablegen oder <u>klicken</u> zum Auswählen</span>
          }
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={(e) => handleFile(e.target.files[0])}
          />
        </div>

        {file && !result && !loading && (
          <button
            onClick={() => runPredict(threshold)}
            style={{
              padding: "0.6rem 1.5rem", background: "#3182ce", color: "#fff",
              border: "none", borderRadius: 8, cursor: "pointer",
              fontWeight: 600, fontSize: "0.95rem",
            }}
          >
            Analysieren
          </button>
        )}

        {loading && (
          <p style={{ marginTop: "0.75rem", color: "#2c5282" }}>
            <span style={spinnerStyle} />Analyse läuft…
          </p>
        )}

        {error && <div style={errorStyle}>{error}</div>}
      </div>

      {result && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem", marginTop: "1.5rem" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
            <PredictionCard result={result} filename={file?.name} />
            <HeatmapViewer originalSrc={previewUrl} heatmapSrc={result.gradcam_image} />
          </div>
          <div>
            {previewUrl && (
              <img
                src={previewUrl}
                alt="Original"
                style={{ width: "100%", borderRadius: 8, border: "1px solid #e2e8f0" }}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
}
