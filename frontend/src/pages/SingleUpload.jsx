import React, { useState, useCallback, useRef } from "react";
import { predictSingle } from "../api/client.js";
import { useOptimalThreshold } from "../hooks/useOptimalThreshold.js";
import { getErrorMessage } from "../utils/prediction.js";
import ThresholdStatusBar from "../components/ThresholdStatusBar.jsx";
import PredictionCard     from "../components/PredictionCard.jsx";
import HeatmapViewer      from "../components/HeatmapViewer.jsx";

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


export default function SingleUpload() {
  const { threshold, setThreshold, optimalThresh, isOptimal } = useOptimalThreshold();
  const [debugOpen, setDebugOpen] = useState(false);
  const [file, setFile]           = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult]       = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);
  const [dragging, setDragging]   = useState(false);
  const inputRef = useRef(null);

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
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }

  async function handleThresholdChange(val) {
    setThreshold(val);
    if (result) await runPredict(val);
  }

  return (
    <div>
      <h1 style={headingStyle}>Einzelbild-Analyse</h1>

      <ThresholdStatusBar
        threshold={threshold}
        optimalThresh={optimalThresh}
        isOptimal={isOptimal}
        debugOpen={debugOpen}
        onDebugToggle={() => setDebugOpen((o) => !o)}
        onThresholdChange={handleThresholdChange}
        onReset={() => handleThresholdChange(optimalThresh)}
      />

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
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem", marginTop: "1.5rem" }}>
          <PredictionCard result={result} filename={file?.name} />
          <HeatmapViewer originalSrc={previewUrl} heatmapSrc={result.gradcam_image} />
        </div>
      )}
    </div>
  );
}
