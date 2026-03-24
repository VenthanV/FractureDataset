import React, { useState, useRef, useEffect } from "react";
import { predictBatch, getModelStats } from "../api/client.js";
import ThresholdSlider   from "../components/ThresholdSlider.jsx";
import BatchResultsTable from "../components/BatchResultsTable.jsx";

const headingStyle = {
  fontSize: "1.4rem",
  fontWeight: 700,
  marginBottom: "1.25rem",
  color: "#1a365d",
};

const dropZoneStyle = {
  border: "2px dashed #90cdf4",
  borderRadius: 12,
  padding: "2rem 1rem",
  textAlign: "center",
  cursor: "pointer",
  background: "#f0f9ff",
  color: "#2c5282",
  marginTop: "1rem",
  marginBottom: "1rem",
};

const btnStyle = {
  padding: "0.6rem 1.5rem",
  background: "#3182ce",
  color: "#fff",
  border: "none",
  borderRadius: 8,
  cursor: "pointer",
  fontWeight: 600,
  fontSize: "0.95rem",
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
  width: 18,
  height: 18,
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

export default function BatchUpload() {
  const [threshold, setThreshold]         = useState(0.5);
  const [optimalThresh, setOptimalThresh]  = useState(null);
  const [debugOpen, setDebugOpen]          = useState(false);
  const [files, setFiles]                  = useState([]);
  const [results, setResults]              = useState([]);
  const [loading, setLoading]              = useState(false);
  const [error, setError]                  = useState(null);
  const inputRef = useRef(null);

  useEffect(() => {
    getModelStats()
      .then((stats) => {
        const t = stats.optimal_threshold ?? 0.5;
        setOptimalThresh(t);
        setThreshold(t);
      })
      .catch(() => {});
  }, []);

  function handleFiles(fileList) {
    setFiles(Array.from(fileList));
    setResults([]);
    setError(null);
  }

  async function runBatch() {
    if (!files.length) return;
    setLoading(true);
    setError(null);
    try {
      const res = await predictBatch(files, threshold);
      setResults(res);
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || "Unbekannter Fehler");
    } finally {
      setLoading(false);
    }
  }

  const isOptimal = optimalThresh !== null && Math.abs(threshold - optimalThresh) < 0.001;

  const summary = results.length
    ? {
        total:    results.length,
        fracture: results.filter((r) => r.label === "fracture").length,
        normal:   results.filter((r) => r.label === "normal").length,
      }
    : null;

  return (
    <div>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      <h1 style={headingStyle}>Batch-Analyse</h1>

      {/* Threshold status bar */}
      <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", flexWrap: "wrap" }}>
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
            onClick={() => { setThreshold(optimalThresh); setResults([]); }}
          >
            ↺ Auf {optimalThresh.toFixed(3)} zurücksetzen
          </button>
        )}
      </div>

      {debugOpen && (
        <div style={{
          padding: "0.75rem", background: "#f7fafc",
          border: "1px solid #e2e8f0", borderRadius: 8,
          marginTop: "0.75rem",
        }}>
          <ThresholdSlider
            value={threshold}
            onChange={(v) => { setThreshold(v); setResults([]); }}
          />
          {optimalThresh !== null && (
            <p style={{ fontSize: "0.75rem", color: "#718096", marginTop: "0.4rem", marginBottom: 0 }}>
              Optimaler Schwellenwert (Youden's J): <strong>{optimalThresh.toFixed(3)}</strong>
              {" "}— niedrigerer Wert = mehr Sensitivität, höherer = mehr Spezifität
            </p>
          )}
        </div>
      )}

      <div
        style={dropZoneStyle}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => { e.preventDefault(); handleFiles(e.dataTransfer.files); }}
      >
        {files.length
          ? <span><strong>{files.length}</strong> Bild(er) ausgewählt &nbsp;(klicken zum Wechseln)</span>
          : <span>Mehrere Röntgenbilder hier ablegen oder <u>klicken</u></span>
        }
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          multiple
          style={{ display: "none" }}
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>

      {files.length > 0 && (
        <button style={btnStyle} onClick={runBatch} disabled={loading}>
          {loading
            ? <><span style={spinnerStyle} />Analysiere {files.length} Bilder…</>
            : `${files.length} Bilder analysieren`
          }
        </button>
      )}

      {error && <div style={errorStyle}>{error}</div>}

      {summary && (
        <div style={{ display: "flex", gap: "1rem", marginTop: "1.5rem", marginBottom: "1rem", flexWrap: "wrap" }}>
          {[
            { label: "Gesamt",    value: summary.total,    color: "#2d3748" },
            { label: "Frakturen", value: summary.fracture, color: "#c53030" },
            { label: "Normal",    value: summary.normal,   color: "#276749" },
          ].map(({ label, value, color }) => (
            <div key={label} style={{
              background: "#fff", border: "1px solid #e2e8f0", borderRadius: 8,
              padding: "0.6rem 1.2rem", textAlign: "center",
            }}>
              <div style={{ fontSize: "1.5rem", fontWeight: 700, color }}>{value}</div>
              <div style={{ fontSize: "0.78rem", color: "#718096" }}>{label}</div>
            </div>
          ))}
        </div>
      )}

      <BatchResultsTable results={results} />
    </div>
  );
}
