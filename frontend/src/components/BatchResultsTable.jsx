import React, { useState } from "react";
import { isUncertain } from "../utils/prediction.js";

const tableStyle = {
  width: "100%",
  borderCollapse: "collapse",
  fontSize: "0.88rem",
};

const thStyle = {
  padding: "0.6rem 0.75rem",
  textAlign: "left",
  background: "#edf2f7",
  borderBottom: "2px solid #cbd5e0",
  cursor: "pointer",
  userSelect: "none",
  fontWeight: 600,
  color: "#2d3748",
};

const tdStyle = {
  padding: "0.55rem 0.75rem",
  borderBottom: "1px solid #e2e8f0",
};

const badgeSmall = (isFracture) => ({
  display: "inline-block",
  padding: "0.15rem 0.55rem",
  borderRadius: 12,
  fontSize: "0.78rem",
  fontWeight: 700,
  background: isFracture ? "#fed7d7" : "#c6f6d5",
  color: isFracture ? "#c53030" : "#276749",
});

const exportBtnStyle = {
  marginTop: "0.75rem",
  padding: "0.45rem 1rem",
  background: "#3182ce",
  color: "#fff",
  border: "none",
  borderRadius: 6,
  cursor: "pointer",
  fontSize: "0.85rem",
  fontWeight: 600,
};

function downloadCSV(rows) {
  const header = ["Dateiname", "Label", "P(Fraktur)", "Schwellenwert"];
  const lines  = rows.map((r) => [
    r.filename,
    r.label,
    r.probability.toFixed(4),
    r.threshold_used.toFixed(2),
  ]);
  const csv = [header, ...lines].map((row) => row.join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href     = url;
  a.download = "fracture_results.csv";
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * BatchResultsTable — sortable results table with CSV export.
 *
 * Props:
 *   results {PredictionResponse[]}
 */
export default function BatchResultsTable({ results }) {
  const [sortKey, setSortKey]   = useState("probability");
  const [sortAsc, setSortAsc]   = useState(false);

  if (!results || results.length === 0) return null;

  function handleSort(key) {
    if (sortKey === key) {
      setSortAsc((a) => !a);
    } else {
      setSortKey(key);
      setSortAsc(false);
    }
  }

  const sorted = [...results].sort((a, b) => {
    const av = a[sortKey];
    const bv = b[sortKey];
    if (typeof av === "number") return sortAsc ? av - bv : bv - av;
    return sortAsc
      ? String(av).localeCompare(String(bv))
      : String(bv).localeCompare(String(av));
  });

  const arrow = (key) =>
    sortKey === key ? (sortAsc ? " ▲" : " ▼") : "";

  return (
    <div>
      <div style={{ overflowX: "auto" }}>
        <table style={tableStyle}>
          <thead>
            <tr>
              <th style={thStyle} onClick={() => handleSort("filename")}>
                Dateiname{arrow("filename")}
              </th>
              <th style={thStyle} onClick={() => handleSort("label")}>
                Befund{arrow("label")}
              </th>
              <th style={thStyle} onClick={() => handleSort("probability")}>
                P(Fraktur){arrow("probability")}
              </th>
              <th style={thStyle} onClick={() => handleSort("threshold_used")}>
                Schwellenwert{arrow("threshold_used")}
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((r, i) => {
              const isFracture  = r.label === "fracture";
              const uncertain = isUncertain(r.probability, r.threshold_used);
              return (
                <tr key={i} style={{ background: uncertain ? "#fffdf0" : i % 2 === 0 ? "#fff" : "#f7fafc" }}>
                  <td style={tdStyle}>{r.filename}</td>
                  <td style={tdStyle}>
                    <span style={badgeSmall(isFracture)}>
                      {isFracture ? "FRAKTUR" : "NORMAL"}
                    </span>
                  </td>
                  <td style={tdStyle}>
                    {(r.probability * 100).toFixed(1)}%
                    {uncertain && (
                      <span title="Unsichere Vorhersage — manuelle Überprüfung empfohlen"
                        style={{ marginLeft: 6, cursor: "help" }}>⚠</span>
                    )}
                  </td>
                  <td style={tdStyle}>{r.threshold_used.toFixed(2)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <button style={exportBtnStyle} onClick={() => downloadCSV(results)}>
        CSV exportieren
      </button>
    </div>
  );
}
