import axios from "axios";

const BASE = "http://localhost:8000";

/**
 * POST /predict — single image inference.
 * @param {File}   file
 * @param {number} threshold  0.0 – 1.0
 */
export async function predictSingle(file, threshold = 0.5) {
  const form = new FormData();
  form.append("file", file);
  form.append("threshold", String(threshold));
  const { data } = await axios.post(`${BASE}/predict`, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

/**
 * POST /predict/batch — multiple image inference.
 * @param {File[]} files
 * @param {number} threshold  0.0 – 1.0
 */
export async function predictBatch(files, threshold = 0.5) {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  form.append("threshold", String(threshold));
  const { data } = await axios.post(`${BASE}/predict/batch`, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

/**
 * GET /model/stats — returns eval_results.json content.
 */
export async function getModelStats() {
  const { data } = await axios.get(`${BASE}/model/stats`);
  return data;
}
