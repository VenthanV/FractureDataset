/** ±5% around threshold → warn radiologist */
export const UNCERTAINTY_MARGIN = 0.05;

export const DEFAULT_THRESHOLD = 0.5;

/**
 * Returns true when the prediction probability is too close to the threshold
 * to be considered a reliable classification.
 */
export function isUncertain(probability, threshold) {
  return Math.abs(probability - threshold) <= UNCERTAINTY_MARGIN;
}

/**
 * Extracts a human-readable error message from an axios error or plain Error.
 */
export function getErrorMessage(err) {
  return err?.response?.data?.detail || err?.message || "Unbekannter Fehler";
}
